"""Generic MLflow run-logging helpers — model-type-agnostic.

All functions are safe to call when MLflow is unavailable:
  - configure_mlflow: sets the global tracking URI from Settings.
  - open_run: context manager that yields an ActiveRun or None on failure.
    Training / scoring continue normally either way; filesystem artifacts
    (via model/io.py) are always written regardless.
  - log_params / log_metrics_step / log_metrics_final / log_artifact_bytes:
    no-ops when there is no active run.

Nothing in this module knows what Telemanom is.  Callers pass generic
params / metrics dicts and receive run objects they can use for registry ops.
"""

from __future__ import annotations

import os
import tempfile
import time
from collections.abc import Generator
from contextlib import contextmanager, suppress
from pathlib import Path
from threading import Event, Lock, Thread
from typing import TYPE_CHECKING, Any, cast

import mlflow
import mlflow.exceptions

from spacecraft_telemetry.core.config import Settings
from spacecraft_telemetry.core.logging import get_logger

if TYPE_CHECKING:
    pass

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# GCP ID-token auth for Cloud Run MLflow backend
# ---------------------------------------------------------------------------

_token_cache: dict[str, tuple[str, float]] = {}  # uri -> (token, expires_at)
_token_lock = Lock()


def _fetch_id_token(audience: str) -> str | None:
    """Fetch a GCP ID token for ``audience``, returning None on failure.

    Two code paths, tried in order:

    1. ``google.oauth2.id_token.fetch_id_token`` — works from GCE/GKE via the
       metadata server, and from local dev when ADC points to a **service
       account** key (GOOGLE_APPLICATION_CREDENTIALS). Produces a token whose
       ``aud`` claim equals ``audience``.

    2. ``gcloud auth print-identity-token`` subprocess — fallback for local dev
       with **user credentials** (``gcloud auth login`` / ``gcloud auth
       application-default login``). ``fetch_id_token`` raises an exception for
       user ADC because there is no private key to sign a JWT assertion.
       ``gcloud`` returns a user OIDC token whose ``aud`` is not the service URL,
       but Cloud Run accepts it as long as the account holds
       ``roles/run.invoker`` on the service.

    The gcloud fallback is intentionally skipped on GKE/GCE where path 1 always
    succeeds — ``gcloud`` may not be installed there, and subprocess overhead
    would be unnecessary.
    """
    # Path 1: google-auth library (service accounts + metadata server).
    try:
        import google.auth.transport.requests
        import google.oauth2.id_token

        req = google.auth.transport.requests.Request()
        token = cast(str, google.oauth2.id_token.fetch_id_token(req, audience))  # type: ignore[no-untyped-call]
        return token
    except Exception as exc1:
        exc1_str = str(exc1)  # save before Python clears `exc1` at except-block exit
        log.debug("mlflow.auth.fetch_id_token_failed", audience=audience, error=exc1_str)

    # Path 2: gcloud subprocess — local dev with user credentials.
    # gcloud print-identity-token does not support --audiences for user accounts;
    # the token's aud is the Google accounts endpoint, not the service URL.
    # Cloud Run still accepts it when the caller has roles/run.invoker.
    try:
        import subprocess

        result = subprocess.run(
            ["gcloud", "auth", "print-identity-token"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        token = result.stdout.strip()
        if result.returncode != 0 or not token:
            raise RuntimeError(result.stderr.strip() or "empty token")
        return token
    except Exception as exc2:
        log.warning(
            "mlflow.auth.id_token_failed",
            audience=audience,
            error=f"fetch_id_token: {exc1_str}; gcloud fallback: {exc2}",
        )
        return None


def _install_id_token_auth(tracking_uri: str) -> None:
    """Set MLFLOW_TRACKING_TOKEN to a fresh GCP ID token for *.run.app URIs.

    The MLflow Cloud Run service is private/authenticated; callers must
    include a GCP ID token in the Authorization header. Setting
    MLFLOW_TRACKING_TOKEN is the standard MLflow mechanism — the HTTP client
    reads it on every request.

    Works transparently from:
    - Cloud Run api service (sa-api via attached service account)
    - GKE Ray pods (sa-ray via Workload Identity metadata server)
    - Local dev with `gcloud auth application-default login` (for testing)

    Token is cached for 50 min (ID tokens expire after 60 min; 10-min buffer
    avoids requests failing at the edge of token validity).  Silently skips
    when google-auth is absent or ADC is unavailable — local SQLite dev never
    hits this path.
    """
    from urllib.parse import urlparse

    if not (urlparse(tracking_uri).hostname or "").endswith(".run.app"):
        return

    with _token_lock:
        cached = _token_cache.get(tracking_uri)
        now = time.monotonic()
        if cached and now < cached[1]:
            token = cached[0]
        else:
            # Audience must be the Cloud Run service base URL (scheme://host),
            # with no path/query — fetch_id_token validates the token's `aud`
            # claim against exactly this string, and Cloud Run issues tokens
            # scoped to the bare service URL.
            parsed = urlparse(tracking_uri)
            audience = f"{parsed.scheme}://{parsed.netloc}"
            fetched = _fetch_id_token(audience)
            if fetched is None:
                return
            token = fetched
            _token_cache[tracking_uri] = (token, now + 50 * 60)

    os.environ["MLFLOW_TRACKING_TOKEN"] = token


def refresh_mlflow_auth() -> None:
    """Refresh the GCP ID token if it is within the expiry buffer.

    Call once per training epoch before log_metrics_step to ensure the token
    in MLFLOW_TRACKING_TOKEN stays valid for the full duration of long training
    runs on Cloud Run.  On the common path (token has ≥10 min remaining) this
    is a single dict lookup and returns immediately.  Near expiry it refetches
    and updates the env var before the next MLflow HTTP request goes out.

    No-op when not targeting a Cloud Run MLflow backend (local SQLite, etc.).
    """
    _install_id_token_auth(mlflow.get_tracking_uri())


@contextmanager
def keep_mlflow_auth_fresh(
    interval_seconds: float = 300.0,
) -> Generator[None, None, None]:
    """Refresh MLFLOW_TRACKING_TOKEN in a background thread for the wrapped block.

    Long-running *driver-side* MLflow work — notably a Ray Tune sweep, where
    ``MLflowLoggerCallback`` logs every trial from inside a single blocking
    ``tuner.fit()`` — makes HTTP requests to the Cloud Run backend continuously
    but exposes no per-iteration hook to refresh the GCP ID token (unlike the
    training epoch loop, which calls ``refresh_mlflow_auth`` directly). Without a
    refresh the token in MLFLOW_TRACKING_TOKEN real-expires at 60 min and the
    tail of any sweep longer than that 401s.

    This context manager runs a daemon thread that calls ``refresh_mlflow_auth``
    every ``interval_seconds`` for the duration of the block. MLFLOW_TRACKING_TOKEN
    is process-global, so the refresh is visible to every thread (the parallel
    per-subsystem sweeps share it) and to MLflow's HTTP client. Refresh exceptions
    are suppressed — a transient metadata-server hiccup must not abort the sweep;
    the next tick retries.

    Cheap when not targeting Cloud Run: ``refresh_mlflow_auth`` early-returns for
    non-``.run.app`` URIs, so the thread just sleeps.
    """
    stop = Event()

    def _loop() -> None:
        # Event.wait returns True once `stop` is set and False on timeout, so
        # looping while it returns False is a cleanly cancellable interval timer
        # (no busy-wait, wakes immediately on stop instead of after a full tick).
        while not stop.wait(interval_seconds):
            with suppress(Exception):
                refresh_mlflow_auth()

    thread = Thread(target=_loop, name="mlflow-auth-refresh", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=5)


def configure_mlflow(settings: Settings) -> None:
    """Apply tracking_uri and registry_uri from Settings to the global MLflow client.

    **Invariant:** one process, one tracking URI.  Under Ray's worker-reuse
    model (a worker process handles multiple tasks sequentially), all tasks
    on the same worker must use the same backend so that runs don't silently
    land in different databases.  Concurrent tasks from different missions
    with different URIs would violate this — the second task's
    ``configure_mlflow`` overwrites the first's global state.

    A warning is emitted if this function is called with a URI that differs
    from whatever MLflow currently has configured.  This makes the bug
    observable without raising (the training loop must never abort due to
    a tracking misconfiguration).

    For the correct invariant, pass a single resolved absolute URI (e.g.
    ``sqlite:////abs/path/mlflow.db``) to all tasks in a sweep.  The
    ``_with_abs_paths`` helper in ``runner.py`` ensures this for Ray tasks.
    """
    uri = settings.mlflow.tracking_uri
    _current = mlflow.get_tracking_uri()
    # Warn when switching away from a previously configured non-default URI.
    # Suppressed only for the "file://" default path — the first call in a
    # fresh process before any URI has been explicitly set.
    # If this warning fires, an entry point is missing configure_mlflow(settings)
    # before its first Evidently or Ray call — fix the calling order rather than
    # adding suppression here.
    if _current and not _current.startswith("file://") and _current != uri:
        log.warning(
            "mlflow.configure.uri_changed",
            previous=_current,
            new=uri,
            note=(
                "Two distinct tracking URIs in one process. "
                "Under Ray worker-reuse, runs from different tasks may land "
                "in different databases. Ensure all tasks share one URI."
            ),
        )
    mlflow.set_tracking_uri(uri)
    if settings.mlflow.registry_uri is not None:
        mlflow.set_registry_uri(settings.mlflow.registry_uri)
    _install_id_token_auth(uri)


@contextmanager
def open_run(
    *,
    experiment: str,
    run_name: str,
    tags: dict[str, str],
    nested: bool = False,
) -> Generator[Any, None, None]:
    """Context manager that opens an MLflow run and yields it, or yields None on failure.

    On any MLflow connectivity or configuration error, a structured warning is
    emitted and None is yielded.  The caller's body executes in both cases —
    filesystem artifacts (via model/io.py) are always produced regardless of
    MLflow availability.

    Callers that need the run_id (e.g. for registry ops) must guard with
    ``if run is not None:``.  The log_* helpers in this module are no-ops when
    there is no active run and do not require that guard.

    Args:
        experiment: MLflow experiment name.  Created if it does not exist.
        run_name:   Human-readable name for this run (e.g. channel_id).
        tags:       Tag dict produced by common_tags() from conventions.py.
        nested:     True when this run is a child of an already-active run.

    Yields:
        mlflow.ActiveRun or None.
    """
    # The thread is used only for set_experiment — the network call most likely
    # to hang on a cold Cloud Run backend.  start_run / end_run must run on the
    # calling thread: MLflow 3.x stores the active-run stack in a ThreadLocal,
    # so a run started on a daemon thread is invisible to log_* helpers called
    # from the main thread.
    _exc: list[Exception] = []

    def _set_experiment() -> None:
        try:
            mlflow.set_experiment(experiment)
        except Exception as exc:
            _exc.append(exc)

    t = Thread(target=_set_experiment, daemon=True)
    t.start()
    t.join(timeout=30)

    if t.is_alive():
        log.warning(
            "mlflow.run.start_failed",
            experiment=experiment,
            error="set_experiment timeout after 30s",
        )
        yield None
        return
    if _exc:
        log.warning("mlflow.run.start_failed", experiment=experiment, error=str(_exc[0]))
        yield None
        return

    _run: Any = None
    try:
        _run = mlflow.start_run(run_name=run_name, tags=tags, nested=nested)
    except Exception as exc:
        log.warning("mlflow.run.start_failed", experiment=experiment, error=str(exc))

    try:
        yield _run
    finally:
        if _run is not None:
            with suppress(mlflow.exceptions.MlflowException, OSError, ConnectionError):
                mlflow.end_run()


def log_params(params: dict[str, Any]) -> None:
    """Log a params dict to the active run; no-op when there is no active run."""
    if mlflow.active_run() is not None:
        mlflow.log_params(params)


def log_metrics_step(metrics: dict[str, float], step: int) -> None:
    """Log metrics for a training step (e.g. per-epoch) to the active run."""
    if mlflow.active_run() is not None:
        mlflow.log_metrics(metrics, step=step)


def log_metrics_final(metrics: dict[str, float | int]) -> None:
    """Log summary metrics (no step) to the active run.

    Accepts both float and int values; casts to float before logging so
    callers (training, scoring, monitoring) don't need to know about MLflow's
    type requirements.
    """
    if mlflow.active_run() is not None:
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})


def log_dict(data: Any, artifact_file: str) -> None:
    """Log a dict or list as a JSON/YAML artifact in the active run.

    The format is inferred from the file extension (.json or .yaml/.yml).
    No-op when there is no active run.

    Args:
        data:          JSON-serialisable Python dict or list.
        artifact_file: Destination filename within the run's artifact root,
                       e.g. "train_log.json" or "normalization_params.json".
    """
    if mlflow.active_run() is not None:
        mlflow.log_dict(data, artifact_file)


def log_artifact_bytes(data: bytes, artifact_file: str) -> None:
    """Write bytes as a named artifact in the active run.

    The artifact appears at ``artifact_file`` relative to the run's artifact
    root.  No-op when there is no active run.

    Args:
        data:          Raw bytes to write.
        artifact_file: Destination path within the artifact store,
                       e.g. "configs/model_config.json".
    """
    if mlflow.active_run() is None:
        return
    artifact_path_obj = Path(artifact_file)
    parent = str(artifact_path_obj.parent)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_file = Path(tmp_dir) / artifact_path_obj.name
        tmp_file.write_bytes(data)
        mlflow.log_artifact(str(tmp_file), artifact_path=parent if parent != "." else None)


def log_input_dataset(
    source: str,
    name: str,
    digest: str | None,
    context: str,
) -> None:
    """Log a Parquet partition as a run input dataset (Dataset column in the UI).

    Records the source path and pre-computed content digest as run metadata.
    No data is read from disk; only the path string and hash are stored.
    No-op when there is no active run, or when ``digest`` is None (skipping
    avoids recording a meaningless constant hash derived from an empty frame).

    Args:
        source:  Local path or GCS URI to the partition directory.
        name:    Human-readable name shown in the MLflow UI Dataset column,
                 e.g. ``"ESA-Mission1-channel_1-train"``.
        digest:  Pre-computed hex fingerprint from ``partition_hash`` or
                 ``training_data_hash``.  When None, the call is silently
                 skipped so that a missing partition never aborts training.
        context: Semantic role of this dataset in the run, e.g. ``"training"``
                 or ``"evaluation"``.
    """
    if mlflow.active_run() is None or digest is None:
        return
    try:
        import pandas as pd
        from mlflow.data.dataset_source_registry import resolve_dataset_source
        from mlflow.data.pandas_dataset import PandasDataset

        # MLflow's dataset digest column is capped at 36 characters (UUID width).
        # The SHA-256 hex digests produced by partition_hash are 64 chars; the
        # first 36 chars are 144 bits of collision resistance — sufficient for
        # data-identity purposes.
        _digest = digest[:36]

        # Read only the Parquet footers (schema + row-count + per-column min/max
        # statistics) — never the actual row data — so this is fast even for
        # 200 MB partitions and works against gs:// URIs via to_upath.
        schema, num_rows, start_date, end_date = _parquet_stats(source)

        # An empty, schema-correct frame carries the column names/dtypes for the
        # dataset's schema; the row count and date range are reported via the
        # overridden profile below (we never materialise the actual rows). When
        # the footer can't be read (schema None), fall back to an empty frame so
        # the dataset still logs with num_rows=0 rather than skipping lineage.
        empty_df = schema.empty_table().to_pandas() if schema is not None else pd.DataFrame()

        profile: dict[str, Any] = {"num_rows": num_rows}
        if start_date is not None:
            profile["start_date"] = start_date
        if end_date is not None:
            profile["end_date"] = end_date

        class _DatedPandasDataset(PandasDataset):
            """PandasDataset whose profile reports row count + slice date range.

            The stock PandasDataset.profile is hardcoded to
            ``{"num_rows", "num_elements"}`` and computes both from the in-memory
            DataFrame — which would be 0 here since we deliberately never load the
            rows. Overriding profile lets us report the true (footer-derived) row
            count plus the start/end timestamps of the slice. num_elements is
            dropped: it is just num_rows x num_columns and adds no lineage signal.
            """

            @property
            def profile(self) -> dict[str, Any]:
                return profile

        dataset = _DatedPandasDataset(
            df=empty_df,
            source=resolve_dataset_source(source),
            name=name,
            digest=_digest,
        )
        mlflow.log_input(dataset, context=context)
    except Exception as exc:
        log.warning(
            "mlflow.dataset.log_failed", name=name, context=context, error=str(exc)
        )


def _parquet_stats(
    partition_dir: str,
) -> tuple[Any, int, str | None, str | None]:
    """Return (schema, num_rows, start_date, end_date) for a Parquet partition.

    Reads only Parquet file footers (schema, row counts, and the
    ``telemetry_timestamp`` column's min/max statistics) — never actual row
    data — so this is fast even for 200 MB partitions.  Works against both local
    paths and ``gs://`` URIs via ``to_upath`` + fsspec file handles.

    Dates are returned as ISO-8601 strings (JSON-serialisable for the MLflow
    profile).  Returns ``(None, 0, None, None)`` when the path does not exist,
    contains no Parquet files, or cannot be read.
    """
    try:
        import pyarrow.parquet as pq

        from spacecraft_telemetry.core.paths import to_upath

        part_dir = to_upath(partition_dir)
        if not part_dir.exists():
            return None, 0, None, None
        files = sorted(part_dir.glob("**/*.parquet"))
        if not files:
            return None, 0, None, None

        with files[0].open("rb") as fh:
            schema = pq.read_schema(fh)

        num_rows = 0
        ts_min: Any = None
        ts_max: Any = None
        ts_idx = (
            schema.names.index("telemetry_timestamp")
            if "telemetry_timestamp" in schema.names
            else None
        )

        for f in files:
            with f.open("rb") as fh:
                md = pq.read_metadata(fh)
            num_rows += md.num_rows
            if ts_idx is None:
                continue
            for rg in range(md.num_row_groups):
                stats = md.row_group(rg).column(ts_idx).statistics
                if stats is None:
                    continue
                if ts_min is None or stats.min < ts_min:
                    ts_min = stats.min
                if ts_max is None or stats.max > ts_max:
                    ts_max = stats.max

        start_date = ts_min.isoformat() if ts_min is not None else None
        end_date = ts_max.isoformat() if ts_max is not None else None
        return schema, num_rows, start_date, end_date
    except Exception:
        return None, 0, None, None
