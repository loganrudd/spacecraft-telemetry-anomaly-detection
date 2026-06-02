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
from threading import Lock, Thread
from typing import TYPE_CHECKING, Any

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
            try:
                import google.auth.transport.requests
                import google.oauth2.id_token

                req = google.auth.transport.requests.Request()
                token = google.oauth2.id_token.fetch_id_token(req, tracking_uri)  # type: ignore[no-untyped-call]
                _token_cache[tracking_uri] = (token, now + 50 * 60)
            except Exception:
                return

    os.environ["MLFLOW_TRACKING_TOKEN"] = token


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
        except Exception as exc:  # noqa: BLE001
            _exc.append(exc)

    t = Thread(target=_set_experiment, daemon=True)
    t.start()
    t.join(timeout=30)

    if t.is_alive():
        log.warning("mlflow.run.start_failed", experiment=experiment, error="set_experiment timeout after 30s")
        yield None
        return
    if _exc:
        log.warning("mlflow.run.start_failed", experiment=experiment, error=str(_exc[0]))
        yield None
        return

    _run: Any = None
    try:
        _run = mlflow.start_run(run_name=run_name, tags=tags, nested=nested)
    except Exception as exc:  # noqa: BLE001
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
