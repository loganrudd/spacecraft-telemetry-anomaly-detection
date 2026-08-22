"""Artifact I/O for Telemanom model files — MLflow-backed.

After the A1 pivot (Phase 6 review), MLflow is the single source of truth for
all model artifacts.  All writes go through MLflow logging APIs inside
training.py and scoring.py.  This module provides the read-path helpers:

- load_model_for_scoring: load the latest registered PyTorch model + window_size.
- download_artifact_bytes: fetch a named artifact from a specific run.
- find_latest_run_for_channel: locate the most recent run for a channel in an experiment.
- find_latest_run_by_tag: same, generalised to any tag (e.g. "subsystem").
- errors_to_bytes / threshold_to_bytes: serialise numpy arrays for log_artifact_bytes.
- bytes_to_errors: deserialise errors bytes back to a numpy array.

Phase 4 note: _write_bytes / _read_bytes no longer exist.  The `gs://`
indirection is handled by the MLflow artifact store — configure
MLFLOW_ARTIFACTS_DESTINATION to a `gs://` bucket for cloud runs.
"""

from __future__ import annotations

import io
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

    from spacecraft_telemetry.model.architecture import TelemanomLSTM


class ModelNotFoundError(Exception):
    """No registered model version exists for a channel.

    Distinct from a genuine scoring failure: scoring a channel that was never
    trained is an *expected* outcome of partial training (e.g. a smoke test
    that trained 3 of 62 channels). Callers map this to status="skipped"
    rather than status="error" so true failures stay visible.
    """


# ---------------------------------------------------------------------------
# Array serialisation helpers (called by scoring.py / tune.py)
# ---------------------------------------------------------------------------


def errors_to_bytes(errors: Any) -> bytes:
    """Serialise a numpy array (smoothed errors) to raw .npy bytes.

    Keeps np.save() out of scoring.py so the discipline check passes.
    """
    import numpy as np

    buf = io.BytesIO()
    np.save(buf, errors)
    return buf.getvalue()


def threshold_to_bytes(threshold: Any) -> bytes:
    """Serialise a numpy threshold array to raw .npy bytes."""
    import numpy as np

    buf = io.BytesIO()
    np.save(buf, threshold)
    return buf.getvalue()


def bytes_to_errors(data: bytes) -> Any:
    """Deserialise raw .npy bytes back to a numpy array."""
    import numpy as np

    return np.load(io.BytesIO(data))


# ---------------------------------------------------------------------------
# Scoring-artifact layout — ONE definition (docs/reviews/021, item B1)
# ---------------------------------------------------------------------------
#
# .claude/rules/pytorch.md mandates that artifact I/O funnel through model.io
# so a layout change is a one-file change. These two path strings escaped that
# rule and were spelled independently in three modules (scoring.py writes them,
# esa_adb/detections.py and ray_fanout/tune.py read them). The failure mode is
# not a clean error: detections.py falls into its `except RuntimeError` branch
# and reports "searched for a multivariate scoring run and found none", which
# points at run discovery when only the naming moved.


def errors_artifact(channel: str | None = None) -> str:
    """Artifact path of a scoring run's smoothed-error array.

    ``channel=None`` is the univariate layout: one array at the run root.
    A channel name gives the multivariate layout, where one run covers a whole
    subsystem group and each member's array is stored separately.
    """
    return "errors.npy" if channel is None else f"errors/{channel}.npy"


def threshold_artifact(channel: str | None = None) -> str:
    """Artifact path of a scoring run's dynamic-threshold array.

    Same univariate/multivariate split as :func:`errors_artifact`; the two
    always move together, which is the reason they live side by side here.
    """
    return "threshold.npy" if channel is None else f"threshold/{channel}.npy"


# ---------------------------------------------------------------------------
# MLflow run lookup
# ---------------------------------------------------------------------------


def find_latest_run_by_tag(
    experiment_name: str,
    tag_name: str,
    tag_value: str,
    tracking_uri: str,
    extra_filter: str | None = None,
) -> Any:
    """Return the most recent MLflow run matching one tag in an experiment, or None.

    General form of find_latest_run_for_channel, which is channel_id-specific
    only because it predates this generalisation and is the overwhelmingly
    common case; that function is now a thin wrapper around this one so its
    existing call sites and mocks are unaffected. Added for the multivariate
    lookup (docs/plans/021-multivariate-telemanom.md): a multivariate scoring
    run carries a ``subsystem`` tag instead of ``channel_id`` (see
    model/scoring.py — it isn't a real channel), so ray_fanout/tune.py needs
    to search on that tag instead.

    Args:
        experiment_name: MLflow experiment name to search within.
        tag_name:        Tag key to filter by, e.g. "channel_id", "subsystem".
        tag_value:        Tag value to match.
        tracking_uri:     MLflow tracking server URI.
        extra_filter:     Optional additional MLflow filter clause, ANDed onto
                          the tag filter (e.g. "tags.data_source = 'nominal'").

    Returns:
        An ``mlflow.entities.Run`` or ``None`` if no matching run is found.
    """
    import mlflow

    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        return None
    filter_string = f"tags.{tag_name} = '{tag_value}'"
    if extra_filter:
        filter_string += f" and {extra_filter}"
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string=filter_string,
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    return runs[0] if runs else None


def find_latest_run_for_channel(
    experiment_name: str,
    channel: str,
    tracking_uri: str,
    extra_filter: str | None = None,
) -> Any:
    """Return the most recent MLflow run for a channel in an experiment, or None.

    Args:
        experiment_name: MLflow experiment name to search within.
        channel:         Channel ID to filter by (matched against tags.channel_id).
        tracking_uri:    MLflow tracking server URI.
        extra_filter:    Optional additional MLflow filter clause, ANDed onto the
                          channel filter (e.g. "tags.data_source = 'nominal'" to
                          distinguish a baseline scoring run from an injected one —
                          see ray_fanout/tune.py's nominal false-positive penalty).

    Returns:
        An ``mlflow.entities.Run`` or ``None`` if no matching run is found.
    """
    return find_latest_run_by_tag(
        experiment_name, "channel_id", channel, tracking_uri, extra_filter
    )


# ---------------------------------------------------------------------------
# Artifact download
# ---------------------------------------------------------------------------


def read_artifact_bytes(path: str) -> bytes:
    """Read a staged artifact's raw bytes from a local path or ``gs://`` URI.

    The tracking-server-free counterpart to download_artifact_bytes: used when
    scoring artifacts have been staged out of the MLflow artifact store and the
    tracking backend is unavailable (see esa_adb/offline.py). Kept in this
    module so all artifact byte reads stay funnelled through model.io.

    Raises:
        FileNotFoundError: If the path does not exist.
    """
    from spacecraft_telemetry.core.paths import to_upath

    p = to_upath(path)
    if not p.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    return p.read_bytes()


# docs/plans/022, stage 022.3: MLflow run artifacts are immutable once
# written, so (run_id, artifact_path) is a genuinely content-stable cache
# key. Lives here — not in scripts/threshold_ceiling.py — because this
# function is the single funnel every error-array read passes through
# (.claude/rules/pytorch.md names model.io as *the* I/O indirection point);
# caching anywhere else would leave other callers uncached.
_DEFAULT_ARTIFACT_CACHE_ROOT = Path(".cache/artifacts")


def _artifact_cache_enabled() -> bool:
    """SPACECRAFT_ARTIFACT_CACHE=0 is the environment-variable escape hatch
    (alongside download_artifact_bytes' use_cache=False parameter) — see its
    docstring."""
    import os

    return os.environ.get("SPACECRAFT_ARTIFACT_CACHE", "1") != "0"


def download_artifact_bytes(
    run_id: str,
    artifact_path: str,
    tracking_uri: str,
    *,
    use_cache: bool = True,
    cache_dir: str | Path | None = None,
) -> bytes:
    """Download a named artifact from an MLflow run and return its raw bytes.

    Reads directly from the run's artifact store when it resolves to a
    ``gs://`` URI (``run.info.artifact_uri`` — the cloud MLflow server logs
    this as the real bucket path, not a proxied ``mlflow-artifacts:`` scheme),
    bypassing ``client.download_artifacts()``, which streams bytes through the
    MLflow tracking server. Measured: errors.npy (~60MB) took 33-49s through
    the proxy (~0.5-0.7 MB/s — a 1 vCPU Cloud Run instance is not built to
    stream large binaries) vs. a direct GCS read.

    Falls back to the tracking-server proxy for any other artifact store
    scheme (e.g. local ``file://`` runs used in tests, or a genuinely proxied
    store) — same behaviour as before this optimisation.

    A local cache sits in front of the network fetch, keyed on
    ``(run_id, artifact_path)`` under ``cache_dir``
    (``.cache/artifacts/{run_id}/{artifact_path}`` by default, gitignored).
    Read-path only: a cache hit returns exactly the bytes a prior fetch wrote,
    so no computed number can change — see scripts/threshold_ceiling.py's
    byte-identity gate (docs/plans/022, stage 022.3).

    The write is atomic (write to a ``.tmp`` sibling, then ``os.replace`` —
    atomic within a filesystem): a crash or Ctrl-C mid-write leaves only an
    orphaned ``.tmp`` file, never a truncated file at the real cache path, so
    a subsequent run sees a clean cache miss rather than silently trusting
    partial bytes forever (docs/reviews/022, item T3) — this cache feeds
    published F0.5 numbers, so a corrupt-but-present entry would be a
    silent-wrong-answer path, not just a slow one.

    Args:
        run_id:        MLflow run ID.
        artifact_path: Path within the run's artifact store, e.g. "errors.npy".
        tracking_uri:  MLflow tracking server URI.
        use_cache:     Check/populate the local cache. Set False (or the
            ``SPACECRAFT_ARTIFACT_CACHE=0`` environment variable, checked when
            this is True) to bypass it entirely — the escape hatch for
            re-validating a suspected-poisoned cache.
        cache_dir:     Cache root directory. Defaults to
            :data:`_DEFAULT_ARTIFACT_CACHE_ROOT` relative to the current
            working directory.

    Returns:
        Raw bytes of the artifact.

    Raises:
        OSError: If the artifact cannot be downloaded.
    """
    enabled = use_cache and _artifact_cache_enabled()
    root = Path(cache_dir) if cache_dir is not None else _DEFAULT_ARTIFACT_CACHE_ROOT
    cached_path = root / run_id / artifact_path

    if enabled and cached_path.exists():
        return cached_path.read_bytes()

    data = _fetch_artifact_bytes(run_id, artifact_path, tracking_uri)

    if enabled:
        cached_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cached_path.with_name(f"{cached_path.name}.tmp-{os.getpid()}")
        tmp_path.write_bytes(data)
        os.replace(tmp_path, cached_path)

    return data


def _fetch_artifact_bytes(run_id: str, artifact_path: str, tracking_uri: str) -> bytes:
    """The uncached network fetch behind download_artifact_bytes' cache."""
    import mlflow

    from spacecraft_telemetry.core.paths import to_upath

    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(run_id)
    artifact_uri = run.info.artifact_uri
    if artifact_uri and artifact_uri.startswith("gs://"):
        return to_upath(f"{artifact_uri.rstrip('/')}/{artifact_path}").read_bytes()

    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = client.download_artifacts(run_id, artifact_path, tmp_dir)
        return Path(local_path).read_bytes()


# ---------------------------------------------------------------------------
# Model load (used by scoring.py and Phase 8 FastAPI)
# ---------------------------------------------------------------------------


def _resolve_model_version(client: Any, name: str, require_champion: bool) -> Any:
    """Pick the ModelVersion that a load should use.

    Extracted so load_model_for_scoring and load_model_contract cannot disagree
    about WHICH version they are describing — a contract read off a different
    version than the weights would be worse than no contract at all.

    Raises:
        ModelNotFoundError: No registered versions exist for ``name``.
        RuntimeError: require_champion=True and no @champion alias is set.
    """
    from mlflow.exceptions import MlflowException

    from spacecraft_telemetry.mlflow_tracking.registry import CHAMPION_ALIAS

    versions = client.search_model_versions(f"name='{name}'")
    if not versions:
        raise ModelNotFoundError(
            f"No registered versions found for model {name!r}. "
            "Run 'model train' for this channel before serving."
        )

    if require_champion:
        try:
            return client.get_model_version_by_alias(name, CHAMPION_ALIAS)
        except MlflowException as err:
            raise RuntimeError(
                f"No @champion alias set for model {name!r}. "
                "Run 'make mlflow-promote MISSION=... CHANNEL=...' before serving."
            ) from err
    return max(versions, key=lambda v: int(v.version))


@dataclass(frozen=True)
class ModelContract:
    """What a registered model version was TRAINED with.

    Scoring must reproduce all three or its errors are meaningless: the
    dataloader's window/target geometry is built from ``window_size`` and
    ``forecast_steps``, and ``channels`` fixes which column of a multivariate
    model's output belongs to which telemetry channel.

    Defaults describe a pre-021 model version faithfully rather than
    defensively: those models genuinely are one-step-ahead univariate
    forecasters, so ``forecast_steps=1`` and ``channels=None`` are the right
    reading of a missing tag, not a fallback that hides one.
    """

    window_size: int
    forecast_steps: int = 1
    channels: tuple[str, ...] | None = None


def load_model_contract(
    name: str,
    tracking_uri: str,
    *,
    require_champion: bool = False,
) -> ModelContract:
    """Read a registered model version's training contract from MLflow.

    Deliberately SEPARATE from load_model_for_scoring rather than widening its
    return type: that function has five callers outside scoring.py (api/app.py
    and three diag scripts) which all unpack the 2-tuple ``(model,
    window_size)``. Widening would drag the deferred serving path and those
    scripts into a review-remediation change; a second function keeps the blast
    radius here.

    Sources, in order of authority — the same dual-sourcing
    load_model_for_scoring already uses for window_size:

    1. The version's associated run's params/tags (written by train_channel).
    2. The model-version tags (written by register_pytorch_model), which
       survive without a run link.

    The cost is one extra registry lookup per scoring task, negligible against
    a forward pass over millions of windows.

    Raises:
        ModelNotFoundError: No registered versions exist for ``name``.
        RuntimeError: window_size cannot be determined (see
            load_model_for_scoring for the remedy).
    """
    import mlflow

    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    mv = _resolve_model_version(client, name, require_champion)
    vtags: dict[str, str] = mv.tags or {}

    run_params: dict[str, str] = {}
    run_tags: dict[str, str] = {}
    if mv.run_id is not None:
        run = client.get_run(mv.run_id)
        run_params = run.data.params
        run_tags = run.data.tags

    window_size = _contract_int(run_params, vtags, "window_size", name, mv.version)
    # 021.7. Absent everywhere on a pre-021.7 version, which IS a single-step
    # model — so 1 is the accurate value, not a guess.
    forecast_steps = int(
        run_params.get("forecast_steps") or vtags.get("forecast_steps") or 1
    )

    # `channels` is a run TAG (not a param) and a version tag — see
    # train_channel's common_tags(extra=...) and _vtags. Absent means
    # univariate, which is why an empty string must not become ("",).
    raw_channels = run_tags.get("channels") or vtags.get("channels")
    channels = (
        tuple(c for c in raw_channels.split(",") if c) if raw_channels else None
    )

    return ModelContract(
        window_size=window_size, forecast_steps=forecast_steps, channels=channels
    )


def _contract_int(
    run_params: dict[str, str],
    version_tags: dict[str, str],
    key: str,
    name: str,
    version: Any,
) -> int:
    """Read a required int from run params, falling back to version tags."""
    raw = run_params.get(key) or version_tags.get(key)
    if raw is None:
        raise RuntimeError(
            f"Cannot determine {key} for model {name!r} version {version}: "
            f"no associated run and no {key!r} version tag. "
            f"Set the tag via: client.set_model_version_tag(name, version, {key!r}, ...)"
        )
    return int(raw)


def load_model_for_scoring(
    name: str,
    device: torch.device,
    tracking_uri: str,
    *,
    require_champion: bool = True,
) -> tuple[TelemanomLSTM, int]:
    """Load a PyTorch model from MLflow for scoring.

    When ``require_champion=True`` (the default, used by the FastAPI serving
    layer), only the version tagged with the ``@champion`` alias is accepted.
    This gates serving behind an explicit promotion step.

    When ``require_champion=False`` (used by the training pipeline's
    score_channel), the most recently registered version is loaded.  This avoids
    a chicken-and-egg problem: you need to score a model to decide whether to
    promote it, but the promotion gate must not block scoring itself.

    After promoting a new version, redeploy the serving layer to pick it up:
    - Cloud: make cloud-deploy
    - Local: restart make serve

    Args:
        name:             Registered model name from registered_model_name().
        device:           Torch device to map the model weights to.
        tracking_uri:     MLflow tracking server URI.
        require_champion: If True, raise unless @champion alias is set.
                          If False, load the latest registered version.

    Returns:
        (model, window_size) — reconstructed TelemanomLSTM and the sequence
        length the model was trained on.

    Raises:
        ModelNotFoundError: If no registered versions exist for ``name`` (an
            expected outcome for untrained channels; callers treat as skipped).
        RuntimeError: When require_champion=True and no @champion alias is set.
    """
    import mlflow
    import mlflow.pytorch as mlflow_pytorch

    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    mv = _resolve_model_version(client, name, require_champion)

    model = mlflow_pytorch.load_model(  # type: ignore[no-untyped-call]
        f"models:/{name}/{mv.version}",
        map_location=device,
    )
    # Prefer run params as the authoritative source; fall back to the version
    # tag written by register_pytorch_model (covers versions registered before
    # the thread-local run-association bug was fixed).
    if mv.run_id is not None:
        run = client.get_run(mv.run_id)
        window_size = int(run.data.params["window_size"])
    elif "window_size" in (mv.tags or {}):
        window_size = int(mv.tags["window_size"])
    else:
        raise RuntimeError(
            f"Cannot determine window_size for model {name!r} version {mv.version}: "
            "no associated run and no 'window_size' version tag. "
            "Set the tag via: client.set_model_version_tag(name, version, 'window_size', '250')"
        )
    return model, window_size


# ---------------------------------------------------------------------------
# Scoring params (used by Phase 8 FastAPI inference engine)
# ---------------------------------------------------------------------------


@dataclass
class ScoringParams:
    """Threshold hyperparameters for online anomaly scoring.

    Loaded from the latest scoring MLflow run for a channel.  Written by
    score_channel() via log_params (see model/scoring.py).
    """

    threshold_window: int
    threshold_z: float
    error_smoothing_window: int
    threshold_min_anomaly_len: int
    # Absolute floor on the smoothed error (ESA-ADB's "pruning"). Pointwise and
    # stateless, so the streaming engine applies it identically to batch
    # scoring — see model.scoring.flag_anomalies. 0.0 = disabled, which is also
    # the behaviour of any scoring run logged before this param existed.
    min_error_value: float = 0.0


def load_scoring_params(
    channel: str,
    mission: str,
    tracking_uri: str,
    model_type: str = "telemanom",
) -> ScoringParams:
    """Fetch the four threshold hyperparameters from the latest scoring run.

    These params are written by score_channel() via log_params (see
    src/spacecraft_telemetry/model/scoring.py).

    Args:
        channel:      Channel ID to look up (e.g. "channel_1").
        mission:      Mission ID (e.g. "ESA-Mission1").
        tracking_uri: MLflow tracking server URI.
        model_type:   Model type prefix used in experiment naming (default "telemanom").

    Returns:
        ScoringParams dataclass with the four threshold hyperparameters.

    Raises:
        RuntimeError: If no scoring run exists for the channel, or required
            params are missing (model trained but never scored).
    """
    from spacecraft_telemetry.mlflow_tracking.conventions import (
        experiment_name as _exp_name,
    )

    scoring_exp = _exp_name(model_type, "scoring", mission)
    run = find_latest_run_for_channel(scoring_exp, channel, tracking_uri)
    if run is None:
        raise RuntimeError(
            f"No scoring run found for channel {channel!r} in mission {mission!r}. "
            "Run `spacecraft-telemetry ray score` for this channel first."
        )
    p = run.data.params
    try:
        return ScoringParams(
            threshold_window=int(p["threshold_window"]),
            threshold_z=float(p["threshold_z"]),
            error_smoothing_window=int(p["error_smoothing_window"]),
            threshold_min_anomaly_len=int(p["threshold_min_anomaly_len"]),
            # Optional via .get(): scoring runs predating the absolute error
            # floor have no such param, and 0.0 reproduces their behaviour.
            min_error_value=float(p.get("min_error_value", 0.0)),
        )
    except KeyError as exc:
        raise RuntimeError(
            f"Scoring run {run.info.run_id!r} for channel {channel!r} is missing "
            f"param {exc.args[0]!r}. "
            "Re-run `spacecraft-telemetry ray score` for this channel."
        ) from exc
