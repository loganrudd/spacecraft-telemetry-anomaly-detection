"""Artifact I/O for Telemanom model files — MLflow-backed.

After the A1 pivot (Phase 6 review), MLflow is the single source of truth for
all model artifacts.  All writes go through MLflow logging APIs inside
training.py and scoring.py.  This module provides the read-path helpers:

- load_model_for_scoring: load the latest registered PyTorch model + window_size.
- download_artifact_bytes: fetch a named artifact from a specific run.
- find_latest_run_for_channel: locate the most recent run for a channel in an experiment.
- errors_to_bytes / threshold_to_bytes: serialise numpy arrays for log_artifact_bytes.
- bytes_to_errors: deserialise errors bytes back to a numpy array.

Phase 4 note: _write_bytes / _read_bytes no longer exist.  The `gs://`
indirection is handled by the MLflow artifact store — configure
MLFLOW_ARTIFACTS_DESTINATION to a `gs://` bucket for cloud runs.
"""

from __future__ import annotations

import io
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
# MLflow run lookup
# ---------------------------------------------------------------------------


def find_latest_run_for_channel(
    experiment_name: str,
    channel: str,
    tracking_uri: str,
) -> Any:
    """Return the most recent MLflow run for a channel in an experiment, or None.

    Args:
        experiment_name: MLflow experiment name to search within.
        channel:         Channel ID to filter by (matched against tags.channel_id).
        tracking_uri:    MLflow tracking server URI.

    Returns:
        An ``mlflow.entities.Run`` or ``None`` if no matching run is found.
    """
    import mlflow

    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        return None
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string=f"tags.channel_id = '{channel}'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    return runs[0] if runs else None


# ---------------------------------------------------------------------------
# Artifact download
# ---------------------------------------------------------------------------


def download_artifact_bytes(
    run_id: str,
    artifact_path: str,
    tracking_uri: str,
) -> bytes:
    """Download a named artifact from an MLflow run and return its raw bytes.

    Args:
        run_id:        MLflow run ID.
        artifact_path: Path within the run's artifact store, e.g. "errors.npy".
        tracking_uri:  MLflow tracking server URI.

    Returns:
        Raw bytes of the artifact.

    Raises:
        OSError: If the artifact cannot be downloaded.
    """
    import mlflow

    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = client.download_artifacts(run_id, artifact_path, tmp_dir)
        return Path(local_path).read_bytes()


# ---------------------------------------------------------------------------
# Model load (used by scoring.py and Phase 8 FastAPI)
# ---------------------------------------------------------------------------


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
    from mlflow.exceptions import MlflowException

    from spacecraft_telemetry.mlflow_tracking.registry import CHAMPION_ALIAS

    client = mlflow.MlflowClient(tracking_uri=tracking_uri)

    versions = client.search_model_versions(f"name='{name}'")
    if not versions:
        raise ModelNotFoundError(
            f"No registered versions found for model {name!r}. "
            "Run 'model train' for this channel before serving."
        )

    if require_champion:
        try:
            mv = client.get_model_version_by_alias(name, CHAMPION_ALIAS)
        except MlflowException as err:
            raise RuntimeError(
                f"No @champion alias set for model {name!r}. "
                "Run 'make mlflow-promote MISSION=... CHANNEL=...' before serving."
            ) from err
    else:
        mv = max(versions, key=lambda v: int(v.version))

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
        )
    except KeyError as exc:
        raise RuntimeError(
            f"Scoring run {run.info.run_id!r} for channel {channel!r} is missing "
            f"param {exc.args[0]!r}. "
            "Re-run `spacecraft-telemetry ray score` for this channel."
        ) from exc


