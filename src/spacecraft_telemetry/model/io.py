"""Artifact I/O for Telemanom model files — MLflow-backed.

After the A1 pivot (Phase 7 review), MLflow is the single source of truth for
all model artifacts.  All writes go through MLflow logging APIs inside
training.py and scoring.py.  This module provides the read-path helpers:

- load_model_for_scoring: load the latest registered PyTorch model + window_size.
- download_artifact_bytes: fetch a named artifact from a specific run.
- find_latest_run_for_channel: locate the most recent run for a channel in an experiment.
- errors_to_bytes / threshold_to_bytes: serialise numpy arrays for log_artifact_bytes.
- bytes_to_errors: deserialise errors bytes back to a numpy array.

Phase 5 note: _write_bytes / _read_bytes no longer exist.  The `gs://`
indirection is handled by the MLflow artifact store — configure
MLFLOW_ARTIFACTS_DESTINATION to a `gs://` bucket for cloud runs.
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

    from spacecraft_telemetry.model.architecture import TelemanomLSTM


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

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
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

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = client.download_artifacts(run_id, artifact_path, tmp_dir)
        return Path(local_path).read_bytes()


# ---------------------------------------------------------------------------
# Model load (used by scoring.py and Phase 9 FastAPI)
# ---------------------------------------------------------------------------


def load_model_for_scoring(
    name: str,
    device: torch.device,
    tracking_uri: str,
) -> tuple[TelemanomLSTM, int]:
    """Load the latest registered PyTorch model from MLflow for scoring.

    Finds the highest version number registered for ``name``, loads the model
    artifact from the run that produced it, and reads ``window_size`` from the
    run's logged params.

    Args:
        name:         Registered model name from registered_model_name().
        device:       Torch device to map the model weights to.
        tracking_uri: MLflow tracking server URI.

    Returns:
        (model, window_size) — reconstructed TelemanomLSTM and the sequence
        length the model was trained on.

    Raises:
        RuntimeError: If no registered version exists for ``name``.
    """
    import mlflow
    import mlflow.pytorch

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    versions = client.search_model_versions(f"name='{name}'")
    if not versions:
        raise RuntimeError(
            f"No registered versions found for model {name!r}. "
            "Run 'model train' for this channel before scoring."
        )
    latest = max(versions, key=lambda v: int(v.version))
    model = mlflow.pytorch.load_model(
        f"runs:/{latest.run_id}/model",
        map_location=device,
    )
    run = client.get_run(latest.run_id)
    window_size = int(run.data.params["window_size"])
    return model, window_size



