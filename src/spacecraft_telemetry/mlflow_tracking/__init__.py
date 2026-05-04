"""MLflow tracking and model registry — Phase 7.

Public API
----------
conventions  experiment_name, registered_model_name, common_tags
hashing      training_data_hash
runs         configure_mlflow, open_run, log_params, log_metrics_step,
             log_metrics_final, log_artifact_bytes
registry     register_pytorch_model, promote, latest_uri
"""

from spacecraft_telemetry.mlflow_tracking.conventions import (
    common_tags,
    experiment_name,
    registered_model_name,
)
from spacecraft_telemetry.mlflow_tracking.hashing import training_data_hash
from spacecraft_telemetry.mlflow_tracking.registry import (
    latest_uri,
    promote,
    register_pytorch_model,
)
from spacecraft_telemetry.mlflow_tracking.runs import (
    configure_mlflow,
    log_artifact_bytes,
    log_metrics_final,
    log_metrics_step,
    log_params,
    open_run,
)

__all__ = [
    "common_tags",
    "configure_mlflow",
    "experiment_name",
    "latest_uri",
    "log_artifact_bytes",
    "log_metrics_final",
    "log_metrics_step",
    "log_params",
    "open_run",
    "promote",
    "register_pytorch_model",
    "registered_model_name",
    "training_data_hash",
]
