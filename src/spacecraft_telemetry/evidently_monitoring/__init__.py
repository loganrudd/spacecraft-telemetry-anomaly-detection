"""evidently_monitoring: batch drift detection and model monitoring — Phase 8.

Public API
----------
build_reference_profile    Build feature DataFrame from train Parquet for a channel.
build_current_profile      Build feature DataFrame from test Parquet for a channel.
save_reference_profile     Persist reference profile to Parquet.
load_reference_profile     Load a previously saved reference profile.
reference_profile_path     Canonical on-disk path for a channel's reference profile.
run_drift_report           Run DataDrift + DataQuality Evidently report.
DriftResult                Dataclass: share_of_drifted_columns, drift_detected, …
log_drift_report           Log drift report + metrics to MLflow.
"""

from spacecraft_telemetry.evidently_monitoring.mlflow_logging import log_drift_report
from spacecraft_telemetry.evidently_monitoring.reference import (
    build_current_profile,
    build_reference_profile,
    load_reference_profile,
    reference_profile_path,
    save_reference_profile,
)
from spacecraft_telemetry.evidently_monitoring.reports import DriftResult, run_drift_report

__all__ = [
    "DriftResult",
    "build_current_profile",
    "build_reference_profile",
    "load_reference_profile",
    "log_drift_report",
    "reference_profile_path",
    "run_drift_report",
    "save_reference_profile",
]
