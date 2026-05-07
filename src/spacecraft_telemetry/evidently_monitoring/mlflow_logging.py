"""Log Evidently drift reports and results to MLflow.

Reuses the generic helpers from ``mlflow_tracking/`` without modification:
``open_run``, ``log_artifact_bytes``, ``log_metrics_final``,
``experiment_name``, ``common_tags``.

One MLflow run is created per (mission, channel) monitoring call.  The run
lives in the ``telemanom-monitoring-{mission}`` experiment so it stays
separate from training and HPO runs while sharing the same tracking server.
"""

from __future__ import annotations

from spacecraft_telemetry.core.config import Settings
from spacecraft_telemetry.evidently_monitoring.reports import DriftResult, report_to_bytes
from spacecraft_telemetry.mlflow_tracking.conventions import common_tags, experiment_name
from spacecraft_telemetry.mlflow_tracking.runs import (
    configure_mlflow,
    log_artifact_bytes,
    log_metrics_final,
    open_run,
)

try:
    from evidently.legacy.report import Report  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover — Evidently not installed in some envs
    Report = object  # type: ignore[assignment, misc]


def log_drift_report(
    report: Report,
    result: DriftResult,
    settings: Settings,
    mission: str,
    channel: str,
) -> str | None:
    """Log a drift report to MLflow and return the run_id, or None on failure.

    Creates (or reuses) the experiment
    ``"telemanom-monitoring-{mission}"`` and opens a child run named
    ``channel``.

    Logged artifacts:
        - ``drift_report.html`` — full Evidently HTML report

    Logged metrics:
        - ``share_of_drifted_columns`` (float)
        - ``n_features`` (int cast to float)
        - ``n_drifted`` (int cast to float)
        - ``drift_detected`` (0.0 / 1.0)
        - ``drift_{feature_name}`` (0.0 / 1.0) for each column in
          ``result.per_column_drift``

    Args:
        report:   The ``Report`` object returned by ``run_drift_report``.
        result:   The ``DriftResult`` summary extracted from that report.
        settings: Project settings (used for MLflow URI configuration).
        mission:  Mission identifier, e.g. ``"ESA-Mission1"``.
        channel:  Channel ID, e.g. ``"channel_1"``.

    Returns:
        The MLflow ``run_id`` string if logging succeeded, else ``None``.
    """
    configure_mlflow(settings)

    exp = experiment_name("telemanom", "monitoring", mission)
    tags = common_tags(
        model_type="telemanom",
        mission=mission,
        phase="monitoring",
        channel=channel,
    )

    run_id: str | None = None

    with open_run(experiment=exp, run_name=channel, tags=tags) as run:
        if run is not None:
            run_id = run.info.run_id

        # Metrics — drift summary
        metrics: dict[str, float | int] = {
            "share_of_drifted_columns": result.share_of_drifted_columns,
            "n_features": result.n_features,
            "n_drifted": result.n_drifted,
            "drift_detected": 1.0 if result.drift_detected else 0.0,
        }
        # Per-column drift flags (0.0 / 1.0)
        for col, drifted in result.per_column_drift.items():
            metrics[f"drift_{col}"] = 1.0 if drifted else 0.0

        log_metrics_final(metrics)

        # HTML report as an artifact
        log_artifact_bytes(report_to_bytes(report), "drift_report.html")

    return run_id
