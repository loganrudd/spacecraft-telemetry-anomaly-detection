"""Evidently drift report generation for spacecraft telemetry (Phase 7).

Wraps the Evidently 0.7.x legacy API (``evidently.legacy.*``) to produce
per-channel drift reports over 14 monitoring columns.  All Evidently-specific
key paths are isolated in ``_extract_drift_result`` — the single place to fix
when upgrading Evidently.

Public API
----------
DriftResult
    Dataclass holding aggregated and per-column drift statistics.
run_drift_report(reference, current, settings) -> (Report, DriftResult)
    Run DataDriftPreset + DataQualityPreset and return the live Report object
    alongside the extracted DriftResult.
report_to_bytes(report) -> bytes
    Render the Evidently Report to HTML bytes for MLflow artifact storage.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

# ---------------------------------------------------------------------------
# Evidently 0.7.x ships the old (Pandas-based) API under evidently.legacy.
# The evidently.report / evidently.metric_preset top-level modules do not
# exist in this version.  All imports must go through evidently.legacy.*.
# ---------------------------------------------------------------------------
from evidently.legacy.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.legacy.pipeline.column_mapping import ColumnMapping
from evidently.legacy.report import Report

from spacecraft_telemetry.core.config import Settings
from spacecraft_telemetry.evidently_monitoring.reference import MONITORING_FEATURE_COLS


@dataclass
class DriftResult:
    """Aggregated and per-column drift statistics from one Evidently run.

    Attributes:
        share_of_drifted_columns: Fraction of columns where Evidently's per-column
            KS test detected drift (p < 0.05).  Range [0.0, 1.0].
        drift_detected: ``True`` when ``share_of_drifted_columns`` exceeds
            ``settings.monitoring.drift_threshold`` (default 0.30).
        n_features: Total number of monitored columns (always 14 for this project).
        n_drifted: Number of columns where drift was detected.
        per_column_drift: Column name → bool; ``True`` = KS test detected drift.
    """

    share_of_drifted_columns: float
    drift_detected: bool
    n_features: int
    n_drifted: int
    per_column_drift: dict[str, bool]


def run_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    settings: Settings,
) -> tuple[Report, DriftResult]:
    """Run DataDriftPreset + DataQualityPreset over the 14 monitoring columns.

    Args:
        reference: Reference profile DataFrame (train split), columns must include
            all entries of ``MONITORING_FEATURE_COLS``.
        current:   Current DataFrame (test split with the same feature columns).
        settings:  Runtime settings; ``settings.monitoring.drift_threshold`` sets
            the share threshold that flips ``DriftResult.drift_detected``.

    Returns:
        ``(report, result)`` where ``report`` is the live Evidently Report object
        (can be rendered to HTML via :func:`report_to_bytes`) and ``result`` is
        the extracted :class:`DriftResult`.
    """
    col_mapping = ColumnMapping(numerical_features=MONITORING_FEATURE_COLS)
    report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
    report.run(
        reference_data=reference[MONITORING_FEATURE_COLS],
        current_data=current[MONITORING_FEATURE_COLS],
        column_mapping=col_mapping,
    )
    result = _extract_drift_result(report, settings)
    return report, result


def report_to_bytes(report: Report) -> bytes:
    """Render the Evidently Report to UTF-8 HTML bytes.

    The returned bytes are suitable for direct upload to MLflow as an artifact
    (``drift_report.html``).

    Args:
        report: A ``Report`` returned by :func:`run_drift_report` (must have been
                ``.run()`` before calling this).

    Returns:
        UTF-8 encoded HTML bytes.
    """
    return str(report.get_html()).encode("utf-8")


def _extract_drift_result(report: Report, settings: Settings) -> DriftResult:
    """Extract DriftResult from the Evidently 0.7.x result dict.

    Evidently 0.7.x result layout (after ``report.run()``):

    .. code-block:: text

        report.as_dict()["metrics"][0]   → DatasetDriftMetric
            result["share_of_drifted_columns"] : float
            result["number_of_drifted_columns"]: int
            result["number_of_columns"]        : int

        report.as_dict()["metrics"][1]   → DataDriftTable
            result["drift_by_columns"]   : dict[str, dict]
                [col]["drift_detected"]  : bool  (KS p < 0.05)

    If the Evidently version changes and this path breaks, fix only here.

    Args:
        report:   Executed ``Report`` object.
        settings: Used to apply ``monitoring.drift_threshold``.

    Returns:
        :class:`DriftResult` populated from the report data.
    """
    metrics = report.as_dict()["metrics"]

    # DatasetDriftMetric (index 0) — aggregate stats
    dataset_metric = metrics[0]["result"]
    n_features: int = dataset_metric["number_of_columns"]
    n_drifted: int = dataset_metric["number_of_drifted_columns"]
    share: float = dataset_metric["share_of_drifted_columns"]

    # DataDriftTable (index 1) — per-column KS results
    drift_table = metrics[1]["result"]
    per_column_drift: dict[str, bool] = {
        col: info["drift_detected"]
        for col, info in drift_table["drift_by_columns"].items()
    }

    drift_detected = share > settings.monitoring.drift_threshold

    return DriftResult(
        share_of_drifted_columns=share,
        drift_detected=drift_detected,
        n_features=n_features,
        n_drifted=n_drifted,
        per_column_drift=per_column_drift,
    )
