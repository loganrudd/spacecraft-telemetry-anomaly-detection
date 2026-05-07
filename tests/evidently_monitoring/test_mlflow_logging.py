"""Integration tests for evidently_monitoring/mlflow_logging.py.

Uses a per-test isolated SQLite MLflow store (mirrors the pattern in
tests/mlflow_tracking/test_runs.py) so every test starts with a clean slate.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest

from spacecraft_telemetry.core.config import Settings
from spacecraft_telemetry.evidently_monitoring.mlflow_logging import log_drift_report
from spacecraft_telemetry.evidently_monitoring.reference import (
    MONITORING_FEATURE_COLS,
    compute_feature_dataframe,
)
from spacecraft_telemetry.evidently_monitoring.reports import DriftResult, run_drift_report

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_MISSION = "ESA-Mission1"
_CHANNEL = "channel_1"

_N = 300  # rows in synthetic series


@pytest.fixture
def mlflow_uri(tmp_path: Path) -> Generator[str, None, None]:
    """Isolated SQLite MLflow store, reset after each test."""
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    mlflow.set_tracking_uri(uri)
    yield uri
    if mlflow.active_run() is not None:
        mlflow.end_run()
    mlflow.set_tracking_uri("")


@pytest.fixture
def isolated_settings(tmp_path: Path) -> Settings:
    """Settings pointing to the per-test SQLite MLflow store."""
    uri = mlflow.get_tracking_uri()
    return Settings().model_copy(
        update={
            "mlflow": Settings().mlflow.model_copy(update={"tracking_uri": uri})
        }
    )


def _make_feature_df(shift: float = 0.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2020-01-01", periods=_N, freq="1s", tz="UTC")
    raw = pd.DataFrame(
        {
            "telemetry_timestamp": timestamps,
            "value_normalized": rng.standard_normal(_N).astype(np.float32),
        }
    )
    df = compute_feature_dataframe(raw, Settings())
    if shift != 0.0:
        df = df + shift
    return df


@pytest.fixture(scope="module")
def nominal_report_and_result() -> tuple:
    """Module-scoped: ref == cur, guaranteed zero drift."""
    ref = _make_feature_df(seed=0)
    cur = ref.copy()
    return run_drift_report(ref, cur, Settings())


@pytest.fixture(scope="module")
def drifted_report_and_result() -> tuple:
    """Module-scoped: 5σ mean-shifted current, guaranteed full drift."""
    ref = _make_feature_df(seed=0)
    cur = _make_feature_df(shift=5.0, seed=0)
    return run_drift_report(ref, cur, Settings())


# ---------------------------------------------------------------------------
# TestLogDriftReportCreatesExperimentAndRun
# ---------------------------------------------------------------------------


class TestLogDriftReportCreatesExperimentAndRun:
    """log_drift_report creates the expected experiment and run in MLflow."""

    def test_returns_run_id_string(
        self,
        mlflow_uri: str,
        isolated_settings: Settings,
        nominal_report_and_result: tuple,
    ) -> None:
        report, result = nominal_report_and_result
        run_id = log_drift_report(report, result, isolated_settings, _MISSION, _CHANNEL)
        assert isinstance(run_id, str)
        assert len(run_id) == 32  # MLflow run IDs are 32-char hex strings

    def test_creates_correct_experiment_name(
        self,
        mlflow_uri: str,
        isolated_settings: Settings,
        nominal_report_and_result: tuple,
    ) -> None:
        report, result = nominal_report_and_result
        log_drift_report(report, result, isolated_settings, _MISSION, _CHANNEL)

        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name(f"telemanom-monitoring-{_MISSION}")
        assert exp is not None

    def test_run_name_equals_channel(
        self,
        mlflow_uri: str,
        isolated_settings: Settings,
        nominal_report_and_result: tuple,
    ) -> None:
        report, result = nominal_report_and_result
        run_id = log_drift_report(report, result, isolated_settings, _MISSION, _CHANNEL)

        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)  # type: ignore[arg-type]
        assert run.info.run_name == _CHANNEL

    def test_run_is_finished(
        self,
        mlflow_uri: str,
        isolated_settings: Settings,
        nominal_report_and_result: tuple,
    ) -> None:
        report, result = nominal_report_and_result
        run_id = log_drift_report(report, result, isolated_settings, _MISSION, _CHANNEL)

        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)  # type: ignore[arg-type]
        assert run.info.status == "FINISHED"


# ---------------------------------------------------------------------------
# TestLogDriftReportTags
# ---------------------------------------------------------------------------


class TestLogDriftReportTags:
    """Expected MLflow tags are set on the run."""

    def test_model_type_tag(
        self,
        mlflow_uri: str,
        isolated_settings: Settings,
        nominal_report_and_result: tuple,
    ) -> None:
        report, result = nominal_report_and_result
        run_id = log_drift_report(report, result, isolated_settings, _MISSION, _CHANNEL)

        run = mlflow.tracking.MlflowClient().get_run(run_id)  # type: ignore[arg-type]
        assert run.data.tags["model_type"] == "telemanom"

    def test_phase_tag(
        self,
        mlflow_uri: str,
        isolated_settings: Settings,
        nominal_report_and_result: tuple,
    ) -> None:
        report, result = nominal_report_and_result
        run_id = log_drift_report(report, result, isolated_settings, _MISSION, _CHANNEL)

        run = mlflow.tracking.MlflowClient().get_run(run_id)  # type: ignore[arg-type]
        assert run.data.tags["phase"] == "monitoring"

    def test_mission_and_channel_tags(
        self,
        mlflow_uri: str,
        isolated_settings: Settings,
        nominal_report_and_result: tuple,
    ) -> None:
        report, result = nominal_report_and_result
        run_id = log_drift_report(report, result, isolated_settings, _MISSION, _CHANNEL)

        run = mlflow.tracking.MlflowClient().get_run(run_id)  # type: ignore[arg-type]
        assert run.data.tags["mission_id"] == _MISSION
        assert run.data.tags["channel_id"] == _CHANNEL


# ---------------------------------------------------------------------------
# TestLogDriftReportMetrics
# ---------------------------------------------------------------------------


class TestLogDriftReportMetrics:
    """Summary drift metrics are logged correctly."""

    def test_share_of_drifted_columns_logged(
        self,
        mlflow_uri: str,
        isolated_settings: Settings,
        drifted_report_and_result: tuple,
    ) -> None:
        report, result = drifted_report_and_result
        run_id = log_drift_report(report, result, isolated_settings, _MISSION, _CHANNEL)

        run = mlflow.tracking.MlflowClient().get_run(run_id)  # type: ignore[arg-type]
        assert "share_of_drifted_columns" in run.data.metrics
        assert run.data.metrics["share_of_drifted_columns"] == pytest.approx(
            result.share_of_drifted_columns, abs=1e-6
        )

    def test_drift_detected_logged_as_float(
        self,
        mlflow_uri: str,
        isolated_settings: Settings,
        drifted_report_and_result: tuple,
    ) -> None:
        report, result = drifted_report_and_result
        run_id = log_drift_report(report, result, isolated_settings, _MISSION, _CHANNEL)

        run = mlflow.tracking.MlflowClient().get_run(run_id)  # type: ignore[arg-type]
        assert run.data.metrics["drift_detected"] == 1.0

    def test_n_features_and_n_drifted_logged(
        self,
        mlflow_uri: str,
        isolated_settings: Settings,
        drifted_report_and_result: tuple,
    ) -> None:
        report, result = drifted_report_and_result
        run_id = log_drift_report(report, result, isolated_settings, _MISSION, _CHANNEL)

        run = mlflow.tracking.MlflowClient().get_run(run_id)  # type: ignore[arg-type]
        assert run.data.metrics["n_features"] == float(result.n_features)
        assert run.data.metrics["n_drifted"] == float(result.n_drifted)

    def test_per_column_drift_metrics_logged(
        self,
        mlflow_uri: str,
        isolated_settings: Settings,
        drifted_report_and_result: tuple,
    ) -> None:
        report, result = drifted_report_and_result
        run_id = log_drift_report(report, result, isolated_settings, _MISSION, _CHANNEL)

        run = mlflow.tracking.MlflowClient().get_run(run_id)  # type: ignore[arg-type]
        for col in result.per_column_drift:
            key = f"drift_{col}"
            assert key in run.data.metrics, f"Expected metric '{key}' not found"
            expected = 1.0 if result.per_column_drift[col] else 0.0
            assert run.data.metrics[key] == expected

    def test_per_column_metric_count(
        self,
        mlflow_uri: str,
        isolated_settings: Settings,
        drifted_report_and_result: tuple,
    ) -> None:
        """Exactly one metric per column in per_column_drift, plus 4 summary metrics."""
        report, result = drifted_report_and_result
        run_id = log_drift_report(report, result, isolated_settings, _MISSION, _CHANNEL)

        run = mlflow.tracking.MlflowClient().get_run(run_id)  # type: ignore[arg-type]
        per_col_keys = [
            k for k in run.data.metrics
            if k.startswith("drift_") and k != "drift_detected"
        ]
        assert len(per_col_keys) == len(result.per_column_drift)

    def test_nominal_drift_detected_is_zero(
        self,
        mlflow_uri: str,
        isolated_settings: Settings,
        nominal_report_and_result: tuple,
    ) -> None:
        report, result = nominal_report_and_result
        run_id = log_drift_report(report, result, isolated_settings, _MISSION, _CHANNEL)

        run = mlflow.tracking.MlflowClient().get_run(run_id)  # type: ignore[arg-type]
        assert run.data.metrics["drift_detected"] == 0.0


# ---------------------------------------------------------------------------
# TestLogDriftReportArtifact
# ---------------------------------------------------------------------------


class TestLogDriftReportArtifact:
    """HTML report is stored as an MLflow artifact."""

    def test_drift_report_html_artifact_exists(
        self,
        mlflow_uri: str,
        isolated_settings: Settings,
        nominal_report_and_result: tuple,
        tmp_path: Path,
    ) -> None:
        report, result = nominal_report_and_result
        run_id = log_drift_report(report, result, isolated_settings, _MISSION, _CHANNEL)

        client = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(run_id)  # type: ignore[arg-type]
        artifact_names = [a.path for a in artifacts]
        assert "drift_report.html" in artifact_names

    def test_artifact_is_non_trivially_large(
        self,
        mlflow_uri: str,
        isolated_settings: Settings,
        nominal_report_and_result: tuple,
        tmp_path: Path,
    ) -> None:
        report, result = nominal_report_and_result
        run_id = log_drift_report(report, result, isolated_settings, _MISSION, _CHANNEL)

        client = mlflow.tracking.MlflowClient()
        artifact_info = client.list_artifacts(run_id)[0]  # type: ignore[arg-type]
        assert artifact_info.file_size > 100_000  # at least 100KB


# ---------------------------------------------------------------------------
# TestLogDriftReportNoActiveRunLeak
# ---------------------------------------------------------------------------


class TestLogDriftReportNoActiveRunLeak:
    """log_drift_report must not leave an active MLflow run open."""

    def test_no_active_run_after_call(
        self,
        mlflow_uri: str,
        isolated_settings: Settings,
        nominal_report_and_result: tuple,
    ) -> None:
        report, result = nominal_report_and_result
        log_drift_report(report, result, isolated_settings, _MISSION, _CHANNEL)
        assert mlflow.active_run() is None
