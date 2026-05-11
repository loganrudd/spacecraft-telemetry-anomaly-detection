"""Tests for evidently_monitoring/reports.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacecraft_telemetry.core.config import MonitoringConfig, Settings
from spacecraft_telemetry.evidently_monitoring.reference import (
    MONITORING_FEATURE_COLS,
    compute_feature_dataframe,
)
from spacecraft_telemetry.evidently_monitoring.reports import (
    DriftResult,
    _extract_drift_result,
    report_to_bytes,
    run_drift_report,
)

# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------

_N = 300  # rows per synthetic dataset — large enough for reliable KS stats


def _make_feature_df(
    shift: float = 0.0,
    scale: float = 1.0,
    seed: int = 0,
    n: int = _N,
) -> pd.DataFrame:
    """Synthetic feature DataFrame with all MONITORING_FEATURE_COLS.

    Builds a raw series, computes features, then optionally applies a mean-shift
    and/or scale change to *all* feature columns to simulate data drift.
    """
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2020-01-01", periods=n, freq="1s", tz="UTC")
    raw = pd.DataFrame(
        {
            "telemetry_timestamp": timestamps,
            "value_normalized": rng.standard_normal(n).astype(np.float32),
        }
    )
    df = compute_feature_dataframe(raw, Settings())
    if shift != 0.0 or scale != 1.0:
        df = df * scale + shift
    return df


# ---------------------------------------------------------------------------
# DriftResult dataclass
# ---------------------------------------------------------------------------


class TestDriftResultDataclass:
    def test_construction(self) -> None:
        result = DriftResult(
            share_of_drifted_columns=0.5,
            drift_detected=True,
            n_features=14,
            n_drifted=7,
            per_column_drift={"a": True, "b": False},
        )
        assert result.share_of_drifted_columns == 0.5
        assert result.drift_detected is True
        assert result.n_features == 14
        assert result.n_drifted == 7
        assert result.per_column_drift == {"a": True, "b": False}


# ---------------------------------------------------------------------------
# run_drift_report — nominal (no drift)
# ---------------------------------------------------------------------------


class TestRunDriftReportNominal:
    """Reference == current (identical DataFrames) → KS test p=1.0 → no drift.

    Using two independent seeds with only 200 rows causes Evidently's KS test to
    flag rolling features as drifted due to autocorrelation within overlapping
    windows.  Identical data is the reliable way to assert zero drift.
    """

    @pytest.fixture(scope="class")
    def nominal_result(self) -> tuple:
        ref = _make_feature_df(seed=0)
        cur = ref.copy()  # identical distributions → guaranteed no drift
        return run_drift_report(ref, cur, Settings())

    def test_returns_tuple_of_two(self, nominal_result: tuple) -> None:
        assert len(nominal_result) == 2

    def test_drift_detected_is_false(self, nominal_result: tuple) -> None:
        _, result = nominal_result
        # Two i.i.d. draws should not exceed the 30% drift threshold
        assert result.drift_detected is False

    def test_share_below_threshold(self, nominal_result: tuple) -> None:
        _, result = nominal_result
        assert result.share_of_drifted_columns <= 0.30

    def test_n_features_equals_14(self, nominal_result: tuple) -> None:
        _, result = nominal_result
        assert result.n_features == 14

    def test_per_column_drift_has_all_cols(self, nominal_result: tuple) -> None:
        _, result = nominal_result
        assert set(result.per_column_drift.keys()) == set(MONITORING_FEATURE_COLS)

    def test_per_column_drift_values_are_bool(self, nominal_result: tuple) -> None:
        _, result = nominal_result
        assert all(isinstance(v, bool) for v in result.per_column_drift.values())

    def test_n_drifted_consistent_with_per_column(self, nominal_result: tuple) -> None:
        _, result = nominal_result
        expected_n = sum(1 for v in result.per_column_drift.values() if v)
        assert result.n_drifted == expected_n

    def test_share_consistent_with_n_drifted(self, nominal_result: tuple) -> None:
        _, result = nominal_result
        expected_share = result.n_drifted / result.n_features
        assert abs(result.share_of_drifted_columns - expected_share) < 1e-6


# ---------------------------------------------------------------------------
# run_drift_report — drifted (5-sigma mean shift on all columns)
# ---------------------------------------------------------------------------


class TestRunDriftReportDrifted:
    """Current data mean-shifted by 5-sigma → all columns should drift."""

    @pytest.fixture(scope="class")
    def drifted_result(self) -> tuple:
        ref = _make_feature_df(shift=0.0, seed=0)
        # 5-sigma shift: with 200+ samples per column, KS test will detect drift
        cur = _make_feature_df(shift=5.0, seed=1)
        return run_drift_report(ref, cur, Settings())

    def test_drift_detected_is_true(self, drifted_result: tuple) -> None:
        _, result = drifted_result
        assert result.drift_detected is True

    def test_share_equals_one(self, drifted_result: tuple) -> None:
        _, result = drifted_result
        assert result.share_of_drifted_columns == 1.0

    def test_all_columns_marked_drifted(self, drifted_result: tuple) -> None:
        _, result = drifted_result
        assert all(result.per_column_drift.values())

    def test_n_drifted_equals_n_features(self, drifted_result: tuple) -> None:
        _, result = drifted_result
        assert result.n_drifted == result.n_features == 14


# ---------------------------------------------------------------------------
# run_drift_report — custom threshold
# ---------------------------------------------------------------------------


def _make_partial_drift_df(seed: int = 0, n: int = _N) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (ref, cur) where cur has a 5-sigma mean shift on the raw signal.

    A pure mean-shift leaves rolling_std and rate_of_change unchanged (both are
    shift-invariant), so roughly 10 of 14 features drift instead of all 14.
    This makes the pair suitable for testing threshold boundary logic.
    """
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2020-01-01", periods=n, freq="1s", tz="UTC")
    raw = pd.DataFrame(
        {
            "telemetry_timestamp": timestamps,
            "value_normalized": rng.standard_normal(n).astype(np.float32),
        }
    )
    ref = compute_feature_dataframe(raw, Settings())
    # Shift the raw signal so features react realistically — rolling_std and
    # rate_of_change don't shift, keeping share below 1.0.
    raw_shifted = raw.copy()
    raw_shifted["value_normalized"] = (raw_shifted["value_normalized"] + 5.0).astype(
        np.float32
    )
    cur = compute_feature_dataframe(raw_shifted, Settings())
    return ref, cur


class TestDriftThreshold:
    def test_low_threshold_triggers_drift_on_partial_shift(self) -> None:
        """With threshold=0.01 even partial drift triggers detection."""
        ref, cur = _make_partial_drift_df(seed=0)
        settings = Settings(monitoring=MonitoringConfig(drift_threshold=0.01))
        _, result = run_drift_report(ref, cur, settings)
        assert result.drift_detected is True

    def test_threshold_above_partial_share_suppresses_detection(self) -> None:
        """A threshold above the actual drifted share suppresses drift detection.

        A 5-sigma raw-signal mean-shift drifts ~10/14 features (share ≈ 0.714).
        Setting threshold=0.80 (above 0.714) means drift is not flagged.
        """
        ref, cur = _make_partial_drift_df(seed=0)
        settings = Settings(monitoring=MonitoringConfig(drift_threshold=0.80))
        _, result = run_drift_report(ref, cur, settings)
        assert result.share_of_drifted_columns > 0.0  # data genuinely drifted
        assert result.drift_detected is False

    def test_threshold_below_partial_share_detects_drift(self) -> None:
        """A threshold below the actual drifted share triggers drift detection.

        Same fixture as above; threshold=0.50 (below ~0.714) → detected.
        """
        ref, cur = _make_partial_drift_df(seed=0)
        settings = Settings(monitoring=MonitoringConfig(drift_threshold=0.50))
        _, result = run_drift_report(ref, cur, settings)
        assert result.drift_detected is True


# ---------------------------------------------------------------------------
# report_to_bytes
# ---------------------------------------------------------------------------


class TestReportToBytes:
    @pytest.fixture(scope="class")
    def report_obj(self):
        ref = _make_feature_df(seed=0)
        cur = _make_feature_df(seed=1)
        report, _ = run_drift_report(ref, cur, Settings())
        return report

    def test_returns_bytes(self, report_obj) -> None:
        result = report_to_bytes(report_obj)
        assert isinstance(result, bytes)

    def test_bytes_non_empty(self, report_obj) -> None:
        result = report_to_bytes(report_obj)
        assert len(result) > 0

    def test_bytes_are_valid_utf8_html(self, report_obj) -> None:
        html = report_to_bytes(report_obj).decode("utf-8")
        assert "<html" in html.lower()

    def test_bytes_contain_evidently_content(self, report_obj) -> None:
        html = report_to_bytes(report_obj).decode("utf-8")
        # Evidently always embeds its brand in reports
        assert "evidently" in html.lower()

    def test_bytes_non_trivially_large(self, report_obj) -> None:
        """HTML report should be at least 100KB — verifies full rendering occurred."""
        assert len(report_to_bytes(report_obj)) > 100_000


# ---------------------------------------------------------------------------
# _extract_drift_result (private, but tested directly to pin the dict path)
# ---------------------------------------------------------------------------


class TestExtractDriftResult:
    """Pin the exact Evidently 0.7.x result dict path used by _extract_drift_result.

    If Evidently changes its result structure these tests will fail fast and
    pinpoint the broken key path.
    """

    @pytest.fixture(scope="class")
    def executed_report(self):
        ref = _make_feature_df(seed=0)
        cur = _make_feature_df(shift=5.0, seed=1)
        report, _ = run_drift_report(ref, cur, Settings())
        return report

    def test_metrics_list_is_non_empty(self, executed_report) -> None:
        d = executed_report.as_dict()
        assert len(d["metrics"]) >= 2

    def test_first_metric_is_dataset_drift(self, executed_report) -> None:
        d = executed_report.as_dict()
        assert d["metrics"][0]["metric"] == "DatasetDriftMetric"

    def test_second_metric_is_data_drift_table(self, executed_report) -> None:
        d = executed_report.as_dict()
        assert d["metrics"][1]["metric"] == "DataDriftTable"

    def test_share_key_present(self, executed_report) -> None:
        d = executed_report.as_dict()
        assert "share_of_drifted_columns" in d["metrics"][0]["result"]

    def test_drift_by_columns_key_present(self, executed_report) -> None:
        d = executed_report.as_dict()
        assert "drift_by_columns" in d["metrics"][1]["result"]

    def test_per_column_has_drift_detected_key(self, executed_report) -> None:
        d = executed_report.as_dict()
        drift_by_cols = d["metrics"][1]["result"]["drift_by_columns"]
        first_col = next(iter(drift_by_cols.values()))
        assert "drift_detected" in first_col

    def test_extract_result_all_cols_present(self, executed_report) -> None:
        result = _extract_drift_result(executed_report, Settings())
        assert set(result.per_column_drift.keys()) == set(MONITORING_FEATURE_COLS)
