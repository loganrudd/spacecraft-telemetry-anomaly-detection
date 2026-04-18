"""Tests for features.definitions — the shared feature registry."""

from __future__ import annotations

import math

import numpy as np
import pytest

from spacecraft_telemetry.features.definitions import (
    FEATURE_DEFINITIONS,
    FeatureDefinition,
    compute_features_numpy,
    get_feature_by_name,
    get_feature_names,
)

# ---------------------------------------------------------------------------
# Registry integrity
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_no_duplicate_names(self) -> None:
        names = get_feature_names()
        assert len(names) == len(set(names)), "duplicate feature names in registry"

    def test_expected_feature_names_present(self) -> None:
        names = set(get_feature_names())
        expected = {
            "rolling_mean_10", "rolling_std_10", "rolling_min_10", "rolling_max_10",
            "rolling_mean_50", "rolling_std_50", "rolling_min_50", "rolling_max_50",
            "rolling_mean_100", "rolling_std_100", "rolling_min_100", "rolling_max_100",
            "rate_of_change",
        }
        assert expected == names

    def test_all_definitions_are_frozen_dataclasses(self) -> None:
        for fd in FEATURE_DEFINITIONS:
            assert isinstance(fd, FeatureDefinition)
            with pytest.raises((AttributeError, TypeError)):
                fd.name = "mutated"  # type: ignore[misc]

    def test_all_dtypes_are_float32(self) -> None:
        for fd in FEATURE_DEFINITIONS:
            assert fd.dtype == "float32", f"{fd.name} has dtype {fd.dtype!r}"

    def test_all_window_sizes_positive(self) -> None:
        for fd in FEATURE_DEFINITIONS:
            assert fd.window_size >= 1, f"{fd.name} has window_size {fd.window_size}"

    def test_get_feature_by_name_known(self) -> None:
        fd = get_feature_by_name("rolling_mean_10")
        assert fd.name == "rolling_mean_10"

    def test_get_feature_by_name_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="unknown_feature"):
            get_feature_by_name("unknown_feature")


# ---------------------------------------------------------------------------
# Rolling mean
# ---------------------------------------------------------------------------


class TestRollingMean:
    def test_exact_window(self) -> None:
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ts = np.arange(5, dtype=np.float64)
        fd = get_feature_by_name("rolling_mean_10")
        # buffer shorter than window → nan
        assert math.isnan(fd.compute_numpy(values, ts))

    def test_full_window(self) -> None:
        values = np.ones(10, dtype=np.float64) * 3.0
        ts = np.arange(10, dtype=np.float64)
        fd = get_feature_by_name("rolling_mean_10")
        assert fd.compute_numpy(values, ts) == pytest.approx(3.0)

    def test_uses_trailing_window(self) -> None:
        # 15 values: first 5 are 0.0, last 10 are 2.0
        values = np.array([0.0] * 5 + [2.0] * 10)
        ts = np.arange(15, dtype=np.float64)
        fd = get_feature_by_name("rolling_mean_10")
        assert fd.compute_numpy(values, ts) == pytest.approx(2.0)

    def test_mean_50(self) -> None:
        values = np.arange(1, 51, dtype=np.float64)  # 1..50
        ts = np.arange(50, dtype=np.float64)
        fd = get_feature_by_name("rolling_mean_50")
        assert fd.compute_numpy(values, ts) == pytest.approx(25.5)


# ---------------------------------------------------------------------------
# Rolling std
# ---------------------------------------------------------------------------


class TestRollingStd:
    def test_insufficient_buffer_is_nan(self) -> None:
        values = np.array([1.0, 2.0])
        ts = np.arange(2, dtype=np.float64)
        fd = get_feature_by_name("rolling_std_10")
        assert math.isnan(fd.compute_numpy(values, ts))

    def test_constant_series_std_is_zero(self) -> None:
        values = np.ones(10, dtype=np.float64)
        ts = np.arange(10, dtype=np.float64)
        fd = get_feature_by_name("rolling_std_10")
        assert fd.compute_numpy(values, ts) == pytest.approx(0.0)

    def test_known_std(self) -> None:
        # window=10, values=[0]*7 + [1,2,3] → sample std of all 10 values
        values = np.array([0.0] * 7 + [1.0, 2.0, 3.0])
        ts = np.arange(10, dtype=np.float64)
        fd = get_feature_by_name("rolling_std_10")
        expected = float(np.std(values, ddof=1))
        assert fd.compute_numpy(values, ts) == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------------------
# Rolling min / max
# ---------------------------------------------------------------------------


class TestRollingMinMax:
    def test_rolling_min(self) -> None:
        values = np.array([5.0, 3.0, 8.0, 1.0, 6.0] + [9.0] * 5)
        ts = np.arange(10, dtype=np.float64)
        fd = get_feature_by_name("rolling_min_10")
        assert fd.compute_numpy(values, ts) == pytest.approx(1.0)

    def test_rolling_max(self) -> None:
        values = np.array([5.0, 3.0, 8.0, 1.0, 6.0] + [2.0] * 5)
        ts = np.arange(10, dtype=np.float64)
        fd = get_feature_by_name("rolling_max_10")
        assert fd.compute_numpy(values, ts) == pytest.approx(8.0)

    def test_insufficient_buffer_is_nan(self) -> None:
        values = np.array([1.0])
        ts = np.array([0.0])
        for name in ("rolling_min_10", "rolling_max_10"):
            assert math.isnan(get_feature_by_name(name).compute_numpy(values, ts))


# ---------------------------------------------------------------------------
# Rate of change
# ---------------------------------------------------------------------------


class TestRateOfChange:
    def test_basic_rate(self) -> None:
        # value goes from 1.0 to 3.0 over 2 seconds → ROC = 1.0 /s
        values = np.array([1.0, 3.0])
        ts = np.array([0.0, 2.0])
        fd = get_feature_by_name("rate_of_change")
        assert fd.compute_numpy(values, ts) == pytest.approx(1.0)

    def test_negative_rate(self) -> None:
        values = np.array([10.0, 4.0])
        ts = np.array([0.0, 3.0])
        fd = get_feature_by_name("rate_of_change")
        assert fd.compute_numpy(values, ts) == pytest.approx(-2.0)

    def test_zero_interval_returns_nan(self) -> None:
        values = np.array([1.0, 2.0])
        ts = np.array([5.0, 5.0])  # same timestamp
        fd = get_feature_by_name("rate_of_change")
        assert math.isnan(fd.compute_numpy(values, ts))

    def test_single_value_returns_nan(self) -> None:
        values = np.array([1.0])
        ts = np.array([0.0])
        fd = get_feature_by_name("rate_of_change")
        assert math.isnan(fd.compute_numpy(values, ts))

    def test_uses_last_two_values(self) -> None:
        # long buffer — ROC should only use the last two entries
        values = np.array([100.0] * 98 + [0.0, 10.0])
        ts = np.array([float(i) for i in range(98)] + [98.0, 99.0])
        fd = get_feature_by_name("rate_of_change")
        assert fd.compute_numpy(values, ts) == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# compute_features_numpy (full pipeline)
# ---------------------------------------------------------------------------


class TestComputeFeaturesNumpy:
    def test_returns_all_features(self) -> None:
        values = np.ones(110, dtype=np.float64)
        ts = np.arange(110, dtype=np.float64)
        result = compute_features_numpy(values, ts)
        assert set(result.keys()) == set(get_feature_names())

    def test_constant_series_means_equal_value(self) -> None:
        values = np.ones(110, dtype=np.float64) * 7.0
        ts = np.arange(110, dtype=np.float64)
        result = compute_features_numpy(values, ts)
        for name in ("rolling_mean_10", "rolling_mean_50", "rolling_mean_100"):
            assert result[name] == pytest.approx(7.0), f"{name} mismatch"

    def test_short_buffer_nans_for_large_windows(self) -> None:
        # Only 5 values — windows 10, 50, 100 should all be NaN
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ts = np.arange(5, dtype=np.float64)
        result = compute_features_numpy(values, ts)
        for name in ("rolling_mean_10", "rolling_mean_50", "rolling_mean_100"):
            assert math.isnan(result[name]), f"{name} should be NaN for short buffer"

    def test_accepts_list_inputs(self) -> None:
        # Should coerce plain Python lists to numpy arrays
        result = compute_features_numpy(
            np.array([1.0] * 110),
            np.arange(110, dtype=np.float64),
        )
        assert isinstance(result, dict)
        assert not math.isnan(result["rolling_mean_100"])
