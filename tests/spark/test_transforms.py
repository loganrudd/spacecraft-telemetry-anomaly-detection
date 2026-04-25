"""Tests for spark.transforms — handle_nulls, detect_gaps, normalize."""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# handle_nulls
# ---------------------------------------------------------------------------


class TestHandleNulls:
    def test_no_nulls_unchanged(self, sample_spark_df) -> None:
        from pyspark.sql import functions as F

        from spacecraft_telemetry.spark.transforms import handle_nulls

        result = handle_nulls(sample_spark_df)
        assert result.count() == 100
        assert result.filter(F.col("value").isNull()).count() == 0

    def test_nulls_are_filled(self, nulls_spark_df) -> None:
        from pyspark.sql import functions as F

        from spacecraft_telemetry.spark.transforms import handle_nulls

        assert nulls_spark_df.filter(F.col("value").isNull()).count() == 5
        result = handle_nulls(nulls_spark_df)
        assert result.filter(F.col("value").isNull()).count() == 0

    def test_row_count_preserved_when_no_leading_nulls(self, nulls_spark_df) -> None:
        from spacecraft_telemetry.spark.transforms import handle_nulls

        # nulls_spark_df has nulls at positions [5,20,35,60,80] — none leading
        result = handle_nulls(nulls_spark_df)
        assert result.count() == 100

    def test_forward_fill_value_matches_previous(self, nulls_spark_df) -> None:
        from spacecraft_telemetry.spark.transforms import handle_nulls

        result = handle_nulls(nulls_spark_df)
        rows = result.orderBy("telemetry_timestamp").collect()

        # values = [float(i % 10) * 0.1 for i in range(100)]
        # pos 4: 4 % 10 * 0.1 = 0.4, pos 5: null → forward-filled to 0.4
        assert rows[5]["value"] == pytest.approx(rows[4]["value"], abs=1e-5)

    def test_leading_nulls_are_dropped(self, spark_session, sample_channel_pd) -> None:
        import pandas as pd
        from pyspark.sql import functions as F

        from spacecraft_telemetry.spark.transforms import handle_nulls

        pdf = sample_channel_pd.reset_index().rename(
            columns={"datetime": "telemetry_timestamp", "channel_1": "value"}
        )
        pdf["channel_id"] = "channel_1"
        pdf["mission_id"] = "ESA-Mission1"
        values: list[float | None] = pdf["value"].astype("float64").tolist()
        values[0] = None
        values[1] = None
        values[2] = None
        pdf["value"] = pd.array(values, dtype="Float64")

        df = spark_session.createDataFrame(pdf)
        # createDataFrame maps pd.NA → IEEE 754 NaN, not Spark null — normalise.
        df = df.withColumn("value", F.when(F.isnan("value"), None).otherwise(F.col("value")))
        result = handle_nulls(df)
        assert result.count() == 97

    def test_unsupported_strategy_raises(self, sample_spark_df) -> None:
        from spacecraft_telemetry.spark.transforms import handle_nulls

        with pytest.raises(ValueError, match="backfill"):
            handle_nulls(sample_spark_df, strategy="backfill")


# ---------------------------------------------------------------------------
# detect_gaps
# ---------------------------------------------------------------------------


class TestDetectGaps:
    def test_columns_added(self, sample_spark_df) -> None:
        from spacecraft_telemetry.spark.transforms import detect_gaps

        result = detect_gaps(sample_spark_df)
        assert "is_gap" in result.columns
        assert "segment_id" in result.columns

    def test_no_intermediate_columns_leaked(self, sample_spark_df) -> None:
        from spacecraft_telemetry.spark.transforms import detect_gaps

        result = detect_gaps(sample_spark_df)
        for col in result.columns:
            assert not col.startswith("_"), f"Intermediate column leaked: {col!r}"

    def test_regular_data_no_gaps(self, sample_spark_df) -> None:
        from pyspark.sql import functions as F

        from spacecraft_telemetry.spark.transforms import detect_gaps

        result = detect_gaps(sample_spark_df)
        assert result.filter(F.col("is_gap")).count() == 0

    def test_regular_data_single_segment(self, sample_spark_df) -> None:
        from spacecraft_telemetry.spark.transforms import detect_gaps

        result = detect_gaps(sample_spark_df)
        distinct = sorted(r[0] for r in result.select("segment_id").distinct().collect())
        assert distinct == [0]

    def test_gap_detected_in_irregular_data(self, irregular_spark_df) -> None:
        from pyspark.sql import functions as F

        from spacecraft_telemetry.spark.transforms import detect_gaps

        result = detect_gaps(irregular_spark_df)
        assert result.filter(F.col("is_gap")).count() == 1

    def test_two_segments_after_gap(self, irregular_spark_df) -> None:
        from spacecraft_telemetry.spark.transforms import detect_gaps

        result = detect_gaps(irregular_spark_df)
        distinct = sorted(r[0] for r in result.select("segment_id").distinct().collect())
        assert distinct == [0, 1]

    def test_segment_counts_split_at_gap(self, irregular_spark_df) -> None:
        from pyspark.sql import functions as F

        from spacecraft_telemetry.spark.transforms import detect_gaps

        result = detect_gaps(irregular_spark_df)
        # Gap is between rows 49 and 50 (0-indexed): 50 rows each side
        assert result.filter(F.col("segment_id") == 0).count() == 50
        assert result.filter(F.col("segment_id") == 1).count() == 50

    def test_first_row_is_not_gap(self, irregular_spark_df) -> None:
        from spacecraft_telemetry.spark.transforms import detect_gaps

        result = detect_gaps(irregular_spark_df)
        first_row = result.orderBy("telemetry_timestamp").first()
        assert first_row is not None
        assert first_row["is_gap"] is False

    def test_row_count_preserved(self, sample_spark_df) -> None:
        from spacecraft_telemetry.spark.transforms import detect_gaps

        result = detect_gaps(sample_spark_df)
        assert result.count() == 100


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------


class TestNormalize:
    def test_adds_value_normalized_column(self, sample_spark_df) -> None:
        from spacecraft_telemetry.spark.transforms import normalize

        result, _ = normalize(sample_spark_df)
        assert "value_normalized" in result.columns

    def test_no_intermediate_columns_leaked(self, sample_spark_df) -> None:
        from spacecraft_telemetry.spark.transforms import normalize

        result, _ = normalize(sample_spark_df)
        for col in result.columns:
            assert not col.startswith("_"), f"Intermediate column leaked: {col!r}"

    def test_returns_params_dict(self, sample_spark_df) -> None:
        from spacecraft_telemetry.spark.transforms import normalize

        _, params = normalize(sample_spark_df)
        assert "channel_1" in params
        assert "mean" in params["channel_1"]
        assert "std" in params["channel_1"]

    def test_params_match_actual_stats(self, sample_spark_df) -> None:
        from pyspark.sql import functions as F

        from spacecraft_telemetry.spark.transforms import normalize

        _, params = normalize(sample_spark_df)
        actual_mean = sample_spark_df.select(F.mean("value")).collect()[0][0]
        actual_std = sample_spark_df.select(F.stddev("value")).collect()[0][0]
        assert params["channel_1"]["mean"] == pytest.approx(actual_mean, rel=1e-4)
        assert params["channel_1"]["std"] == pytest.approx(actual_std, rel=1e-4)

    def test_normalized_mean_approx_zero(self, sample_spark_df) -> None:
        from pyspark.sql import functions as F

        from spacecraft_telemetry.spark.transforms import normalize

        result, _ = normalize(sample_spark_df)
        mean_val = result.select(F.mean("value_normalized")).collect()[0][0]
        assert abs(mean_val) < 1e-4

    def test_normalized_std_approx_one(self, sample_spark_df) -> None:
        from pyspark.sql import functions as F

        from spacecraft_telemetry.spark.transforms import normalize

        result, _ = normalize(sample_spark_df)
        std_val = result.select(F.stddev("value_normalized")).collect()[0][0]
        assert std_val == pytest.approx(1.0, rel=0.01)

    def test_row_count_preserved(self, sample_spark_df) -> None:
        from spacecraft_telemetry.spark.transforms import normalize

        result, _ = normalize(sample_spark_df)
        assert result.count() == 100

    def test_constant_channel_no_nulls(self, spark_session) -> None:
        import pandas as pd
        from pyspark.sql import functions as F

        from spacecraft_telemetry.spark.transforms import normalize

        pdf = pd.DataFrame(
            {
                "telemetry_timestamp": pd.date_range("2000-01-01", periods=10, freq="90s"),
                "value": [5.0] * 10,
                "channel_id": ["const_ch"] * 10,
                "mission_id": ["ESA-Mission1"] * 10,
            }
        )
        df = spark_session.createDataFrame(pdf)
        result, params = normalize(df)
        assert result.filter(F.col("value_normalized").isNull()).count() == 0
        assert params["const_ch"]["mean"] == pytest.approx(5.0)

    def test_unsupported_method_raises(self, sample_spark_df) -> None:
        from spacecraft_telemetry.spark.transforms import normalize

        with pytest.raises(ValueError, match="min-max"):
            normalize(sample_spark_df, method="min-max")

    def test_spark_matches_normalize_value(self, spark_session) -> None:
        """Spark normalize() output must match the normalize_value() reference impl row-by-row."""
        import pandas as pd

        from spacecraft_telemetry.features.definitions import normalize_value
        from spacecraft_telemetry.spark.transforms import normalize

        raw_values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        pdf = pd.DataFrame(
            {
                "telemetry_timestamp": pd.date_range("2000-01-01", periods=10, freq="90s"),
                "value": raw_values,
                "channel_id": ["ch_test"] * 10,
                "mission_id": ["ESA-Mission1"] * 10,
            }
        )
        df = spark_session.createDataFrame(pdf)
        result, params = normalize(df)

        mean = params["ch_test"]["mean"]
        std = params["ch_test"]["std"]

        rows = result.orderBy("telemetry_timestamp").collect()
        for row, x in zip(rows, raw_values, strict=True):
            expected = normalize_value(x, mean=mean, std=std)
            assert float(row["value_normalized"]) == pytest.approx(expected, rel=1e-5), (
                f"value={x}: spark={row['value_normalized']}, normalize_value={expected}"
            )

    def test_constant_channel_normalize_value_matches_spark(self, spark_session) -> None:
        """std=0 edge case: both Spark and normalize_value must return 0.0."""
        import pandas as pd

        from spacecraft_telemetry.features.definitions import normalize_value
        from spacecraft_telemetry.spark.transforms import normalize

        pdf = pd.DataFrame(
            {
                "telemetry_timestamp": pd.date_range("2000-01-01", periods=5, freq="90s"),
                "value": [42.0] * 5,
                "channel_id": ["const_ch"] * 5,
                "mission_id": ["ESA-Mission1"] * 5,
            }
        )
        df = spark_session.createDataFrame(pdf)
        result, params = normalize(df)

        std = params["const_ch"]["std"]
        mean = params["const_ch"]["mean"]
        assert std == 0.0

        for row in result.collect():
            spark_val = float(row["value_normalized"])
            ref_val = normalize_value(row["value"], mean=mean, std=std)
            assert spark_val == ref_val == 0.0


# ---------------------------------------------------------------------------
# add_rolling_features
# ---------------------------------------------------------------------------


class TestAddRollingFeatures:
    """Tests for add_rolling_features().

    Uses a minimal synthetic DataFrame with the required columns already present
    (value_normalized, channel_id, segment_id, telemetry_timestamp) so these
    tests are isolated from upstream transforms.
    """

    @pytest.fixture()
    def normalized_df(self, spark_session):
        """110 rows of value_normalized = [0.0, 1.0, ..., 109.0], single segment.

        110 rows is enough for the largest window (100) to produce at least one
        fully-populated row.
        """
        import pandas as pd

        n = 110
        pdf = pd.DataFrame(
            {
                "telemetry_timestamp": pd.date_range("2000-01-01", periods=n, freq="90s"),
                "value_normalized": [float(i) for i in range(n)],
                "channel_id": ["ch1"] * n,
                "mission_id": ["m1"] * n,
                "segment_id": [0] * n,
            }
        )
        return spark_session.createDataFrame(pdf)

    # ------------------------------------------------------------------
    # Schema / structural
    # ------------------------------------------------------------------

    def test_all_feature_columns_added(self, normalized_df) -> None:
        from spacecraft_telemetry.features.definitions import get_feature_names
        from spacecraft_telemetry.spark.transforms import add_rolling_features

        result = add_rolling_features(normalized_df)
        for name in get_feature_names():
            assert name in result.columns, f"Missing column: {name}"

    def test_no_intermediate_columns_leaked(self, normalized_df) -> None:
        from spacecraft_telemetry.spark.transforms import add_rolling_features

        result = add_rolling_features(normalized_df)
        for col in result.columns:
            assert not col.startswith("_"), f"Intermediate column leaked: {col!r}"

    def test_row_count_preserved(self, normalized_df) -> None:
        from spacecraft_telemetry.spark.transforms import add_rolling_features

        result = add_rolling_features(normalized_df)
        assert result.count() == 110

    # ------------------------------------------------------------------
    # Correctness — rolling stats
    # ------------------------------------------------------------------

    def test_rolling_mean_known_value(self, normalized_df) -> None:
        from pyspark.sql import functions as F

        from spacecraft_telemetry.spark.transforms import add_rolling_features

        result = add_rolling_features(normalized_df)
        # Last row: mean of [100, 101, ..., 109] = 104.5
        last = result.orderBy(F.col("telemetry_timestamp").desc()).first()
        assert last is not None
        assert last["rolling_mean_10"] == pytest.approx(104.5, rel=1e-4)

    def test_rolling_std_known_value(self, normalized_df) -> None:
        import math

        import numpy as np
        from pyspark.sql import functions as F

        from spacecraft_telemetry.spark.transforms import add_rolling_features

        result = add_rolling_features(normalized_df)
        last = result.orderBy(F.col("telemetry_timestamp").desc()).first()
        assert last is not None
        expected = float(np.std(list(range(100, 110)), ddof=1))
        assert not math.isnan(last["rolling_std_10"])
        assert last["rolling_std_10"] == pytest.approx(expected, rel=1e-3)

    def test_rolling_min_max_known_values(self, normalized_df) -> None:
        from pyspark.sql import functions as F

        from spacecraft_telemetry.spark.transforms import add_rolling_features

        result = add_rolling_features(normalized_df)
        last = result.orderBy(F.col("telemetry_timestamp").desc()).first()
        assert last is not None
        assert last["rolling_min_10"] == pytest.approx(100.0, rel=1e-4)
        assert last["rolling_max_10"] == pytest.approx(109.0, rel=1e-4)

    # ------------------------------------------------------------------
    # Null behaviour for short buffers
    # ------------------------------------------------------------------

    def test_first_n_minus_1_rows_null_for_rolling_mean_10(self, normalized_df) -> None:

        from spacecraft_telemetry.spark.transforms import add_rolling_features

        result = add_rolling_features(normalized_df)
        rows = result.orderBy("telemetry_timestamp").collect()
        # Rows 0-8 (rn 1-9) must be null; row 9 (rn 10) must be non-null.
        for i in range(9):
            assert rows[i]["rolling_mean_10"] is None, f"row {i} should be null"
        assert rows[9]["rolling_mean_10"] is not None

    def test_rate_of_change_first_row_null(self, normalized_df) -> None:
        from spacecraft_telemetry.spark.transforms import add_rolling_features

        result = add_rolling_features(normalized_df)
        first = result.orderBy("telemetry_timestamp").first()
        assert first is not None
        assert first["rate_of_change"] is None

    def test_rate_of_change_known_value(self, normalized_df) -> None:

        from spacecraft_telemetry.spark.transforms import add_rolling_features

        result = add_rolling_features(normalized_df)
        # Δvalue = 1.0, Δtime = 90 s → ROC = 1/90
        second = result.orderBy("telemetry_timestamp").collect()[1]
        assert second["rate_of_change"] == pytest.approx(1.0 / 90.0, rel=1e-4)

    # ------------------------------------------------------------------
    # Segment boundary — windows must not cross gap boundaries
    # ------------------------------------------------------------------

    def test_windows_do_not_cross_segment_boundaries(self, spark_session) -> None:
        import pandas as pd
        from pyspark.sql import functions as F

        from spacecraft_telemetry.spark.transforms import add_rolling_features

        # Two segments: segment 0 has values [0..9], segment 1 has values [100..109].
        # If windowing respected boundaries, rolling_mean_10 for the last row of
        # segment 1 should be mean([100..109]) = 104.5, not a mixture of both.
        n = 10
        pdf = pd.DataFrame(
            {
                "telemetry_timestamp": pd.date_range("2000-01-01", periods=n * 2, freq="90s"),
                "value_normalized": [float(i) for i in range(n)]
                + [float(100 + i) for i in range(n)],
                "channel_id": ["ch1"] * (n * 2),
                "mission_id": ["m1"] * (n * 2),
                "segment_id": [0] * n + [1] * n,
            }
        )
        df = spark_session.createDataFrame(pdf)
        result = add_rolling_features(df)

        # Last row of segment 1 — full window of 10 rows, all from segment 1.
        last_seg1 = (
            result.filter(F.col("segment_id") == 1)
            .orderBy(F.col("telemetry_timestamp").desc())
            .first()
        )
        assert last_seg1 is not None
        assert last_seg1["rolling_mean_10"] == pytest.approx(104.5, rel=1e-4)

        # First row of segment 1 — insufficient buffer within the segment → null.
        first_seg1 = result.filter(F.col("segment_id") == 1).orderBy("telemetry_timestamp").first()
        assert first_seg1 is not None
        assert first_seg1["rolling_mean_10"] is None

    # ------------------------------------------------------------------
    # Equivalence: Spark output must match the numpy reference impl
    # (train-serve skew prevention test)
    # ------------------------------------------------------------------

    def test_equivalence_with_numpy_reference(self, normalized_df) -> None:
        import math

        import numpy as np
        import pandas as pd
        from pyspark.sql import functions as F

        from spacecraft_telemetry.features.definitions import (
            compute_features_numpy,
            get_feature_names,
        )
        from spacecraft_telemetry.spark.transforms import add_rolling_features

        result = add_rolling_features(normalized_df)
        last_row = result.orderBy(F.col("telemetry_timestamp").desc()).first()
        assert last_row is not None

        # Build the same buffer the numpy reference sees.
        n = 110
        vals = np.array([float(i) for i in range(n)], dtype=np.float64)
        ts_s = np.array(
            [t.timestamp() for t in pd.date_range("2000-01-01", periods=n, freq="90s")],
            dtype=np.float64,
        )
        expected = compute_features_numpy(vals, ts_s)

        for name in get_feature_names():
            spark_val = last_row[name]
            numpy_val = expected[name]
            assert not math.isnan(numpy_val), f"{name} should not be NaN with 110-row buffer"
            assert spark_val is not None, f"{name} should not be null for last row"
            assert float(spark_val) == pytest.approx(numpy_val, rel=1e-3), (
                f"{name}: spark={spark_val}, numpy={numpy_val}"
            )


# ---------------------------------------------------------------------------
# create_windows
# ---------------------------------------------------------------------------


class TestCreateWindows:
    """Tests for create_windows().

    Uses small synthetic DataFrames (≤20 rows) with value_normalized already
    set, isolated from upstream transforms.
    """

    @pytest.fixture()
    def ten_row_df(self, spark_session):
        """10 rows, single segment, value_normalized = [0.0 .. 9.0]."""
        import pandas as pd

        pdf = pd.DataFrame(
            {
                "telemetry_timestamp": pd.date_range("2000-01-01", periods=10, freq="90s"),
                "value_normalized": [float(i) for i in range(10)],
                "channel_id": ["ch1"] * 10,
                "mission_id": ["m1"] * 10,
                "segment_id": [0] * 10,
            }
        )
        return spark_session.createDataFrame(pdf)

    # ------------------------------------------------------------------
    # Row count
    # ------------------------------------------------------------------

    def test_window_count_10_rows_window_3(self, ten_row_df) -> None:
        from spacecraft_telemetry.spark.transforms import create_windows

        # N=10, ws=3, ph=1 → max(0, 10 - 3 - 1 + 1) = 7 windows
        result = create_windows(ten_row_df, window_size=3, prediction_horizon=1)
        assert result.count() == 7

    def test_short_segment_produces_zero_windows(self, spark_session) -> None:
        import pandas as pd

        from spacecraft_telemetry.spark.transforms import create_windows

        # 2 rows < window_size=3 + prediction_horizon=1 → 0 windows
        pdf = pd.DataFrame(
            {
                "telemetry_timestamp": pd.date_range("2000-01-01", periods=2, freq="90s"),
                "value_normalized": [1.0, 2.0],
                "channel_id": ["ch1"] * 2,
                "mission_id": ["m1"] * 2,
                "segment_id": [0] * 2,
            }
        )
        df = spark_session.createDataFrame(pdf)
        result = create_windows(df, window_size=3, prediction_horizon=1)
        assert result.count() == 0

    def test_exact_minimum_segment_produces_one_window(self, spark_session) -> None:
        import pandas as pd

        from spacecraft_telemetry.spark.transforms import create_windows

        # N = window_size + prediction_horizon = 3 + 1 = 4 → exactly 1 window
        pdf = pd.DataFrame(
            {
                "telemetry_timestamp": pd.date_range("2000-01-01", periods=4, freq="90s"),
                "value_normalized": [0.0, 1.0, 2.0, 3.0],
                "channel_id": ["ch1"] * 4,
                "mission_id": ["m1"] * 4,
                "segment_id": [0] * 4,
            }
        )
        df = spark_session.createDataFrame(pdf)
        result = create_windows(df, window_size=3, prediction_horizon=1)
        assert result.count() == 1

    def test_prediction_horizon_2_reduces_count(self, ten_row_df) -> None:
        from spacecraft_telemetry.spark.transforms import create_windows

        # N=10, ws=3, ph=2 → max(0, 10 - 3 - 2 + 1) = 6 windows
        result = create_windows(ten_row_df, window_size=3, prediction_horizon=2)
        assert result.count() == 6

    # ------------------------------------------------------------------
    # Values array correctness
    # ------------------------------------------------------------------

    def test_values_array_length_equals_window_size(self, ten_row_df) -> None:
        from pyspark.sql import functions as F

        from spacecraft_telemetry.spark.transforms import create_windows

        result = create_windows(ten_row_df, window_size=3, prediction_horizon=1)
        bad = result.filter(F.size("values") != 3).count()
        assert bad == 0

    def test_first_window_values(self, ten_row_df) -> None:
        from spacecraft_telemetry.spark.transforms import create_windows

        result = create_windows(ten_row_df, window_size=3, prediction_horizon=1)
        first = result.orderBy("window_start_ts").first()
        assert first is not None
        assert list(first["values"]) == pytest.approx([0.0, 1.0, 2.0], abs=1e-5)

    def test_last_window_values(self, ten_row_df) -> None:
        from pyspark.sql import functions as F

        from spacecraft_telemetry.spark.transforms import create_windows

        result = create_windows(ten_row_df, window_size=3, prediction_horizon=1)
        last = result.orderBy(F.col("window_start_ts").desc()).first()
        assert last is not None
        assert list(last["values"]) == pytest.approx([6.0, 7.0, 8.0], abs=1e-5)

    def test_values_are_oldest_first(self, ten_row_df) -> None:
        """collect_list with an ordered window must preserve ascending order."""
        from spacecraft_telemetry.spark.transforms import create_windows

        result = create_windows(ten_row_df, window_size=3, prediction_horizon=1)
        rows = result.orderBy("window_start_ts").collect()
        for row in rows:
            vals = list(row["values"])
            assert vals == sorted(vals), f"Values not ascending: {vals}"

    # ------------------------------------------------------------------
    # Target correctness
    # ------------------------------------------------------------------

    def test_first_window_target(self, ten_row_df) -> None:
        from spacecraft_telemetry.spark.transforms import create_windows

        result = create_windows(ten_row_df, window_size=3, prediction_horizon=1)
        first = result.orderBy("window_start_ts").first()
        assert first is not None
        # values=[0,1,2], next value is 3.0
        assert first["target"] == pytest.approx(3.0, abs=1e-5)

    def test_prediction_horizon_2_target(self, ten_row_df) -> None:
        from spacecraft_telemetry.spark.transforms import create_windows

        result = create_windows(ten_row_df, window_size=3, prediction_horizon=2)
        first = result.orderBy("window_start_ts").first()
        assert first is not None
        # values=[0,1,2], target 2 steps ahead = 4.0
        assert first["target"] == pytest.approx(4.0, abs=1e-5)

    def test_no_null_targets(self, ten_row_df) -> None:
        from pyspark.sql import functions as F

        from spacecraft_telemetry.spark.transforms import create_windows

        result = create_windows(ten_row_df, window_size=3, prediction_horizon=1)
        assert result.filter(F.col("target").isNull()).count() == 0

    # ------------------------------------------------------------------
    # Timestamps
    # ------------------------------------------------------------------

    def test_window_start_before_window_end(self, ten_row_df) -> None:
        from pyspark.sql import functions as F

        from spacecraft_telemetry.spark.transforms import create_windows

        result = create_windows(ten_row_df, window_size=3, prediction_horizon=1)
        bad = result.filter(F.col("window_start_ts") >= F.col("window_end_ts")).count()
        assert bad == 0

    # ------------------------------------------------------------------
    # Segment boundary isolation
    # ------------------------------------------------------------------

    def test_windows_do_not_cross_segment_boundaries(self, spark_session) -> None:
        import pandas as pd
        from pyspark.sql import functions as F

        from spacecraft_telemetry.spark.transforms import create_windows

        # Two segments: segment 0 values [0..9], segment 1 values [100..109]
        n = 10
        pdf = pd.DataFrame(
            {
                "telemetry_timestamp": pd.date_range("2000-01-01", periods=n * 2, freq="90s"),
                "value_normalized": [float(i) for i in range(n)]
                + [float(100 + i) for i in range(n)],
                "channel_id": ["ch1"] * (n * 2),
                "mission_id": ["m1"] * (n * 2),
                "segment_id": [0] * n + [1] * n,
            }
        )
        df = spark_session.createDataFrame(pdf)
        result = create_windows(df, window_size=3, prediction_horizon=1)

        assert result.count() == 14  # 7 from each segment

        # Every value in segment 0 windows must be < 100
        for row in result.filter(F.col("segment_id") == 0).collect():
            assert all(v < 100 for v in row["values"]), (
                f"Cross-segment leak in seg 0: {row['values']}"
            )

        # Every value in segment 1 windows must be >= 100
        for row in result.filter(F.col("segment_id") == 1).collect():
            assert all(v >= 100 for v in row["values"]), (
                f"Cross-segment leak in seg 1: {row['values']}"
            )

    # ------------------------------------------------------------------
    # Output schema
    # ------------------------------------------------------------------

    def test_output_column_names(self, ten_row_df) -> None:
        from spacecraft_telemetry.spark.transforms import create_windows

        result = create_windows(ten_row_df, window_size=3, prediction_horizon=1)
        expected = {
            "window_id",
            "channel_id",
            "mission_id",
            "segment_id",
            "window_start_ts",
            "window_end_ts",
            "values",
            "target",
        }
        assert set(result.columns) == expected

    def test_window_id_is_unique(self, ten_row_df) -> None:
        from spacecraft_telemetry.spark.transforms import create_windows

        result = create_windows(ten_row_df, window_size=3, prediction_horizon=1)
        total = result.count()
        distinct = result.select("window_id").distinct().count()
        assert total == distinct


# ---------------------------------------------------------------------------
# temporal_train_test_split
# ---------------------------------------------------------------------------


class TestTemporalTrainTestSplit:
    """Tests for temporal_train_test_split().

    Uses sample_spark_df (100 rows, 90s intervals) for the primary split tests.
    """

    def test_train_count_80(self, sample_spark_df) -> None:
        from spacecraft_telemetry.spark.transforms import temporal_train_test_split

        train, _ = temporal_train_test_split(sample_spark_df, train_fraction=0.8)
        assert train.count() == 80

    def test_test_count_20(self, sample_spark_df) -> None:
        from spacecraft_telemetry.spark.transforms import temporal_train_test_split

        _, test = temporal_train_test_split(sample_spark_df, train_fraction=0.8)
        assert test.count() == 20

    def test_total_rows_preserved(self, sample_spark_df) -> None:
        from spacecraft_telemetry.spark.transforms import temporal_train_test_split

        train, test = temporal_train_test_split(sample_spark_df, train_fraction=0.8)
        assert train.count() + test.count() == 100

    def test_train_timestamps_before_test(self, sample_spark_df) -> None:
        from pyspark.sql import functions as F

        from spacecraft_telemetry.spark.transforms import temporal_train_test_split

        train, test = temporal_train_test_split(sample_spark_df, train_fraction=0.8)
        max_train = train.select(F.max("telemetry_timestamp")).collect()[0][0]
        min_test = test.select(F.min("telemetry_timestamp")).collect()[0][0]
        assert max_train < min_test

    def test_no_timestamp_overlap(self, sample_spark_df) -> None:
        from spacecraft_telemetry.spark.transforms import temporal_train_test_split

        train, test = temporal_train_test_split(sample_spark_df, train_fraction=0.8)
        train_ts = {r[0] for r in train.select("telemetry_timestamp").collect()}
        test_ts = {r[0] for r in test.select("telemetry_timestamp").collect()}
        assert train_ts.isdisjoint(test_ts)

    def test_no_intermediate_columns_leaked(self, sample_spark_df) -> None:
        from spacecraft_telemetry.spark.transforms import temporal_train_test_split

        train, test = temporal_train_test_split(sample_spark_df, train_fraction=0.8)
        for df, label in [(train, "train"), (test, "test")]:
            for col in df.columns:
                assert not col.startswith("_"), f"{label}: intermediate column leaked: {col!r}"


# ---------------------------------------------------------------------------
# join_anomaly_labels
# ---------------------------------------------------------------------------


class TestJoinAnomalyLabels:
    """Tests for join_anomaly_labels().

    Uses a minimal windows DataFrame (no 'values' array) since the function
    only requires window_id, channel_id, window_start_ts, window_end_ts.
    """

    @pytest.fixture()
    def windows_df(self, spark_session):
        """4 windows: 3 overlapping the label segments, 1 nominal."""
        import pandas as pd

        base = pd.Timestamp("2000-01-01")
        # Label segments (from labels_pd fixture):
        #   seg 1: rows 10-14 → base+900s .. base+1260s
        #   seg 2: rows 40-44 → base+3600s .. base+3960s
        #   seg 3: rows 70-74 → base+6300s .. base+6660s
        pdf = pd.DataFrame(
            {
                "window_id": [0, 1, 2, 3],
                "channel_id": ["channel_1"] * 4,
                "mission_id": ["ESA-Mission1"] * 4,
                "segment_id": [0] * 4,
                "window_start_ts": [
                    base + pd.Timedelta(seconds=900),  # overlaps label seg 1
                    base + pd.Timedelta(seconds=3600),  # overlaps label seg 2
                    base + pd.Timedelta(seconds=6300),  # overlaps label seg 3
                    base + pd.Timedelta(seconds=2000),  # nominal (between labels)
                ],
                "window_end_ts": [
                    base + pd.Timedelta(seconds=1260),
                    base + pd.Timedelta(seconds=3960),
                    base + pd.Timedelta(seconds=6660),
                    base + pd.Timedelta(seconds=2270),
                ],
                "target": [1.0] * 4,
            }
        )
        return spark_session.createDataFrame(pdf)

    @pytest.fixture()
    def labels_spark_df(self, spark_session):
        """Spark labels DF matching the conftest labels_pd segments."""
        import pandas as pd

        base = pd.Timestamp("2000-01-01")
        pdf = pd.DataFrame(
            {
                "anomaly_id": ["id_1", "id_1", "id_2"],
                "channel_id": ["channel_1", "channel_1", "channel_1"],
                "start_time": [
                    base + pd.Timedelta(seconds=90 * 10),
                    base + pd.Timedelta(seconds=90 * 40),
                    base + pd.Timedelta(seconds=90 * 70),
                ],
                "end_time": [
                    base + pd.Timedelta(seconds=90 * 14),
                    base + pd.Timedelta(seconds=90 * 44),
                    base + pd.Timedelta(seconds=90 * 74),
                ],
            }
        )
        return spark_session.createDataFrame(pdf)

    def test_is_anomaly_column_added(self, windows_df, labels_spark_df) -> None:
        from spacecraft_telemetry.spark.transforms import join_anomaly_labels

        result = join_anomaly_labels(windows_df, labels_spark_df)
        assert "is_anomaly" in result.columns

    def test_row_count_preserved(self, windows_df, labels_spark_df) -> None:
        from spacecraft_telemetry.spark.transforms import join_anomaly_labels

        result = join_anomaly_labels(windows_df, labels_spark_df)
        assert result.count() == windows_df.count()

    def test_overlapping_windows_flagged(self, windows_df, labels_spark_df) -> None:
        from pyspark.sql import functions as F

        from spacecraft_telemetry.spark.transforms import join_anomaly_labels

        result = join_anomaly_labels(windows_df, labels_spark_df)
        anomaly_ids = {
            r["window_id"]
            for r in result.filter(F.col("is_anomaly")).select("window_id").collect()
        }
        assert 0 in anomaly_ids  # overlaps label seg 1
        assert 1 in anomaly_ids  # overlaps label seg 2
        assert 2 in anomaly_ids  # overlaps label seg 3

    def test_nominal_window_not_flagged(self, windows_df, labels_spark_df) -> None:
        from pyspark.sql import functions as F

        from spacecraft_telemetry.spark.transforms import join_anomaly_labels

        result = join_anomaly_labels(windows_df, labels_spark_df)
        row = result.filter(F.col("window_id") == 3).first()
        assert row is not None
        assert row["is_anomaly"] is False

    def test_empty_labels_all_false(self, windows_df, spark_session) -> None:
        from pyspark.sql import functions as F
        from pyspark.sql.types import (
            StringType,
            StructField,
            StructType,
            TimestampType,
        )

        from spacecraft_telemetry.spark.transforms import join_anomaly_labels

        empty_labels = spark_session.createDataFrame(
            [],
            schema=StructType(
                [
                    StructField("anomaly_id", StringType()),
                    StructField("channel_id", StringType()),
                    StructField("start_time", TimestampType()),
                    StructField("end_time", TimestampType()),
                ]
            ),
        )
        result = join_anomaly_labels(windows_df, empty_labels)
        assert result.filter(F.col("is_anomaly")).count() == 0


# ---------------------------------------------------------------------------
# exclude_anomalies_from_train
# ---------------------------------------------------------------------------


class TestExcludeAnomaliesFromTrain:
    """Tests for exclude_anomalies_from_train().

    Uses a tiny synthetic DataFrame with a known mix of True/False is_anomaly flags.
    """

    @pytest.fixture()
    def flagged_df(self, spark_session):
        """5 rows: 2 anomalies (ids 0, 2) and 3 nominal (ids 1, 3, 4)."""
        import pandas as pd

        pdf = pd.DataFrame(
            {
                "window_id": list(range(5)),
                "is_anomaly": [True, False, True, False, False],
                "value_normalized": [float(i) for i in range(5)],
            }
        )
        return spark_session.createDataFrame(pdf)

    def test_anomaly_rows_removed(self, flagged_df) -> None:
        from spacecraft_telemetry.spark.transforms import exclude_anomalies_from_train

        result = exclude_anomalies_from_train(flagged_df)
        assert result.count() == 3

    def test_no_anomaly_rows_in_result(self, flagged_df) -> None:
        from pyspark.sql import functions as F

        from spacecraft_telemetry.spark.transforms import exclude_anomalies_from_train

        result = exclude_anomalies_from_train(flagged_df)
        assert result.filter(F.col("is_anomaly")).count() == 0

    def test_nominal_window_ids_preserved(self, flagged_df) -> None:
        from spacecraft_telemetry.spark.transforms import exclude_anomalies_from_train

        result = exclude_anomalies_from_train(flagged_df)
        ids = {r["window_id"] for r in result.select("window_id").collect()}
        assert ids == {1, 3, 4}

    def test_all_nominal_unchanged(self, spark_session) -> None:
        """DataFrame with no anomalies is returned as-is."""
        import pandas as pd

        from spacecraft_telemetry.spark.transforms import exclude_anomalies_from_train

        pdf = pd.DataFrame(
            {
                "window_id": list(range(3)),
                "is_anomaly": [False, False, False],
                "value_normalized": [1.0, 2.0, 3.0],
            }
        )
        df = spark_session.createDataFrame(pdf)
        result = exclude_anomalies_from_train(df)
        assert result.count() == 3

    def test_all_anomaly_returns_empty_with_correct_schema(self, spark_session) -> None:
        """T2: all-anomaly input produces an empty DataFrame with the right columns."""
        import pandas as pd
        from pyspark.sql import functions as F

        from spacecraft_telemetry.spark.transforms import exclude_anomalies_from_train

        pdf = pd.DataFrame(
            {
                "window_id": list(range(5)),
                "is_anomaly": [True, True, True, True, True],
                "value_normalized": [float(i) for i in range(5)],
                "channel_id": ["ch1"] * 5,
            }
        )
        df = spark_session.createDataFrame(pdf)
        result = exclude_anomalies_from_train(df)

        assert result.count() == 0
        assert "window_id" in result.columns
        assert "is_anomaly" in result.columns
        assert "value_normalized" in result.columns
        assert result.filter(F.col("is_anomaly")).count() == 0


# ---------------------------------------------------------------------------
# T1: Value-level correctness tests
# ---------------------------------------------------------------------------


class TestHandleNullsValues:
    """T1 additions: assert the computed VALUES, not just counts/columns."""

    def test_forward_fill_not_backward(self, spark_session) -> None:
        """Null must take the PRIOR value, not the next one.

        Arrange: [1.0, null, 3.0] → forward fill gives [1.0, 1.0, 3.0].
        If the fill were backward the null would become 3.0.
        """
        import pandas as pd
        from pyspark.sql import functions as F

        from spacecraft_telemetry.spark.transforms import handle_nulls

        pdf = pd.DataFrame(
            {
                "telemetry_timestamp": pd.date_range("2000-01-01", periods=3, freq="90s"),
                "value": [1.0, None, 3.0],
                "channel_id": ["ch1"] * 3,
                "mission_id": ["m1"] * 3,
            }
        )
        df = spark_session.createDataFrame(pdf)
        # createDataFrame maps Python None → IEEE 754 NaN, not Spark null — normalise.
        df = df.withColumn("value", F.when(F.isnan("value"), None).otherwise(F.col("value")))
        result = handle_nulls(df)
        rows = result.orderBy("telemetry_timestamp").collect()
        assert float(rows[1]["value"]) == pytest.approx(1.0, abs=1e-5), (
            f"Expected forward-fill value 1.0, got {rows[1]['value']}"
        )

    def test_all_null_returns_empty(self, spark_session) -> None:
        """Q3 runtime check: a channel with all nulls should produce zero rows."""
        from pyspark.sql import Row
        from pyspark.sql.types import FloatType, StringType, StructField, StructType, TimestampType

        from spacecraft_telemetry.spark.transforms import handle_nulls

        import datetime

        schema = StructType(
            [
                StructField("telemetry_timestamp", TimestampType(), nullable=False),
                StructField("value", FloatType(), nullable=True),
                StructField("channel_id", StringType(), nullable=False),
                StructField("mission_id", StringType(), nullable=False),
            ]
        )
        base_ts = datetime.datetime(2000, 1, 1)
        rows = [
            Row(
                telemetry_timestamp=base_ts + datetime.timedelta(seconds=90 * i),
                value=None,
                channel_id="ch1",
                mission_id="m1",
            )
            for i in range(4)
        ]
        df = spark_session.createDataFrame(rows, schema=schema)
        result = handle_nulls(df)
        assert result.count() == 0


class TestDetectGapsValues:
    """T1 additions: verify the gap threshold is applied correctly."""

    def test_interval_exactly_at_threshold_is_not_a_gap(self, spark_session) -> None:
        """interval == gap_multiplier * median is NOT a gap (strictly greater required)."""
        import pandas as pd
        from pyspark.sql import functions as F

        from spacecraft_telemetry.spark.transforms import detect_gaps

        # median interval = 10s, gap_multiplier = 3.0 → threshold = 30s
        # row 2 interval = exactly 30s → NOT a gap
        pdf = pd.DataFrame(
            {
                "telemetry_timestamp": [
                    pd.Timestamp("2000-01-01 00:00:00"),
                    pd.Timestamp("2000-01-01 00:00:10"),  # +10s (median)
                    pd.Timestamp("2000-01-01 00:00:40"),  # +30s == threshold, NOT gap
                    pd.Timestamp("2000-01-01 00:00:50"),  # +10s
                ],
                "value": [1.0, 2.0, 3.0, 4.0],
                "channel_id": ["ch1"] * 4,
                "mission_id": ["m1"] * 4,
            }
        )
        df = spark_session.createDataFrame(pdf)
        result = detect_gaps(df, gap_multiplier=3.0)
        assert result.filter(F.col("is_gap")).count() == 0

    def test_interval_just_above_threshold_is_a_gap(self, spark_session) -> None:
        """interval > gap_multiplier * median IS a gap."""
        import pandas as pd
        from pyspark.sql import functions as F

        from spacecraft_telemetry.spark.transforms import detect_gaps

        # median interval = 10s, gap_multiplier = 3.0 → threshold = 30s
        # row 2 interval = 31s > 30s → IS a gap
        pdf = pd.DataFrame(
            {
                "telemetry_timestamp": [
                    pd.Timestamp("2000-01-01 00:00:00"),
                    pd.Timestamp("2000-01-01 00:00:10"),  # +10s (median)
                    pd.Timestamp("2000-01-01 00:00:41"),  # +31s > threshold → gap
                    pd.Timestamp("2000-01-01 00:00:51"),  # +10s
                ],
                "value": [1.0, 2.0, 3.0, 4.0],
                "channel_id": ["ch1"] * 4,
                "mission_id": ["m1"] * 4,
            }
        )
        df = spark_session.createDataFrame(pdf)
        result = detect_gaps(df, gap_multiplier=3.0)
        gaps = result.filter(F.col("is_gap")).orderBy("telemetry_timestamp").collect()
        assert len(gaps) == 1
        assert float(gaps[0]["value"]) == pytest.approx(3.0, abs=1e-5)


# ---------------------------------------------------------------------------
# T3: create_windows parametrized over realistic window sizes
# ---------------------------------------------------------------------------


class TestCreateWindowsLargeWindows:
    """T3: Verify window count and values-array length hold across realistic window sizes.

    Tests use a 300-row fixture so window_size=250 produces windows.
    Expected count formula: max(0, N - window_size - prediction_horizon + 1).
    """

    @pytest.fixture()
    def three_hundred_row_df(self, spark_session):
        import pandas as pd

        n = 300
        pdf = pd.DataFrame(
            {
                "telemetry_timestamp": pd.date_range("2000-01-01", periods=n, freq="90s"),
                "value_normalized": [float(i) for i in range(n)],
                "channel_id": ["ch1"] * n,
                "mission_id": ["m1"] * n,
                "segment_id": [0] * n,
            }
        )
        return spark_session.createDataFrame(pdf)

    @pytest.mark.parametrize(
        "window_size",
        [3, 10, 50, 250],
        ids=["ws3", "ws10", "ws50", "ws250"],
    )
    def test_window_count_formula(self, three_hundred_row_df, window_size) -> None:
        from spacecraft_telemetry.spark.transforms import create_windows

        ph = 1
        n = 300
        expected = max(0, n - window_size - ph + 1)
        result = create_windows(
            three_hundred_row_df, window_size=window_size, prediction_horizon=ph
        )
        assert result.count() == expected

    @pytest.mark.parametrize(
        "window_size",
        [3, 10, 50, 250],
        ids=["ws3", "ws10", "ws50", "ws250"],
    )
    def test_values_array_length(self, three_hundred_row_df, window_size) -> None:
        from pyspark.sql import functions as F

        from spacecraft_telemetry.spark.transforms import create_windows

        result = create_windows(
            three_hundred_row_df, window_size=window_size, prediction_horizon=1
        )
        bad = result.filter(F.size("values") != window_size).count()
        assert bad == 0, f"Some windows had values array != {window_size} elements"


# ---------------------------------------------------------------------------
# T4: join_anomaly_labels boundary semantics
# ---------------------------------------------------------------------------


class TestJoinAnomalyLabelsBoundaries:
    """T4: Verify the strict half-open interval semantics of join_anomaly_labels.

    The join uses strict > and <, so boundary-touching windows are NOT anomalous.
    """

    @pytest.fixture()
    def label_df(self, spark_session):
        """Single label: start=100s, end=200s from epoch."""
        import pandas as pd

        base = pd.Timestamp("2000-01-01")
        pdf = pd.DataFrame(
            {
                "anomaly_id": ["L1"],
                "channel_id": ["ch1"],
                "start_time": [base + pd.Timedelta(seconds=100)],
                "end_time": [base + pd.Timedelta(seconds=200)],
            }
        )
        return spark_session.createDataFrame(pdf)

    @pytest.mark.parametrize(
        "win_start_s,win_end_s,expected_anomaly,description",
        [
            # Boundary-touching: window ends exactly when label starts → NOT anomaly
            (50, 100, False, "window_end == label_start"),
            # Boundary-touching: window starts exactly when label ends → NOT anomaly
            (200, 250, False, "window_start == label_end"),
            # Clearly inside
            (120, 180, True, "window fully inside label"),
            # Window starts exactly when label starts → IS anomaly
            (100, 150, True, "window_start == label_start"),
            # Window ends exactly when label ends → IS anomaly
            (150, 200, True, "window_end == label_end"),
            # Clearly outside before label
            (0, 50, False, "window entirely before label"),
            # Clearly outside after label
            (250, 300, False, "window entirely after label"),
        ],
    )
    def test_boundary_semantics(
        self,
        spark_session,
        label_df,
        win_start_s,
        win_end_s,
        expected_anomaly,
        description,
    ) -> None:
        import pandas as pd

        from spacecraft_telemetry.spark.transforms import join_anomaly_labels

        base = pd.Timestamp("2000-01-01")
        pdf = pd.DataFrame(
            {
                "window_id": [0],
                "channel_id": ["ch1"],
                "mission_id": ["m1"],
                "segment_id": [0],
                "window_start_ts": [base + pd.Timedelta(seconds=win_start_s)],
                "window_end_ts": [base + pd.Timedelta(seconds=win_end_s)],
                "target": [0.0],
            }
        )
        df = spark_session.createDataFrame(pdf)
        result = join_anomaly_labels(df, label_df)
        row = result.collect()[0]
        assert row["is_anomaly"] is expected_anomaly, (
            f"{description}: expected is_anomaly={expected_anomaly}, got {row['is_anomaly']}"
        )
