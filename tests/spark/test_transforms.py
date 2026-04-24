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

    def test_row_count_preserved_when_no_leading_nulls(
        self, nulls_spark_df
    ) -> None:
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

        pdf = pd.DataFrame({
            "telemetry_timestamp": pd.date_range("2000-01-01", periods=10, freq="90s"),
            "value": [5.0] * 10,
            "channel_id": ["const_ch"] * 10,
            "mission_id": ["ESA-Mission1"] * 10,
        })
        df = spark_session.createDataFrame(pdf)
        result, params = normalize(df)
        assert result.filter(F.col("value_normalized").isNull()).count() == 0
        assert params["const_ch"]["mean"] == pytest.approx(5.0)

    def test_unsupported_method_raises(self, sample_spark_df) -> None:
        from spacecraft_telemetry.spark.transforms import normalize

        with pytest.raises(ValueError, match="min-max"):
            normalize(sample_spark_df, method="min-max")


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
        pdf = pd.DataFrame({
            "telemetry_timestamp": pd.date_range("2000-01-01", periods=n, freq="90s"),
            "value_normalized": [float(i) for i in range(n)],
            "channel_id": ["ch1"] * n,
            "mission_id": ["m1"] * n,
            "segment_id": [0] * n,
        })
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

    def test_first_n_minus_1_rows_null_for_rolling_mean_10(
        self, normalized_df
    ) -> None:
        from pyspark.sql import functions as F

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
        from pyspark.sql import functions as F

        from spacecraft_telemetry.spark.transforms import add_rolling_features

        result = add_rolling_features(normalized_df)
        # Δvalue = 1.0, Δtime = 90 s → ROC = 1/90
        second = result.orderBy("telemetry_timestamp").collect()[1]
        assert second["rate_of_change"] == pytest.approx(1.0 / 90.0, rel=1e-4)

    # ------------------------------------------------------------------
    # Segment boundary — windows must not cross gap boundaries
    # ------------------------------------------------------------------

    def test_windows_do_not_cross_segment_boundaries(
        self, spark_session
    ) -> None:
        import pandas as pd
        from pyspark.sql import functions as F

        from spacecraft_telemetry.spark.transforms import add_rolling_features

        # Two segments: segment 0 has values [0..9], segment 1 has values [100..109].
        # If windowing respected boundaries, rolling_mean_10 for the last row of
        # segment 1 should be mean([100..109]) = 104.5, not a mixture of both.
        n = 10
        pdf = pd.DataFrame({
            "telemetry_timestamp": pd.date_range("2000-01-01", periods=n * 2, freq="90s"),
            "value_normalized": [float(i) for i in range(n)] + [float(100 + i) for i in range(n)],
            "channel_id": ["ch1"] * (n * 2),
            "mission_id": ["m1"] * (n * 2),
            "segment_id": [0] * n + [1] * n,
        })
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
        first_seg1 = (
            result.filter(F.col("segment_id") == 1)
            .orderBy("telemetry_timestamp")
            .first()
        )
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
            assert float(spark_val) == pytest.approx(
                numpy_val, rel=1e-3
            ), f"{name}: spark={spark_val}, numpy={numpy_val}"
