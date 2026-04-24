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
