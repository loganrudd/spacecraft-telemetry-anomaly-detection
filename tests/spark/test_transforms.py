"""Tests for spark.transforms — handle_nulls, detect_gaps, normalize, label_timesteps."""

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
# label_timesteps
# ---------------------------------------------------------------------------


class TestLabelTimesteps:
    """Tests for label_timesteps().

    label_timesteps replaces join_anomaly_labels + exclude_anomalies_from_train.
    It adds an is_anomaly boolean per timestep via a broadcast join on
    (channel_id, ts >= start_time, ts < end_time).
    """

    @pytest.fixture()
    def series_df(self, spark_session):
        """10 timestep rows for channel_1, regular 90s intervals."""
        import pandas as pd

        base = pd.Timestamp("2000-01-01")
        pdf = pd.DataFrame(
            {
                "telemetry_timestamp": [base + pd.Timedelta(seconds=90 * i) for i in range(10)],
                "value_normalized": [float(i) for i in range(10)],
                "channel_id": ["channel_1"] * 10,
                "mission_id": ["ESA-Mission1"] * 10,
                "segment_id": [0] * 10,
            }
        )
        return spark_session.createDataFrame(pdf)

    @pytest.fixture()
    def labels_df(self, spark_session):
        """One anomaly segment covering rows 3-5 (ts 270s-450s)."""
        import pandas as pd

        base = pd.Timestamp("2000-01-01")
        pdf = pd.DataFrame(
            {
                "anomaly_id": ["a1"],
                "channel_id": ["channel_1"],
                "start_time": [base + pd.Timedelta(seconds=270)],  # row 3
                "end_time": [base + pd.Timedelta(seconds=450)],    # row 5
            }
        )
        return spark_session.createDataFrame(pdf)

    def test_is_anomaly_column_added(self, series_df, labels_df) -> None:
        from spacecraft_telemetry.spark.transforms import label_timesteps

        result = label_timesteps(series_df, labels_df)
        assert "is_anomaly" in result.columns

    def test_row_count_preserved(self, series_df, labels_df) -> None:
        from spacecraft_telemetry.spark.transforms import label_timesteps

        result = label_timesteps(series_df, labels_df)
        assert result.count() == series_df.count()

    def test_anomalous_rows_flagged(self, spark_session, series_df, labels_df) -> None:
        """Rows 3 and 4 fall inside [270s, 450s); row 5 at ts=450s is outside (half-open)."""
        from pyspark.sql import functions as F

        from spacecraft_telemetry.spark.transforms import label_timesteps

        result = label_timesteps(series_df, labels_df)
        anomalous = {
            int(r["value_normalized"])
            for r in result.filter(F.col("is_anomaly")).collect()
        }
        assert anomalous == {3, 4}  # rows with value_normalized 3.0 and 4.0

    def test_nominal_rows_not_flagged(self, series_df, labels_df) -> None:
        from pyspark.sql import functions as F

        from spacecraft_telemetry.spark.transforms import label_timesteps

        result = label_timesteps(series_df, labels_df)
        nominal_count = result.filter(~F.col("is_anomaly")).count()
        assert nominal_count == 8  # 10 total - 2 anomalous

    def test_empty_labels_all_false(self, series_df, spark_session) -> None:
        from pyspark.sql import functions as F
        from pyspark.sql.types import (
            StringType,
            StructField,
            StructType,
            TimestampType,
        )

        from spacecraft_telemetry.spark.transforms import label_timesteps

        empty_labels = spark_session.createDataFrame(
            [],
            schema=StructType([
                StructField("anomaly_id", StringType()),
                StructField("channel_id", StringType()),
                StructField("start_time", TimestampType()),
                StructField("end_time", TimestampType()),
            ]),
        )
        result = label_timesteps(series_df, empty_labels)
        assert result.filter(F.col("is_anomaly")).count() == 0

    def test_missing_channel_all_false(self, series_df, spark_session) -> None:
        """Labels for a different channel_id do not flag this channel's rows."""
        import pandas as pd
        from pyspark.sql import functions as F

        from spacecraft_telemetry.spark.transforms import label_timesteps

        base = pd.Timestamp("2000-01-01")
        pdf = pd.DataFrame(
            {
                "anomaly_id": ["a1"],
                "channel_id": ["other_channel"],  # different channel
                "start_time": [base + pd.Timedelta(seconds=270)],
                "end_time": [base + pd.Timedelta(seconds=450)],
            }
        )
        other_labels = spark_session.createDataFrame(pdf)
        result = label_timesteps(series_df, other_labels)
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
        import datetime

        from pyspark.sql import Row
        from pyspark.sql.types import FloatType, StringType, StructField, StructType, TimestampType

        from spacecraft_telemetry.spark.transforms import handle_nulls

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

