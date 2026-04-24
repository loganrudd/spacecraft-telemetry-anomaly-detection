"""Tests for spark.io — read_channel, read_labels, write_windows, write_features."""

from __future__ import annotations

from pathlib import Path

import pytest

from spacecraft_telemetry.spark.io import (
    read_channel,
    read_labels,
    write_features,
    write_windows,
)


# ---------------------------------------------------------------------------
# read_channel
# ---------------------------------------------------------------------------


class TestReadChannel:
    def test_column_names(self, spark_session, sample_channel_parquet: Path) -> None:
        df = read_channel(spark_session, sample_channel_parquet, "channel_1", "ESA-Mission1")
        assert df.columns == ["telemetry_timestamp", "value", "channel_id", "mission_id"]

    def test_row_count(self, spark_session, sample_channel_parquet: Path) -> None:
        df = read_channel(spark_session, sample_channel_parquet, "channel_1", "ESA-Mission1")
        assert df.count() == 100

    def test_telemetry_timestamp_is_timestamp_type(
        self, spark_session, sample_channel_parquet: Path
    ) -> None:
        from pyspark.sql.types import TimestampType

        df = read_channel(spark_session, sample_channel_parquet, "channel_1", "ESA-Mission1")
        assert isinstance(df.schema["telemetry_timestamp"].dataType, TimestampType)

    def test_value_is_float_type(
        self, spark_session, sample_channel_parquet: Path
    ) -> None:
        from pyspark.sql.types import FloatType

        df = read_channel(spark_session, sample_channel_parquet, "channel_1", "ESA-Mission1")
        assert isinstance(df.schema["value"].dataType, FloatType)

    def test_channel_id_literal(
        self, spark_session, sample_channel_parquet: Path
    ) -> None:
        df = read_channel(spark_session, sample_channel_parquet, "channel_1", "ESA-Mission1")
        distinct = [r["channel_id"] for r in df.select("channel_id").distinct().collect()]
        assert distinct == ["channel_1"]

    def test_mission_id_literal(
        self, spark_session, sample_channel_parquet: Path
    ) -> None:
        df = read_channel(spark_session, sample_channel_parquet, "channel_1", "ESA-Mission1")
        distinct = [r["mission_id"] for r in df.select("mission_id").distinct().collect()]
        assert distinct == ["ESA-Mission1"]

    def test_no_nulls_in_sample_data(
        self, spark_session, sample_channel_parquet: Path
    ) -> None:
        from pyspark.sql import functions as F

        df = read_channel(spark_session, sample_channel_parquet, "channel_1", "ESA-Mission1")
        null_count = df.filter(F.col("value").isNull()).count()
        assert null_count == 0

    def test_wrong_channel_id_raises(
        self, spark_session, sample_channel_parquet: Path
    ) -> None:
        with pytest.raises(ValueError, match="channel_99"):
            read_channel(spark_session, sample_channel_parquet, "channel_99", "ESA-Mission1")

    def test_timestamps_are_ordered(
        self, spark_session, sample_channel_parquet: Path
    ) -> None:
        from pyspark.sql import functions as F
        from pyspark.sql.window import Window

        df = read_channel(spark_session, sample_channel_parquet, "channel_1", "ESA-Mission1")
        # Add row numbers by original order and by timestamp order — they should match
        w_orig = Window.orderBy(F.monotonically_increasing_id())
        w_ts = Window.orderBy("telemetry_timestamp")
        df_orig = df.withColumn("rn_orig", F.row_number().over(w_orig))
        df_ts = df.withColumn("rn_ts", F.row_number().over(w_ts))
        # Both orderings should give the same sequence of timestamps
        orig_ts = [r[0] for r in df_orig.orderBy("rn_orig").select("telemetry_timestamp").collect()]
        ts_ts = [r[0] for r in df_ts.orderBy("rn_ts").select("telemetry_timestamp").collect()]
        assert orig_ts == ts_ts


# ---------------------------------------------------------------------------
# read_labels
# ---------------------------------------------------------------------------


class TestReadLabels:
    def test_column_names(self, spark_session, labels_csv: Path) -> None:
        df = read_labels(spark_session, labels_csv)
        assert df.columns == ["anomaly_id", "channel_id", "start_time", "end_time"]

    def test_row_count(self, spark_session, labels_csv: Path) -> None:
        df = read_labels(spark_session, labels_csv)
        assert df.count() == 3

    def test_start_time_is_timestamp(self, spark_session, labels_csv: Path) -> None:
        from pyspark.sql.types import TimestampType

        df = read_labels(spark_session, labels_csv)
        assert isinstance(df.schema["start_time"].dataType, TimestampType)

    def test_end_time_is_timestamp(self, spark_session, labels_csv: Path) -> None:
        from pyspark.sql.types import TimestampType

        df = read_labels(spark_session, labels_csv)
        assert isinstance(df.schema["end_time"].dataType, TimestampType)

    def test_channel_id_values(self, spark_session, labels_csv: Path) -> None:
        df = read_labels(spark_session, labels_csv)
        channels = {r["channel_id"] for r in df.select("channel_id").collect()}
        assert channels == {"channel_1"}

    def test_start_before_end(self, spark_session, labels_csv: Path) -> None:
        from pyspark.sql import functions as F

        df = read_labels(spark_session, labels_csv)
        invalid = df.filter(F.col("start_time") >= F.col("end_time")).count()
        assert invalid == 0

    def test_no_null_timestamps(self, spark_session, labels_csv: Path) -> None:
        from pyspark.sql import functions as F

        df = read_labels(spark_session, labels_csv)
        nulls = df.filter(
            F.col("start_time").isNull() | F.col("end_time").isNull()
        ).count()
        assert nulls == 0


# ---------------------------------------------------------------------------
# write_windows
# ---------------------------------------------------------------------------


class TestWriteWindows:
    def test_writes_parquet_files(
        self, spark_session, sample_spark_df, tmp_path: Path
    ) -> None:
        out = tmp_path / "windows"
        write_windows(sample_spark_df, out)
        result = spark_session.read.parquet(str(out))
        assert result.count() == 100

    def test_partition_directories_created(
        self, sample_spark_df, tmp_path: Path
    ) -> None:
        out = tmp_path / "windows"
        write_windows(sample_spark_df, out)
        assert (out / "mission_id=ESA-Mission1").is_dir()
        assert (out / "mission_id=ESA-Mission1" / "channel_id=channel_1").is_dir()

    def test_overwrite_does_not_duplicate(
        self, spark_session, sample_spark_df, tmp_path: Path
    ) -> None:
        out = tmp_path / "windows"
        write_windows(sample_spark_df, out)
        write_windows(sample_spark_df, out)  # second write overwrites, not appends
        result = spark_session.read.parquet(str(out))
        assert result.count() == 100

    def test_schema_roundtrip(
        self, spark_session, sample_spark_df, tmp_path: Path
    ) -> None:
        out = tmp_path / "windows"
        write_windows(sample_spark_df, out)
        result = spark_session.read.parquet(str(out))
        # Partition columns (mission_id, channel_id) are recovered from path metadata
        assert "telemetry_timestamp" in result.columns
        assert "value" in result.columns


# ---------------------------------------------------------------------------
# write_features
# ---------------------------------------------------------------------------


class TestWriteFeatures:
    def test_writes_parquet_files(
        self, spark_session, sample_spark_df, tmp_path: Path
    ) -> None:
        out = tmp_path / "features"
        write_features(sample_spark_df, out)
        result = spark_session.read.parquet(str(out))
        assert result.count() == 100

    def test_partition_directories_created(
        self, sample_spark_df, tmp_path: Path
    ) -> None:
        out = tmp_path / "features"
        write_features(sample_spark_df, out)
        assert (out / "mission_id=ESA-Mission1").is_dir()
        assert (out / "mission_id=ESA-Mission1" / "channel_id=channel_1").is_dir()
