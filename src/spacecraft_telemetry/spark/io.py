"""Spark I/O: read raw channel Parquet + labels CSV, write processed output.

All reads produce DataFrames in the standard column schema defined in schemas.py.
All writes partition by mission_id and channel_id for downstream parallel reads.
"""

from __future__ import annotations

from pathlib import Path

from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType

from spacecraft_telemetry.core.logging import get_logger

log = get_logger(__name__)

# ISO 8601 format used in labels.csv StartTime/EndTime columns.
# ESA labels may include sub-second precision (e.g. "2004-12-01T20:42:15.429Z").
# We strip the fractional part before parsing so one format handles both variants.
_LABEL_TS_FORMAT = "yyyy-MM-dd'T'HH:mm:ss'Z'"


def read_channel(
    spark: SparkSession,
    path: Path,
    channel_id: str,
    mission_id: str,
) -> DataFrame:
    """Read a raw channel Parquet file and return a standardised DataFrame.

    Raw files written by ingest/sample.py have two columns:
      - One named after the channel (e.g. "channel_1") — float32 telemetry values
      - One named "datetime" — the pandas DatetimeIndex stored as a regular column
        by PyArrow when writing Parquet

    Output schema matches RAW_CHANNEL_SCHEMA in schemas.py:
        telemetry_timestamp  TimestampType
        value                FloatType  (nullable — nulls cleaned in transforms step)
        channel_id           StringType
        mission_id           StringType

    Args:
        spark: Active SparkSession.
        path: Path to the .parquet file.
        channel_id: Expected value column name (e.g. "channel_1"). Must match the
                    column name inside the file.
        mission_id: Mission identifier added as a literal column (e.g. "ESA-Mission1").

    Raises:
        ValueError: If either expected column is absent from the file.
    """
    df = spark.read.parquet(str(path))

    if channel_id not in df.columns:
        raise ValueError(
            f"Expected value column {channel_id!r} not found in {path}. "
            f"Available columns: {df.columns}"
        )
    if "datetime" not in df.columns:
        raise ValueError(
            f"Expected timestamp column 'datetime' not found in {path}. "
            f"Available columns: {df.columns}"
        )

    log.info("read_channel", path=str(path), channel_id=channel_id, mission_id=mission_id)

    return (
        df.withColumnRenamed("datetime", "telemetry_timestamp")
        .withColumnRenamed(channel_id, "value")
        .withColumn("value", F.col("value").cast(FloatType()))
        .withColumn("channel_id", F.lit(channel_id))
        .withColumn("mission_id", F.lit(mission_id))
        .select("telemetry_timestamp", "value", "channel_id", "mission_id")
    )


def read_labels(spark: SparkSession, path: Path) -> DataFrame:
    """Read a labels CSV and return a standardised DataFrame.

    Raw CSV has columns: ID, Channel, StartTime, EndTime
    StartTime/EndTime are ISO 8601 strings (e.g. "2000-01-01T00:15:00Z").

    Output schema matches LABELS_SCHEMA in schemas.py:
        anomaly_id   StringType
        channel_id   StringType
        start_time   TimestampType
        end_time     TimestampType

    Args:
        spark: Active SparkSession.
        path: Path to the labels.csv file.
    """
    df = spark.read.option("header", "true").csv(str(path))

    log.info("read_labels", path=str(path))

    def _parse_ts(col_name: str) -> Column:
        # Strip optional sub-second component (.NNN) so one format handles both
        # whole-second ("...15Z") and millisecond ("...15.429Z") timestamps.
        normalized = F.regexp_replace(F.col(col_name), r"\.\d+Z$", "Z")
        return F.to_timestamp(normalized, _LABEL_TS_FORMAT)

    return (
        df.withColumnRenamed("ID", "anomaly_id")
        .withColumnRenamed("Channel", "channel_id")
        .withColumn("start_time", _parse_ts("StartTime"))
        .withColumn("end_time", _parse_ts("EndTime"))
        .select("anomaly_id", "channel_id", "start_time", "end_time")
    )


def write_series(
    df: DataFrame,
    output_path: Path,
    mode: str = "overwrite",
) -> None:
    """Write per-timestep series data as partitioned Parquet.

    Partition layout: {output_path}/mission_id={M}/channel_id={C}/part-*.parquet
    PyTorch DataLoader reads a single channel partition and windows on-the-fly,
    so window_size is not baked into the on-disk schema (see Plan 002.5).

    Args:
        df: DataFrame conforming to SERIES_SCHEMA. Must have mission_id and
            channel_id columns — they become the partition directory names.
        output_path: Root output directory (e.g. data/processed/ESA-Mission1/train/).
        mode: Spark write mode. "overwrite" replaces existing data (default).
    """
    log.info("write_series", output_path=str(output_path), mode=mode)
    (df.write.mode(mode).partitionBy("mission_id", "channel_id").parquet(str(output_path)))
