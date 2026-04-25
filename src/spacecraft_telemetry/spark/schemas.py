"""Explicit PySpark StructType schemas for each stage of the preprocessing pipeline.

Defining schemas here (rather than relying on inference) ensures:
- Type errors surface at read time, not buried in transforms
- ANSI mode strict typing is satisfied from the start
- Feast feature view definitions have a single authoritative type reference
"""

from __future__ import annotations

from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    FloatType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

# ---------------------------------------------------------------------------
# Raw channel input
# ---------------------------------------------------------------------------
# Schema after reading a sample Parquet and renaming columns to standard names.
# Original files have [channel_N: float, datetime: timestamp] — renamed here.

RAW_CHANNEL_SCHEMA = StructType(
    [
        StructField("telemetry_timestamp", TimestampType(), nullable=False),
        StructField("value", FloatType(), nullable=True),  # nulls handled in cleaning step
        StructField("channel_id", StringType(), nullable=False),
        StructField("mission_id", StringType(), nullable=False),
    ]
)

# ---------------------------------------------------------------------------
# Labels input
# ---------------------------------------------------------------------------
# Schema after reading labels.csv and parsing ISO timestamp strings.

LABELS_SCHEMA = StructType(
    [
        StructField("anomaly_id", StringType(), nullable=False),  # original "ID" column
        StructField("channel_id", StringType(), nullable=False),
        StructField("start_time", TimestampType(), nullable=False),
        StructField("end_time", TimestampType(), nullable=False),
    ]
)

# ---------------------------------------------------------------------------
# Cleaned channel (after null handling, gap detection, normalization)
# ---------------------------------------------------------------------------

CLEANED_CHANNEL_SCHEMA = StructType(
    [
        StructField("telemetry_timestamp", TimestampType(), nullable=False),
        StructField("value", FloatType(), nullable=False),  # no nulls after cleaning
        StructField("value_normalized", FloatType(), nullable=False),
        StructField("channel_id", StringType(), nullable=False),
        StructField("mission_id", StringType(), nullable=False),
        StructField("segment_id", IntegerType(), nullable=False),  # contiguous run ID
        StructField("is_gap", BooleanType(), nullable=False),
    ]
)

# ---------------------------------------------------------------------------
# Feature output (written to data/processed/{mission}/features/)
# ---------------------------------------------------------------------------
# One row per timestamp. Ingested by Feast offline store.
# Feature column names come from features.definitions.FEATURE_DEFINITIONS —
# this schema must stay in sync with that registry.

FEATURE_SCHEMA = StructType(
    [
        StructField("telemetry_timestamp", TimestampType(), nullable=False),
        StructField("channel_id", StringType(), nullable=False),
        StructField("mission_id", StringType(), nullable=False),
        StructField("value_normalized", FloatType(), nullable=False),
        # Rolling stats — window sizes match SparkConfig.feature_windows default [10, 50, 100]
        StructField("rolling_mean_10", FloatType(), nullable=True),
        StructField("rolling_std_10", FloatType(), nullable=True),
        StructField("rolling_min_10", FloatType(), nullable=True),
        StructField("rolling_max_10", FloatType(), nullable=True),
        StructField("rolling_mean_50", FloatType(), nullable=True),
        StructField("rolling_std_50", FloatType(), nullable=True),
        StructField("rolling_min_50", FloatType(), nullable=True),
        StructField("rolling_max_50", FloatType(), nullable=True),
        StructField("rolling_mean_100", FloatType(), nullable=True),
        StructField("rolling_std_100", FloatType(), nullable=True),
        StructField("rolling_min_100", FloatType(), nullable=True),
        StructField("rolling_max_100", FloatType(), nullable=True),
        StructField("rate_of_change", FloatType(), nullable=True),
    ]
)

# ---------------------------------------------------------------------------
# Windowed LSTM input (written to data/processed/{mission}/train/ and test/)
# ---------------------------------------------------------------------------
# One row per sliding window. Consumed by PyTorch Dataset in Phase 4.

WINDOW_SCHEMA = StructType(
    [
        StructField("window_id", LongType(), nullable=False),
        StructField("channel_id", StringType(), nullable=False),
        StructField("mission_id", StringType(), nullable=False),
        StructField("segment_id", IntegerType(), nullable=False),
        StructField("window_start_ts", TimestampType(), nullable=False),
        StructField("window_end_ts", TimestampType(), nullable=False),
        StructField("values", ArrayType(FloatType(), containsNull=False), nullable=False),
        StructField("target", FloatType(), nullable=False),
        StructField("is_anomaly", BooleanType(), nullable=False),
    ]
)
