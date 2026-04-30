"""Explicit PySpark StructType schemas for each stage of the preprocessing pipeline.

Defining schemas here (rather than relying on inference) ensures:
- Type errors surface at read time, not buried in transforms
- ANSI mode strict typing is satisfied from the start
- Feast feature view definitions have a single authoritative type reference
"""

from __future__ import annotations

from pyspark.sql.types import (
    BooleanType,
    FloatType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from spacecraft_telemetry.features.definitions import FEATURE_DEFINITIONS

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
# Feature columns are derived from FEATURE_DEFINITIONS so this schema stays
# in sync automatically when new features are added to the registry.

def _build_feature_schema() -> StructType:
    base = [
        StructField("telemetry_timestamp", TimestampType(), nullable=False),
        StructField("channel_id", StringType(), nullable=False),
        StructField("mission_id", StringType(), nullable=False),
        StructField("value_normalized", FloatType(), nullable=False),
    ]
    feature_fields = [
        StructField(fd.name, FloatType(), nullable=True) for fd in FEATURE_DEFINITIONS
    ]
    return StructType(base + feature_fields)


FEATURE_SCHEMA = _build_feature_schema()

# ---------------------------------------------------------------------------
# Per-timestep series (written to data/processed/{mission}/train/ and test/)
# ---------------------------------------------------------------------------
# One row per telemetry timestep. Windows are constructed on-the-fly in the
# PyTorch DataLoader (Phase 4+), so window_size is a DataLoader parameter,
# not baked into the on-disk schema. This avoids the 250× storage inflation
# of pre-materialized windows (see Plan 002.5).

SERIES_SCHEMA = StructType(
    [
        StructField("telemetry_timestamp", TimestampType(), nullable=False),
        StructField("value_normalized",    FloatType(),     nullable=False),
        StructField("channel_id",          StringType(),    nullable=False),
        StructField("mission_id",          StringType(),    nullable=False),
        StructField("segment_id",          IntegerType(),   nullable=False),
        StructField("is_anomaly",          BooleanType(),   nullable=False),
    ]
)
