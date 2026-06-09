"""PyArrow schemas for each stage of the pandas preprocessing pipeline.

PyArrow schemas for the preprocessing stages. The SERIES_FILE_SCHEMA is the
load-bearing contract: it
describes the columns written to each Hive partition file (excluding mission_id
and channel_id, which are encoded in the directory path).
"""

from __future__ import annotations

import pyarrow as pa

# ---------------------------------------------------------------------------
# Raw channel input (after reading raw Parquet and renaming columns)
# ---------------------------------------------------------------------------

RAW_CHANNEL_SCHEMA = pa.schema(
    [
        pa.field("telemetry_timestamp", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("value", pa.float32(), nullable=True),  # nulls cleaned in transforms
        pa.field("channel_id", pa.string(), nullable=False),
        pa.field("mission_id", pa.string(), nullable=False),
    ]
)

# ---------------------------------------------------------------------------
# Labels input (after reading labels.csv and parsing ISO timestamps)
# ---------------------------------------------------------------------------

LABELS_SCHEMA = pa.schema(
    [
        pa.field("anomaly_id", pa.string(), nullable=False),
        pa.field("channel_id", pa.string(), nullable=False),
        pa.field("start_time", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("end_time", pa.timestamp("us", tz="UTC"), nullable=False),
    ]
)

# ---------------------------------------------------------------------------
# Per-timestep series — written to {mission}/{split}/mission_id=M/channel_id=C/
# ---------------------------------------------------------------------------
# Partition columns (mission_id, channel_id) are encoded in directory names,
# NOT stored in the Parquet file itself. SERIES_FILE_SCHEMA covers only the
# columns that appear inside each part.parquet file.
#
# Downstream readers (model/dataset.py, ray_fanout/conftest.py) expect exactly
# these four columns in this order.

SERIES_FILE_SCHEMA = pa.schema(
    [
        pa.field("telemetry_timestamp", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("value_normalized", pa.float32(), nullable=False),
        pa.field("segment_id", pa.int32(), nullable=False),
        pa.field("is_anomaly", pa.bool_(), nullable=False),
    ]
)

# Full column list for the in-memory series DataFrame (before writing).
SERIES_SCHEMA_COLS = [
    "telemetry_timestamp",
    "value_normalized",
    "channel_id",
    "mission_id",
    "segment_id",
    "is_anomaly",
]
