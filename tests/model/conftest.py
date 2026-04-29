"""Shared fixtures for Telemanom model tests.

Key fixture: tiny_windowed_parquet — writes synthetic windowed Parquet in the
Hive-partitioned layout produced by the Spark pipeline (Phase 2), so model
tests can run without Spark. Matches WINDOW_SCHEMA from spark/schemas.py.

Layout:
    {processed_dir}/ESA-Mission1/train/mission_id=ESA-Mission1/channel_id=channel_1/part.parquet
    {processed_dir}/ESA-Mission1/test/mission_id=ESA-Mission1/channel_id=channel_1/part.parquet

Partition columns (mission_id, channel_id) are NOT in the Parquet files — they
are encoded in directory names, exactly as Spark writes them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MISSION = "ESA-Mission1"
_CHANNEL = "channel_1"
_WINDOW_SIZE = 10
_N_TRAIN = 50
_N_TEST = 20

# PyArrow schema matching WINDOW_SCHEMA from spark/schemas.py,
# minus the Hive partition columns (mission_id, channel_id).
_WINDOW_FILE_SCHEMA = pa.schema(
    [
        pa.field("window_id", pa.int64()),
        pa.field("segment_id", pa.int32()),
        pa.field("window_start_ts", pa.timestamp("us", tz="UTC")),
        pa.field("window_end_ts", pa.timestamp("us", tz="UTC")),
        pa.field("values", pa.list_(pa.float32())),
        pa.field("target", pa.float32()),
        pa.field("is_anomaly", pa.bool_()),
    ]
)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowedParquetFixture:
    processed_dir: Path
    mission: str
    channel: str
    window_size: int
    n_train: int
    n_test: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_window_table(
    n: int,
    window_size: int,
    id_offset: int = 0,
    anomaly_fraction: float = 0.0,
    split: Literal["train", "test"] = "train",
) -> pa.Table:
    """Build a PyArrow Table with n synthetic windows."""
    _BASE_DT = datetime(2000, 1, 1, tzinfo=timezone.utc)
    base_ts = pa.array([_BASE_DT] * n, type=pa.timestamp("us", tz="UTC"))
    rng = np.random.default_rng(seed=42 + id_offset)
    raw_values = rng.standard_normal((n, window_size)).astype(np.float32)
    targets = rng.standard_normal(n).astype(np.float32)

    # Mark the last anomaly_fraction of test windows as anomalies
    is_anomaly = np.zeros(n, dtype=bool)
    if anomaly_fraction > 0 and split == "test":
        anomaly_start = int(n * (1 - anomaly_fraction))
        is_anomaly[anomaly_start:] = True

    return pa.table(
        {
            "window_id": pa.array(np.arange(id_offset, id_offset + n, dtype=np.int64)),
            "segment_id": pa.array(np.zeros(n, dtype=np.int32)),
            "window_start_ts": base_ts,
            "window_end_ts": base_ts,
            "values": pa.array(raw_values.tolist(), type=pa.list_(pa.float32())),
            "target": pa.array(targets, type=pa.float32()),
            "is_anomaly": pa.array(is_anomaly),
        },
        schema=_WINDOW_FILE_SCHEMA,
    )


def _write_partition(table: pa.Table, partition_dir: Path) -> None:
    partition_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, partition_dir / "part.parquet")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tiny_windowed_parquet(tmp_path: Path) -> WindowedParquetFixture:
    """Write synthetic windowed Parquet in Spark's Hive-partitioned layout.

    50 train windows + 20 test windows, window_size=10, single channel.
    Test windows have 25% anomaly rate (last 5 rows marked is_anomaly=True).
    """
    processed_dir = tmp_path / "processed"

    train_dir = processed_dir / _MISSION / "train" / f"mission_id={_MISSION}" / f"channel_id={_CHANNEL}"
    test_dir = processed_dir / _MISSION / "test" / f"mission_id={_MISSION}" / f"channel_id={_CHANNEL}"

    _write_partition(_make_window_table(_N_TRAIN, _WINDOW_SIZE, id_offset=0, split="train"), train_dir)
    _write_partition(_make_window_table(_N_TEST, _WINDOW_SIZE, id_offset=_N_TRAIN, anomaly_fraction=0.25, split="test"), test_dir)

    # Write normalization_params.json so train_channel can copy it into artifacts.
    norm_file = processed_dir / _MISSION / "normalization_params.json"
    norm_file.parent.mkdir(parents=True, exist_ok=True)
    norm_file.write_text(json.dumps({_CHANNEL: {"mean": 0.0, "std": 1.0}}))

    return WindowedParquetFixture(
        processed_dir=processed_dir,
        mission=_MISSION,
        channel=_CHANNEL,
        window_size=_WINDOW_SIZE,
        n_train=_N_TRAIN,
        n_test=_N_TEST,
    )
