"""Shared fixtures for Telemanom model tests.

Key fixture: tiny_series_parquet — writes synthetic per-timestep series Parquet in
the Hive-partitioned layout produced by the Spark pipeline (Phase 2.5+), so model
tests can run without Spark.  Matches SERIES_SCHEMA from spark/schemas.py.

Layout:
    {processed_dir}/ESA-Mission1/train/mission_id=ESA-Mission1/channel_id=channel_1/part.parquet
    {processed_dir}/ESA-Mission1/test/mission_id=ESA-Mission1/channel_id=channel_1/part.parquet

Partition columns (mission_id, channel_id) are NOT in the Parquet files — they
are encoded in directory names, exactly as Spark writes them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Generator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import mlflow

from spacecraft_telemetry.core.config import load_settings as _load_settings

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MISSION = "ESA-Mission1"
_CHANNEL = "channel_1"

# Read window params from test.yaml so fixtures stay in sync with config.
_test_model_cfg = _load_settings("test").model
_WINDOW_SIZE = _test_model_cfg.window_size
_PREDICTION_HORIZON = _test_model_cfg.prediction_horizon
_SPAN = _WINDOW_SIZE + _PREDICTION_HORIZON

# Train split: 3 clean segments so that on-the-fly windowing produces a
# predictable number of valid windows.
#   segment 0: 60 rows  → max(0, 60 - span + 1) valid windows
#   segment 1: 30 rows  → max(0, 30 - span + 1) valid windows
#   segment 2: 10 rows  → max(0, 10 - span + 1) valid windows (0 when span > 10)
_SEG_SIZES_TRAIN = (60, 30, 10)
_N_TRAIN_WINDOWS = sum(max(0, s - _SPAN + 1) for s in _SEG_SIZES_TRAIN)

# Test split: 1 segment of 30 rows → last 5 rows are anomalous.
_N_TEST_ROWS = 30
_N_TEST_WINDOWS = max(0, _N_TEST_ROWS - _SPAN + 1)
_ANOMALY_ROWS = 5       # last 5 rows of test segment are anomalous

# PyArrow schema matching SERIES_SCHEMA from spark/schemas.py,
# minus the Hive partition columns (mission_id, channel_id).
_SERIES_FILE_SCHEMA = pa.schema(
    [
        pa.field("telemetry_timestamp", pa.timestamp("us", tz="UTC")),
        pa.field("value_normalized", pa.float32()),
        pa.field("segment_id", pa.int32()),
        pa.field("is_anomaly", pa.bool_()),
    ]
)


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeriesParquetFixture:
    processed_dir: Path
    mission: str
    channel: str
    window_size: int
    prediction_horizon: int
    n_train_windows: int
    n_test_rows: int
    n_test_windows: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_series_table(
    seg_sizes: tuple[int, ...],
    anomaly_tail: int = 0,
) -> pa.Table:
    """Build a PyArrow Table with per-timestep rows.

    ``seg_sizes``: tuple of row counts per segment (segment IDs = 0, 1, 2, …).
    ``anomaly_tail``: number of rows at the end of the *last* segment to mark
        as anomalous.
    """
    _BASE_DT = datetime(2000, 1, 1, tzinfo=UTC)
    rng = np.random.default_rng(seed=42)

    timestamps: list[pa.Scalar] = []
    values: list[float] = []
    seg_ids: list[int] = []
    is_anomaly: list[bool] = []

    t_offset = 0
    for seg_id, size in enumerate(seg_sizes):
        for i in range(size):
            ts = pa.scalar(
                _BASE_DT.timestamp() + (t_offset + i) * 90,  # 90-second cadence
                type=pa.timestamp("s", tz="UTC"),
            ).cast(pa.timestamp("us", tz="UTC"))
            timestamps.append(ts)
            values.append(float(rng.standard_normal(1)[0]))
            seg_ids.append(seg_id)
            is_anomaly.append(False)
        t_offset += size

    # Mark the tail of the last segment as anomalous.
    if anomaly_tail > 0:
        for i in range(len(is_anomaly) - anomaly_tail, len(is_anomaly)):
            is_anomaly[i] = True

    return pa.table(
        {
            "telemetry_timestamp": pa.array(timestamps, type=pa.timestamp("us", tz="UTC")),
            "value_normalized": pa.array(values, type=pa.float32()),
            "segment_id": pa.array(seg_ids, type=pa.int32()),
            "is_anomaly": pa.array(is_anomaly, type=pa.bool_()),
        },
        schema=_SERIES_FILE_SCHEMA,
    )


def _write_partition(table: pa.Table, partition_dir: Path) -> None:
    partition_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, partition_dir / "part.parquet")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mlflow_uri(tmp_path: Path) -> Generator[str, None, None]:
    """Per-test isolated SQLite MLflow store for model tests."""
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    mlflow.set_tracking_uri(uri)
    yield uri
    if mlflow.active_run() is not None:
        mlflow.end_run()
    mlflow.set_tracking_uri("")


@pytest.fixture()
def tiny_series_parquet(tmp_path: Path) -> SeriesParquetFixture:
    """Write synthetic series Parquet in Spark's Hive-partitioned layout.

    Train: 3 segments (60 + 30 + 10 rows).  Segment 2 is too short for any
    window, so only segments 0 and 1 contribute valid windows (50 + 20 = 70).
    Test: 1 segment of 30 rows, last 5 marked anomalous.  All 20 windows
    cross no boundaries, so all are included in the test index.
    """
    processed_dir = tmp_path / "processed"

    train_dir = (
        processed_dir / _MISSION / "train"
        / f"mission_id={_MISSION}" / f"channel_id={_CHANNEL}"
    )
    test_dir = (
        processed_dir / _MISSION / "test"
        / f"mission_id={_MISSION}" / f"channel_id={_CHANNEL}"
    )

    _write_partition(
        _make_series_table(_SEG_SIZES_TRAIN, anomaly_tail=0),
        train_dir,
    )
    _write_partition(
        _make_series_table((_N_TEST_ROWS,), anomaly_tail=_ANOMALY_ROWS),
        test_dir,
    )

    # normalization_params.json — needed by save_norm_params in training.
    norm_file = processed_dir / _MISSION / "normalization_params.json"
    norm_file.parent.mkdir(parents=True, exist_ok=True)
    norm_file.write_text(json.dumps({_CHANNEL: {"mean": 0.0, "std": 1.0}}))

    return SeriesParquetFixture(
        processed_dir=processed_dir,
        mission=_MISSION,
        channel=_CHANNEL,
        window_size=_WINDOW_SIZE,
        prediction_horizon=_PREDICTION_HORIZON,
        n_train_windows=_N_TRAIN_WINDOWS,
        n_test_rows=_N_TEST_ROWS,
        n_test_windows=_N_TEST_WINDOWS,
    )
