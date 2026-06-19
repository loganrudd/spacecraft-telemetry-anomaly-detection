"""Tests for ingest.collector_io — Parquet flush, schema, and partition layout."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from spacecraft_telemetry.ingest.collector_io import (
    RAW_TICK_SCHEMA,
    flush_buffer,
    shard_path,
)

_TS1 = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)
_CHANNEL = "S1000003"


def _make_rows(n: int = 3) -> list[dict]:
    return [
        {
            "telemetry_timestamp": datetime(2024, 6, 1, 12, i, 0, tzinfo=UTC),
            "value": float(i),
            "aos_timestamp": 4076.0 + i,
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# shard_path
# ---------------------------------------------------------------------------


def test_shard_path_structure(tmp_path: Path) -> None:
    bucket = datetime(2024, 6, 1, 12, 5, 3, tzinfo=UTC)
    p = shard_path(tmp_path, _CHANNEL, bucket)
    assert f"ISS/ticks/channel_id={_CHANNEL}/" in str(p)
    assert "20240601T120503" in str(p)
    assert str(p).endswith(".parquet")


def test_shard_path_preserves_gcs_uri() -> None:
    """shard_path must not mangle gs:// URIs (no pathlib.Path round-trip)."""
    bucket = datetime(2024, 6, 1, 12, 5, 0, tzinfo=UTC)
    p = shard_path("gs://my-project-raw-data", _CHANNEL, bucket)
    assert str(p).startswith("gs://my-project-raw-data/ISS/ticks/"), (
        f"gs:// URI mangled: {str(p)!r}"
    )


def test_shard_path_second_granularity(tmp_path: Path) -> None:
    # Second-level filename prevents a shutdown-flush from truncating a
    # same-minute periodic flush (open("wb") overwrites).
    b1 = datetime(2024, 6, 1, 12, 5, 0, tzinfo=UTC)
    b2 = datetime(2024, 6, 1, 12, 5, 1, tzinfo=UTC)
    assert shard_path(tmp_path, _CHANNEL, b1) != shard_path(tmp_path, _CHANNEL, b2)


def test_shard_path_different_hours_differ(tmp_path: Path) -> None:
    b1 = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)
    b2 = datetime(2024, 6, 1, 13, 0, 0, tzinfo=UTC)
    assert shard_path(tmp_path, _CHANNEL, b1) != shard_path(tmp_path, _CHANNEL, b2)


def test_shard_path_different_channels_differ(tmp_path: Path) -> None:
    bucket = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)
    p1 = shard_path(tmp_path, "S1000003", bucket)
    p2 = shard_path(tmp_path, "P1000003", bucket)
    assert p1 != p2


# ---------------------------------------------------------------------------
# flush_buffer
# ---------------------------------------------------------------------------


def test_flush_empty_returns_none(tmp_path: Path) -> None:
    result = flush_buffer([], tmp_path, _CHANNEL, bucket_ts=_TS1)
    assert result is None


def test_flush_creates_parquet(tmp_path: Path) -> None:
    rows = _make_rows(3)
    path = flush_buffer(rows, tmp_path, _CHANNEL, bucket_ts=_TS1)
    assert path is not None
    assert path.exists()


def test_flush_schema_matches(tmp_path: Path) -> None:
    rows = _make_rows(3)
    path = flush_buffer(rows, tmp_path, _CHANNEL, bucket_ts=_TS1)
    assert path is not None
    # partitioning=None prevents PyArrow 18+ from auto-adding the channel_id
    # Hive partition column when reading a file inside a channel_id=... directory.
    table = pq.read_table(str(path), partitioning=None)
    assert table.schema.equals(RAW_TICK_SCHEMA)


def test_flush_row_count(tmp_path: Path) -> None:
    rows = _make_rows(5)
    path = flush_buffer(rows, tmp_path, _CHANNEL, bucket_ts=_TS1)
    assert path is not None
    table = pq.read_table(str(path), partitioning=None)
    assert len(table) == 5
    # Exactly one shard written for this channel/bucket — no duplicates.
    shards = list((tmp_path / "ISS" / "ticks").glob(f"channel_id={_CHANNEL}/*.parquet"))
    assert len(shards) == 1


def test_flush_column_values(tmp_path: Path) -> None:
    rows = [
        {
            "telemetry_timestamp": _TS1,
            "value": 42.0,
            "aos_timestamp": 4076.5,
        }
    ]
    path = flush_buffer(rows, tmp_path, _CHANNEL, bucket_ts=_TS1)
    assert path is not None
    table = pq.read_table(str(path))
    assert table["value"][0].as_py() == pytest.approx(42.0, abs=1e-3)
    assert table["aos_timestamp"][0].as_py() == pytest.approx(4076.5)


def test_flush_timestamps_are_utc(tmp_path: Path) -> None:
    rows = _make_rows(1)
    path = flush_buffer(rows, tmp_path, _CHANNEL, bucket_ts=_TS1)
    assert path is not None
    table = pq.read_table(str(path))
    tz_str = str(table.schema.field("telemetry_timestamp").type)
    assert "UTC" in tz_str or "utc" in tz_str.lower()


def test_flush_creates_parent_dirs(tmp_path: Path) -> None:
    deep = tmp_path / "a" / "b" / "c"
    rows = _make_rows(1)
    path = flush_buffer(rows, deep, _CHANNEL, bucket_ts=_TS1)
    assert path is not None
    assert path.exists()


def test_flush_uses_now_when_bucket_ts_omitted(tmp_path: Path) -> None:
    rows = _make_rows(1)
    path = flush_buffer(rows, tmp_path, _CHANNEL)
    assert path is not None
    assert path.exists()


def test_flush_partition_path(tmp_path: Path) -> None:
    rows = _make_rows(2)
    path = flush_buffer(rows, tmp_path, "P4000001", bucket_ts=_TS1)
    assert path is not None
    assert "channel_id=P4000001" in str(path)
    assert "ISS/ticks" in str(path)


def test_flush_context_item(tmp_path: Path) -> None:
    rows = [
        {
            "telemetry_timestamp": _TS1,
            "value": 1234.5,
            "aos_timestamp": 4076.5,
        }
    ]
    path = flush_buffer(rows, tmp_path, "TIME_000001", bucket_ts=_TS1)
    assert path is not None
    assert "channel_id=TIME_000001" in str(path)


def test_flush_aos_timestamp_nullable(tmp_path: Path) -> None:
    rows = [
        {
            "telemetry_timestamp": _TS1,
            "value": 1.0,
            "aos_timestamp": None,
        }
    ]
    path = flush_buffer(rows, tmp_path, _CHANNEL, bucket_ts=_TS1)
    assert path is not None
    table = pq.read_table(str(path), partitioning=None)
    assert table["aos_timestamp"][0].as_py() is None
