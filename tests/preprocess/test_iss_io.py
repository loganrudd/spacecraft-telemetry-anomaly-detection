"""Tests for ISS-specific I/O functions in preprocess/io.py.

All tests use synthetic in-memory fixtures written to tmp_path.
No real ISS data or network access required.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from spacecraft_telemetry.preprocess.io import (
    discover_iss_channels,
    read_all_iss_ticks_for_los,
    read_iss_ticks,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RAW_TICK_SCHEMA = pa.schema(
    [
        pa.field("telemetry_timestamp", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("value", pa.float32(), nullable=False),
        pa.field("aos_timestamp", pa.float64(), nullable=True),
    ]
)


def _write_tick_shard(dest_dir: Path, filename: str, rows: list[dict]) -> Path:
    """Write a list of tick row dicts to a Parquet shard."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / filename
    table = pa.Table.from_pylist(rows, schema=RAW_TICK_SCHEMA)
    pq.write_table(table, str(path))
    return path


def _make_ticks(
    channel_id: str,
    start: str = "2026-06-01T00:00:00Z",
    n: int = 10,
    interval_s: int = 2,
) -> list[dict]:
    base = pd.Timestamp(start, tz="UTC")
    return [
        {
            "telemetry_timestamp": base + pd.Timedelta(seconds=i * interval_s),
            "value": float(i),
            "aos_timestamp": None,
        }
        for i in range(n)
    ]


def _write_channel_dir(root: Path, channel_id: str, n_shards: int = 1) -> Path:
    channel_dir = root / "ISS" / "ticks" / f"channel_id={channel_id}"
    for shard_idx in range(n_shards):
        rows = _make_ticks(
            channel_id,
            start=f"2026-06-0{shard_idx + 1}T00:00:00Z",
            n=10,
        )
        _write_tick_shard(channel_dir, f"2026060{shard_idx + 1}T000000.parquet", rows)
    return channel_dir


# ---------------------------------------------------------------------------
# TestReadIssTicks
# ---------------------------------------------------------------------------


class TestReadIssTicks:
    def test_reads_single_shard(self, tmp_path: Path) -> None:
        _write_channel_dir(tmp_path, "S1000003", n_shards=1)
        df = read_iss_ticks(tmp_path, "S1000003")
        assert len(df) == 10
        assert list(df.columns) == ["telemetry_timestamp", "value", "aos_timestamp"]

    def test_reads_multiple_shards(self, tmp_path: Path) -> None:
        _write_channel_dir(tmp_path, "P4000001", n_shards=3)
        df = read_iss_ticks(tmp_path, "P4000001")
        assert len(df) == 30

    def test_sorted_by_timestamp(self, tmp_path: Path) -> None:
        _write_channel_dir(tmp_path, "S1000003", n_shards=2)
        df = read_iss_ticks(tmp_path, "S1000003")
        assert df["telemetry_timestamp"].is_monotonic_increasing

    def test_telemetry_timestamp_is_utc(self, tmp_path: Path) -> None:
        _write_channel_dir(tmp_path, "S1000003", n_shards=1)
        df = read_iss_ticks(tmp_path, "S1000003")
        assert str(df["telemetry_timestamp"].dt.tz) == "UTC"

    def test_value_is_float32(self, tmp_path: Path) -> None:
        _write_channel_dir(tmp_path, "S1000003", n_shards=1)
        df = read_iss_ticks(tmp_path, "S1000003")
        assert df["value"].dtype == "float32"

    def test_raises_if_dir_missing(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            read_iss_ticks(tmp_path, "NONEXISTENT")

    def test_raises_if_no_parquets(self, tmp_path: Path) -> None:
        channel_dir = tmp_path / "ISS" / "ticks" / "channel_id=EMPTY"
        channel_dir.mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="No Parquet shards"):
            read_iss_ticks(tmp_path, "EMPTY")

    def test_no_hive_column_injection(self, tmp_path: Path) -> None:
        # PyArrow 18+ injects a channel_id column from the directory name when
        # partitioning is not disabled.  read_iss_ticks must use partitioning=None.
        _write_channel_dir(tmp_path, "S1000003", n_shards=1)
        df = read_iss_ticks(tmp_path, "S1000003")
        assert "channel_id" not in df.columns


# ---------------------------------------------------------------------------
# TestDiscoverIssChannels
# ---------------------------------------------------------------------------


class TestDiscoverIssChannels:
    def test_finds_available_channels(self, tmp_path: Path) -> None:
        for ch in ["P4000001", "S1000003"]:
            _write_channel_dir(tmp_path, ch, n_shards=1)
        channels = discover_iss_channels(tmp_path)
        assert "P4000001" in channels
        assert "S1000003" in channels

    def test_returns_sorted(self, tmp_path: Path) -> None:
        for ch in ["S1000003", "P4000001", "USLAB000018"]:
            _write_channel_dir(tmp_path, ch, n_shards=1)
        channels = discover_iss_channels(tmp_path)
        assert channels == sorted(channels)

    def test_excludes_context_items_by_default(self, tmp_path: Path) -> None:
        for ch in ["S1000003", "TIME_000001", "USLAB000086"]:
            _write_channel_dir(tmp_path, ch, n_shards=1)
        channels = discover_iss_channels(tmp_path)
        assert "TIME_000001" not in channels
        assert "USLAB000086" not in channels
        assert "S1000003" in channels

    def test_custom_exclude_list(self, tmp_path: Path) -> None:
        for ch in ["P4000001", "S1000003"]:
            _write_channel_dir(tmp_path, ch, n_shards=1)
        channels = discover_iss_channels(tmp_path, exclude=["P4000001"])
        assert "P4000001" not in channels
        assert "S1000003" in channels

    def test_empty_dir_returns_empty_list(self, tmp_path: Path) -> None:
        ticks_root = tmp_path / "ISS" / "ticks"
        ticks_root.mkdir(parents=True)
        channels = discover_iss_channels(tmp_path)
        assert channels == []


# ---------------------------------------------------------------------------
# TestReadAllIssTicksForLos
# ---------------------------------------------------------------------------


class TestReadAllIssTicksForLos:
    def test_returns_timestamp_and_channel_id_only(self, tmp_path: Path) -> None:
        _write_channel_dir(tmp_path, "S1000003", n_shards=1)
        df = read_all_iss_ticks_for_los(tmp_path, ["S1000003"])
        assert set(df.columns) == {"telemetry_timestamp", "channel_id"}

    def test_concatenates_multiple_channels(self, tmp_path: Path) -> None:
        for ch in ["S1000003", "P4000001"]:
            _write_channel_dir(tmp_path, ch, n_shards=1)
        df = read_all_iss_ticks_for_los(tmp_path, ["S1000003", "P4000001"])
        assert len(df) == 20
        assert set(df["channel_id"].unique()) == {"S1000003", "P4000001"}

    def test_empty_channel_list_returns_empty_df(self, tmp_path: Path) -> None:
        df = read_all_iss_ticks_for_los(tmp_path, [])
        assert df.empty
        assert "telemetry_timestamp" in df.columns
        assert "channel_id" in df.columns

    def test_drops_value_and_aos_columns(self, tmp_path: Path) -> None:
        _write_channel_dir(tmp_path, "S1000003", n_shards=1)
        df = read_all_iss_ticks_for_los(tmp_path, ["S1000003"])
        assert "value" not in df.columns
        assert "aos_timestamp" not in df.columns
