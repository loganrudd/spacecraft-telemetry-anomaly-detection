"""Tests for ingest.collector — mocked Lightstreamer client, no network."""

from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pyarrow.parquet as pq
import pytest

from spacecraft_telemetry.core.config import CollectorConfig
from spacecraft_telemetry.ingest.collector import (
    LightstreamerCollector,
    _ISSSubscriptionListener,
    parse_timestamp,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TS_STR = "2024-06-01T12:00:00.000Z"
_TS_DT = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)
_INGEST = datetime(2024, 6, 1, 12, 0, 5, tzinfo=UTC)


def _make_update(item_name: str, value: str, timestamp: str) -> MagicMock:
    """Return a mock ItemUpdate with getValue() and getItemName()."""
    update = MagicMock()
    update.getItemName.return_value = item_name
    update.getValue.side_effect = lambda field: {
        "Value": value,
        "TimeStamp": timestamp,
    }.get(field)
    return update


def _make_listener(
    items: list[str] | None = None,
    staleness_seconds: float = 60.0,
) -> _ISSSubscriptionListener:
    items = items or ["S1000003", "TIME_000001"]
    buffers: dict[str, list[Any]] = {ch: [] for ch in items}
    lock = threading.Lock()
    return _ISSSubscriptionListener(buffers, lock, staleness_seconds)


# ---------------------------------------------------------------------------
# parse_timestamp
# ---------------------------------------------------------------------------


def test_parse_timestamp_iso_with_ms() -> None:
    ts = parse_timestamp("2024-06-01T12:00:00.500Z")
    assert ts is not None
    assert ts.tzinfo is UTC
    assert ts.year == 2024


def test_parse_timestamp_iso_no_ms() -> None:
    ts = parse_timestamp("2024-06-01T12:00:00Z")
    assert ts is not None
    assert ts.second == 0


def test_parse_timestamp_unix_epoch() -> None:
    epoch = datetime(1970, 1, 1, 0, 0, 0, tzinfo=UTC)
    ts = parse_timestamp("0.0")
    assert ts is not None
    assert ts == epoch


def test_parse_timestamp_empty_returns_none() -> None:
    assert parse_timestamp("") is None


def test_parse_timestamp_garbage_returns_none() -> None:
    assert parse_timestamp("NOT_A_TIMESTAMP") is None


# ---------------------------------------------------------------------------
# _ISSSubscriptionListener.onItemUpdate
# ---------------------------------------------------------------------------


def test_on_item_update_appends_to_buffer() -> None:
    listener = _make_listener(["S1000003", "TIME_000001"])
    update = _make_update("S1000003", "21.5", _TS_STR)
    listener.onItemUpdate(update)
    assert len(listener._buffers["S1000003"]) == 1
    row = listener._buffers["S1000003"][0]
    assert row["value"] == pytest.approx(21.5)
    assert row["telemetry_timestamp"] == _TS_DT


def test_on_item_update_multiple_ticks_accumulate() -> None:
    listener = _make_listener(["S1000003", "TIME_000001"])
    for i in range(5):
        update = _make_update("S1000003", str(float(i)), _TS_STR)
        listener.onItemUpdate(update)
    assert len(listener._buffers["S1000003"]) == 5


def test_on_item_update_none_value_skipped() -> None:
    listener = _make_listener(["S1000003", "TIME_000001"])
    update = MagicMock()
    update.getItemName.return_value = "S1000003"
    update.getValue.return_value = None
    listener.onItemUpdate(update)
    assert len(listener._buffers["S1000003"]) == 0


def test_on_item_update_bad_value_skipped() -> None:
    listener = _make_listener(["S1000003", "TIME_000001"])
    update = _make_update("S1000003", "NOT_A_FLOAT", _TS_STR)
    listener.onItemUpdate(update)
    assert len(listener._buffers["S1000003"]) == 0


def test_on_item_update_bad_timestamp_skipped() -> None:
    listener = _make_listener(["S1000003", "TIME_000001"])
    update = _make_update("S1000003", "21.5", "BAD_TS")
    listener.onItemUpdate(update)
    assert len(listener._buffers["S1000003"]) == 0


def test_on_item_update_ingest_time_is_set() -> None:
    listener = _make_listener(["S1000003", "TIME_000001"])
    before = datetime.now(UTC)
    update = _make_update("S1000003", "1.0", _TS_STR)
    listener.onItemUpdate(update)
    after = datetime.now(UTC)
    row = listener._buffers["S1000003"][0]
    assert before <= row["ingest_time"] <= after


# ---------------------------------------------------------------------------
# LOS staleness detection
# ---------------------------------------------------------------------------


def test_los_onset_logged_when_time_stale(caplog: pytest.LogCaptureFixture) -> None:
    import logging
    listener = _make_listener(["S1000003", "TIME_000001"], staleness_seconds=1.0)
    # Seed TIME_000001 with an old timestamp
    old_ts = datetime(2024, 6, 1, 11, 0, 0, tzinfo=UTC)
    with listener._lock:
        listener._last_ts["TIME_000001"] = old_ts

    now = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)  # 1 hour later → stale
    with caplog.at_level(logging.INFO):
        listener.check_staleness(now)

    assert listener._in_los is True


def test_los_recovery_logged_on_new_time_tick(caplog: pytest.LogCaptureFixture) -> None:
    import logging
    listener = _make_listener(["S1000003", "TIME_000001"], staleness_seconds=1.0)
    listener._in_los = True
    # Seed last_ts so recovery check has something to compare against
    with listener._lock:
        listener._last_ts["TIME_000001"] = datetime(2024, 6, 1, 11, 59, 59, tzinfo=UTC)

    with caplog.at_level(logging.INFO):
        update = _make_update("TIME_000001", "1.0", _TS_STR)
        listener.onItemUpdate(update)

    assert listener._in_los is False


def test_no_los_when_time_advances_within_threshold() -> None:
    listener = _make_listener(["S1000003", "TIME_000001"], staleness_seconds=60.0)
    recent = datetime.now(UTC) - timedelta(seconds=30)
    with listener._lock:
        listener._last_ts["TIME_000001"] = recent
    listener.check_staleness(datetime.now(UTC))
    assert listener._in_los is False


# ---------------------------------------------------------------------------
# LightstreamerCollector flush integration
# ---------------------------------------------------------------------------


def test_collector_flush_writes_parquet(tmp_path: Path) -> None:
    """Drive on_item_update directly then trigger _flush_all; verify output."""
    config = CollectorConfig(
        channel_set="validation",
        raw_ticks_dir=str(tmp_path),
        flush_interval_seconds=300.0,
    )
    collector = LightstreamerCollector(config, dest_dir=tmp_path)

    # Feed 3 ticks for S1000003 via the listener
    for i in range(3):
        ts_str = f"2024-06-01T12:0{i}:00.000Z"
        update = _make_update("S1000003", str(float(i)), ts_str)
        collector._listener.onItemUpdate(update)

    collector._flush_all(final=True)

    # Find written shard(s)
    shards = list((tmp_path / "ISS" / "ticks").glob("channel_id=S1000003/*.parquet"))
    assert shards, "No Parquet shard written for S1000003"

    table = pq.read_table(str(shards[0]), partitioning=None)
    assert len(table) == 3


def test_collector_flush_empties_buffer(tmp_path: Path) -> None:
    config = CollectorConfig(
        channel_set="validation",
        raw_ticks_dir=str(tmp_path),
        flush_interval_seconds=300.0,
    )
    collector = LightstreamerCollector(config, dest_dir=tmp_path)
    update = _make_update("S1000003", "1.0", _TS_STR)
    collector._listener.onItemUpdate(update)

    assert len(collector._buffers["S1000003"]) == 1
    collector._flush_all(final=False)
    assert len(collector._buffers["S1000003"]) == 0


def test_collector_context_item_flushed(tmp_path: Path) -> None:
    config = CollectorConfig(
        channel_set="validation",
        raw_ticks_dir=str(tmp_path),
        flush_interval_seconds=300.0,
    )
    collector = LightstreamerCollector(config, dest_dir=tmp_path)
    update = _make_update("TIME_000001", "1234.5", _TS_STR)
    collector._listener.onItemUpdate(update)
    collector._flush_all(final=True)

    shards = list((tmp_path / "ISS" / "ticks").glob("channel_id=TIME_000001/*.parquet"))
    assert shards, "No shard written for TIME_000001"


def test_collector_run_raises_without_lightstreamer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run() raises RuntimeError when lightstreamer-client-lib is not installed."""
    import sys
    from importlib.abc import MetaPathFinder

    class _BlockLightstreamer(MetaPathFinder):
        """Intercepts import machinery to simulate the package being absent."""

        def find_spec(self, fullname: str, path: object, target: object = None) -> None:
            if "lightstreamer" in fullname:
                raise ImportError(f"No module named {fullname!r}")
            return None

    config = CollectorConfig(channel_set="validation", raw_ticks_dir=str(tmp_path))
    collector = LightstreamerCollector(config, dest_dir=tmp_path)

    # Remove cached modules so the lazy import inside run() re-enters find_spec.
    for key in list(sys.modules.keys()):
        if "lightstreamer" in key:
            monkeypatch.delitem(sys.modules, key)

    # Insert before the standard finders so it fires first.
    blocker = _BlockLightstreamer()
    sys.meta_path.insert(0, blocker)
    try:
        with pytest.raises(RuntimeError, match="lightstreamer-client-lib"):
            collector.run(seconds=0)
    finally:
        sys.meta_path.remove(blocker)
