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
    parse_aos_timestamp,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Real ISSLive TimeStamp format: a decimal AOS value, not a wall-clock time.
_AOS_STR = "4076.449722222222"


def _make_update(item_name: str, value: str, timestamp: str = _AOS_STR) -> MagicMock:
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
# parse_aos_timestamp
# ---------------------------------------------------------------------------


def test_parse_aos_timestamp_decimal() -> None:
    assert parse_aos_timestamp("4076.449722222222") == pytest.approx(4076.449722222222)


def test_parse_aos_timestamp_integer() -> None:
    assert parse_aos_timestamp("4076") == pytest.approx(4076.0)


def test_parse_aos_timestamp_empty_returns_none() -> None:
    assert parse_aos_timestamp("") is None


def test_parse_aos_timestamp_garbage_returns_none() -> None:
    assert parse_aos_timestamp("NOT_A_NUMBER") is None


# ---------------------------------------------------------------------------
# _ISSSubscriptionListener.onItemUpdate
# ---------------------------------------------------------------------------


def test_on_item_update_appends_to_buffer() -> None:
    listener = _make_listener(["S1000003", "TIME_000001"])
    before = datetime.now(UTC)
    update = _make_update("S1000003", "21.5")
    listener.onItemUpdate(update)
    after = datetime.now(UTC)
    assert len(listener._buffers["S1000003"]) == 1
    row = listener._buffers["S1000003"][0]
    assert row["value"] == pytest.approx(21.5)
    # telemetry_timestamp is the UTC receipt time, not the feed's AOS value.
    assert before <= row["telemetry_timestamp"] <= after
    assert row["aos_timestamp"] == pytest.approx(4076.449722222222)


def test_on_item_update_multiple_ticks_accumulate() -> None:
    listener = _make_listener(["S1000003", "TIME_000001"])
    for i in range(5):
        update = _make_update("S1000003", str(float(i)))
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
    update = _make_update("S1000003", "NOT_A_FLOAT")
    listener.onItemUpdate(update)
    assert len(listener._buffers["S1000003"]) == 0


def test_on_item_update_bad_aos_timestamp_kept_with_none() -> None:
    # A bad AOS TimeStamp no longer drops the tick — telemetry_timestamp comes
    # from receipt time, so the row is kept with aos_timestamp=None.
    listener = _make_listener(["S1000003", "TIME_000001"])
    update = _make_update("S1000003", "21.5", "BAD_TS")
    listener.onItemUpdate(update)
    assert len(listener._buffers["S1000003"]) == 1
    assert listener._buffers["S1000003"][0]["aos_timestamp"] is None


def test_on_item_update_missing_timestamp_kept_with_none() -> None:
    listener = _make_listener(["S1000003", "TIME_000001"])
    update = MagicMock()
    update.getItemName.return_value = "S1000003"
    update.getValue.side_effect = lambda field: {"Value": "21.5"}.get(field)
    listener.onItemUpdate(update)
    assert len(listener._buffers["S1000003"]) == 1
    assert listener._buffers["S1000003"][0]["aos_timestamp"] is None


# ---------------------------------------------------------------------------
# LOS staleness detection (wall-clock arrival gap on TIME_000001)
# ---------------------------------------------------------------------------


def test_los_onset_logged_when_no_time_arrivals() -> None:
    listener = _make_listener(["S1000003", "TIME_000001"], staleness_seconds=1.0)
    # Last TIME_000001 arrival was an hour ago (wall clock).
    old_arrival = datetime(2024, 6, 1, 11, 0, 0, tzinfo=UTC)
    with listener._lock:
        listener._last_time_arrival = old_arrival

    now = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)  # 1 hour later → stale
    listener.check_staleness(now)
    assert listener._in_los is True


def test_los_recovery_on_new_time_arrival() -> None:
    listener = _make_listener(["S1000003", "TIME_000001"], staleness_seconds=1.0)
    listener._in_los = True

    update = _make_update("TIME_000001", "14675219000")
    listener.onItemUpdate(update)

    assert listener._in_los is False
    assert listener._last_time_arrival is not None


def test_no_los_when_arrivals_recent() -> None:
    listener = _make_listener(["S1000003", "TIME_000001"], staleness_seconds=60.0)
    recent = datetime.now(UTC) - timedelta(seconds=30)
    with listener._lock:
        listener._last_time_arrival = recent
    listener.check_staleness(datetime.now(UTC))
    assert listener._in_los is False


def test_non_time_channel_does_not_reset_los_clock() -> None:
    # Only TIME_000001 arrivals reset the LOS clock; telemetry channels don't.
    listener = _make_listener(["S1000003", "TIME_000001"])
    listener.onItemUpdate(_make_update("S1000003", "21.5"))
    assert listener._last_time_arrival is None


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
        update = _make_update("S1000003", str(float(i)), str(4076.0 + i))
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
    update = _make_update("S1000003", "1.0")
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
    update = _make_update("TIME_000001", "1234.5")
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
