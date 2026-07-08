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
# LOS staleness detection (wall-clock gap on ANY subscribed item)
# ---------------------------------------------------------------------------


def test_los_onset_when_feed_silent() -> None:
    listener = _make_listener(["S1000003", "TIME_000001"], staleness_seconds=1.0)
    # Last update from any channel was an hour ago.
    old_arrival = datetime(2024, 6, 1, 11, 0, 0, tzinfo=UTC)
    with listener._lock:
        listener._last_any_arrival = old_arrival

    now = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)  # 1 hour later → stale
    listener.check_staleness(now)
    assert listener._in_los is True


def test_los_recovery_on_any_channel_arrival() -> None:
    # LOS clears on ANY channel update, not just TIME_000001.
    listener = _make_listener(["S1000003", "TIME_000001"], staleness_seconds=1.0)
    listener._in_los = True

    update = _make_update("S1000003", "21.5")
    listener.onItemUpdate(update)

    assert listener._in_los is False
    assert listener._last_any_arrival is not None


def test_los_recovery_on_time_channel_too() -> None:
    listener = _make_listener(["S1000003", "TIME_000001"], staleness_seconds=1.0)
    listener._in_los = True

    update = _make_update("TIME_000001", "14675219000")
    listener.onItemUpdate(update)

    assert listener._in_los is False


def test_no_los_when_arrivals_recent() -> None:
    listener = _make_listener(["S1000003", "TIME_000001"], staleness_seconds=60.0)
    recent = datetime.now(UTC) - timedelta(seconds=30)
    with listener._lock:
        listener._last_any_arrival = recent
    listener.check_staleness(datetime.now(UTC))
    assert listener._in_los is False


def test_any_channel_resets_los_clock() -> None:
    # Empirically: TIME_000001 stalled 5 min in a dry-run while other channels
    # kept updating. The LOS clock must advance on any channel.
    listener = _make_listener(["S1000003", "TIME_000001"])
    assert listener._last_any_arrival is None
    listener.onItemUpdate(_make_update("S1000003", "21.5"))
    assert listener._last_any_arrival is not None


def test_elapsed_since_last_arrival_none_before_first_tick() -> None:
    listener = _make_listener(["S1000003", "TIME_000001"])
    assert listener.elapsed_since_last_arrival(datetime.now(UTC)) is None


def test_elapsed_since_last_arrival_reports_seconds() -> None:
    listener = _make_listener(["S1000003", "TIME_000001"])
    old_arrival = datetime(2024, 6, 1, 11, 0, 0, tzinfo=UTC)
    with listener._lock:
        listener._last_any_arrival = old_arrival
    now = datetime(2024, 6, 1, 11, 30, 0, tzinfo=UTC)
    assert listener.elapsed_since_last_arrival(now) == pytest.approx(1800.0)


# ---------------------------------------------------------------------------
# Fatal-staleness self-restart (recovers a WEDGED Lightstreamer session that
# never exits on its own, unlike a crash -- see CollectorConfig docstring)
# ---------------------------------------------------------------------------


def test_fatal_staleness_triggers_stop(tmp_path: Path) -> None:
    config = CollectorConfig(
        channel_set="validation",
        raw_ticks_dir=str(tmp_path),
        los_staleness_seconds=60.0,
        fatal_staleness_seconds=120.0,
    )
    collector = LightstreamerCollector(config, dest_dir=tmp_path)
    old_arrival = datetime(2024, 6, 1, 11, 0, 0, tzinfo=UTC)
    with collector._listener._lock:
        collector._listener._last_any_arrival = old_arrival

    now = old_arrival + timedelta(seconds=121)  # just past fatal_staleness_seconds
    assert not collector._stop_event.is_set()
    collector._check_fatal_staleness(now)
    assert collector._stop_event.is_set()


def test_fatal_staleness_does_not_fire_within_threshold(tmp_path: Path) -> None:
    config = CollectorConfig(
        channel_set="validation",
        raw_ticks_dir=str(tmp_path),
        los_staleness_seconds=60.0,
        fatal_staleness_seconds=120.0,
    )
    collector = LightstreamerCollector(config, dest_dir=tmp_path)
    old_arrival = datetime(2024, 6, 1, 11, 0, 0, tzinfo=UTC)
    with collector._listener._lock:
        collector._listener._last_any_arrival = old_arrival

    # Past los_staleness_seconds (routine LOS) but within fatal_staleness_seconds.
    now = old_arrival + timedelta(seconds=90)
    collector._check_fatal_staleness(now)
    assert not collector._stop_event.is_set()


def test_fatal_staleness_no_op_before_first_tick(tmp_path: Path) -> None:
    config = CollectorConfig(channel_set="validation", raw_ticks_dir=str(tmp_path))
    collector = LightstreamerCollector(config, dest_dir=tmp_path)
    collector._check_fatal_staleness(datetime.now(UTC))
    assert not collector._stop_event.is_set()


def test_fatal_staleness_config_validates_above_los() -> None:
    with pytest.raises(ValueError, match="must be greater than"):
        CollectorConfig(los_staleness_seconds=60.0, fatal_staleness_seconds=60.0)


def test_log_heartbeat_does_not_raise_before_first_tick(tmp_path: Path) -> None:
    config = CollectorConfig(channel_set="validation", raw_ticks_dir=str(tmp_path))
    collector = LightstreamerCollector(config, dest_dir=tmp_path)
    collector._log_heartbeat()  # must not raise when nothing has arrived yet


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


def test_flush_error_retains_rows_for_next_flush(tmp_path: Path) -> None:
    """On a write failure, rows are re-queued and survive to the next flush."""
    from unittest.mock import patch

    import spacecraft_telemetry.ingest.collector as _cmod

    config = CollectorConfig(
        channel_set="validation",
        raw_ticks_dir=str(tmp_path),
        flush_interval_seconds=300.0,
    )
    collector = LightstreamerCollector(config, dest_dir=tmp_path)

    update = _make_update("S1000003", "21.5")
    collector._listener.onItemUpdate(update)

    # Patch flush_buffer in the collector module's namespace (it's imported directly).
    with patch.object(_cmod, "flush_buffer", side_effect=OSError("GCS 503")):
        collector._flush_all(final=False)

    # Rows must be retained — not silently discarded.
    assert len(collector._buffers["S1000003"]) == 1, (
        "Rows were discarded on flush error instead of being re-queued"
    )


def test_flush_error_buffer_overflow_drops_oldest(tmp_path: Path) -> None:
    """When re-buffered rows exceed the cap, oldest are dropped and overflow is logged."""
    from unittest.mock import patch

    import spacecraft_telemetry.ingest.collector as _cmod
    from spacecraft_telemetry.ingest.collector import _MAX_BUFFERED_ROWS

    config = CollectorConfig(
        channel_set="validation",
        raw_ticks_dir=str(tmp_path),
        flush_interval_seconds=300.0,
    )
    collector = LightstreamerCollector(config, dest_dir=tmp_path)

    # Pre-fill the buffer beyond the cap.
    with collector._lock:
        collector._buffers["S1000003"] = [
            {"telemetry_timestamp": None, "value": float(i), "aos_timestamp": None}
            for i in range(_MAX_BUFFERED_ROWS)
        ]

    # A single new tick arrives during the flush attempt.
    update = _make_update("S1000003", "99.0")
    collector._listener.onItemUpdate(update)

    with patch.object(_cmod, "flush_buffer", side_effect=OSError("GCS 503")):
        collector._flush_all(final=False)

    # Buffer should be capped, not growing beyond the limit.
    assert len(collector._buffers["S1000003"]) <= _MAX_BUFFERED_ROWS


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
