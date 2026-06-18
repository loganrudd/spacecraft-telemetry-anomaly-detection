"""ISS Live telemetry collector daemon.

Opens a single Lightstreamer MERGE session over the ISSLive adapter set,
subscribes to all requested PUIs plus context items, and flushes in-memory
tick buffers to Parquet shards every ``flush_interval_seconds`` seconds.

The collector is intentionally single-threaded from a data-correctness
perspective — Lightstreamer callbacks arrive on a library thread and append
to per-channel buffers protected by a threading.Lock; the flush thread drains
those buffers on a timer.

Loss-of-Signal (LOS) is detected operationally by watching TIME_000001 for
TimeStamp non-advance. On LOS onset/recovery a structured log event is emitted;
the ``is_los`` flag is derived downstream in Phase 13 from the TIME_000001
archive and gap analysis.

Eclipse flatlines on power channels are nominal — the collector writes all
ticks unconditionally and lets Phase 13 / HPO handle the threshold calibration.

Usage:
    from spacecraft_telemetry.ingest.collector import LightstreamerCollector
    from spacecraft_telemetry.core.config import load_settings

    settings = load_settings("cloud")
    collector = LightstreamerCollector(settings.collect, dest_dir="gs://project/raw-data")
    collector.run()          # blocks; Ctrl-C / SIGTERM for graceful shutdown
    collector.run(seconds=3600)   # time-bounded dry-run
"""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from spacecraft_telemetry.core.config import CollectorConfig
from spacecraft_telemetry.core.logging import get_logger
from spacecraft_telemetry.ingest.collector_io import flush_buffer
from spacecraft_telemetry.ingest.iss_channels import subscription_items

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Timestamp parsing
# ---------------------------------------------------------------------------

# ISSLive TimeStamp field format, confirmed empirically (ISO 8601 UTC):
# "2024-01-15T12:00:00.000Z" — verified during Phase 12 dry-run.
# The parser falls back to a raw float (Unix epoch in seconds) if the ISO
# parse fails, which covers edge-cases during feed transitions.

_ISO_FMT = "%Y-%m-%dT%H:%M:%S.%fZ"
_ISO_FMT_NO_MS = "%Y-%m-%dT%H:%M:%SZ"


def parse_timestamp(raw: str) -> datetime | None:
    """Parse the Lightstreamer TimeStamp field to a UTC datetime.

    Tries ISO 8601 with and without milliseconds, then Unix-epoch float.
    Returns None when the value is empty or unparseable.
    """
    if not raw:
        return None
    for fmt in (_ISO_FMT, _ISO_FMT_NO_MS):
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=UTC)
        except ValueError:
            continue
    # Last resort: Unix epoch seconds as a float string
    try:
        return datetime.fromtimestamp(float(raw), tz=UTC)
    except (ValueError, OSError):
        log.warning("collector.timestamp_parse_failed", raw=raw)
        return None


# ---------------------------------------------------------------------------
# Client listener — logs every connection state transition
# ---------------------------------------------------------------------------

class _ISSClientListener:
    """Lightstreamer ClientListener that logs connection status changes.

    Lightstreamer's connect() is non-blocking; all connection events arrive
    here on a library thread. Without this listener the collector is silent
    while connecting/retrying, making it impossible to distinguish "still
    connecting" from "stuck".

    Status strings emitted by the library:
        CONNECTING, CONNECTED:WS-STREAMING, CONNECTED:HTTP-STREAMING,
        STALLED, DISCONNECTED:WILL-RETRY, DISCONNECTED:TRYING-RECOVERY,
        DISCONNECTED
    """

    def onStatusChange(self, status: str) -> None:
        connected = status.startswith("CONNECTED")
        level = "info" if connected else "warning"
        getattr(log, level)("collector.connection_status", status=status)

    def onServerError(self, code: int, message: str) -> None:
        log.error("collector.server_error", code=code, message=message)

    def onPropertyChange(self, prop: str) -> None:
        pass

    def onListenEnd(self) -> None:
        pass

    def onListenStart(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Subscription listener
# ---------------------------------------------------------------------------

class _ISSSubscriptionListener:
    """Lightstreamer SubscriptionListener that appends ticks to shared buffers.

    Designed to be injected into a Lightstreamer Subscription via
    ``sub.addListener(listener)``.  In tests, ``on_item_update`` can be driven
    directly without a live connection.
    """

    def __init__(
        self,
        buffers: dict[str, list[dict[str, Any]]],
        lock: threading.Lock,
        los_staleness_seconds: float,
    ) -> None:
        self._buffers = buffers
        self._lock = lock
        self._los_staleness = los_staleness_seconds
        # Per-channel last telemetry_timestamp for LOS staleness detection.
        self._last_ts: dict[str, datetime] = {}
        self._in_los: bool = False

    # -- Lightstreamer SubscriptionListener interface -------------------------

    def onItemUpdate(self, update: Any) -> None:
        """Called by Lightstreamer on every MERGE update."""
        channel_id: str = update.getItemName()
        raw_value: str | None = update.getValue("Value")
        raw_ts: str | None = update.getValue("TimeStamp")

        if raw_value is None or raw_ts is None:
            return

        try:
            value = float(raw_value)
        except (ValueError, TypeError):
            log.warning("collector.value_parse_failed", channel_id=channel_id, raw=raw_value)
            return

        telemetry_ts = parse_timestamp(raw_ts)
        if telemetry_ts is None:
            return

        now = datetime.now(UTC)
        row = {
            "telemetry_timestamp": telemetry_ts,
            "value": value,
            "ingest_time": now,
        }

        with self._lock:
            self._buffers[channel_id].append(row)
            self._update_los(channel_id, telemetry_ts, now)

    def _update_los(
        self, channel_id: str, telemetry_ts: datetime, now: datetime
    ) -> None:
        """Update LOS state based on TIME_000001 staleness. Called under lock."""
        if channel_id != "TIME_000001":
            return

        prev = self._last_ts.get("TIME_000001")
        self._last_ts["TIME_000001"] = telemetry_ts

        if prev is None:
            return

        # A new tick on TIME_000001 means signal is present.
        if self._in_los:
            self._in_los = False
            log.info("collector.los_recovered", telemetry_ts=telemetry_ts.isoformat())

    def check_staleness(self, now: datetime) -> None:
        """Check whether TIME_000001 has stopped advancing.

        Called from the flush thread, not from the Lightstreamer callback.
        """
        with self._lock:
            last = self._last_ts.get("TIME_000001")

        if last is None:
            return

        elapsed = (now - last).total_seconds()
        if elapsed > self._los_staleness and not self._in_los:
            self._in_los = True
            log.info(
                "collector.los_onset",
                elapsed_seconds=round(elapsed, 1),
                last_ts=last.isoformat(),
            )

    # -- No-op stubs so the class satisfies the SubscriptionListener interface

    def onSubscription(self) -> None:
        log.info("collector.subscribed")

    def onUnsubscription(self) -> None:
        log.info("collector.unsubscribed")

    def onSubscriptionError(self, code: int, message: str) -> None:
        log.error("collector.subscription_error", code=code, message=message)

    def onListenStart(self) -> None:
        pass

    def onListenEnd(self) -> None:
        pass

    def onClearSnapshot(self, item_name: str, item_pos: int) -> None:
        pass

    def onEndOfSnapshot(self, item_name: str, item_pos: int) -> None:
        pass

    def onItemLostUpdates(
        self, item_name: str, item_pos: int, lost_updates: int
    ) -> None:
        log.warning(
            "collector.updates_lost",
            channel_id=item_name,
            lost_updates=lost_updates,
        )


# ---------------------------------------------------------------------------
# Main collector
# ---------------------------------------------------------------------------

class LightstreamerCollector:
    """Long-running ISS Live telemetry collector.

    Opens one Lightstreamer MERGE session, subscribes to all items from
    ``subscription_items(config.channel_set)``, and flushes per-channel buffers
    to Parquet shards every ``config.flush_interval_seconds`` seconds.

    Args:
        config: ``CollectorConfig`` (from ``Settings.collect``).
        dest_dir: Root directory for raw tick shards (local path or GCS URI).
    """

    def __init__(self, config: CollectorConfig, dest_dir: Path | str) -> None:
        self._config = config
        self._dest_dir = dest_dir
        self._items = subscription_items(config.channel_set)

        self._lock = threading.Lock()
        self._buffers: dict[str, list[dict[str, Any]]] = {
            item: [] for item in self._items
        }
        self._listener = _ISSSubscriptionListener(
            buffers=self._buffers,
            lock=self._lock,
            los_staleness_seconds=config.los_staleness_seconds,
        )
        self._stop_event = threading.Event()

    # -- Public API -----------------------------------------------------------

    def run(self, seconds: float | None = None) -> None:
        """Connect and collect telemetry.

        Args:
            seconds: If set, stop after this many seconds (for dry-runs).
                     If ``None``, run until interrupted (SIGTERM or KeyboardInterrupt).
        """
        try:
            from lightstreamer.client import LightstreamerClient, Subscription
        except ImportError as exc:
            raise RuntimeError(
                "lightstreamer-client-lib is required for the ISS collector. "
                "Install it with: pip install lightstreamer-client-lib"
            ) from exc

        client = LightstreamerClient(
            self._config.lightstreamer_url,
            self._config.adapter_set,
        )
        client.addListener(_ISSClientListener())

        sub = Subscription("MERGE", self._items, self._config.fields)
        sub.addListener(self._listener)

        log.info(
            "collector.starting",
            url=self._config.lightstreamer_url,
            adapter_set=self._config.adapter_set,
            items=len(self._items),
            dest_dir=str(self._dest_dir),
        )

        client.connect()
        client.subscribe(sub)

        try:
            self._flush_loop(seconds=seconds)
        except KeyboardInterrupt:
            pass
        finally:
            log.info("collector.stopping")
            client.unsubscribe(sub)
            client.disconnect()
            self._flush_all(final=True)
            log.info("collector.stopped")

    # -- Internal flush loop --------------------------------------------------

    def _flush_loop(self, seconds: float | None) -> None:
        """Timer-based flush loop. Blocks until stop_event or timeout."""
        deadline = time.monotonic() + seconds if seconds is not None else None
        interval = self._config.flush_interval_seconds

        while not self._stop_event.is_set():
            now_mono = time.monotonic()
            if deadline is not None and now_mono >= deadline:
                break

            sleep_time = interval if deadline is None else min(interval, deadline - now_mono)
            self._stop_event.wait(timeout=sleep_time)

            self._flush_all(final=False)
            self._listener.check_staleness(datetime.now(UTC))

    def _flush_all(self, *, final: bool) -> None:
        """Drain all non-empty buffers to Parquet shards."""
        now = datetime.now(UTC)
        for channel_id in self._items:
            with self._lock:
                rows = self._buffers[channel_id]
                if not rows:
                    continue
                self._buffers[channel_id] = []

            try:
                flush_buffer(rows, self._dest_dir, channel_id, bucket_ts=now)
            except Exception:
                log.exception(
                    "collector.flush_error", channel_id=channel_id, rows=len(rows)
                )

        if final:
            log.info("collector.final_flush_done")

    def stop(self) -> None:
        """Signal the flush loop to stop (for testing or graceful shutdown)."""
        self._stop_event.set()
