"""ISS Live telemetry collector daemon.

Opens a single Lightstreamer MERGE session over the ISSLive adapter set,
subscribes to all requested PUIs plus context items, and flushes in-memory
tick buffers to Parquet shards every ``flush_interval_seconds`` seconds.

The collector is intentionally single-threaded from a data-correctness
perspective — Lightstreamer callbacks arrive on a library thread and append
to per-channel buffers protected by a threading.Lock; the flush thread drains
those buffers on a timer.

Loss-of-Signal (LOS) is detected operationally by watching for wall-clock
silence across ALL subscribed items — real LOS silences the entire feed.
On LOS onset/recovery a structured log event is emitted; the ``is_los`` flag
is derived downstream in Phase 13 from the raw-tick archive and gap analysis.

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

import os
import signal
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
# TLS certificate bundle
# ---------------------------------------------------------------------------

def ensure_ssl_cert_env() -> None:
    """Point SSL_CERT_FILE at certifi's bundle if it is not already set.

    uv-managed CPython on macOS does not load the system keychain, so
    ``ssl.create_default_context()`` (used by the Lightstreamer client's aiohttp
    transport) fails verification against push.lightstreamer.com and the
    connection silently retries forever. certifi is always present (httpx,
    a core dependency, requires it). ``setdefault`` semantics mean an explicit
    SSL_CERT_FILE — including the Debian system store inside the Docker image —
    still wins.
    """
    if os.environ.get("SSL_CERT_FILE"):
        return
    import certifi

    os.environ["SSL_CERT_FILE"] = certifi.where()
    log.info("collector.ssl_cert_file_set", path=certifi.where())


# ---------------------------------------------------------------------------
# AOS timestamp parsing
# ---------------------------------------------------------------------------

# The ISSLive "TimeStamp" field is a decimal AOS (Acquisition of Signal)
# timestamp, e.g. "4076.449722222222" — NOT an absolute wall-clock time, and
# its epoch is not publicly documented. We preserve it verbatim as a float
# (aos_timestamp) for possible later decoding, but anchor the canonical
# telemetry_timestamp on ingest_time (true UTC receipt) instead — the feed is a
# near-real-time MERGE push, so receipt time tracks measurement time within
# seconds, which is more than precise enough for the 30 s resample grid.


def parse_aos_timestamp(raw: str) -> float | None:
    """Parse the raw ISSLive TimeStamp field to a float. None if unparseable."""
    if not raw or not raw.strip():
        return None
    try:
        return float(raw)
    except (ValueError, TypeError):
        log.debug("collector.aos_timestamp_parse_failed", raw=raw)
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
        # Wall-clock time of the most recent update on ANY subscribed item.
        # LOS drops the entire feed; using any-channel arrival is more reliable
        # than TIME_000001 alone — a 1-hour dry-run showed TIME_000001 can stall
        # for ~5 min with full signal present while all other channels keep
        # updating normally.
        self._last_any_arrival: datetime | None = None
        self._in_los: bool = False

    # -- Lightstreamer SubscriptionListener interface -------------------------

    def onItemUpdate(self, update: Any) -> None:
        """Called by Lightstreamer on every MERGE update."""
        channel_id: str = update.getItemName()
        raw_value: str | None = update.getValue("Value")
        raw_ts: str | None = update.getValue("TimeStamp")

        if raw_value is None:
            return

        try:
            value = float(raw_value)
        except (ValueError, TypeError):
            log.warning("collector.value_parse_failed", channel_id=channel_id, raw=raw_value)
            return

        # Anchor the canonical timestamp on receipt time (true UTC); preserve
        # the raw AOS decimal verbatim for possible later decoding.
        now = datetime.now(UTC)
        row = {
            "telemetry_timestamp": now,
            "value": value,
            "aos_timestamp": parse_aos_timestamp(raw_ts) if raw_ts is not None else None,
        }

        with self._lock:
            self._buffers[channel_id].append(row)
            self._note_arrival(now)

    def _note_arrival(self, now: datetime) -> None:
        """Record that at least one item updated; clear LOS if active. Called under lock."""
        self._last_any_arrival = now

        if self._in_los:
            self._in_los = False
            log.info("collector.los_recovered", recovered_at=now.isoformat())

    def check_staleness(self, now: datetime) -> None:
        """Check whether ANY subscribed item has stopped arriving.

        Called from the flush thread, not from the Lightstreamer callback.
        Real LOS silences the entire feed, so using any-channel arrival avoids
        false positives from individual channel stalls (e.g. TIME_000001 pausing
        5 min while other channels remain active).
        """
        with self._lock:
            last = self._last_any_arrival

        if last is None:
            return

        elapsed = (now - last).total_seconds()
        if elapsed > self._los_staleness and not self._in_los:
            self._in_los = True
            log.warning(
                "collector.los_onset",
                elapsed_seconds=round(elapsed, 1),
                last_arrival=last.isoformat(),
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


# Maximum rows retained per channel when GCS is unavailable. At ~2 s median
# cadence and 28 channels this caps memory at roughly 6 hours of ticks per
# channel (< 50 MB total) before we start dropping-oldest and logging an
# overflow warning. Prevents OOM on the 2 GB e2-small during a prolonged
# GCS outage.
_MAX_BUFFERED_ROWS = 10_800  # ~6 h at 2 s cadence


class LightstreamerCollector:
    """Long-running ISS Live telemetry collector.

    Opens one Lightstreamer MERGE session, subscribes to all items from
    ``subscription_items(config.channel_set)``, and flushes per-channel buffers
    to Parquet shards every ``config.flush_interval_seconds`` seconds.

    On a transient flush error the tick rows are re-queued at the front of the
    channel buffer (write-first / clear-on-success) so no data is silently
    dropped. The buffer is capped at ``_MAX_BUFFERED_ROWS`` per channel to bound
    memory during prolonged GCS outages.

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
        # Must run BEFORE importing the Lightstreamer client: its transport
        # resolves the TLS cert bundle at import time, so a later env change
        # would be ignored.
        ensure_ssl_cert_env()

        try:
            from lightstreamer.client import LightstreamerClient, Subscription
        except ImportError as exc:
            raise RuntimeError(
                "lightstreamer-client-lib is required for the ISS collector. "
                "Install it with: pip install lightstreamer-client-lib"
            ) from exc

        # SIGTERM (docker stop / VM shutdown) must trigger graceful shutdown so
        # the finally-block final-flush runs. signal.signal only works on the
        # main thread, which is where the CLI invokes run().
        original_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, lambda _sig, _frame: self.stop())

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
            signal.signal(signal.SIGTERM, original_sigterm)
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
        """Drain all non-empty buffers to Parquet shards.

        On write failure, rows are re-queued at the front of the channel buffer
        (write-first / clear-on-success). The buffer is capped at
        ``_MAX_BUFFERED_ROWS`` per channel to prevent OOM during prolonged
        GCS outages.
        """
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
                # Re-queue: prepend the failed rows so the next flush retries.
                with self._lock:
                    merged = rows + self._buffers[channel_id]
                    if len(merged) > _MAX_BUFFERED_ROWS:
                        dropped = len(merged) - _MAX_BUFFERED_ROWS
                        merged = merged[-_MAX_BUFFERED_ROWS:]
                        log.warning(
                            "collector.buffer_overflow",
                            channel_id=channel_id,
                            dropped_rows=dropped,
                        )
                    self._buffers[channel_id] = merged

        if final:
            log.info("collector.final_flush_done")

    def stop(self) -> None:
        """Signal the flush loop to stop (for testing or graceful shutdown)."""
        self._stop_event.set()
