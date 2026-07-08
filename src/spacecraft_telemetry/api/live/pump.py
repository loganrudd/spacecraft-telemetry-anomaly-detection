"""Live ISS telemetry pump — Lightstreamer → EventBroadcaster bridge.

LivePump opens a single Lightstreamer MERGE session subscribed to all ISS PUIs,
archives raw ticks for every channel to GCS, and publishes two SSE event types:

  event: raw        — per native-cadence tick, normalized value only.
                      Drives the continuous live chart line at 1-10 s cadence.
  event: telemetry  — one per closed 30-second grid bucket, with prediction
                      and anomaly flags.  Drives the anomaly overlay.
  event: drift      — via drift_feed.step_drift, fed the same fault-adjusted
                      bucket value as event: telemetry.  One shared monitor
                      per channel backs both this pump and the ESA replay
                      loop, so drift tracking is unaffected by LOS fallback.

On Loss-of-Signal (TDRS handover / zone-of-exclusion), the pump pauses live
emission and optionally starts run_shared_loop() as a replay fallback so the
chart stays alive.  On recovery, replay is cancelled, engines are re-primed
from the last window_size collected grid buckets, and live emission resumes.

Fault injection (POST /api/inject) works unchanged:
  - apply_fault() is called on each raw normalized tick → spike appears on the
    chart line immediately.
  - The fault-adjusted value flows into the 30-second bucket aggregator →
    the model also sees it at the next grid boundary.

Thread-safety: Lightstreamer callbacks arrive on library threads.  All asyncio
state (broadcaster, engines, resampler) lives on the event loop and is only
touched via run_coroutine_threadsafe or call_soon_threadsafe.  The archive
flush follows the same discipline: draining and re-queuing ``_archive_buffers``
(``_drain_archive`` / ``_requeue_failed``) always run on the loop thread; only
the blocking Parquet write (``_write_archive``) is offloaded via
``asyncio.to_thread``, so no lock is needed on the shared buffers.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable, Coroutine
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from spacecraft_telemetry.api.drift_feed import step_drift
from spacecraft_telemetry.api.live.normalization import NormalizationParams, normalize
from spacecraft_telemetry.api.live.resampler import OnlineGridResampler
from spacecraft_telemetry.api.models import RawTelemetryEvent
from spacecraft_telemetry.core.config import CollectorConfig
from spacecraft_telemetry.core.logging import get_logger
from spacecraft_telemetry.ingest.collector_io import flush_buffer
from spacecraft_telemetry.ingest.iss_channels import subscription_items
from spacecraft_telemetry.ingest.lightstreamer import (
    MAX_BUFFERED_ROWS,
    ISSClientListener,
    ensure_ssl_cert_env,
)

if TYPE_CHECKING:
    from spacecraft_telemetry.api.broadcast import EventBroadcaster
    from spacecraft_telemetry.api.inference import ChannelInferenceEngine
    from spacecraft_telemetry.api.state import AppState

log = get_logger("api.live.pump")

# Maximum rows buffered per channel between archive flushes (shared cap with
# the collector daemon — see ingest/lightstreamer.py).
_MAX_ARCHIVE_ROWS = MAX_BUFFERED_ROWS


class _PumpSubscriptionListener:
    """Lightstreamer SubscriptionListener that bridges ticks to the asyncio loop.

    Parses each MERGE update on the Lightstreamer library thread and schedules
    ``pump._on_tick`` on the event loop via ``run_coroutine_threadsafe``.
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        on_tick: Callable[[str, datetime, float], Coroutine[Any, Any, None]],
    ) -> None:
        self._loop = loop
        self._on_tick = on_tick

    def onItemUpdate(self, update: Any) -> None:
        channel_id: str = update.getItemName()
        raw_value: str | None = update.getValue("Value")
        if raw_value is None:
            return
        try:
            value = float(raw_value)
        except (ValueError, TypeError):
            log.warning("pump.value_parse_failed", channel_id=channel_id, raw=raw_value)
            return
        now = datetime.now(UTC)
        asyncio.run_coroutine_threadsafe(self._on_tick(channel_id, now, value), self._loop)

    # No-op stubs for the SubscriptionListener interface.
    def onSubscription(self) -> None:
        log.info("pump.subscribed")

    def onUnsubscription(self) -> None:
        log.info("pump.unsubscribed")

    def onSubscriptionError(self, code: int, message: str) -> None:
        log.error("pump.subscription_error", code=code, message=message)

    def onListenStart(self) -> None:
        pass

    def onListenEnd(self) -> None:
        pass

    def onClearSnapshot(self, item_name: str, item_pos: int) -> None:
        pass

    def onEndOfSnapshot(self, item_name: str, item_pos: int) -> None:
        pass

    def onItemLostUpdates(self, item_name: str, item_pos: int, lost_updates: int) -> None:
        log.warning("pump.updates_lost", channel_id=item_name, lost_updates=lost_updates)


class LivePump:
    """Live ISS telemetry pump that feeds EventBroadcaster from Lightstreamer.

    Args:
        loop:               The asyncio event loop (used for thread→loop bridge).
        broadcaster:        SSE fan-out hub (shared with the replay loop).
        engines:            Mapping of channel_id → ChannelInferenceEngine for
                            channels with ``@champion`` models.  The pump runs
                            inference only for these channels.
        norm_params:        Normalization parameters from ``normalization_params.json``.
        collect_config:     Collector configuration (URL, channel set, intervals).
        state:              AppState reference for LOS fallback replay loop.
                            When ``None``, LOS onset emits a status event but no
                            fallback replay is started.
        archive_to_gcs:     Whether to archive raw ticks to GCS via flush_buffer.
        raw_ticks_dir:      Root directory for raw tick shards (GCS URI or local).
        los_stats_median_s: Historical median LOS duration in seconds, surfaced as
                            ``expected_resume_in_s`` in the status event.
        _fallback_start_fn: Injectable for testing.  Defaults to run_shared_loop(state).
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        broadcaster: EventBroadcaster,
        engines: dict[str, ChannelInferenceEngine],
        norm_params: NormalizationParams,
        collect_config: CollectorConfig,
        state: AppState | None = None,
        archive_to_gcs: bool = False,
        raw_ticks_dir: Path | str | None = None,
        los_stats_median_s: float | None = None,
        _fallback_start_fn: Callable[[], Coroutine[Any, Any, None]] | None = None,
    ) -> None:
        self._loop = loop
        self._broadcaster = broadcaster
        self._engines: dict[str, ChannelInferenceEngine] = dict(engines)
        self._served_channels: frozenset[str] = frozenset(engines.keys())
        self._norm_params = norm_params
        self._collect_config = collect_config
        self._state = state
        self._archive_to_gcs = archive_to_gcs
        self._raw_ticks_dir = raw_ticks_dir
        self._los_stats_median_s = los_stats_median_s

        # Per-channel resamplers (served channels only).
        self._resamplers: dict[str, OnlineGridResampler] = {
            ch: OnlineGridResampler(collect_config.grid_interval_seconds)
            for ch in self._served_channels
        }

        # Recent grid-bucket values per channel for LOS-recovery re-prime.
        # maxlen = window_size + threshold_window so prime_with_scoring() can
        # warm both the LSTM input buffer AND the EWMA/threshold ring buffer on
        # LOS recovery — without enough history, threshold stays inf for
        # threshold_window x grid_interval after each handover (up to ~2 h),
        # and with 16 LOS/day the detector would never warm up during live ops.
        # On the first few LOS events the deque may hold fewer than the full
        # warmup depth; prime_with_scoring() handles partial history gracefully
        # (threshold warms faster once live ticks resume).
        self._recent_buckets: dict[str, deque[float]] = {
            ch: deque(
                maxlen=(
                    getattr(engines[ch], "window_size", 128)
                    + getattr(getattr(engines[ch], "params", None), "threshold_window", 250)
                )
            )
            for ch in self._served_channels
        }

        # Archive buffers (all subscribed channels).
        self._archive_buffers: dict[str, list[dict[str, Any]]] = {}

        # LOS state.
        self._last_any_arrival: datetime | None = None
        self._in_los: bool = False
        self._replay_task: asyncio.Task[None] | None = None

        # Per-channel flag: any tick in the current bucket was injected.
        self._bucket_injected: dict[str, bool] = {}

        # Newest grid bucket for which end_tick() has been called.  The
        # injection elapsed counter must advance once per grid interval —
        # NOT once per channel-bucket close: with N served channels an
        # unconditional end_tick() in the bucket-close loop advances
        # ``elapsed`` N times per interval, silently dividing every injected
        # fault's duration by N (a "60-tick" demo fault lasted ~10 buckets
        # with 6 channels) and detaching live faults from the 20-120-bucket
        # population the HPO thresholds are calibrated on.
        self._last_ended_bucket: datetime | None = None

        # Background asyncio tasks.
        self._los_watchdog_task: asyncio.Task[None] | None = None
        self._archive_flush_task: asyncio.Task[None] | None = None

        # Lightstreamer objects — created in start().
        self._ls_client: Any = None
        self._ls_sub: Any = None

        # Fallback function for LOS replay.
        if _fallback_start_fn is not None:
            self._fallback_fn: Callable[[], Coroutine[Any, Any, None]] | None = _fallback_start_fn
        elif state is not None:
            from spacecraft_telemetry.api.broadcast import run_shared_loop

            self._fallback_fn = lambda: run_shared_loop(state)
        else:
            self._fallback_fn = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Connect Lightstreamer, start archive-flush and LOS-watchdog tasks."""
        ensure_ssl_cert_env()

        items = subscription_items(self._collect_config.channel_set)
        self._archive_buffers = {item: [] for item in items}

        try:
            from lightstreamer.client import LightstreamerClient, Subscription
        except ImportError:
            log.warning(
                "pump.lightstreamer_not_installed",
                hint="install lightstreamer-client-lib",
            )
            return

        self._ls_client = LightstreamerClient(
            self._collect_config.lightstreamer_url,
            self._collect_config.adapter_set,
        )
        self._ls_client.addListener(ISSClientListener())

        listener = _PumpSubscriptionListener(loop=self._loop, on_tick=self._on_tick)
        self._ls_sub = Subscription("MERGE", items, self._collect_config.fields)
        self._ls_sub.addListener(listener)
        self._ls_client.connect()
        self._ls_client.subscribe(self._ls_sub)

        self._los_watchdog_task = asyncio.create_task(self._los_watchdog())
        if self._archive_to_gcs and self._raw_ticks_dir:
            self._archive_flush_task = asyncio.create_task(self._archive_flush_loop())

        log.info(
            "pump.started",
            served_channels=len(self._served_channels),
            archive_to_gcs=self._archive_to_gcs,
        )

    async def stop(self) -> None:
        """Disconnect, cancel background tasks, run a final archive flush."""
        if self._los_watchdog_task:
            self._los_watchdog_task.cancel()
        if self._archive_flush_task:
            self._archive_flush_task.cancel()
        if self._replay_task:
            self._replay_task.cancel()

        if self._ls_client is not None:
            if self._ls_sub is not None:
                self._ls_client.unsubscribe(self._ls_sub)
            self._ls_client.disconnect()

        if self._archive_to_gcs and self._raw_ticks_dir:
            await self._flush_archive_once()

        log.info("pump.stopped")

    # ------------------------------------------------------------------
    # Core tick handler — public so tests can drive it directly
    # ------------------------------------------------------------------

    async def _on_tick(self, channel: str, ts: datetime, raw_value: float) -> None:
        """Process one tick from any subscribed channel.

        Called on the asyncio event loop (via run_coroutine_threadsafe in
        production, or directly in tests).
        """
        # 1. Update LOS detection timestamp (ALL channels, including context items).
        self._last_any_arrival = ts

        # 2. Archive tick.
        if channel in self._archive_buffers:
            self._archive_buffers[channel].append(
                {
                    "telemetry_timestamp": ts,
                    "value": float(raw_value),
                    "aos_timestamp": None,
                }
            )

        # 3. Don't process live events during LOS (replay loop is driving engines).
        if self._in_los:
            return

        # 4. Skip channels with no champion model.
        if channel not in self._served_channels:
            return

        # 5. Z-score normalize.
        try:
            normalized = normalize(channel, raw_value, self._norm_params)
        except (KeyError, ZeroDivisionError):
            log.warning("pump.normalize_failed", channel=channel)
            return

        # 6. Activate any pending injection so the spike appears immediately on
        #    the raw chart line rather than waiting for the next bucket close.
        self._broadcaster.begin_tick()

        # 7. Apply fault injection.  (The per-channel _bucket_injected flag is
        #    recorded AFTER step 9's bucket-close loop — see comment there.)
        injected_normalized, is_injected = self._broadcaster.apply_fault(channel, normalized)

        # 8. Publish event: raw (immediate visual feedback on the chart line).
        raw_event = RawTelemetryEvent(
            timestamp=ts, channel=channel, value_normalized=injected_normalized
        )
        self._broadcaster.publish(
            channel,
            f"event: raw\ndata: {raw_event.model_dump_json()}\n\n".encode(),
        )

        # 9. Push fault-adjusted value to resampler; emit telemetry on bucket close.
        for bucket_ts, bucket_mean in self._resamplers[channel].push(ts, injected_normalized):
            bucket_injected = self._bucket_injected.pop(channel, False)
            event = self._engines[channel].step(bucket_mean, bucket_ts, bucket_injected)
            self._broadcaster.publish(
                channel,
                f"event: telemetry\ndata: {event.model_dump_json()}\n\n".encode(),
            )
            self._recent_buckets[channel].append(bucket_mean)
            if self._state is not None:
                await step_drift(self._state, channel, bucket_mean)
            # Advance the injection elapsed counter once per grid bucket.
            # Channels close the same 30s bucket at slightly different
            # wall-times (each closes when its own next tick arrives), so
            # only the FIRST closure of a new bucket advances the shared
            # clock; later channels closing the same (or an older, laggier)
            # bucket must not — see _last_ended_bucket in __init__.
            if self._last_ended_bucket is None or bucket_ts > self._last_ended_bucket:
                self._broadcaster.end_tick()
                self._last_ended_bucket = bucket_ts

        # Record the injected flag for THIS tick's (still-open) bucket only
        # after the closes above: the pop in the loop consumes flags set by
        # prior ticks.  Setting the flag before the push let a tick's flag be
        # consumed by the close of the PREVIOUS bucket, shifting is_anomaly
        # labels one bucket early and dropping the label on a fault's final
        # bucket entirely.
        if is_injected:
            self._bucket_injected[channel] = True

    # ------------------------------------------------------------------
    # LOS state machine
    # ------------------------------------------------------------------

    async def _on_los_onset(self) -> None:
        """Transition to LOS: pause live emission, start replay fallback."""
        self._in_los = True
        log.warning("pump.los_onset")
        self._broadcaster.publish_status(
            "los",
            mode="replay",
            expected_resume_in_s=self._los_stats_median_s,
        )
        if self._fallback_fn is not None:
            self._replay_task = asyncio.create_task(self._fallback_fn())

    async def _on_los_recovery(self) -> None:
        """Transition from LOS: cancel replay, re-prime engines, resume live."""
        if self._replay_task is not None:
            self._replay_task.cancel()
            self._replay_task = None

        # Re-prime each engine from the last collected buckets.
        # prime_with_scoring() warms both the input window and the EWMA/threshold
        # ring buffer so detection resumes immediately after handover rather than
        # waiting threshold_window x 30s for the threshold to go finite again.
        for ch, engine in self._engines.items():
            if self._recent_buckets[ch]:
                engine.prime_with_scoring(list(self._recent_buckets[ch]))

        # Reset resamplers: their _current_bucket is frozen at the last
        # pre-LOS bucket (no ticks reached push() during LOS). Without this,
        # the first post-recovery tick lands far ahead of _current_bucket and
        # push()'s ffill loop backfills one bucket per grid step across the
        # entire outage — a flood of stale telemetry events that immediately
        # overwrites the window just re-primed above.
        self._resamplers = {
            ch: OnlineGridResampler(self._collect_config.grid_interval_seconds)
            for ch in self._served_channels
        }
        self._bucket_injected.clear()

        self._in_los = False
        log.info("pump.los_recovered")
        self._broadcaster.publish_status("resumed")

    # ------------------------------------------------------------------
    # Background tasks
    # ------------------------------------------------------------------

    async def _los_watchdog(self) -> None:
        """Periodically check for LOS and drive state transitions."""
        cadence = min(self._collect_config.los_staleness_seconds / 3.0, 20.0)
        while True:
            await asyncio.sleep(cadence)
            last = self._last_any_arrival
            if last is None:
                continue
            now = datetime.now(UTC)
            elapsed = (now - last).total_seconds()
            if not self._in_los and elapsed > self._collect_config.los_staleness_seconds:
                await self._on_los_onset()
            elif self._in_los and elapsed <= self._collect_config.los_staleness_seconds:
                await self._on_los_recovery()

    async def _archive_flush_loop(self) -> None:
        """Periodically flush archive buffers to GCS."""
        while True:
            await asyncio.sleep(self._collect_config.flush_interval_seconds)
            await self._flush_archive_once()

    async def _flush_archive_once(self) -> None:
        """Drain buffers on the loop thread, write via a worker thread.

        Splitting drain (mutates ``_archive_buffers``, which ``_on_tick`` also
        appends to) from write (blocking Parquet I/O) keeps every mutation of
        shared state on the event loop, so no lock is needed: ``_on_tick`` and
        ``_drain_archive``/``_requeue_failed`` never run concurrently with each
        other. Only the actual I/O happens off-loop.
        """
        if not self._raw_ticks_dir:
            return
        drained = self._drain_archive()
        if not drained:
            return
        failed = await asyncio.to_thread(self._write_archive, drained)
        if failed:
            self._requeue_failed(failed)

    def _drain_archive(self) -> dict[str, list[dict[str, Any]]]:
        """Pop all non-empty archive buffers. Loop-thread only."""
        drained: dict[str, list[dict[str, Any]]] = {}
        for channel_id, rows in self._archive_buffers.items():
            if rows:
                drained[channel_id] = rows
                self._archive_buffers[channel_id] = []
        return drained

    def _write_archive(
        self, drained: dict[str, list[dict[str, Any]]]
    ) -> dict[str, list[dict[str, Any]]]:
        """Write drained rows to Parquet shards (blocking I/O; runs in a worker thread).

        Returns the subset of ``drained`` that failed to write, for the caller
        to re-queue on the loop thread.
        """
        assert self._raw_ticks_dir is not None  # caller checks before invoking
        now = datetime.now(UTC)
        failed: dict[str, list[dict[str, Any]]] = {}
        for channel_id, rows in drained.items():
            try:
                flush_buffer(rows, self._raw_ticks_dir, channel_id, bucket_ts=now)
            except Exception:
                log.exception("pump.archive_flush_error", channel_id=channel_id)
                failed[channel_id] = rows
        return failed

    def _requeue_failed(self, failed: dict[str, list[dict[str, Any]]]) -> None:
        """Re-queue failed-write rows at the front of each buffer. Loop-thread only."""
        for channel_id, rows in failed.items():
            existing = self._archive_buffers.get(channel_id, [])
            merged = rows + existing
            if len(merged) > _MAX_ARCHIVE_ROWS:
                dropped = len(merged) - _MAX_ARCHIVE_ROWS
                merged = merged[-_MAX_ARCHIVE_ROWS:]
                log.warning(
                    "pump.archive_buffer_overflow",
                    channel_id=channel_id,
                    dropped_rows=dropped,
                )
            self._archive_buffers[channel_id] = merged
