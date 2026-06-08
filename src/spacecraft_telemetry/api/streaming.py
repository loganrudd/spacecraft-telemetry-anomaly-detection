"""SSE telemetry and drift stream composers.

``telemetry_stream`` is an async generator that:

1. Allocates one bounded ``asyncio.Queue`` per selected channel (the *pump*
   queue).  A single shared queue would head-of-line-block every pump when
   any one channel's consumer falls behind; per-channel queues limit
   backpressure to the offending channel only.
2. Spawns one ``asyncio.Task`` per channel.  Each pump iterates
   ``replay_channel`` and pushes SSE-encoded payloads into its channel's
   queue.
3. A merging consumer races all active queues with
   ``asyncio.wait(FIRST_COMPLETED)`` and yields whichever payload arrives
   first.  This keeps latency fair across channels regardless of their
   individual tick rates.
4. Client disconnect is detected via a dedicated watcher task that awaits
   the ASGI ``http.disconnect`` message and sets an ``asyncio.Event``.
   The merge loop races its getter tasks against that event so exit is
   push-based rather than polled on each iteration.
5. On any exit path (disconnect, exhaustion, exception) the pump tasks are
   cancelled and awaited so no coroutines are leaked.

``drift_stream`` is fully self-contained: each connection creates its own
per-request ``RollingDriftMonitor`` instances and drives its own
``replay_channel`` iteration independently of any telemetry stream clients.
Multiple concurrent drift clients each maintain isolated monitor state so
tick counts are never doubled by a second connection.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

import structlog.contextvars
from starlette.requests import Request

from spacecraft_telemetry.api.drift import DriftSnapshot, RollingDriftMonitor
from spacecraft_telemetry.api.models import DriftEvent, DriftFeature
from spacecraft_telemetry.api.replay import replay_channel
from spacecraft_telemetry.api.state import AppState

# EventBroadcaster imported lazily in subscriber_stream to avoid circular
# imports; AppState.broadcaster is typed Any at runtime via TYPE_CHECKING.
from spacecraft_telemetry.core.logging import get_logger

_log = get_logger("api.streaming")

# Minimum per-channel queue slots.  Prevents the floor from hitting zero when
# stream_buffer_max_events / N rounds down to very small values.
_MIN_PER_CHANNEL_SLOTS = 8

# How often to attach subsystem-level aggregation to a drift event.
_SUBSYSTEM_SUMMARY_EVERY_N_EVENTS = 10


async def _merge_queues(
    request: Request,
    tasks: dict[str, asyncio.Task[None]],
    queues: dict[str, asyncio.Queue[bytes]],
    channels: list[str],
    log_event: str,
) -> AsyncGenerator[bytes, None]:
    """Race per-channel queues, yield payloads as they arrive, clean up on exit.

    Shared merge loop used by both ``telemetry_stream`` and ``drift_stream``.

    Long-lived getter design: one persistent Task per channel queue, re-armed
    only when its result is consumed.  This avoids the per-iteration
    create/cancel churn of the previous batch-getter approach.

    A disconnect-watcher task receives ASGI messages and sets an
    ``asyncio.Event`` on ``http.disconnect``, so exit is push-based rather
    than polled.  The merge loop races all getter tasks against a single
    ``disconnect.wait()`` task so disconnect latency is bounded by
    ``asyncio.wait`` timeout (1 s), not by the next queue item.
    """
    disconnect = asyncio.Event()

    async def _watch_disconnect() -> None:
        while True:
            msg = await request.receive()
            if msg["type"] == "http.disconnect":
                disconnect.set()
                return

    watcher = asyncio.create_task(_watch_disconnect())
    # One persistent getter per channel; re-armed after each consumed result.
    getters: dict[str, asyncio.Task[bytes]] = {
        ch: asyncio.create_task(queues[ch].get()) for ch in channels
    }
    disc_task: asyncio.Task[bool] = asyncio.create_task(disconnect.wait())

    try:
        while getters:
            if disconnect.is_set():
                break

            # Channels whose pump finished and queue drained will never produce
            # another item — their getter would block forever.  Remove them now.
            # Guard: skip getters that are already done (they hold a result not
            # yet yielded — cancelling them would lose that last event).
            for ch in [
                ch for ch in list(getters)
                if tasks[ch].done() and queues[ch].empty() and not getters[ch].done()
            ]:
                getters[ch].cancel()
                del getters[ch]
            if not getters:
                break

            done, _ = await asyncio.wait(
                [*list(getters.values()), disc_task],
                return_when=asyncio.FIRST_COMPLETED,
                timeout=1.0,
            )

            if disc_task in done:
                break

            for ch, getter in list(getters.items()):
                if getter not in done:
                    continue
                yield getter.result()
                # Re-arm if pump is still running or queue still has items.
                if not tasks[ch].done() or not queues[ch].empty():
                    getters[ch] = asyncio.create_task(queues[ch].get())
                else:
                    del getters[ch]

    finally:
        for g in getters.values():
            g.cancel()
        disc_task.cancel()
        watcher.cancel()
        await asyncio.gather(*getters.values(), return_exceptions=True)
        await asyncio.gather(disc_task, watcher, return_exceptions=True)
        for pump_task in tasks.values():
            pump_task.cancel()
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        for ch, result in zip(channels, results, strict=False):
            if isinstance(result, Exception) and not isinstance(
                result, asyncio.CancelledError
            ):
                _log.error(log_event, channel=ch, error=str(result))


async def subscriber_stream(
    state: AppState,
    request: Request,
    selected_channels: list[str],
) -> AsyncGenerator[bytes, None]:
    """Attach to the shared replay loop and yield events for ``selected_channels``.

    Each SSE connection subscribes to the shared ``EventBroadcaster`` (started
    once at container startup) rather than spawning its own engine pumps.
    On connect, the broadcaster's per-channel backlog is drained first so the
    chart appears mid-flight instead of starting from an empty canvas.
    Subsequent live ticks follow without gap or duplication.
    On disconnect the subscription is cleaned up automatically.

    Falls through to ``telemetry_stream`` when no broadcaster is available
    (test fixtures that construct AppState directly without lifespan).
    """
    broadcaster = state.broadcaster
    assert broadcaster is not None  # caller checks this

    client_id, q, backlog = await broadcaster.subscribe_with_backlog(
        frozenset(selected_channels)
    )

    disconnect = asyncio.Event()

    async def _watch_disconnect() -> None:
        while True:
            msg = await request.receive()
            if msg["type"] == "http.disconnect":
                disconnect.set()
                return

    watcher = asyncio.create_task(_watch_disconnect())
    try:
        # Drain the backlog first so the chart fills immediately on connect.
        for payload in backlog:
            if disconnect.is_set():
                return
            yield payload

        # Then follow the live broadcast.
        while not disconnect.is_set():
            try:
                payload = await asyncio.wait_for(q.get(), timeout=1.0)
                yield payload
            except TimeoutError:
                pass  # re-check disconnect flag
    finally:
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)
        await broadcaster.unsubscribe(client_id)


async def telemetry_stream(
    state: AppState,
    request: Request,
    speed: float,
    selected_channels: list[str],
) -> AsyncGenerator[bytes, None]:
    """Async generator yielding SSE-formatted bytes for the telemetry stream.

    Spawns one asyncio task per channel, each driving ``replay_channel`` and
    pushing encoded SSE payloads into that channel's dedicated bounded queue.
    A merging consumer races all queues and yields the first available payload
    per iteration, preserving per-channel fairness.

    SSE format per event::

        event: telemetry\\n
        data: <TelemetryEvent JSON>\\n
        \\n

    Args:
        state:             Runtime application state (engines + settings).
        request:           Starlette request — used for disconnect detection.
        speed:             Replay speed multiplier (> 0).
        selected_channels: Ordered list of channel IDs to stream.
    """
    n = max(1, len(selected_channels))
    per_ch_maxsize = max(
        _MIN_PER_CHANNEL_SLOTS,
        state.settings.api.stream_buffer_max_events // n,
    )

    queues: dict[str, asyncio.Queue[bytes]] = {
        ch: asyncio.Queue(maxsize=per_ch_maxsize) for ch in selected_channels
    }

    async def pump(channel: str) -> None:
        """Replay one channel and push SSE payloads into its queue."""
        structlog.contextvars.bind_contextvars(channel_id=channel)
        engine = state.engines[channel]
        # Reset rolling state so each stream connection starts cold. Without
        # this, state from a previous stream leaks in and anomaly timing is
        # non-deterministic across page refreshes.
        engine.reset()
        async for ts, val, anom_true in replay_channel(
            state.settings.preprocess.processed_data_dir,
            state.mission,
            channel,
            speed=speed,
            tick_interval_seconds=state.settings.api.replay_tick_interval_seconds,
            cached_data=state.replay_data.get(channel),
            warmup_rows=state.settings.api.replay_warmup_rows,
            max_rows=state.settings.api.replay_max_rows,
        ):
            event = engine.step(val, ts, anom_true)
            payload = (
                f"event: telemetry\ndata: {event.model_dump_json()}\n\n".encode()
            )
            await queues[channel].put(payload)

    tasks: dict[str, asyncio.Task[None]] = {
        ch: asyncio.create_task(pump(ch)) for ch in selected_channels
    }

    async for chunk in _merge_queues(request, tasks, queues, selected_channels, "api.pump.error"):
        yield chunk


async def drift_stream(
    state: AppState,
    request: Request,
    selected_channels: list[str],
    speed: float,
) -> AsyncGenerator[bytes, None]:
    """Async generator yielding SSE-formatted bytes for the drift stream.

    Fully self-contained: each connection creates per-request
    ``RollingDriftMonitor`` instances and drives its own ``replay_channel``
    loop.  Multiple concurrent clients maintain independent monitor state so
    tick counts are never doubled and the stream works without a telemetry
    stream client being open.

    When the monitor fires (every ``tick_interval`` ticks after the window is
    full), the ``DriftSnapshot`` is converted to a ``DriftEvent`` SSE frame.
    Every ``_SUBSYSTEM_SUMMARY_EVERY_N_EVENTS`` events the subsystem-level
    aggregation fields are attached.

    SSE format per event::

        event: drift\\n
        data: <DriftEvent JSON>\\n
        \\n

    Args:
        state:             Runtime application state (drift_references + settings).
        request:           Starlette request — used for disconnect detection.
        selected_channels: Ordered list of channel IDs to monitor.
        speed:             Replay speed multiplier passed to ``replay_channel``.
    """
    n = max(1, len(selected_channels))
    per_ch_maxsize = max(
        _MIN_PER_CHANNEL_SLOTS,
        state.settings.api.stream_buffer_max_events // n,
    )
    queues: dict[str, asyncio.Queue[bytes]] = {
        ch: asyncio.Queue(maxsize=per_ch_maxsize) for ch in selected_channels
    }

    # Shared dict — each pump task writes its latest snapshot so the
    # subsystem aggregation can read across channels.
    latest_snapshots: dict[str, object] = {}

    async def pump(channel: str) -> None:
        """Replay one channel, push ticks into a fresh monitor, emit drift events."""
        structlog.contextvars.bind_contextvars(channel_id=channel)
        monitor = RollingDriftMonitor(
            channel=channel,
            reference=state.drift_references[channel],
            window_size=state.settings.drift.window_size,
            tick_interval=state.settings.drift.tick_interval,
            feature_drift_threshold=state.settings.drift.feature_drift_threshold,
            channel_drift_threshold=state.settings.drift.drift_alert_threshold,
        )
        drift_event_count = 0

        async for _ts, val, _anom in replay_channel(
            state.settings.preprocess.processed_data_dir,
            state.mission,
            channel,
            speed=speed,
            tick_interval_seconds=state.settings.api.replay_tick_interval_seconds,
            cached_data=state.replay_data.get(channel),
            warmup_rows=state.settings.api.replay_warmup_rows,
            max_rows=state.settings.api.replay_max_rows,
        ):
            monitor.push({"value_normalized": float(val)})

            if not monitor.should_run():
                continue

            snapshot = await monitor.run()
            if snapshot is None:
                continue

            latest_snapshots[channel] = snapshot
            drift_event_count += 1

            sub_pct: float | None = None
            sub_alert: bool | None = None
            if drift_event_count % _SUBSYSTEM_SUMMARY_EVERY_N_EVENTS == 0:
                typed = {
                    ch: s
                    for ch, s in latest_snapshots.items()
                    if isinstance(s, DriftSnapshot)
                }
                if typed:
                    n_drifted = sum(1 for s in typed.values() if s.drifted)
                    sub_pct = n_drifted / len(typed)
                    sub_alert = sub_pct >= state.settings.drift.drift_alert_threshold

            event = DriftEvent(
                timestamp=snapshot.timestamp,
                mission=state.mission,
                channel=snapshot.channel,
                features=[
                    DriftFeature(
                        feature=f.feature,
                        score=f.score,
                        drifted=f.drifted,
                    )
                    for f in snapshot.features
                ],
                percent_drifted=snapshot.percent_drifted,
                drifted=snapshot.drifted,
                subsystem_percent_drifted=sub_pct,
                subsystem_alert=sub_alert,
            )
            payload = f"event: drift\ndata: {event.model_dump_json()}\n\n".encode()
            await queues[channel].put(payload)

    tasks: dict[str, asyncio.Task[None]] = {
        ch: asyncio.create_task(pump(ch)) for ch in selected_channels
    }

    async for chunk in _merge_queues(
        request, tasks, queues, selected_channels, "api.drift_pump.error"
    ):
        yield chunk
