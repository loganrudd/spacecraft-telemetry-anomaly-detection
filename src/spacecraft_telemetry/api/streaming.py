"""SSE telemetry and drift stream composers.

``telemetry_stream`` is an async generator that:

1. Allocates one bounded ``asyncio.Queue`` per selected channel (the *pump*
   queue).  A single shared queue would head-of-line-block every pump when
   any one channel's consumer falls behind; per-channel queues limit
   backpressure to the offending channel only.
2. Spawns one ``asyncio.Task`` per channel.  Each pump iterates
   ``replay_channel`` and pushes SSE-encoded payloads into its channel's
   queue.  After each ``engine.step()`` it also appends a minimal tick record
   to ``state.tick_buses[channel]`` (if present) for the drift stream.
3. A merging consumer races all active queues with
   ``asyncio.wait(FIRST_COMPLETED)`` and yields whichever payload arrives
   first.  This keeps latency fair across channels regardless of their
   individual tick rates.
4. Client disconnect is detected via ``request.is_disconnected()`` at the
   top of each merge iteration.
5. On any exit path (disconnect, exhaustion, exception) the pump tasks are
   cancelled and awaited so no coroutines are leaked.

``drift_stream`` reads from the per-channel tick buses populated by
``telemetry_stream`` pumps, drives ``RollingDriftMonitor`` instances, and
emits ``drift`` SSE events at the configured cadence (every N ticks).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

import structlog.contextvars
from starlette.requests import Request

from spacecraft_telemetry.api.models import DriftEvent, DriftFeature
from spacecraft_telemetry.api.replay import replay_channel
from spacecraft_telemetry.api.state import AppState
from spacecraft_telemetry.core.logging import get_logger

_log = get_logger("api.streaming")

# Minimum per-channel queue slots.  Prevents the floor from hitting zero when
# stream_buffer_max_events / N rounds down to very small values.
_MIN_PER_CHANNEL_SLOTS = 8

# How often to attach subsystem-level aggregation to a drift event.
_SUBSYSTEM_SUMMARY_EVERY_N_EVENTS = 10

# Poll interval for the drift pump to drain the tick bus.
_DRIFT_POLL_SECONDS = 0.05


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
        tick_bus = state.tick_buses.get(channel)
        async for ts, val, anom_true in replay_channel(
            state.settings.spark.processed_data_dir,
            state.mission,
            channel,
            speed=speed,
            tick_interval_seconds=state.settings.api.replay_tick_interval_seconds,
            cached_data=state.replay_data.get(channel),
        ):
            event = engine.step(val, ts, anom_true)
            payload = (
                f"event: telemetry\ndata: {event.model_dump_json()}\n\n".encode()
            )
            await queues[channel].put(payload)
            # Publish a minimal tick to the drift bus — non-blocking append,
            # deque maxlen silently evicts oldest if drift consumer falls behind.
            if tick_bus is not None:
                tick_bus.append({
                    "value_normalized": float(event.value_normalized),
                })

    tasks: dict[str, asyncio.Task[None]] = {
        ch: asyncio.create_task(pump(ch)) for ch in selected_channels
    }

    try:
        while True:
            if await request.is_disconnected():
                break

            # Channels still active: pump is running OR its queue has buffered events.
            active: dict[str, asyncio.Queue[bytes]] = {
                ch: q
                for ch, q in queues.items()
                if not tasks[ch].done() or not q.empty()
            }
            if not active:
                break  # All pumps finished and all queues drained.

            # Race every active queue — yield the first payload that arrives.
            # The payload is self-describing (contains channel in the JSON), so
            # we don't need a task→channel mapping here.
            getter_tasks: list[asyncio.Task[bytes]] = [
                asyncio.create_task(q.get()) for q in active.values()
            ]
            done, pending = await asyncio.wait(
                getter_tasks,
                return_when=asyncio.FIRST_COMPLETED,
                timeout=1.0,
            )

            # Cancel the getters that lost the race (they held no event).
            # Await them so CancelledError is processed before the tasks go out
            # of scope — prevents "Task was destroyed but it is pending!" warnings.
            for t in pending:
                t.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

            if not done:
                # 1-second timeout with no events — recheck active queues.
                continue

            for t in done:
                yield t.result()

    finally:
        for pump_task in tasks.values():
            pump_task.cancel()
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        for ch, result in zip(selected_channels, results, strict=False):
            if isinstance(result, Exception) and not isinstance(
                result, asyncio.CancelledError
            ):
                _log.error("api.pump.error", channel=ch, error=str(result))


async def drift_stream(
    state: AppState,
    request: Request,
    selected_channels: list[str],
) -> AsyncGenerator[bytes, None]:
    """Async generator yielding SSE-formatted bytes for the drift stream.

    Reads from ``state.tick_buses`` populated by ``telemetry_stream`` pumps.
    Drives one ``RollingDriftMonitor`` per selected channel.  When the monitor
    fires (every ``tick_interval`` ticks after the window is full), the resulting
    ``DriftSnapshot`` is converted to a ``DriftEvent`` and emitted as an SSE
    frame.  Every ``_SUBSYSTEM_SUMMARY_EVERY_N_EVENTS`` events the subsystem-level
    ``percent_drifted`` and alert flag are attached to the outgoing event.

    SSE format per event::

        event: drift\\n
        data: <DriftEvent JSON>\\n
        \\n

    The drift pump polls the tick bus with a short sleep rather than blocking on
    a queue, so the telemetry pump is never stalled waiting for drift consumers.

    Args:
        state:             Runtime application state (drift_monitors + tick_buses).
        request:           Starlette request — used for disconnect detection.
        selected_channels: Ordered list of channel IDs to monitor.
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
        """Drain tick bus → monitor → emit drift SSE frames."""
        from spacecraft_telemetry.api.drift import DriftSnapshot  # local to avoid circular

        structlog.contextvars.bind_contextvars(channel_id=channel)
        monitor = state.drift_monitors[channel]
        tick_bus = state.tick_buses[channel]
        drift_event_count = 0

        while True:
            # Drain any new ticks into the monitor (O(1) per tick, cooperative).
            while tick_bus:
                monitor.push(tick_bus.popleft())

            if monitor.should_run():
                snapshot = await monitor.run()
                if snapshot is not None:
                    latest_snapshots[channel] = snapshot
                    drift_event_count += 1

                    sub_pct: float | None = None
                    sub_alert = False
                    if drift_event_count % _SUBSYSTEM_SUMMARY_EVERY_N_EVENTS == 0:
                        typed = {
                            ch: s
                            for ch, s in latest_snapshots.items()
                            if isinstance(s, DriftSnapshot)
                        }
                        if typed:
                            n_drifted = sum(1 for s in typed.values() if s.drifted)
                            sub_pct = n_drifted / len(typed)
                            sub_alert = sub_pct >= state.settings.drift.subsystem_alert_threshold

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

            await asyncio.sleep(_DRIFT_POLL_SECONDS)

    tasks: dict[str, asyncio.Task[None]] = {
        ch: asyncio.create_task(pump(ch)) for ch in selected_channels
    }

    try:
        while True:
            if await request.is_disconnected():
                break

            active: dict[str, asyncio.Queue[bytes]] = {
                ch: q
                for ch, q in queues.items()
                if not tasks[ch].done() or not q.empty()
            }
            if not active:
                break

            getter_tasks: list[asyncio.Task[bytes]] = [
                asyncio.create_task(q.get()) for q in active.values()
            ]
            done, pending = await asyncio.wait(
                getter_tasks,
                return_when=asyncio.FIRST_COMPLETED,
                timeout=1.0,
            )
            for t in pending:
                t.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            if not done:
                continue
            for t in done:
                yield t.result()

    finally:
        for pump_task in tasks.values():
            pump_task.cancel()
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        for ch, result in zip(selected_channels, results, strict=False):
            if isinstance(result, Exception) and not isinstance(
                result, asyncio.CancelledError
            ):
                _log.error("api.drift_pump.error", channel=ch, error=str(result))
