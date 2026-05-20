"""SSE telemetry stream composer.

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
4. Client disconnect is detected via ``request.is_disconnected()`` at the
   top of each merge iteration.
5. On any exit path (disconnect, exhaustion, exception) the pump tasks are
   cancelled and awaited so no coroutines are leaked.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

import structlog.contextvars
from starlette.requests import Request

from spacecraft_telemetry.api.replay import replay_channel
from spacecraft_telemetry.api.state import AppState
from spacecraft_telemetry.core.logging import get_logger

_log = get_logger("api.streaming")

# Minimum per-channel queue slots.  Prevents the floor from hitting zero when
# stream_buffer_max_events / N rounds down to very small values.
_MIN_PER_CHANNEL_SLOTS = 8


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
            for t in pending:
                t.cancel()

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
