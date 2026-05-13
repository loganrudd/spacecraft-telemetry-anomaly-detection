"""SSE telemetry stream composer.

``telemetry_stream`` is an async generator that:

1. Spawns one ``asyncio.Task`` per selected channel (the *pump* coroutine).
2. Each pump iterates ``replay_channel`` and pushes SSE-encoded payloads into
   a shared ``asyncio.Queue``.
3. The generator drains the queue and yields the payloads to the
   ``StreamingResponse`` iterator.
4. Client disconnect is detected via ``request.is_disconnected()`` before each
   ``queue.get()`` — if True the generator exits early.
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


async def telemetry_stream(
    state: AppState,
    request: Request,
    speed: float,
    selected_channels: list[str],
) -> AsyncGenerator[bytes, None]:
    """Async generator yielding SSE-formatted bytes for the telemetry stream.

    Spawns one asyncio task per channel, each driving ``replay_channel`` and
    pushing encoded SSE payloads into a shared bounded queue.  Exits when the
    client disconnects or when all channel tasks have finished and the queue is
    fully drained.

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
    queue: asyncio.Queue[bytes] = asyncio.Queue(
        maxsize=state.settings.api.stream_buffer_max_events
    )
    tasks: list[asyncio.Task[None]] = []

    async def pump(channel: str) -> None:
        """Replay one channel and push SSE payloads into the shared queue."""
        structlog.contextvars.bind_contextvars(channel_id=channel)
        engine = state.engines[channel]
        async for ts, val, anom_true in replay_channel(
            state.settings.spark.processed_data_dir,
            state.mission,
            channel,
            speed=speed,
            tick_interval_seconds=state.settings.api.replay_tick_interval_seconds,
        ):
            event = engine.step(val, ts, anom_true)
            payload = (
                f"event: telemetry\ndata: {event.model_dump_json()}\n\n".encode()
            )
            await queue.put(payload)

    for ch in selected_channels:
        tasks.append(asyncio.create_task(pump(ch)))

    try:
        while True:
            if await request.is_disconnected():
                break
            try:
                payload = await asyncio.wait_for(queue.get(), timeout=1.0)
            except TimeoutError:
                # Timeout while waiting — check if all work is done.
                if all(t.done() for t in tasks) and queue.empty():
                    break
                continue
            yield payload
            # After yielding: check if all tasks are done and nothing remains.
            if all(t.done() for t in tasks) and queue.empty():
                break
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
