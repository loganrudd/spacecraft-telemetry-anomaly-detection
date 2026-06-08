"""Shared telemetry replay loop and SSE fan-out broadcaster.

Architecture
------------
A single ``run_shared_loop`` task runs continuously from container startup,
advancing all channel engines one tick at a time on a shared clock and
publishing ``TelemetryEvent`` payloads to every connected SSE subscriber.

This replaces the previous per-connection design (one pump task per channel
per client) with:

  - One engine set, reset once per loop pass (not per page load).
  - O(1) CPU regardless of viewer count — N viewers share one set of forward
    passes instead of multiplying them.
  - No per-refresh warmup: subscribers attach to the in-progress stream.

``EventBroadcaster`` manages subscriber queues.  Each SSE connection
subscribes with a channel filter and an ``asyncio.Queue``; events are
``put_nowait`` (drops on full rather than blocking the loop).

The loop restarts automatically when the replay slice is exhausted, resetting
all engines so the next pass starts cold.
"""

from __future__ import annotations

import asyncio
import uuid
from collections import deque
from contextlib import suppress
from typing import TYPE_CHECKING, Any

import pandas as pd

from spacecraft_telemetry.core.logging import get_logger

if TYPE_CHECKING:
    from spacecraft_telemetry.api.state import AppState

_log = get_logger("api.broadcast")

# Per-subscriber queue capacity.  Slow clients drop events rather than
# back-pressuring the shared loop.
_SUBSCRIBER_QUEUE_SIZE = 256

# Rolling backlog depth per channel.  New subscribers receive this many
# recent events before live ticks so the chart appears mid-flight on
# page refresh instead of starting from an empty canvas.  Matches the
# chart's CHART_WINDOW constant (200) so a fresh connection sees exactly
# what was visible in the running chart.
_BACKLOG_SIZE = 200


class EventBroadcaster:
    """Fan-out hub: one payload → N subscriber queues.

    Thread-safe via asyncio.Lock (all callers run on the same event loop).
    """

    def __init__(self) -> None:
        self._subscribers: dict[str, tuple[frozenset[str], asyncio.Queue[bytes]]] = {}
        self._lock = asyncio.Lock()
        # Rolling per-channel backlog for new-subscriber catch-up.
        self._backlogs: dict[str, deque[bytes]] = {}

    async def subscribe(
        self,
        channels: frozenset[str],
    ) -> tuple[str, asyncio.Queue[bytes]]:
        """Register a new subscriber; return (client_id, queue).

        ``channels`` is the set of channel IDs this client wants. An empty
        frozenset means "all channels".
        """
        client_id = str(uuid.uuid4())
        q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=_SUBSCRIBER_QUEUE_SIZE)
        async with self._lock:
            self._subscribers[client_id] = (channels, q)
        return client_id, q

    async def subscribe_with_backlog(
        self,
        channels: frozenset[str],
    ) -> tuple[str, asyncio.Queue[bytes], list[bytes]]:
        """Register a subscriber and atomically snapshot the current backlog.

        Returns ``(client_id, queue, backlog_events)`` where ``backlog_events``
        is a flat list of pre-serialised SSE payloads covering the last
        ``_BACKLOG_SIZE`` ticks for each requested channel.

        Atomicity guarantee: subscriber registration and backlog snapshot both
        happen inside the same ``async with self._lock`` block with no
        ``await`` in between.  Because asyncio is single-threaded and
        ``publish()`` is synchronous, no new events can be inserted into the
        backlog or fan-out to this subscriber's queue between the two
        operations — so callers that drain backlog first, then drain the queue,
        see a gapless, duplicate-free event stream.
        """
        client_id = str(uuid.uuid4())
        q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=_SUBSCRIBER_QUEUE_SIZE)
        async with self._lock:
            self._subscribers[client_id] = (channels, q)
            backlog: list[bytes] = []
            for ch, b_deque in self._backlogs.items():
                if not channels or ch in channels:
                    backlog.extend(b_deque)
        return client_id, q, backlog

    async def unsubscribe(self, client_id: str) -> None:
        """Remove a subscriber.  No-op if already gone."""
        async with self._lock:
            self._subscribers.pop(client_id, None)

    def publish(self, channel: str, payload: bytes) -> None:
        """Push ``payload`` to every subscriber that wants ``channel``.

        Called synchronously from the loop; uses ``put_nowait`` so a slow
        subscriber never blocks the shared clock.  Also appends to the
        per-channel backlog so new subscribers can catch up on connect.
        """
        backlog = self._backlogs.get(channel)
        if backlog is None:
            backlog = deque(maxlen=_BACKLOG_SIZE)
            self._backlogs[channel] = backlog
        backlog.append(payload)

        for _cid, (channels, q) in list(self._subscribers.items()):
            if not channels or channel in channels:
                with suppress(asyncio.QueueFull):  # slow subscriber — drop event
                    q.put_nowait(payload)

    def clear_backlogs(self) -> None:
        """Discard all per-channel backlog history.

        Called at the start of each replay pass so a new subscriber never
        receives a mix of tail events from the previous pass and head events
        from the new one.
        """
        self._backlogs.clear()

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)


async def run_shared_loop(state: AppState) -> None:
    """Continuously replay all channels and broadcast events to subscribers.

    Runs forever as an asyncio background task:
      - Resets all channel engines at the start of each pass.
      - Steps every channel engine once per tick, advancing the shared clock.
      - Sleeps ``tick_interval / speed`` between ticks.
      - Loops immediately when the slice is exhausted.

    Logged at INFO on each loop pass so Cloud Logging captures restarts.
    The task is cancelled on lifespan shutdown (FastAPI's asyncio cleanup).
    """
    broadcaster = state.broadcaster
    if broadcaster is None:
        _log.error("broadcast.loop.no_broadcaster")
        return

    channels = state.channels_loaded
    settings = state.settings
    delay = settings.api.replay_tick_interval_seconds / settings.api.replay_speed_default

    # Pre-extract replay arrays once (immutable after startup).
    replay: dict[str, Any] = {
        ch: state.replay_data[ch]
        for ch in channels
        if ch in state.replay_data
    }
    if not replay:
        _log.warning("broadcast.loop.no_replay_data", channels=channels)
        return

    n_ticks = min(len(data[0]) for data in replay.values())
    pass_count = 0

    _log.info(
        "broadcast.loop.start",
        channels=len(channels),
        ticks_per_pass=n_ticks,
        speed=settings.api.replay_speed_default,
        delay_ms=round(delay * 1000, 1),
    )

    while True:
        pass_count += 1
        _log.info("broadcast.loop.pass_start", pass_count=pass_count)

        # Reset all engines and clear backlogs so each pass starts from a
        # clean state.  Clearing the backlogs prevents a new subscriber from
        # receiving a mix of tail events from the previous pass and head events
        # from the new one, which would give the chart a discontinuous jump.
        broadcaster.clear_backlogs()
        for ch in channels:
            if ch in state.engines:
                state.engines[ch].reset()

        for i in range(n_ticks):
            for ch in channels:
                if ch not in replay or ch not in state.engines:
                    continue
                values, anom, timestamps = replay[ch]
                ts = pd.Timestamp(timestamps[i]).to_pydatetime()
                event = state.engines[ch].step(
                    float(values[i]), ts, bool(anom[i])
                )
                payload = (
                    f"event: telemetry\ndata: {event.model_dump_json()}\n\n"
                    .encode()
                )
                broadcaster.publish(ch, payload)
            await asyncio.sleep(delay)

        _log.info(
            "broadcast.loop.pass_end",
            pass_count=pass_count,
            subscribers=broadcaster.subscriber_count,
        )
