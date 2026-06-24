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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd

from spacecraft_telemetry.core.logging import get_logger

if TYPE_CHECKING:
    from spacecraft_telemetry.api.state import AppState

_log = get_logger("api.broadcast")


@dataclass
class _ActiveInjection:
    """Transient fault injection applied tick-by-tick by the shared replay loop.

    Fields
    ------
    fault_type:       "spike" | "drift_inject" | "flatline"
    channels:         frozenset of channel IDs to inject; empty = all channels.
    magnitude_sigma:  Additive offset in z-score units (unused for flatline).
    total_ticks:      Number of ticks the fault lasts.
    elapsed:          Ticks elapsed so far (advanced by end_tick).
    _flatline_values: Per-channel value captured on first apply (sensor-death anchor).
    """

    fault_type: str
    channels: frozenset[str]
    magnitude_sigma: float
    total_ticks: int
    elapsed: int = 0
    _flatline_values: dict[str, float] = field(default_factory=dict)

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
        # Fault injection state — both set/read only from the asyncio event loop,
        # so no additional locking is required (asyncio is single-threaded).
        self._pending_injection: _ActiveInjection | None = None
        self._active_injection: _ActiveInjection | None = None

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

    # ------------------------------------------------------------------
    # Fault injection — called by the shared replay loop and the endpoint
    # ------------------------------------------------------------------

    def request_injection(
        self,
        fault_type: str,
        channels: frozenset[str],
        magnitude_sigma: float,
        total_ticks: int,
    ) -> None:
        """Queue a fault injection; replaces any already-pending request.

        Called by the POST /api/inject endpoint from the asyncio event loop.
        The loop activates it at the start of the next tick via begin_tick().
        """
        self._pending_injection = _ActiveInjection(
            fault_type=fault_type,
            channels=channels,
            magnitude_sigma=magnitude_sigma,
            total_ticks=total_ticks,
        )

    def begin_tick(self) -> None:
        """Activate any pending injection.  Call once at the start of each tick.

        Separating activation from advancement ensures a pending injection
        requested during the previous tick is applied starting from tick 0,
        not tick 1 (elapsed=0 when apply_fault is first called).
        """
        if self._pending_injection is not None:
            self._active_injection = self._pending_injection
            self._pending_injection = None

    def apply_fault(self, channel: str, value: float) -> tuple[float, bool]:
        """Apply the active injection (if any) to a single channel value.

        Returns (modified_value, is_injected).  Called once per channel per
        tick from run_shared_loop; the returned is_injected flag is ORed into
        the engine's is_anomaly argument so the detector sees the label.

        Fault math (all in z-score space, matching injection/faults.py):
          spike:       additive offset = magnitude_sigma for every tick.
          drift_inject: linear ramp 0 → magnitude_sigma over total_ticks.
          flatline:    hold the value captured on the first call per channel.
        """
        inj = self._active_injection
        if inj is None:
            return value, False
        if inj.channels and channel not in inj.channels:
            return value, False

        if inj.fault_type == "spike":
            return value + inj.magnitude_sigma, True

        if inj.fault_type == "drift_inject":
            progress = inj.elapsed / max(1, inj.total_ticks - 1)
            return value + inj.magnitude_sigma * progress, True

        # flatline: anchor to each channel's first-seen value independently
        if channel not in inj._flatline_values:
            inj._flatline_values[channel] = value
        return inj._flatline_values[channel], True

    def end_tick(self) -> None:
        """Advance injection elapsed counter.  Call once after all channels are processed.

        Clears the active injection when its duration expires so apply_fault
        returns clean values on the next tick.
        """
        if self._active_injection is not None:
            self._active_injection.elapsed += 1
            if self._active_injection.elapsed >= self._active_injection.total_ticks:
                self._active_injection = None


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
            broadcaster.begin_tick()
            for ch in channels:
                if ch not in replay or ch not in state.engines:
                    continue
                values, anom, timestamps = replay[ch]
                ts = pd.Timestamp(timestamps[i]).to_pydatetime()
                raw_value = float(values[i])
                injected_value, is_injected = broadcaster.apply_fault(ch, raw_value)
                event = state.engines[ch].step(
                    injected_value, ts, bool(anom[i]) or is_injected
                )
                payload = (
                    f"event: telemetry\ndata: {event.model_dump_json()}\n\n"
                    .encode()
                )
                broadcaster.publish(ch, payload)
            broadcaster.end_tick()
            await asyncio.sleep(delay)

        _log.info(
            "broadcast.loop.pass_end",
            pass_count=pass_count,
            subscribers=broadcaster.subscriber_count,
        )
