"""Shared drift-feed helper — feeds RollingDriftMonitor from either producer.

Both the ESA replay loop (``broadcast.run_shared_loop``) and the ISS live
pump (``api.live.pump.LivePump``) call ``step_drift`` once per closed tick so
a single set of monitors backs the drift SSE stream regardless of which
producer is active — mirrors the shared-engine pattern already used for
telemetry (O(1) cost, consistent across viewers, injection and LOS-fallback
replay come for free since both flow through the same tick).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from spacecraft_telemetry.api.drift import DriftSnapshot
from spacecraft_telemetry.api.models import DriftEvent, DriftFeature

if TYPE_CHECKING:
    from spacecraft_telemetry.api.state import AppState

# How often (in fired drift runs, per channel) to attach subsystem-level
# aggregation to a drift event.
_SUBSYSTEM_SUMMARY_EVERY_N_EVENTS = 10


async def step_drift(state: AppState, channel: str, value_normalized: float) -> None:
    """Feed one tick to ``channel``'s shared drift monitor; publish on fire.

    No-op when drift monitoring is disabled or ``channel`` has no monitor
    (e.g. unpromoted channel, or no reference profile was loaded for it).
    Also a no-op without a broadcaster to publish to (test fixtures that
    construct AppState directly).
    """
    monitor = state.drift_monitors.get(channel)
    if monitor is None:
        return

    monitor.push({"value_normalized": value_normalized})
    if not monitor.should_run():
        return

    snapshot = await monitor.run()
    if snapshot is None:
        return

    monitor.latest = snapshot
    monitor.event_count += 1

    sub_pct: float | None = None
    sub_alert: bool | None = None
    if monitor.event_count % _SUBSYSTEM_SUMMARY_EVERY_N_EVENTS == 0:
        typed = {
            ch: m.latest
            for ch, m in state.drift_monitors.items()
            if isinstance(m.latest, DriftSnapshot)
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
            DriftFeature(feature=f.feature, score=f.score, drifted=f.drifted)
            for f in snapshot.features
        ],
        percent_drifted=snapshot.percent_drifted,
        drifted=snapshot.drifted,
        subsystem_percent_drifted=sub_pct,
        subsystem_alert=sub_alert,
    )
    payload = f"event: drift\ndata: {event.model_dump_json()}\n\n".encode()
    if state.broadcaster is not None:
        state.broadcaster.publish(channel, payload)
