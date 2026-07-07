"""Tests for api.drift_feed.step_drift — the shared producer-side drift feed."""

from __future__ import annotations

import json
import time
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest
import torch

from spacecraft_telemetry.api.broadcast import EventBroadcaster
from spacecraft_telemetry.api.drift import RollingDriftMonitor
from spacecraft_telemetry.api.drift_feed import step_drift
from spacecraft_telemetry.api.state import AppState
from spacecraft_telemetry.core.config import load_settings

_MISSION = "test-mission"


def _make_reference(n_rows: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    v = pd.Series(rng.standard_normal(n_rows).astype(float))
    return pd.DataFrame({"value_normalized": v, "rate_of_change": v.diff().fillna(0.0)})


def _make_monitor(
    channel: str, window_size: int = 4, tick_interval: int = 1
) -> RollingDriftMonitor:
    return RollingDriftMonitor(
        channel=channel,
        reference=_make_reference(),
        window_size=window_size,
        tick_interval=tick_interval,
        feature_drift_threshold=0.10,
        channel_drift_threshold=0.30,
    )


def _make_state(
    drift_monitors: dict[str, RollingDriftMonitor],
    broadcaster: EventBroadcaster | None = None,
) -> AppState:
    settings = load_settings("test")
    return AppState(
        settings=settings,
        mission=_MISSION,
        subsystems=None,
        device=torch.device("cpu"),
        engines=MappingProxyType({}),
        channel_subsystem_map=MappingProxyType({}),
        replay_data=MappingProxyType({}),
        startup_monotonic_ns=time.monotonic_ns(),
        mlflow_tracking_uri=settings.mlflow.tracking_uri,
        drift_monitors=MappingProxyType(drift_monitors),
        broadcaster=broadcaster,
    )


class TestStepDrift:
    @pytest.mark.asyncio
    async def test_no_monitor_is_noop(self) -> None:
        state = _make_state({})
        await step_drift(state, "missing-channel", 0.5)  # must not raise

    @pytest.mark.asyncio
    async def test_no_publish_before_should_run(self) -> None:
        monitor = _make_monitor("ch1", window_size=4, tick_interval=1)
        broadcaster = EventBroadcaster()
        state = _make_state({"ch1": monitor}, broadcaster=broadcaster)

        await step_drift(state, "ch1", 0.1)
        await step_drift(state, "ch1", 0.2)

        assert monitor.latest is None
        assert "ch1" not in broadcaster._backlogs

    @pytest.mark.asyncio
    async def test_publishes_event_on_fire(self) -> None:
        monitor = _make_monitor("ch1", window_size=4, tick_interval=1)
        broadcaster = EventBroadcaster()
        state = _make_state({"ch1": monitor}, broadcaster=broadcaster)

        for v in (0.1, 0.2, 0.3, 0.4):
            await step_drift(state, "ch1", v)

        assert monitor.latest is not None
        assert monitor.event_count == 1
        backlog = broadcaster._backlogs.get("ch1")
        assert backlog is not None and len(backlog) == 1
        assert backlog[0].startswith(b"event: drift")
        payload = json.loads(backlog[0].decode().split("data:", 1)[1].strip())
        assert payload["channel"] == "ch1"
        assert payload["mission"] == _MISSION

    @pytest.mark.asyncio
    async def test_noop_broadcaster_still_updates_monitor(self) -> None:
        """No broadcaster to publish to must not prevent monitor bookkeeping."""
        monitor = _make_monitor("ch1", window_size=4, tick_interval=1)
        state = _make_state({"ch1": monitor}, broadcaster=None)

        for v in (0.1, 0.2, 0.3, 0.4):
            await step_drift(state, "ch1", v)

        assert monitor.latest is not None
        assert monitor.event_count == 1

    @pytest.mark.asyncio
    async def test_subsystem_summary_every_n_events(self) -> None:
        monitors = {
            ch: _make_monitor(ch, window_size=1, tick_interval=1) for ch in ("ch0", "ch1")
        }
        broadcaster = EventBroadcaster()
        state = _make_state(monitors, broadcaster=broadcaster)

        events: list[dict[str, object]] = []
        for _tick in range(10):
            for ch in monitors:
                await step_drift(state, ch, 0.1)
                backlog = broadcaster._backlogs.get(ch)
                assert backlog is not None
                events.append(json.loads(backlog[-1].decode().split("data:", 1)[1].strip()))

        assert len(events) == 20
        # Every fire before the 10th event on a channel omits subsystem fields.
        assert events[0]["subsystem_percent_drifted"] is None
        # The 10th event on the last channel attaches the aggregation.
        assert events[-1]["subsystem_percent_drifted"] is not None
        assert isinstance(events[-1]["subsystem_alert"], bool)
