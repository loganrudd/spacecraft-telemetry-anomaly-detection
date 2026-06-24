"""Tests for POST /api/inject and EventBroadcaster injection state machine."""

from __future__ import annotations

import asyncio
import time
from contextlib import suppress
from datetime import datetime
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from spacecraft_telemetry.api.broadcast import EventBroadcaster, run_shared_loop
from spacecraft_telemetry.api.endpoints import router
from spacecraft_telemetry.api.logging_middleware import CorrelationIdMiddleware
from spacecraft_telemetry.api.models import TelemetryEvent
from spacecraft_telemetry.api.state import AppState
from spacecraft_telemetry.core.config import load_settings

_MISSION = "test-mission"
_CHANNEL = "ch1"
_CHANNEL_B = "ch2"
_SUBSYSTEM = "test-sub"


# ---------------------------------------------------------------------------
# Broadcaster unit tests — injection state machine
# ---------------------------------------------------------------------------


class TestEventBroadcasterInjection:
    def test_no_injection_returns_value_unchanged(self) -> None:
        b = EventBroadcaster()
        val, injected = b.apply_fault(_CHANNEL, 1.0)
        assert val == pytest.approx(1.0)
        assert injected is False

    def test_spike_adds_magnitude(self) -> None:
        b = EventBroadcaster()
        b.request_injection("spike", frozenset(), 2.0, 5)
        b.begin_tick()
        val, injected = b.apply_fault(_CHANNEL, 0.0)
        assert val == pytest.approx(2.0)
        assert injected is True

    def test_spike_channel_filter(self) -> None:
        b = EventBroadcaster()
        b.request_injection("spike", frozenset([_CHANNEL]), 2.0, 5)
        b.begin_tick()
        val_a, inj_a = b.apply_fault(_CHANNEL, 0.0)
        val_b, inj_b = b.apply_fault(_CHANNEL_B, 0.0)
        assert val_a == pytest.approx(2.0) and inj_a is True
        assert val_b == pytest.approx(0.0) and inj_b is False

    def test_drift_ramps_then_holds(self) -> None:
        b = EventBroadcaster()
        # total_ticks=10: ramp_len=5 (ticks 0-4), hold at max (ticks 5-9)
        b.request_injection("drift", frozenset(), 4.0, 10)
        results = []
        for _ in range(10):
            b.begin_tick()
            val, _ = b.apply_fault(_CHANNEL, 0.0)
            results.append(val)
            b.end_tick()
        # ramp: elapsed 0-4, progress = 0, 0.25, 0.5, 0.75, 1.0
        # hold: elapsed 5-9, all at 4.0
        assert results[0] == pytest.approx(0.0)
        assert results[2] == pytest.approx(2.0)
        assert results[4] == pytest.approx(4.0)
        assert results[-1] == pytest.approx(4.0)

    def test_flatline_holds_first_value_per_channel(self) -> None:
        b = EventBroadcaster()
        b.request_injection("flatline", frozenset(), 0.0, 3)
        b.begin_tick()
        val1, _ = b.apply_fault(_CHANNEL, 5.0)   # anchors at 5.0
        val2, _ = b.apply_fault(_CHANNEL, 9.0)   # same channel, still 5.0
        assert val1 == pytest.approx(5.0)
        assert val2 == pytest.approx(5.0)

    def test_end_tick_clears_after_duration(self) -> None:
        b = EventBroadcaster()
        b.request_injection("spike", frozenset(), 1.0, 2)
        for _ in range(2):
            b.begin_tick()
            b.apply_fault(_CHANNEL, 0.0)
            b.end_tick()
        # After 2 ticks the injection is done
        b.begin_tick()
        val, injected = b.apply_fault(_CHANNEL, 0.0)
        assert val == pytest.approx(0.0)
        assert injected is False

    def test_pending_activation_on_begin_tick(self) -> None:
        b = EventBroadcaster()
        b.request_injection("spike", frozenset(), 1.0, 1)
        # Before begin_tick, _active_injection is still None
        assert b._active_injection is None
        b.begin_tick()
        assert b._active_injection is not None

    def test_second_request_replaces_pending(self) -> None:
        b = EventBroadcaster()
        b.request_injection("spike", frozenset(), 1.0, 5)
        b.request_injection("flatline", frozenset(), 0.0, 3)
        b.begin_tick()
        assert b._active_injection is not None
        assert b._active_injection.fault_type == "flatline"


# ---------------------------------------------------------------------------
# Endpoint tests
# ---------------------------------------------------------------------------


def _make_app(with_broadcaster: bool = False) -> FastAPI:
    settings = load_settings("test")
    broadcaster = EventBroadcaster() if with_broadcaster else None
    app = FastAPI()
    app.state.settings = settings
    app.state.app_state = AppState(
        settings=settings,
        mission=_MISSION,
        subsystems=[_SUBSYSTEM],
        device=torch.device("cpu"),
        engines=MappingProxyType({_CHANNEL: object()}),  # type: ignore[arg-type]
        channel_subsystem_map=MappingProxyType({_CHANNEL: _SUBSYSTEM}),
        replay_data=MappingProxyType({}),
        startup_monotonic_ns=time.monotonic_ns(),
        mlflow_tracking_uri=settings.mlflow.tracking_uri,
        broadcaster=broadcaster,
    )
    app.add_middleware(CorrelationIdMiddleware)
    app.include_router(router)
    return app


class TestInjectEndpoint:
    def test_503_without_broadcaster(self) -> None:
        app = _make_app(with_broadcaster=False)
        resp = TestClient(app).post(
            "/api/inject",
            json={"fault_type": "spike"},
        )
        assert resp.status_code == 503

    def test_200_with_broadcaster(self) -> None:
        app = _make_app(with_broadcaster=True)
        resp = TestClient(app).post(
            "/api/inject",
            json={"fault_type": "spike", "magnitude_sigma": 3.0, "duration_ticks": 5},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "accepted"
        assert body["fault_type"] == "spike"
        assert body["duration_ticks"] == 5

    def test_400_for_unknown_channel(self) -> None:
        app = _make_app(with_broadcaster=True)
        resp = TestClient(app).post(
            "/api/inject",
            json={"fault_type": "spike", "channels": ["nonexistent"]},
        )
        assert resp.status_code == 400
        assert "nonexistent" in resp.json()["detail"]

    def test_channels_expand_to_all_when_empty(self) -> None:
        app = _make_app(with_broadcaster=True)
        resp = TestClient(app).post(
            "/api/inject",
            json={"fault_type": "flatline"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert _CHANNEL in body["channels"]

    def test_drift_accepted(self) -> None:
        app = _make_app(with_broadcaster=True)
        resp = TestClient(app).post(
            "/api/inject",
            json={"fault_type": "drift", "magnitude_sigma": 2.5, "duration_ticks": 20},
        )
        assert resp.status_code == 200
        assert resp.json()["fault_type"] == "drift"

    def test_invalid_magnitude_returns_422(self) -> None:
        app = _make_app(with_broadcaster=True)
        resp = TestClient(app).post(
            "/api/inject",
            json={"fault_type": "spike", "magnitude_sigma": -1.0},
        )
        assert resp.status_code == 422

    def test_invalid_duration_returns_422(self) -> None:
        app = _make_app(with_broadcaster=True)
        resp = TestClient(app).post(
            "/api/inject",
            json={"fault_type": "spike", "duration_ticks": 0},
        )
        assert resp.status_code == 422

    def test_queues_injection_on_broadcaster(self) -> None:
        app = _make_app(with_broadcaster=True)
        client = TestClient(app)
        client.post(
            "/api/inject",
            json={"fault_type": "spike", "magnitude_sigma": 3.0, "duration_ticks": 10},
        )
        broadcaster = app.state.app_state.broadcaster
        assert broadcaster._pending_injection is not None
        assert broadcaster._pending_injection.fault_type == "spike"
        assert broadcaster._pending_injection.magnitude_sigma == pytest.approx(3.0)

    def test_invalid_magnitude_upper_bound_returns_422(self) -> None:
        app = _make_app(with_broadcaster=True)
        resp = TestClient(app).post(
            "/api/inject",
            json={"fault_type": "spike", "magnitude_sigma": 51.0},
        )
        assert resp.status_code == 422

    def test_invalid_duration_upper_bound_returns_422(self) -> None:
        app = _make_app(with_broadcaster=True)
        resp = TestClient(app).post(
            "/api/inject",
            json={"fault_type": "spike", "duration_ticks": 1001},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# run_shared_loop integration — injection wiring end-to-end
# ---------------------------------------------------------------------------


class _RecordingEngine:
    """Minimal stand-in for ChannelInferenceEngine that records step() calls."""

    def __init__(self, mission: str, channel: str) -> None:
        self._mission = mission
        self._channel = channel
        self.calls: list[tuple[float, bool]] = []

    def step(self, value: float, ts: datetime, is_anomaly: bool) -> TelemetryEvent:
        self.calls.append((value, is_anomaly))
        return TelemetryEvent(
            timestamp=ts,
            mission=self._mission,
            channel=self._channel,
            value_normalized=value,
            prediction=None,
            residual=None,
            smoothed_error=None,
            threshold=None,
            is_anomaly_predicted=is_anomaly,
            is_anomaly=is_anomaly,
        )

    def reset(self) -> None:
        pass


@pytest.mark.asyncio
async def test_run_shared_loop_injection_wiring() -> None:
    """run_shared_loop must deliver injected values and is_anomaly=True to engine.step.

    This pins the begin_tick / apply_fault / end_tick wiring inside the loop:
    deleting any of the three calls would leave the 16 unit tests green while
    silently breaking the demo.
    """
    _CH = "loop-ch"
    _M = "test-mission"
    _SUB = "test-sub"

    n = 10
    values = np.zeros(n, dtype=np.float32)  # all 0.0 → spike sign = +1
    anom = np.zeros(n, dtype=bool)
    timestamps = pd.date_range("2024-01-01", periods=n, freq="30s").values

    settings = load_settings("test")
    settings = settings.model_copy(
        update={
            "api": settings.api.model_copy(
                update={
                    "replay_tick_interval_seconds": 0.005,
                    "replay_speed_default": 1.0,
                }
            )
        }
    )

    broadcaster = EventBroadcaster()
    broadcaster.request_injection("spike", frozenset(), 3.0, 2)

    engine = _RecordingEngine(_M, _CH)
    state = AppState(
        settings=settings,
        mission=_M,
        subsystems=[_SUB],
        device=torch.device("cpu"),
        engines=MappingProxyType({_CH: engine}),  # type: ignore[arg-type]
        channel_subsystem_map=MappingProxyType({_CH: _SUB}),
        replay_data=MappingProxyType({_CH: (values, anom, timestamps)}),
        startup_monotonic_ns=time.monotonic_ns(),
        mlflow_tracking_uri=settings.mlflow.tracking_uri,
        broadcaster=broadcaster,
    )

    task = asyncio.create_task(run_shared_loop(state))
    # 10 ticks * 0.005 s = 50 ms; wait 300 ms for comfortable margin
    await asyncio.sleep(0.3)
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task

    assert len(engine.calls) >= 2, f"expected >= 2 ticks, got {len(engine.calls)}"

    # First two ticks had spike injection: base=0.0, sign=+1, magnitude=3.0
    injected_value_0, is_anomaly_0 = engine.calls[0]
    injected_value_1, is_anomaly_1 = engine.calls[1]
    assert injected_value_0 == pytest.approx(3.0), "spike must add magnitude to base value"
    assert is_anomaly_0 is True, "injected tick must carry is_anomaly=True"
    assert injected_value_1 == pytest.approx(3.0)
    assert is_anomaly_1 is True

    # After 2-tick injection expires the value returns to nominal
    if len(engine.calls) >= 3:
        clean_value, clean_is_anomaly = engine.calls[2]
        assert clean_value == pytest.approx(0.0), "post-injection ticks must be unmodified"
        assert clean_is_anomaly is False
