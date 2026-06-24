"""Tests for POST /api/inject and EventBroadcaster injection state machine."""

from __future__ import annotations

import time
from types import MappingProxyType

import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from spacecraft_telemetry.api.broadcast import EventBroadcaster
from spacecraft_telemetry.api.endpoints import router
from spacecraft_telemetry.api.logging_middleware import CorrelationIdMiddleware
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

    def test_drift_ramps_linearly(self) -> None:
        b = EventBroadcaster()
        b.request_injection("drift_inject", frozenset(), 4.0, 5)
        results = []
        for _ in range(5):
            b.begin_tick()
            val, _ = b.apply_fault(_CHANNEL, 0.0)
            results.append(val)
            b.end_tick()
        # elapsed goes 0,1,2,3,4 → progress = 0, 0.25, 0.5, 0.75, 1.0
        assert results[0] == pytest.approx(0.0)
        assert results[-1] == pytest.approx(4.0)
        assert results[2] == pytest.approx(2.0)

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

    def test_drift_inject_accepted(self) -> None:
        app = _make_app(with_broadcaster=True)
        resp = TestClient(app).post(
            "/api/inject",
            json={"fault_type": "drift_inject", "magnitude_sigma": 2.5, "duration_ticks": 20},
        )
        assert resp.status_code == 200
        assert resp.json()["fault_type"] == "drift_inject"

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
