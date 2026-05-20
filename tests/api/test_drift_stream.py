"""Tests for the GET /api/stream/drift endpoint and drift_stream composer."""

from __future__ import annotations

import json
import time
from collections import deque
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from spacecraft_telemetry.api.drift import RollingDriftMonitor
from spacecraft_telemetry.api.endpoints import router
from spacecraft_telemetry.api.logging_middleware import CorrelationIdMiddleware
from spacecraft_telemetry.api.state import AppState
from spacecraft_telemetry.core.config import Settings, load_settings
from spacecraft_telemetry.evidently_monitoring.reference import MONITORING_FEATURE_COLS

_MISSION = "test-mission"
_CHANNEL = "test-ch"
_SUBSYSTEM = "test-sub"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_reference(n_rows: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {col: rng.standard_normal(n_rows).astype(float) for col in MONITORING_FEATURE_COLS}
    )


def _make_drift_settings(*, enabled: bool = True) -> Settings:
    settings = load_settings("test")
    return settings.model_copy(
        update={
            "drift": settings.drift.model_copy(
                update={
                    "enabled": enabled,
                    "window_size": 32,
                    "tick_interval": 5,
                }
            )
        }
    )


def _make_app(settings: Settings, *, drift_enabled: bool = True) -> FastAPI:
    """Build a minimal FastAPI app with drift state pre-loaded."""
    app = FastAPI()
    app.state.settings = settings

    drift_monitors: dict[str, RollingDriftMonitor] = {}
    tick_buses: dict[str, deque[dict[str, float]]] = {}

    if drift_enabled:
        ref = _make_reference()
        monitor = RollingDriftMonitor(
            channel=_CHANNEL,
            reference=ref,
            window_size=settings.drift.window_size,
            tick_interval=settings.drift.tick_interval,
            feature_drift_threshold=settings.drift.feature_drift_threshold,
            channel_drift_threshold=settings.drift.subsystem_alert_threshold,
        )
        drift_monitors[_CHANNEL] = monitor
        tick_buses[_CHANNEL] = deque(maxlen=settings.drift.window_size)

    app.state.app_state = AppState(
        settings=settings,
        mission=_MISSION,
        subsystem=_SUBSYSTEM,
        device=torch.device("cpu"),
        engines=MappingProxyType({}),
        channel_subsystem_map=MappingProxyType({}),
        replay_data=MappingProxyType({}),
        startup_monotonic_ns=time.monotonic_ns(),
        mlflow_tracking_uri=settings.mlflow.tracking_uri,
        drift_monitors=drift_monitors,
        tick_buses=tick_buses,
    )
    app.add_middleware(CorrelationIdMiddleware)
    app.include_router(router)
    return app


# ---------------------------------------------------------------------------
# 503 and 400 error paths (fast — no Evidently)
# ---------------------------------------------------------------------------


class TestDriftStreamErrors:
    def test_503_when_no_drift_monitors(self) -> None:
        settings = _make_drift_settings(enabled=False)
        app = _make_app(settings, drift_enabled=False)
        with TestClient(app) as client:
            resp = client.get("/api/stream/drift")
        assert resp.status_code == 503
        assert "drift" in resp.json()["detail"].lower()

    def test_400_unknown_channel(self) -> None:
        settings = _make_drift_settings()
        app = _make_app(settings)
        with TestClient(app) as client:
            resp = client.get("/api/stream/drift?channels=no-such-channel")
        assert resp.status_code == 400
        assert "no-such-channel" in resp.json()["detail"]



# ---------------------------------------------------------------------------
# Drift event emission (slow — drives Evidently)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestDriftStreamEvents:
    """Pre-fill the tick bus so the monitor fires without waiting for telemetry."""

    def _fill_tick_bus(
        self,
        app: FastAPI,
        *,
        n_extra: int = 10,
        drifted: bool = False,
    ) -> None:
        """Push window_size + n_extra rows into the tick bus."""
        state: AppState = app.state.app_state
        bus = state.tick_buses[_CHANNEL]
        ref = _make_reference()
        rng = np.random.default_rng(99)
        window_size = state.settings.drift.window_size
        for _ in range(window_size + n_extra):
            if drifted:
                ref_mean = float(ref["value_normalized"].mean())
                ref_std = max(float(ref["value_normalized"].std()), 1e-6)
                val = ref_mean + 5.0 * ref_std
            else:
                idx = int(rng.integers(0, len(ref)))
                val = float(ref.iloc[idx]["value_normalized"])
            bus.append({"value_normalized": val})

    def test_200_content_type_and_fields(self) -> None:
        """Verify status 200 + SSE content-type and check required event fields."""
        settings = _make_drift_settings()
        app = _make_app(settings)
        self._fill_tick_bus(app)
        with TestClient(app) as client:
            events = []
            with client.stream("GET", f"/api/stream/drift?channels={_CHANNEL}") as resp:
                assert resp.status_code == 200
                assert "text/event-stream" in resp.headers["content-type"]
                for line in resp.iter_lines():
                    if line.startswith("data:"):
                        payload = json.loads(line[len("data:"):].strip())
                        events.append(payload)
                        if len(events) >= 1:
                            break
        assert len(events) >= 1

    def test_drift_events_have_required_fields(self) -> None:
        settings = _make_drift_settings()
        app = _make_app(settings)
        self._fill_tick_bus(app)
        with TestClient(app) as client:
            events = []
            with client.stream("GET", f"/api/stream/drift?channels={_CHANNEL}") as resp:
                assert resp.status_code == 200
                for line in resp.iter_lines():
                    if line.startswith("data:"):
                        payload = json.loads(line[len("data:"):].strip())
                        events.append(payload)
                        if len(events) >= 1:
                            break
        assert len(events) >= 1
        ev = events[0]
        assert "channel" in ev
        assert "features" in ev
        assert "percent_drifted" in ev
        assert "drifted" in ev
        assert 0.0 <= ev["percent_drifted"] <= 1.0
        assert isinstance(ev["drifted"], bool)
        assert len(ev["features"]) == len(MONITORING_FEATURE_COLS)

    def test_drift_event_timestamps_monotone(self) -> None:
        settings = _make_drift_settings()
        app = _make_app(settings)
        self._fill_tick_bus(app, n_extra=30)
        with TestClient(app) as client:
            timestamps = []
            with client.stream("GET", f"/api/stream/drift?channels={_CHANNEL}") as resp:
                for line in resp.iter_lines():
                    if line.startswith("data:"):
                        payload = json.loads(line[len("data:"):].strip())
                        timestamps.append(payload["timestamp"])
                        if len(timestamps) >= 2:
                            break
        assert timestamps == sorted(timestamps)

    def test_disconnect_no_traceback(self) -> None:
        settings = _make_drift_settings()
        app = _make_app(settings)
        # Connect and immediately close — must not raise.
        with TestClient(app, raise_server_exceptions=True) as client, client.stream(
            "GET", f"/api/stream/drift?channels={_CHANNEL}"
        ) as resp:
            assert resp.status_code == 200
            # Read nothing and let the context manager close the connection.
