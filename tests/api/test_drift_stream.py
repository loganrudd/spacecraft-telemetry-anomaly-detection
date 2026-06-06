"""Tests for the GET /api/stream/drift endpoint and drift_stream composer."""

from __future__ import annotations

import asyncio
import json
import time
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from spacecraft_telemetry.api.endpoints import router
from spacecraft_telemetry.api.logging_middleware import CorrelationIdMiddleware
from spacecraft_telemetry.api.state import AppState
from spacecraft_telemetry.core.config import Settings, load_settings
from spacecraft_telemetry.evidently_monitoring.reference import (
    MONITORING_FEATURE_COLS,
    REALTIME_FEATURE_COLS,
)

_MISSION = "test-mission"
_CHANNEL = "test-ch"
_SUBSYSTEM = "test-sub"

# Run the replay fast enough that tests complete in <30 s.
_TEST_SPEED = 1e6

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
                    "window_size": 30,  # divisible by tick_interval → should_run() fires
                    "tick_interval": 5,
                }
            )
        }
    )


def _make_app(settings: Settings, *, drift_enabled: bool = True) -> FastAPI:
    """Build a minimal FastAPI app with drift reference profiles and replay data."""
    app = FastAPI()
    app.state.settings = settings

    drift_references: dict[str, pd.DataFrame] = {}
    replay_data_map: dict[str, object] = {}

    if drift_enabled:
        ref = _make_reference()
        drift_references[_CHANNEL] = ref

        # Synthetic replay: enough ticks to fill the window several times so
        # drift_stream can emit multiple events during a test.
        rng = np.random.default_rng(42)
        n = settings.drift.window_size * 6
        values = rng.choice(ref["value_normalized"].values, size=n).astype(np.float64)
        anom = np.zeros(n, dtype=bool)
        timestamps = pd.date_range("2020-01-01", periods=n, freq="1s").to_numpy()
        replay_data_map[_CHANNEL] = (values, anom, timestamps)

    app.state.app_state = AppState(
        settings=settings,
        mission=_MISSION,
        subsystems=[_SUBSYSTEM],
        device=torch.device("cpu"),
        engines=MappingProxyType({}),
        channel_subsystem_map=MappingProxyType({}),
        replay_data=MappingProxyType(replay_data_map),
        startup_monotonic_ns=time.monotonic_ns(),
        mlflow_tracking_uri=settings.mlflow.tracking_uri,
        drift_references=drift_references,
    )
    app.add_middleware(CorrelationIdMiddleware)
    app.include_router(router)
    return app


# ---------------------------------------------------------------------------
# 503 and 400 error paths (fast — no Evidently)
# ---------------------------------------------------------------------------


class TestDriftStreamErrors:
    def test_503_when_no_drift_references(self) -> None:
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
    """Drive the drift_stream generator directly — bypasses TestClient body collection.

    Starlette's TestClient.handle_request() calls portal.call(app, ...) which
    blocks until the ASGI app completes.  For an infinite SSE stream this never
    returns, so client.stream().__enter__() hangs forever.  Calling the async
    generator directly lets the event loop drive pump tasks and collect events
    without any HTTP transport.

    drift_stream is now self-contained: each call creates its own per-request
    RollingDriftMonitor and drives replay_channel from AppState.replay_data, so
    no tick-bus pre-filling is needed.
    """

    async def _collect_events(
        self,
        app: FastAPI,
        *,
        n: int = 1,
        timeout: float = 30.0,
    ) -> list[dict]:
        """Invoke drift_stream directly and collect *n* parsed event payloads."""
        from spacecraft_telemetry.api.streaming import drift_stream

        state: AppState = app.state.app_state

        class _NeverDisconnects:
            async def is_disconnected(self) -> bool:
                return False

        events: list[dict] = []

        async def _drain() -> None:
            gen = drift_stream(
                state, _NeverDisconnects(), selected_channels=[_CHANNEL], speed=_TEST_SPEED
            )
            try:
                async for chunk in gen:
                    for line in chunk.decode().splitlines():
                        if line.startswith("data:"):
                            events.append(json.loads(line[5:].strip()))
                    if len(events) >= n:
                        return
            finally:
                await gen.aclose()

        await asyncio.wait_for(_drain(), timeout=timeout)
        return events

    async def test_200_content_type_and_fields(self) -> None:
        """Generator emits at least one event driven by its own replay."""
        settings = _make_drift_settings()
        app = _make_app(settings)
        events = await self._collect_events(app, n=1)
        assert len(events) >= 1

    async def test_drift_events_have_required_fields(self) -> None:
        settings = _make_drift_settings()
        app = _make_app(settings)
        events = await self._collect_events(app, n=1)
        assert len(events) >= 1
        ev = events[0]
        # Per-channel fields always present.
        assert ev["channel"] == _CHANNEL
        assert ev["mission"] == _MISSION
        assert "features" in ev
        assert "percent_drifted" in ev
        assert "drifted" in ev
        assert 0.0 <= ev["percent_drifted"] <= 1.0
        assert isinstance(ev["drifted"], bool)
        assert len(ev["features"]) == len(REALTIME_FEATURE_COLS)
        # Subsystem fields are None on per-channel events (not a summary tick).
        assert ev["subsystem_percent_drifted"] is None
        assert ev["subsystem_alert"] is None

    async def test_drift_event_timestamps_monotone(self) -> None:
        settings = _make_drift_settings()
        app = _make_app(settings)
        events = await self._collect_events(app, n=2)
        timestamps = [ev["timestamp"] for ev in events]
        # Strict monotone — equal timestamps would mean the clock is frozen.
        for i in range(len(timestamps) - 1):
            assert timestamps[i] <= timestamps[i + 1], (
                f"timestamp[{i}]={timestamps[i]} > timestamp[{i + 1}]={timestamps[i + 1]}"
            )

    async def test_disconnect_no_traceback(self) -> None:
        """A disconnect on the first iteration must not raise."""
        from spacecraft_telemetry.api.streaming import drift_stream

        settings = _make_drift_settings()
        app = _make_app(settings)
        state: AppState = app.state.app_state

        class _DisconnectsImmediately:
            async def is_disconnected(self) -> bool:
                return True

        gen = drift_stream(
            state, _DisconnectsImmediately(), selected_channels=[_CHANNEL], speed=_TEST_SPEED
        )
        async for _ in gen:
            break  # pragma: no cover
        await gen.aclose()


# ---------------------------------------------------------------------------
# HTTP boundary smoke test (slow — goes through ASGI transport)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestDriftStreamHTTPBoundary:
    """Verify StreamingResponse headers and SSE wire format via httpx ASGITransport.

    The TestDriftStreamEvents tests drive drift_stream as a raw async generator,
    bypassing the HTTP layer entirely.  This class adds one end-to-end check that
    the endpoint wires up the correct Content-Type, Cache-Control, and
    X-Accel-Buffering headers and that the SSE wire format uses the ``event: drift``
    prefix — regressions in endpoints.py would not be caught by the generator tests.
    """

    async def test_sse_headers_and_framing(self) -> None:
        from httpx import ASGITransport, AsyncClient

        settings = _make_drift_settings()
        app = _make_app(settings)

        raw = b""

        async def _read_one_frame() -> dict[str, str]:
            nonlocal raw
            transport = ASGITransport(app=app)
            async with (
                AsyncClient(transport=transport, base_url="http://test") as client,
                client.stream("GET", f"/api/stream/drift?speed={int(_TEST_SPEED)}") as resp,
            ):
                headers = dict(resp.headers)
                async for chunk in resp.aiter_bytes():
                    raw += chunk
                    if b"\n\n" in raw:
                        return headers
            return {}  # pragma: no cover

        headers = await asyncio.wait_for(_read_one_frame(), timeout=30.0)

        assert "text/event-stream" in headers["content-type"]
        assert headers.get("cache-control") == "no-cache"
        assert headers.get("x-accel-buffering") == "no"

        frame = raw[: raw.index(b"\n\n") + 2].decode()
        assert frame.startswith("event: drift\n"), f"unexpected SSE prefix: {frame[:60]!r}"
        assert "\ndata:" in frame
