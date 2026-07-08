"""Tests for the GET /api/stream/drift endpoint and drift_stream composer.

drift_stream is now a thin subscriber over the shared EventBroadcaster (see
streaming.py): event: drift frames are published by the shared producer
(run_shared_loop for ESA / the ISS live pump) via drift_feed.step_drift, one
RollingDriftMonitor per channel feeding every connected viewer.

Design notes
------------
- Event-emission tests drive drift_stream directly as an async generator
  against a real run_shared_loop producer task, rather than going through a
  full ASGI transport. Driving both the hot producer loop (test settings use
  1000x replay speed) and a consumer through httpx's ASGITransport +
  BaseHTTPMiddleware in the same event loop was observed to starve the
  consumer indefinitely (a 15s collection would still return nothing).
  Calling the generator directly is the same pattern the pre-rewrite test
  suite used for this exact reason -- see the historical note in
  TestDriftStreamEvents._collect_events.
- Only the HTTP-boundary test (headers/SSE framing) goes through a real
  ASGITransport; it publishes one synthetic frame directly rather than
  running a producer, so it isn't subject to the same contention.
"""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import suppress
from datetime import datetime
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from spacecraft_telemetry.api.broadcast import EventBroadcaster, run_shared_loop
from spacecraft_telemetry.api.drift import RollingDriftMonitor
from spacecraft_telemetry.api.endpoints import router
from spacecraft_telemetry.api.logging_middleware import CorrelationIdMiddleware
from spacecraft_telemetry.api.models import TelemetryEvent
from spacecraft_telemetry.api.state import AppState
from spacecraft_telemetry.core.config import Settings, load_settings
from spacecraft_telemetry.evidently_monitoring.reference import REALTIME_FEATURE_COLS

_MISSION = "test-mission"
_CHANNEL = "test-ch"
_SUBSYSTEM = "test-sub"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _RecordingEngine:
    """Minimal stand-in for ChannelInferenceEngine, driven by run_shared_loop."""

    def __init__(self, mission: str, channel: str) -> None:
        self._mission = mission
        self._channel = channel

    def step(self, value: float, ts: datetime, is_anomaly: bool) -> TelemetryEvent:
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


class _FakeRequest:
    """Minimal stand-in for starlette.Request.

    drift_stream's disconnect watcher only calls the ASGI-level ``.receive()``
    (not ``.is_disconnected()``); this fake supplies exactly that.
    """

    def __init__(self, disconnect: bool = False) -> None:
        self._disconnect = disconnect

    async def receive(self) -> dict[str, str]:
        if self._disconnect:
            return {"type": "http.disconnect"}
        await asyncio.Event().wait()  # blocks until the caller cancels us
        return {}  # pragma: no cover


def _make_reference(n_rows: int = 1200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    v = pd.Series(rng.standard_normal(n_rows).astype(float))
    return pd.DataFrame({"value_normalized": v, "rate_of_change": v.diff().fillna(0.0)})


def _make_monitor(
    channel: str, reference: pd.DataFrame, window_size: int = 4, tick_interval: int = 1
) -> RollingDriftMonitor:
    assert list(REALTIME_FEATURE_COLS) == ["value_normalized", "rate_of_change"]
    return RollingDriftMonitor(
        channel=channel,
        reference=reference,
        window_size=window_size,
        tick_interval=tick_interval,
        feature_drift_threshold=0.10,
        channel_drift_threshold=0.30,
    )


def _make_settings(api_overrides: dict[str, Any] | None = None) -> Settings:
    settings = load_settings("test")
    if not api_overrides:
        return settings
    return settings.model_copy(
        update={"api": settings.api.model_copy(update=api_overrides)}
    )


def _make_app(
    *,
    with_broadcaster: bool = False,
    with_drift_monitors: bool = False,
    with_drift_references: bool = True,
    n_rows: int = 60,
    window_size: int = 4,
    tick_interval: int = 1,
    api_overrides: dict[str, Any] | None = None,
) -> tuple[FastAPI, AppState]:
    """Build a minimal FastAPI app + AppState for drift-stream tests.

    ``with_drift_monitors`` additionally wires an engine + replay data for
    ``_CHANNEL`` so ``run_shared_loop`` can drive it end-to-end.
    """
    settings = _make_settings(api_overrides)
    broadcaster = EventBroadcaster() if with_broadcaster else None

    drift_references: dict[str, object] = {}
    drift_monitors: dict[str, RollingDriftMonitor] = {}
    engines: dict[str, object] = {}
    replay_data: dict[str, object] = {}

    reference = _make_reference()
    if with_drift_references:
        drift_references[_CHANNEL] = reference
    if with_drift_monitors:
        drift_monitors[_CHANNEL] = _make_monitor(
            _CHANNEL, reference, window_size, tick_interval
        )
        engines[_CHANNEL] = _RecordingEngine(_MISSION, _CHANNEL)
        # Sample replay values from the reference distribution itself (not an
        # independently-seeded draw) so a "baseline" window is genuinely
        # nominal rather than spuriously drifted by sampling noise between
        # two unrelated N(0,1) realizations at these small window sizes.
        rng = np.random.default_rng(42)
        values = rng.choice(reference["value_normalized"].to_numpy(), size=n_rows)
        anom = np.zeros(n_rows, dtype=bool)
        timestamps = pd.date_range("2020-01-01", periods=n_rows, freq="1s").to_numpy()
        replay_data[_CHANNEL] = (values, anom, timestamps)

    state = AppState(
        settings=settings,
        mission=_MISSION,
        subsystems=[_SUBSYSTEM],
        device=torch.device("cpu"),
        engines=MappingProxyType(engines),
        channel_subsystem_map=MappingProxyType({_CHANNEL: _SUBSYSTEM}),
        replay_data=MappingProxyType(replay_data),
        startup_monotonic_ns=time.monotonic_ns(),
        mlflow_tracking_uri=settings.mlflow.tracking_uri,
        drift_references=MappingProxyType(drift_references),
        drift_monitors=MappingProxyType(drift_monitors),
        broadcaster=broadcaster,
    )
    app = FastAPI()
    app.state.settings = settings
    app.state.app_state = state
    app.add_middleware(CorrelationIdMiddleware)
    app.include_router(router)
    return app, state


async def _collect_drift_events(
    state: AppState, channels: list[str], *, n: int = 1, timeout: float = 15.0
) -> list[dict[str, Any]]:
    """Drive drift_stream directly and collect *n* parsed event payloads.

    See the module docstring for why this bypasses the ASGI/HTTP layer.
    """
    from spacecraft_telemetry.api.streaming import drift_stream

    events: list[dict[str, Any]] = []

    async def _drain() -> None:
        gen = drift_stream(state, _FakeRequest(), selected_channels=channels)
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


# ---------------------------------------------------------------------------
# 503 and 400 error paths (fast — no producer running)
# ---------------------------------------------------------------------------


class TestDriftStreamErrors:
    def test_503_when_no_drift_references(self) -> None:
        app, _ = _make_app(with_drift_references=False, with_drift_monitors=False)
        with TestClient(app) as client:
            resp = client.get("/api/stream/drift")
        assert resp.status_code == 503
        assert "drift" in resp.json()["detail"].lower()

    def test_503_when_no_broadcaster(self) -> None:
        """Drift now requires the shared producer -- no per-connection fallback."""
        app, _ = _make_app(with_broadcaster=False, with_drift_references=True)
        with TestClient(app) as client:
            resp = client.get("/api/stream/drift")
        assert resp.status_code == 503

    def test_400_unknown_channel(self) -> None:
        app, _ = _make_app(
            with_broadcaster=True, with_drift_monitors=True, with_drift_references=True
        )
        with TestClient(app) as client:
            resp = client.get("/api/stream/drift?channels=no-such-channel")
        assert resp.status_code == 400
        assert "no-such-channel" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Drift event emission — drives the real producer (run_shared_loop)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestDriftStreamEvents:
    async def _run_and_collect(
        self, state: AppState, *, n: int = 1, timeout: float = 15.0
    ) -> list[dict[str, Any]]:
        loop_task = asyncio.create_task(run_shared_loop(state))
        try:
            return await _collect_drift_events(state, [_CHANNEL], n=n, timeout=timeout)
        finally:
            loop_task.cancel()
            with suppress(asyncio.CancelledError):
                await loop_task

    async def test_drift_events_have_required_fields(self) -> None:
        _app, state = _make_app(with_broadcaster=True, with_drift_monitors=True)
        events = await self._run_and_collect(state, n=1)
        assert len(events) >= 1
        ev = events[0]
        assert ev["channel"] == _CHANNEL
        assert ev["mission"] == _MISSION
        assert "features" in ev
        assert 0.0 <= ev["percent_drifted"] <= 1.0
        assert isinstance(ev["drifted"], bool)
        assert len(ev["features"]) == len(REALTIME_FEATURE_COLS)

    async def test_drift_event_timestamps_monotone_backlog_and_live(self) -> None:
        """Timestamps stay non-decreasing across the backlog->live handoff.

        Sleeping before subscribing ensures the backlog is non-empty, so the
        collected sequence spans both the pre-connect backlog and live ticks --
        a gap or duplicate at that boundary would show up as a timestamp
        regression here.
        """
        _app, state = _make_app(with_broadcaster=True, with_drift_monitors=True)
        loop_task = asyncio.create_task(run_shared_loop(state))
        try:
            await asyncio.sleep(0.1)
            events = await _collect_drift_events(state, [_CHANNEL], n=5, timeout=15.0)
        finally:
            loop_task.cancel()
            with suppress(asyncio.CancelledError):
                await loop_task
        timestamps = [ev["timestamp"] for ev in events]
        for i in range(len(timestamps) - 1):
            assert timestamps[i] <= timestamps[i + 1], (
                f"timestamp[{i}]={timestamps[i]} > timestamp[{i + 1}]={timestamps[i + 1]}"
            )

    async def test_injection_raises_percent_drifted(self) -> None:
        """request_injection (POST /api/inject) must reach the shared drift
        monitor -- the bug this whole rework fixes: injection used to mutate
        the broadcaster while drift ran its own disconnected replay."""
        # window_size=256 (matching test_drift_monitor.py's proven-stable ref=1200/
        # window=256 pair) is needed for a reliably nominal baseline: at n=64 the
        # finite-sample Wasserstein distance between a bootstrap window and its own
        # generating distribution is ~O(1/sqrt(n)) ~= 0.12-0.20 even with zero real
        # drift, comfortably above the 0.10 threshold on sampling noise alone.
        _app, state = _make_app(
            with_broadcaster=True, with_drift_monitors=True, window_size=256, n_rows=300
        )
        assert state.broadcaster is not None
        loop_task = asyncio.create_task(run_shared_loop(state))
        try:
            baseline = await _collect_drift_events(state, [_CHANNEL], n=1, timeout=15.0)
            baseline_pct = baseline[0]["percent_drifted"]

            state.broadcaster.request_injection("spike", frozenset(), 10.0, 200)
            await asyncio.sleep(0.2)  # let several injected ticks flow through

            spiked = await _collect_drift_events(state, [_CHANNEL], n=1, timeout=15.0)
            assert spiked[0]["percent_drifted"] > baseline_pct
        finally:
            loop_task.cancel()
            with suppress(asyncio.CancelledError):
                await loop_task

    async def test_priming_avoids_cold_start_blank(self) -> None:
        """A monitor primed like app.py primes it (push window_size seed values
        before the producer starts) fires on the first live tick; an unprimed
        one needs the window to fill from empty first."""
        window_size = 8
        fast_api = {"replay_tick_interval_seconds": 0.05, "replay_speed_default": 1.0}

        _app_primed, state_primed = _make_app(
            with_broadcaster=True,
            with_drift_monitors=True,
            window_size=window_size,
            n_rows=80,
            api_overrides=fast_api,
        )
        for _ in range(window_size):
            state_primed.drift_monitors[_CHANNEL].push({"value_normalized": 0.0})

        _app_cold, state_cold = _make_app(
            with_broadcaster=True,
            with_drift_monitors=True,
            window_size=window_size,
            n_rows=80,
            api_overrides=fast_api,
        )

        primed_task = asyncio.create_task(run_shared_loop(state_primed))
        cold_task = asyncio.create_task(run_shared_loop(state_cold))
        try:
            await asyncio.sleep(0.15)  # ~3 ticks: enough for a primed fire, not a cold one
            assert state_primed.drift_monitors[_CHANNEL].latest is not None
            assert state_cold.drift_monitors[_CHANNEL].latest is None
        finally:
            for t in (primed_task, cold_task):
                t.cancel()
            for t in (primed_task, cold_task):
                with suppress(asyncio.CancelledError):
                    await t

    async def test_disconnect_no_traceback(self) -> None:
        """A disconnect on the first iteration must not raise."""
        from spacecraft_telemetry.api.streaming import drift_stream

        _app, state = _make_app(with_broadcaster=True, with_drift_monitors=True)
        gen = drift_stream(state, _FakeRequest(disconnect=True), selected_channels=[_CHANNEL])
        async for _ in gen:
            break  # pragma: no cover
        await gen.aclose()


# ---------------------------------------------------------------------------
# HTTP boundary smoke test — headers and SSE wire format
# ---------------------------------------------------------------------------


class _FakeAppStateHolder:
    """Stands in for ``request.app.state`` -- just needs an ``app_state`` attr."""

    def __init__(self, app_state: AppState) -> None:
        self.app_state = app_state


class _FakeApp:
    """Stands in for ``request.app`` -- just needs a ``.state`` attr."""

    def __init__(self, app_state: AppState) -> None:
        self.state = _FakeAppStateHolder(app_state)


class _FakeRequestWithApp(_FakeRequest):
    """Adds the ``request.app.state.app_state`` chain ``_get_ready_state`` needs.

    Calling the ``stream_drift`` endpoint function directly (no ASGI, no
    FastAPI routing) drives the exact same production code -- StreamingResponse
    construction, headers, and the generator's first frame -- without an ASGI
    transport. This sidesteps a real deadlock: BaseHTTPMiddleware (used for
    CorrelationIdMiddleware) does not compose safely with infinite SSE
    StreamingResponses under TestClient/httpx's ASGITransport -- confirmed to
    reproduce identically with the pre-existing, unmodified subscriber_stream
    (the production telemetry path), so it is a latent, pre-existing gap in
    test infra rather than something introduced by this rework. No existing
    test anywhere in the suite exercises subscriber_stream through a real ASGI
    transport for the same reason.
    """

    def __init__(self, app_state: AppState, disconnect: bool = False) -> None:
        super().__init__(disconnect=disconnect)
        self.app = _FakeApp(app_state)


class TestDriftStreamHTTPBoundary:
    """A synthetic frame published directly avoids depending on real drift
    computation timing -- this class only checks the endpoint's response
    construction (headers, media type, first SSE frame)."""

    async def test_response_headers_and_first_frame(self) -> None:
        from spacecraft_telemetry.api.endpoints import stream_drift
        from spacecraft_telemetry.api.models import StreamQueryParams

        _app, state = _make_app(with_broadcaster=True, with_drift_monitors=True)
        assert state.broadcaster is not None
        state.broadcaster.publish(_CHANNEL, b"event: drift\ndata: {}\n\n")

        request = _FakeRequestWithApp(state)
        response = await stream_drift(request, StreamQueryParams())

        assert response.media_type == "text/event-stream"
        assert response.headers["cache-control"] == "no-cache"
        assert response.headers["x-accel-buffering"] == "no"

        try:
            first_chunk = await response.body_iterator.__anext__()
        finally:
            await response.body_iterator.aclose()
        if isinstance(first_chunk, str):
            first_chunk = first_chunk.encode()
        assert first_chunk.startswith(b"event: drift\n"), (
            f"unexpected SSE prefix: {first_chunk[:60]!r}"
        )
        assert b"\ndata:" in first_chunk
