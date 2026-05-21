"""FastAPI router — health check and SSE stream endpoints.

Routes
------
GET /health
    Returns service health status, loaded channels, and uptime.
    200 OK  → HealthResponse (all channels loaded)
    503 Service Unavailable → degraded JSON when no engines are ready.

GET /api/stream/telemetry
    SSE stream replaying the test-split Parquet at the requested speed
    with per-tick LSTM inference.  One ``telemetry`` event per tick per
    channel.
    400 Bad Request → unknown channel name(s).

    Query params (all optional):
        speed    - override replay speed multiplier (default from settings).
        channels - comma-separated channel IDs (default: all loaded channels).

GET /api/stream/drift
    SSE drift-monitoring stream.  Requires drift.enabled=true and at least one
    reference profile loaded.  Emits ``drift`` events at the configured cadence
    (every N telemetry ticks per channel, after the window is full).
    503 Service Unavailable → drift disabled or no reference profiles loaded.
    400 Bad Request → unknown channel name(s).

    Query params (all optional):
        channels - comma-separated channel IDs (default: all monitored channels).
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from spacecraft_telemetry.api.models import HealthResponse, StreamQueryParams
from spacecraft_telemetry.api.state import AppState
from spacecraft_telemetry.api.streaming import drift_stream, telemetry_stream

router = APIRouter()


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


@router.get("/health")
async def health(request: Request) -> JSONResponse:
    """Return service health.

    Returns 200 with loaded channel list when at least one engine is ready.
    Returns 503 when no engines are loaded (e.g. degraded startup or
    all channels failed to load).
    """
    state: AppState = request.app.state.app_state
    if not state.channels_loaded:
        return JSONResponse(
            status_code=503,
            content={
                "status": "degraded",
                "detail": "no channels loaded",
            },
        )
    loaded = set(state.channels_loaded)
    return JSONResponse(
        content=HealthResponse(
            status="ok",
            mission=state.mission,
            subsystem=state.subsystem,
            channels_loaded=state.channels_loaded,
            channel_subsystems={
                ch: sub
                for ch, sub in state.channel_subsystem_map.items()
                if ch in loaded
            },
            uptime_s=state.uptime_seconds(),
            mlflow_tracking_uri=state.mlflow_tracking_uri,
        ).model_dump(),
    )


# ---------------------------------------------------------------------------
# GET /api/stream/telemetry
# ---------------------------------------------------------------------------


@router.get("/api/stream/telemetry")
async def stream(
    request: Request,
    params: Annotated[StreamQueryParams, Depends()],
) -> StreamingResponse:
    """SSE telemetry stream.

    Replays pre-processed test-split Parquet at the requested speed,
    running per-tick LSTM inference and emitting one ``telemetry`` SSE
    event per tick per channel.

    Events are newline-delimited JSON (``TelemetryEvent`` schema)::

        event: telemetry
        data: {"timestamp": ..., "channel": "A-1", ...}

    """
    state: AppState = request.app.state.app_state
    effective_speed = params.speed or state.settings.api.replay_speed_default

    selected = (
        [c.strip() for c in params.channels.split(",") if c.strip()]
        if params.channels
        else state.channels_loaded
    )

    unknown = sorted(set(selected) - set(state.channels_loaded))
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"unknown channels: {unknown}",
        )

    return StreamingResponse(
        telemetry_stream(
            state,
            request,
            speed=effective_speed,
            selected_channels=selected,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# GET /api/stream/drift
# ---------------------------------------------------------------------------


@router.get("/api/stream/drift")
async def stream_drift(
    request: Request,
    params: Annotated[StreamQueryParams, Depends()],
) -> StreamingResponse:
    """SSE drift-monitoring stream.

    Emits ``drift`` SSE events driven by the ``RollingDriftMonitor`` instances
    loaded at startup.  Each event carries per-feature drift scores computed by
    Evidently against the Phase 7 reference profile.

    Returns 503 when drift monitoring is disabled or no reference profiles were
    loaded at startup.  Returns 400 for unknown channel names.
    """
    state: AppState = request.app.state.app_state

    if not state.drift_monitors:
        raise HTTPException(
            status_code=503,
            detail="drift monitoring disabled or no reference profiles loaded",
        )

    selected = (
        [c.strip() for c in params.channels.split(",") if c.strip()]
        if params.channels
        else sorted(state.drift_monitors.keys())
    )

    unknown = sorted(set(selected) - set(state.drift_monitors.keys()))
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"unknown channels: {unknown}",
        )

    return StreamingResponse(
        drift_stream(state, request, selected_channels=selected),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
