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

from spacecraft_telemetry.api.models import (
    HealthResponse,
    InjectRequest,
    InjectResponse,
    StreamQueryParams,
)
from spacecraft_telemetry.api.state import AppState, LoadingState
from spacecraft_telemetry.api.streaming import drift_stream, subscriber_stream, telemetry_stream

router = APIRouter()


def _get_ready_state(request: Request) -> AppState:
    """Return AppState when engines are loaded, or raise 503 while loading."""
    app_state: AppState | None = getattr(request.app.state, "app_state", None)
    if app_state is not None:
        return app_state
    loading: LoadingState | None = getattr(request.app.state, "loading_state", None)
    if loading and not loading.error:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Service loading: {loading.channels_ready}/{loading.channels_total} "
                "channels ready — try again in a moment"
            ),
        )
    raise HTTPException(status_code=503, detail="Service unavailable")


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


@router.get("/health")
async def health(request: Request) -> JSONResponse:
    """Return service health and loading progress.

    Always returns 200 so the container passes liveness checks while models
    are loading.  Callers should inspect ``status``:

    * ``"loading"``  — background task in progress; use channels_ready /
                       channels_total to drive a progress bar.
    * ``"ok"``       — all engines ready; channels_loaded is the full list.
    * ``"degraded"`` — loading finished but no engines could be loaded.
    """
    app_state: AppState | None = getattr(request.app.state, "app_state", None)
    loading: LoadingState | None = getattr(request.app.state, "loading_state", None)
    settings = request.app.state.settings

    if app_state is None:
        if loading is None:
            return JSONResponse(
                status_code=503,
                content={"status": "degraded", "detail": "not initialized"},
            )
        if loading.error:
            return JSONResponse(
                status_code=503,
                content=HealthResponse(
                    status="degraded",
                    mission=settings.api.mission,
                    subsystems=settings.api.subsystems,
                    channels_total=loading.channels_total,
                    channels_ready=loading.channels_ready,
                    uptime_s=loading.uptime_seconds(),
                    available_missions=settings.api.available_missions,
                ).model_dump(),
            )
        return JSONResponse(
            content=HealthResponse(
                status="loading",
                mission=settings.api.mission,
                subsystems=settings.api.subsystems,
                channels_total=loading.channels_total,
                channels_ready=loading.channels_ready,
                uptime_s=loading.uptime_seconds(),
                available_missions=settings.api.available_missions,
            ).model_dump(),
        )

    if not app_state.channels_loaded:
        return JSONResponse(
            status_code=503,
            content=HealthResponse(
                status="degraded",
                mission=app_state.mission,
                subsystems=app_state.subsystems,
                channels_total=loading.channels_total if loading else 0,
                uptime_s=app_state.uptime_seconds(),
                mlflow_tracking_uri=app_state.mlflow_tracking_uri,
                available_missions=app_state.settings.api.available_missions,
            ).model_dump(),
        )

    loaded = set(app_state.channels_loaded)
    missing = sorted(set(app_state.resolved_channels) - loaded)
    n = len(loaded)
    status = "degraded" if missing else "ok"
    return JSONResponse(
        content=HealthResponse(
            status=status,
            mission=app_state.mission,
            subsystems=app_state.subsystems,
            channels_loaded=app_state.channels_loaded,
            channels_total=len(app_state.resolved_channels) or n,
            channels_ready=n,
            missing=missing,
            channel_subsystems={
                ch: sub
                for ch, sub in app_state.channel_subsystem_map.items()
                if ch in loaded
            },
            uptime_s=app_state.uptime_seconds(),
            mlflow_tracking_uri=app_state.mlflow_tracking_uri,
            available_missions=app_state.settings.api.available_missions,
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
    state = _get_ready_state(request)
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

    # Use the shared broadcaster loop when available (production lifespan).
    # Fall back to the per-connection pump for test fixtures that construct
    # AppState directly without starting the loop.
    generator = (
        subscriber_stream(state, request, selected)
        if state.broadcaster is not None
        else telemetry_stream(state, request, speed=effective_speed, selected_channels=selected)
    )
    return StreamingResponse(
        generator,
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

    Creates per-request ``RollingDriftMonitor`` instances from the reference
    profiles loaded at startup and drives its own replay, so the stream works
    independently of any telemetry stream clients.

    Returns 503 when drift monitoring is disabled or no reference profiles were
    loaded at startup.  Returns 400 for unknown channel names.
    """
    state = _get_ready_state(request)

    if not state.drift_references:
        raise HTTPException(
            status_code=503,
            detail="drift monitoring disabled or no reference profiles loaded",
        )

    selected = (
        [c.strip() for c in params.channels.split(",") if c.strip()]
        if params.channels
        else sorted(state.drift_references.keys())
    )

    unknown = sorted(set(selected) - set(state.drift_references.keys()))
    if unknown:
        raise HTTPException(
            status_code=400,
            detail=f"unknown channels: {unknown}",
        )

    effective_speed = params.speed or state.settings.api.replay_speed_default

    return StreamingResponse(
        drift_stream(state, request, selected_channels=selected, speed=effective_speed),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# POST /api/inject
# ---------------------------------------------------------------------------


@router.post("/api/inject")
async def inject_fault(request: Request, body: InjectRequest) -> JSONResponse:
    """Inject a transient fault into the running shared replay loop.

    The fault is applied to the next ``body.duration_ticks`` ticks in the
    shared broadcaster loop, modifying the values seen by every connected SSE
    subscriber simultaneously (correct behaviour for a mission console — all
    viewers see the same fault).

    Fault types mirror ``injection/faults.py`` so the demo uses the same math
    as the offline HPO evaluation:
      spike        — additive offset of ``magnitude_sigma`` sigmas for each tick.
      drift_inject — linear ramp 0 → ``magnitude_sigma`` sigmas over the duration.
      flatline     — hold each channel's current value (simulates sensor death).

    Returns 503 when the shared loop is not running (test / no-lifespan mode).
    Returns 400 for unknown channel names.
    """
    state = _get_ready_state(request)
    broadcaster = state.broadcaster
    if broadcaster is None:
        raise HTTPException(
            status_code=503,
            detail="inject not available: shared loop is not running",
        )

    if body.channels:
        unknown = sorted(set(body.channels) - set(state.channels_loaded))
        if unknown:
            raise HTTPException(status_code=400, detail=f"unknown channels: {unknown}")

    effective_channels = body.channels or state.channels_loaded
    broadcaster.request_injection(
        fault_type=body.fault_type,
        channels=frozenset(body.channels),  # empty frozenset = all channels in loop
        magnitude_sigma=body.magnitude_sigma,
        total_ticks=body.duration_ticks,
    )
    return JSONResponse(
        InjectResponse(
            status="accepted",
            fault_type=body.fault_type,
            channels=effective_channels,
            magnitude_sigma=body.magnitude_sigma,
            duration_ticks=body.duration_ticks,
        ).model_dump()
    )
