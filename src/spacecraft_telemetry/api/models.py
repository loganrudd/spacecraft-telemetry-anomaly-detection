"""Pydantic v2 request/response models for the FastAPI serving layer."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, field_validator

from spacecraft_telemetry.core.config import MissionLink


class TelemetryEvent(BaseModel):
    """Per-tick inference result emitted by the SSE telemetry stream."""

    timestamp: datetime
    mission: str
    channel: str
    value_normalized: float
    prediction: float | None
    residual: float | None
    smoothed_error: float | None
    threshold: float | None  # math.inf rendered as None
    is_anomaly_predicted: bool
    is_anomaly: bool


class HealthResponse(BaseModel):
    """Response model for GET /health.

    status="loading"  → background task is still loading models;
                        channels_ready / channels_total show progress.
    status="ok"       → all engines ready; channels_loaded is the full list.
    status="degraded" → loading finished but no engines could be loaded.
    """

    status: Literal["ok", "degraded", "loading"]
    mission: str
    subsystems: list[str] | None = None
    channels_loaded: list[str] = []
    channels_total: int = 0    # target channel count (set once channels are resolved)
    channels_ready: int = 0    # successfully loaded so far (== len(channels_loaded) when ok)
    missing: list[str] = []    # channels that failed to load (non-empty → status=degraded)
    channel_subsystems: dict[str, str] = {}
    uptime_s: float = 0.0
    mlflow_tracking_uri: str = ""
    # Sibling-mission entries for the dashboard mission selector. Empty list
    # hides the selector (single-mission deploys, all existing tests).
    available_missions: list[MissionLink] = []


class StreamQueryParams(BaseModel):
    """Query parameters for the SSE stream endpoint (validated at boundary)."""

    speed: float | None = None
    channels: str | None = None  # CSV of channel names

    @field_validator("speed")
    @classmethod
    def speed_positive(cls, v: float | None) -> float | None:
        if v is not None and v <= 0:
            raise ValueError(f"speed must be > 0, got {v}")
        return v


class DriftFeature(BaseModel):
    """Drift score for a single monitored feature column."""

    feature: str
    score: float    # Wasserstein distance reported by Evidently
    drifted: bool


class DriftEvent(BaseModel):
    """Per-channel drift result emitted by the SSE drift stream."""

    timestamp: datetime
    mission: str
    channel: str
    features: list[DriftFeature]
    percent_drifted: float      # fraction of features drifted, in [0, 1]
    drifted: bool               # True if percent_drifted >= channel drift threshold
    # Populated only on periodic subsystem-summary ticks; None on per-channel events.
    subsystem_percent_drifted: float | None = None
    subsystem_alert: bool | None = None

    @field_validator("percent_drifted")
    @classmethod
    def percent_drifted_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"percent_drifted must be in [0, 1], got {v}")
        return v


class RawTelemetryEvent(BaseModel):
    """Per-tick raw event emitted at native Lightstreamer cadence (event: raw).

    Carries the normalized value immediately on each received tick before the
    30-second grid bucket closes.  Drives the continuous live chart line.
    No prediction or anomaly fields — those arrive via TelemetryEvent at 30 s.
    """

    timestamp: datetime
    channel: str
    value_normalized: float


class InjectRequest(BaseModel):
    """Request body for POST /api/inject."""

    fault_type: Literal["spike", "drift", "flatline"]
    # Channel IDs to inject; empty list = all loaded channels.
    channels: list[str] = []
    # Additive offset in z-score units (ignored for flatline).
    # 5-sigma default matches the drift fault-type default — large enough to climb
    # past the EWMA-attenuated threshold but physically plausible.
    magnitude_sigma: float = 5.0
    # Number of replay ticks the fault lasts. 60 ticks = 30 min at 30s grid —
    # gives the EWMA ramp-up and K consecutive crossings needed for drift.
    duration_ticks: int = 60

    @field_validator("magnitude_sigma")
    @classmethod
    def mag_positive(cls, v: float) -> float:
        if not 0 < v <= 50:
            raise ValueError(f"magnitude_sigma must be in (0, 50], got {v}")
        return v

    @field_validator("duration_ticks")
    @classmethod
    def dur_positive(cls, v: int) -> int:
        if not 1 <= v <= 1000:
            raise ValueError(f"duration_ticks must be in [1, 1000], got {v}")
        return v


class InjectResponse(BaseModel):
    """Response body for POST /api/inject."""

    status: str  # "accepted"
    fault_type: str
    channels: list[str]  # effective channel list (expanded from [] to all loaded)
    magnitude_sigma: float
    duration_ticks: int


class ErrorResponse(BaseModel):
    """RFC 7807 Problem Details error response."""

    detail: str
    correlation_id: str | None = None
