"""Pydantic v2 request/response models for the FastAPI serving layer."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, field_validator


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
    subsystem: str | None = None
    channels_loaded: list[str] = []
    channels_total: int = 0    # target channel count (set once channels are resolved)
    channels_ready: int = 0    # successfully loaded so far (== len(channels_loaded) when ok)
    channel_subsystems: dict[str, str] = {}
    uptime_s: float = 0.0
    mlflow_tracking_uri: str = ""


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


class ErrorResponse(BaseModel):
    """RFC 7807 Problem Details error response."""

    detail: str
    correlation_id: str | None = None
