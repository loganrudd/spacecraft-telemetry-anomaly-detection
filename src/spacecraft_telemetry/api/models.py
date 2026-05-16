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
    is_anomaly_true: bool


class HealthResponse(BaseModel):
    """Response model for GET /health."""

    status: Literal["ok", "degraded"]
    mission: str
    subsystem: str
    channels_loaded: list[str]
    uptime_s: float
    mlflow_tracking_uri: str


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


class ErrorResponse(BaseModel):
    """RFC 7807 Problem Details error response."""

    detail: str
    correlation_id: str | None = None
