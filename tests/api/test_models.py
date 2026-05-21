"""Tests for api.models — Pydantic round-trips and field validators."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from spacecraft_telemetry.api.models import (
    DriftEvent,
    DriftFeature,
    ErrorResponse,
    HealthResponse,
    StreamQueryParams,
    TelemetryEvent,
)

_TS = datetime(2000, 1, 1, tzinfo=UTC)


# ---------------------------------------------------------------------------
# TelemetryEvent
# ---------------------------------------------------------------------------


class TestTelemetryEvent:
    def _base(self, **overrides: object) -> dict[str, object]:
        data: dict[str, object] = {
            "timestamp": _TS,
            "mission": "ESA-Mission1",
            "channel": "A-1",
            "value_normalized": 0.5,
            "prediction": 0.4,
            "residual": 0.1,
            "smoothed_error": 0.08,
            "threshold": 0.2,
            "is_anomaly_predicted": False,
            "is_anomaly": False,
        }
        data.update(overrides)
        return data

    def test_round_trip(self) -> None:
        event = TelemetryEvent(**self._base())  # type: ignore[arg-type]
        assert event.mission == "ESA-Mission1"
        assert event.channel == "A-1"
        assert event.value_normalized == pytest.approx(0.5)

    def test_nullable_fields_accept_none(self) -> None:
        event = TelemetryEvent(
            **self._base(prediction=None, residual=None, smoothed_error=None, threshold=None)  # type: ignore[arg-type]
        )
        assert event.prediction is None
        assert event.residual is None
        assert event.smoothed_error is None
        assert event.threshold is None

    def test_json_roundtrip(self) -> None:
        event = TelemetryEvent(**self._base())  # type: ignore[arg-type]
        reloaded = TelemetryEvent.model_validate_json(event.model_dump_json())
        assert reloaded.mission == event.mission
        assert reloaded.channel == event.channel
        assert reloaded.timestamp == event.timestamp

    def test_bool_fields(self) -> None:
        event = TelemetryEvent(**self._base(is_anomaly_predicted=True, is_anomaly=True))  # type: ignore[arg-type]
        assert event.is_anomaly_predicted is True
        assert event.is_anomaly is True


# ---------------------------------------------------------------------------
# HealthResponse
# ---------------------------------------------------------------------------


class TestHealthResponse:
    def test_ok_status(self) -> None:
        resp = HealthResponse(
            status="ok",
            mission="ESA-Mission1",
            subsystem="subsystem_6",
            channels_loaded=["A-1", "A-2"],
            channel_subsystems={"A-1": "subsystem_6", "A-2": "subsystem_6"},
            uptime_s=42.0,
            mlflow_tracking_uri="sqlite:///mlflow.db",
        )
        assert resp.status == "ok"
        assert resp.channels_loaded == ["A-1", "A-2"]
        assert resp.uptime_s == pytest.approx(42.0)

    def test_degraded_status(self) -> None:
        resp = HealthResponse(
            status="degraded",
            mission="ESA-Mission1",
            subsystem="subsystem_6",
            channels_loaded=[],
            channel_subsystems={},
            uptime_s=0.0,
            mlflow_tracking_uri="sqlite:///mlflow.db",
        )
        assert resp.status == "degraded"
        assert resp.channels_loaded == []

    def test_invalid_status_rejected(self) -> None:
        with pytest.raises(ValidationError):
            HealthResponse(
                status="unknown",  # type: ignore[arg-type]
                mission="ESA-Mission1",
                subsystem="subsystem_6",
                channels_loaded=[],
                uptime_s=0.0,
                mlflow_tracking_uri="sqlite:///mlflow.db",
            )


# ---------------------------------------------------------------------------
# StreamQueryParams
# ---------------------------------------------------------------------------


class TestStreamQueryParams:
    def test_defaults(self) -> None:
        params = StreamQueryParams()
        assert params.speed is None
        assert params.channels is None

    def test_valid_speed(self) -> None:
        params = StreamQueryParams(speed=5.0)
        assert params.speed == pytest.approx(5.0)

    def test_speed_zero_rejected(self) -> None:
        with pytest.raises(ValidationError, match="speed must be > 0"):
            StreamQueryParams(speed=0.0)

    def test_negative_speed_rejected(self) -> None:
        with pytest.raises(ValidationError, match="speed must be > 0"):
            StreamQueryParams(speed=-1.0)

    def test_none_speed_allowed(self) -> None:
        params = StreamQueryParams(speed=None)
        assert params.speed is None

    def test_channels_csv_stored(self) -> None:
        params = StreamQueryParams(channels="A-1,A-2,A-3")
        assert params.channels == "A-1,A-2,A-3"

    def test_channels_none(self) -> None:
        params = StreamQueryParams(channels=None)
        assert params.channels is None


# ---------------------------------------------------------------------------
# ErrorResponse
# ---------------------------------------------------------------------------


class TestErrorResponse:
    def test_minimal(self) -> None:
        err = ErrorResponse(detail="Something went wrong")
        assert err.detail == "Something went wrong"
        assert err.correlation_id is None

    def test_with_correlation_id(self) -> None:
        err = ErrorResponse(detail="Not found", correlation_id="abc-123")
        assert err.correlation_id == "abc-123"

    def test_json_roundtrip(self) -> None:
        err = ErrorResponse(detail="oops", correlation_id="xyz")
        reloaded = ErrorResponse.model_validate_json(err.model_dump_json())
        assert reloaded.detail == err.detail
        assert reloaded.correlation_id == err.correlation_id


# ---------------------------------------------------------------------------
# DriftFeature + DriftEvent
# ---------------------------------------------------------------------------


_FEATURE = DriftFeature(feature="value_normalized", score=0.12, drifted=True)


class TestDriftFeature:
    def test_fields(self) -> None:
        assert _FEATURE.feature == "value_normalized"
        assert _FEATURE.score == pytest.approx(0.12)
        assert _FEATURE.drifted is True


class TestDriftEvent:
    def _base(self, **overrides: object) -> dict[str, object]:
        data: dict[str, object] = {
            "timestamp": _TS,
            "mission": "ESA-Mission1",
            "channel": "channel_1",
            "features": [{"feature": "value_normalized", "score": 0.12, "drifted": True}],
            "percent_drifted": 0.5,
            "drifted": True,
        }
        data.update(overrides)
        return data

    def test_round_trip(self) -> None:
        event = DriftEvent(**self._base())  # type: ignore[arg-type]
        assert event.channel == "channel_1"
        assert event.drifted is True
        assert event.percent_drifted == pytest.approx(0.5)
        assert event.subsystem_percent_drifted is None
        assert event.subsystem_alert is None

    def test_optional_subsystem_fields(self) -> None:
        event = DriftEvent(
            **self._base(subsystem_percent_drifted=0.35, subsystem_alert=True)  # type: ignore[arg-type]
        )
        assert event.subsystem_percent_drifted == pytest.approx(0.35)
        assert event.subsystem_alert is True

    def test_percent_drifted_zero(self) -> None:
        event = DriftEvent(**self._base(percent_drifted=0.0, drifted=False))  # type: ignore[arg-type]
        assert event.percent_drifted == pytest.approx(0.0)

    def test_percent_drifted_one(self) -> None:
        event = DriftEvent(**self._base(percent_drifted=1.0))  # type: ignore[arg-type]
        assert event.percent_drifted == pytest.approx(1.0)

    def test_percent_drifted_out_of_range_rejected(self) -> None:
        with pytest.raises(Exception, match="percent_drifted"):
            DriftEvent(**self._base(percent_drifted=1.1))  # type: ignore[arg-type]
        with pytest.raises(Exception, match="percent_drifted"):
            DriftEvent(**self._base(percent_drifted=-0.1))  # type: ignore[arg-type]

    def test_json_roundtrip(self) -> None:
        event = DriftEvent(**self._base(subsystem_percent_drifted=0.2))  # type: ignore[arg-type]
        reloaded = DriftEvent.model_validate_json(event.model_dump_json())
        assert reloaded.channel == event.channel
        assert reloaded.percent_drifted == pytest.approx(event.percent_drifted)
        assert reloaded.subsystem_percent_drifted == pytest.approx(0.2)  # type: ignore[arg-type]
        assert reloaded.features[0].feature == "value_normalized"
