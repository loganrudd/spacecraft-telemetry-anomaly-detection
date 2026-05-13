"""Tests for GET /health endpoint."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


class TestHealthOk:
    def test_returns_200(self, running_app: FastAPI) -> None:
        resp = TestClient(running_app).get("/health")
        assert resp.status_code == 200

    def test_status_field_ok(self, running_app: FastAPI) -> None:
        body = TestClient(running_app).get("/health").json()
        assert body["status"] == "ok"

    def test_channels_loaded_non_empty(self, running_app: FastAPI) -> None:
        body = TestClient(running_app).get("/health").json()
        assert len(body["channels_loaded"]) >= 1

    def test_mission_field_present(self, running_app: FastAPI) -> None:
        body = TestClient(running_app).get("/health").json()
        assert "mission" in body

    def test_uptime_non_negative(self, running_app: FastAPI) -> None:
        body = TestClient(running_app).get("/health").json()
        assert body["uptime_s"] >= 0.0

    def test_mlflow_uri_present(self, running_app: FastAPI) -> None:
        body = TestClient(running_app).get("/health").json()
        assert "mlflow_tracking_uri" in body


class TestHealthDegraded:
    def test_returns_503_when_no_engines(self, running_app: FastAPI) -> None:
        """Clear all engines — health should return 503."""
        running_app.state.app_state.engines.clear()
        resp = TestClient(running_app).get("/health")
        assert resp.status_code == 503

    def test_503_body_has_degraded_status(self, running_app: FastAPI) -> None:
        running_app.state.app_state.engines.clear()
        body = TestClient(running_app).get("/health").json()
        assert body["status"] == "degraded"


class TestHealthLifespanFailure:
    def test_lifespan_raises_when_registry_empty(self) -> None:
        """create_app lifespan raises RuntimeError when no channels load from registry."""
        from spacecraft_telemetry.api.app import create_app
        from spacecraft_telemetry.core.config import load_settings

        settings = load_settings("test")
        app = create_app(settings)
        with pytest.raises(RuntimeError), TestClient(app, raise_server_exceptions=True):
            pass  # pragma: no cover
