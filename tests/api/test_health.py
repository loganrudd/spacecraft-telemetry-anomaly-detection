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

    def test_channel_subsystems_present(self, running_app: FastAPI) -> None:
        body = TestClient(running_app).get("/health").json()
        assert "channel_subsystems" in body

    def test_channel_subsystems_maps_loaded_channels(self, running_app: FastAPI) -> None:
        body = TestClient(running_app).get("/health").json()
        # Every loaded channel must appear in channel_subsystems.
        for ch in body["channels_loaded"]:
            assert ch in body["channel_subsystems"]

    def test_subsystem_field_present(self, running_app: FastAPI) -> None:
        body = TestClient(running_app).get("/health").json()
        # subsystem may be a string or null (whole-mission mode).
        assert "subsystem" in body


class TestHealthDegraded:
    def test_returns_503_when_no_engines(self, running_app_empty: FastAPI) -> None:
        """App with no loaded engines — health should return 503."""
        resp = TestClient(running_app_empty).get("/health")
        assert resp.status_code == 503

    def test_503_body_has_degraded_status(self, running_app_empty: FastAPI) -> None:
        body = TestClient(running_app_empty).get("/health").json()
        assert body["status"] == "degraded"


class TestHealthLifespanFailure:
    def test_lifespan_raises_when_registry_empty(self, tmp_path) -> None:
        """create_app lifespan raises RuntimeError when no channels load from registry."""
        from spacecraft_telemetry.api.app import create_app
        from spacecraft_telemetry.core.config import load_settings

        settings = load_settings("test")
        # Use a fresh empty database — mlflow.test.db may have accumulated
        # models from previous training runs and is not hermetic for this test.
        settings = settings.model_copy(
            update={
                "mlflow": settings.mlflow.model_copy(
                    update={"tracking_uri": f"sqlite:///{tmp_path}/empty.db"}
                )
            }
        )
        app = create_app(settings)
        with pytest.raises(RuntimeError), TestClient(app, raise_server_exceptions=True):
            pass  # pragma: no cover
