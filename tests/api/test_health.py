"""Tests for GET /health endpoint."""

from __future__ import annotations

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
        assert "subsystems" in body


class TestHealthDegraded:
    def test_returns_503_when_no_engines(self, running_app_empty: FastAPI) -> None:
        """App with no loaded engines — health should return 503."""
        resp = TestClient(running_app_empty).get("/health")
        assert resp.status_code == 503

    def test_503_body_has_degraded_status(self, running_app_empty: FastAPI) -> None:
        body = TestClient(running_app_empty).get("/health").json()
        assert body["status"] == "degraded"


class TestHealthLifespanFailure:
    def test_no_channels_gives_degraded_health(self, tmp_path) -> None:
        """When registry is empty, background loading records an error and
        /health returns 503 degraded (lifespan no longer raises)."""
        import time

        from spacecraft_telemetry.api.app import create_app
        from spacecraft_telemetry.core.config import load_settings

        settings = load_settings("test")
        settings = settings.model_copy(
            update={
                "mlflow": settings.mlflow.model_copy(
                    update={"tracking_uri": f"sqlite:///{tmp_path}/empty.db"}
                )
            }
        )
        app = create_app(settings)
        with TestClient(app, raise_server_exceptions=False) as client:
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                loading = getattr(app.state, "loading_state", None)
                if loading and loading.is_complete:
                    break
                time.sleep(0.02)
            resp = client.get("/health")
        assert resp.status_code == 503
        assert resp.json()["status"] == "degraded"


class TestHealthAvailableMissions:
    def test_empty_by_default(self, running_app: FastAPI) -> None:
        body = TestClient(running_app).get("/health").json()
        assert body["available_missions"] == []

    def test_echoes_configured_missions(self, tmp_path, test_settings) -> None:
        import time
        from types import MappingProxyType

        from spacecraft_telemetry.api.endpoints import router
        from spacecraft_telemetry.api.logging_middleware import CorrelationIdMiddleware
        from spacecraft_telemetry.api.state import AppState
        from spacecraft_telemetry.core.config import MissionLink

        missions = [
            MissionLink(id="ESA-Mission1", label="ESA Mission 1", url="http://localhost:8000"),
            MissionLink(id="ISS", label="ISS Live", url="http://localhost:8001"),
        ]
        settings = test_settings.model_copy(
            update={"api": test_settings.api.model_copy(update={"available_missions": missions})}
        )
        app = FastAPI()
        app.state.settings = settings
        app.state.app_state = AppState(
            settings=settings,
            mission="test-mission",
            subsystems=None,
            device=__import__("torch").device("cpu"),
            engines=MappingProxyType({}),
            channel_subsystem_map=MappingProxyType({}),
            replay_data=MappingProxyType({}),
            startup_monotonic_ns=time.monotonic_ns(),
            mlflow_tracking_uri=settings.mlflow.tracking_uri,
        )
        app.add_middleware(CorrelationIdMiddleware)
        app.include_router(router)

        body = TestClient(app).get("/health").json()
        returned = body["available_missions"]
        assert len(returned) == 2
        assert returned[0]["id"] == "ESA-Mission1"
        assert returned[1]["id"] == "ISS"
        assert returned[1]["url"] == "http://localhost:8001"
