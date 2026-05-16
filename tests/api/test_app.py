"""Tests for api.app and api.logging_middleware."""

from __future__ import annotations

import uuid

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from spacecraft_telemetry.api.app import create_app
from spacecraft_telemetry.api.logging_middleware import CorrelationIdMiddleware
from spacecraft_telemetry.core.config import load_settings

# ---------------------------------------------------------------------------
# create_app
# ---------------------------------------------------------------------------


class TestCreateApp:
    def test_returns_fastapi_instance(self) -> None:
        settings = load_settings("test")
        app = create_app(settings)
        assert isinstance(app, FastAPI)

    def test_app_has_settings_attached(self) -> None:
        settings = load_settings("test")
        app = create_app(settings)
        assert app.state.settings is settings

    def test_app_has_title(self) -> None:
        settings = load_settings("test")
        app = create_app(settings)
        assert "Telemetry" in app.title

    def test_middleware_registered(self) -> None:
        settings = load_settings("test")
        app = create_app(settings)
        middleware_classes = [m.cls for m in app.user_middleware]
        assert CorrelationIdMiddleware in middleware_classes


# ---------------------------------------------------------------------------
# CorrelationIdMiddleware
# ---------------------------------------------------------------------------


def _make_echo_app() -> FastAPI:
    """Minimal FastAPI app with CorrelationIdMiddleware for middleware tests."""
    echo = FastAPI()
    echo.add_middleware(CorrelationIdMiddleware)

    @echo.get("/ping")
    async def ping() -> dict[str, str]:
        return {"status": "ok"}

    @echo.get("/error")
    async def error() -> None:
        raise ValueError("test error")

    return echo


class TestCorrelationIdMiddleware:
    @pytest.fixture()
    def client(self) -> TestClient:
        return TestClient(_make_echo_app(), raise_server_exceptions=False)

    def test_adds_correlation_id_header(self, client: TestClient) -> None:
        resp = client.get("/ping")
        assert resp.status_code == 200
        assert "X-Correlation-Id" in resp.headers

    def test_correlation_id_is_uuid4(self, client: TestClient) -> None:
        resp = client.get("/ping")
        value = resp.headers["X-Correlation-Id"]
        # Should parse as a valid UUID without raising.
        parsed = uuid.UUID(value)
        assert parsed.version == 4

    def test_forwards_incoming_correlation_id(self, client: TestClient) -> None:
        custom_id = "my-trace-id-abc-123"
        resp = client.get("/ping", headers={"X-Correlation-Id": custom_id})
        assert resp.headers["X-Correlation-Id"] == custom_id

    def test_different_requests_get_different_ids(self, client: TestClient) -> None:
        resp1 = client.get("/ping")
        resp2 = client.get("/ping")
        assert resp1.headers["X-Correlation-Id"] != resp2.headers["X-Correlation-Id"]

    def test_server_error_still_completes(self, client: TestClient) -> None:
        """Middleware should not suppress or swallow unhandled exceptions."""
        resp = client.get("/error")
        assert resp.status_code == 500


# ---------------------------------------------------------------------------
# Lifespan failure path (no MLflow registry)
# ---------------------------------------------------------------------------


class TestLifespanFailure:
    def test_lifespan_raises_when_no_channels(self) -> None:
        """create_app() instantiation is fine; lifespan raises on empty registry."""
        settings = load_settings("test")
        app = create_app(settings)
        # Entering the lifespan context (TestClient startup) should raise because
        # there are no trained + scored channels in the test MLflow registry.
        with pytest.raises(RuntimeError), TestClient(app, raise_server_exceptions=True):
            pass  # pragma: no cover
