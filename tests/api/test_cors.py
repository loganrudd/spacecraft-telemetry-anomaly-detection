"""Tests for CORS middleware configuration.

CORS is config-gated via ApiConfig.cors_allowed_origins:
  - Empty list  → no CORSMiddleware added; no CORS headers on any response.
  - Non-empty   → CORSMiddleware added; allowed origins receive headers.

Tests use minimal FastAPI apps (no lifespan, no router) — CORS is middleware-only
and doesn't depend on route handlers.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

_ALLOWED_ORIGIN = "http://localhost:5173"
_DISALLOWED_ORIGIN = "http://evil.example.com"

# Headers required to trigger a CORS preflight response.
_PREFLIGHT_HEADERS = {
    "Origin": _ALLOWED_ORIGIN,
    "Access-Control-Request-Method": "GET",
    "Access-Control-Request-Headers": "Content-Type",
}


def _app_with_origins(origins: list[str]) -> FastAPI:
    """Create a minimal app with the given CORS origins (no lifespan)."""
    app = FastAPI()
    if origins:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=False,
            allow_methods=["GET"],
            allow_headers=["*"],
            expose_headers=["X-Correlation-Id"],
        )

    @app.get("/ping")
    async def ping() -> dict[str, str]:
        return {"ok": "true"}

    return app


class TestCorsEnabled:
    """CORS middleware is active when cors_allowed_origins is non-empty."""

    @pytest.fixture()
    def client(self) -> TestClient:
        return TestClient(_app_with_origins([_ALLOWED_ORIGIN]))

    def test_preflight_returns_200(self, client: TestClient) -> None:
        resp = client.options("/ping", headers=_PREFLIGHT_HEADERS)
        assert resp.status_code == 200

    def test_preflight_includes_allow_origin(self, client: TestClient) -> None:
        resp = client.options("/ping", headers=_PREFLIGHT_HEADERS)
        assert resp.headers.get("access-control-allow-origin") == _ALLOWED_ORIGIN

    def test_get_with_allowed_origin_includes_header(self, client: TestClient) -> None:
        resp = client.get("/ping", headers={"Origin": _ALLOWED_ORIGIN})
        assert resp.headers.get("access-control-allow-origin") == _ALLOWED_ORIGIN

    def test_disallowed_origin_not_reflected(self, client: TestClient) -> None:
        resp = client.options(
            "/ping",
            headers={
                "Origin": _DISALLOWED_ORIGIN,
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "Content-Type",
            },
        )
        allow = resp.headers.get("access-control-allow-origin", "")
        assert _DISALLOWED_ORIGIN not in allow


class TestCorsDisabled:
    """No CORS headers present when cors_allowed_origins is empty."""

    @pytest.fixture()
    def client(self) -> TestClient:
        return TestClient(_app_with_origins([]))

    def test_get_has_no_cors_header(self, client: TestClient) -> None:
        resp = client.get("/ping", headers={"Origin": _ALLOWED_ORIGIN})
        assert "access-control-allow-origin" not in resp.headers

    def test_options_has_no_cors_header(self, client: TestClient) -> None:
        resp = client.options("/ping", headers=_PREFLIGHT_HEADERS)
        assert "access-control-allow-origin" not in resp.headers
