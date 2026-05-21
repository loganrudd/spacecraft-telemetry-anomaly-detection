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
    def test_lifespan_raises_when_no_channels(self, tmp_path) -> None:
        """create_app() instantiation is fine; lifespan raises on empty registry."""
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


# ---------------------------------------------------------------------------
# Lifespan branch coverage via stubs
# ---------------------------------------------------------------------------

_STUB_MAP = {"ch-a": "sub1", "ch-b": "sub2", "ch-c": "sub1"}


@pytest.fixture()
def _lifespan_patches(mocker):
    """Patch all external I/O in the lifespan to boot with no real deps."""
    import numpy as np
    import torch

    from spacecraft_telemetry.model.io import ScoringParams

    class _Zero(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.zeros(x.shape[0], 1)

    stub_model = _Zero()
    stub_model.eval()
    stub_params = ScoringParams(
        threshold_window=5,
        threshold_z=2.0,
        error_smoothing_window=5,
        threshold_min_anomaly_len=2,
    )
    N = 20
    stub_replay = (
        np.zeros(N, dtype=np.float32),   # values
        np.zeros(N, dtype=np.int32),     # segment_ids
        np.zeros(N, dtype=bool),         # is_anomaly
        np.empty(N, dtype=object),       # timestamps
    )

    mocker.patch("spacecraft_telemetry.api.app.configure_mlflow")
    mocker.patch(
        "spacecraft_telemetry.api.app.load_model_for_scoring",
        return_value=(stub_model, 10),
    )
    mocker.patch(
        "spacecraft_telemetry.api.app.load_scoring_params",
        return_value=stub_params,
    )
    mocker.patch(
        "spacecraft_telemetry.api.app.load_series_parquet",
        return_value=stub_replay,
    )


class TestLifespanBranchCoverage:
    def test_explicit_channels_override_subsystem(self, mocker, _lifespan_patches) -> None:
        """When api.channels is set, only those channels are loaded."""
        mocker.patch(
            "spacecraft_telemetry.api.app.load_channel_subsystem_map",
            return_value=_STUB_MAP,
        )
        settings = load_settings("test")
        settings = settings.model_copy(
            update={"api": settings.api.model_copy(
                update={"channels": ["ch-a", "ch-b"], "subsystem": "sub1"}
            )}
        )
        app = create_app(settings)
        with TestClient(app):
            assert set(app.state.app_state.engines.keys()) == {"ch-a", "ch-b"}

    def test_subsystem_filter_selects_matching_channels(self, mocker, _lifespan_patches) -> None:
        """With no explicit channels, lifespan filters by configured subsystem."""
        mocker.patch(
            "spacecraft_telemetry.api.app.load_channel_subsystem_map",
            return_value=_STUB_MAP,
        )
        settings = load_settings("test")
        settings = settings.model_copy(
            update={"api": settings.api.model_copy(
                update={"channels": [], "subsystem": "sub1"}
            )}
        )
        app = create_app(settings)
        with TestClient(app):
            assert set(app.state.app_state.engines.keys()) == {"ch-a", "ch-c"}

    def test_per_channel_error_is_skipped(self, mocker, _lifespan_patches) -> None:
        """A channel that fails to load is warned+skipped; others still boot."""
        import torch


        mocker.patch(
            "spacecraft_telemetry.api.app.load_channel_subsystem_map",
            return_value={"ch-good": "sub1", "ch-bad": "sub1"},
        )

        class _Zero(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.zeros(x.shape[0], 1)

        good_model = _Zero()
        good_model.eval()

        def _raise_on_bad(name: str, device: object, tracking_uri: str) -> object:
            if "ch-bad" in name:
                raise RuntimeError("simulated load failure")
            return (good_model, 10)

        mocker.patch(
            "spacecraft_telemetry.api.app.load_model_for_scoring",
            side_effect=_raise_on_bad,
        )
        settings = load_settings("test")
        settings = settings.model_copy(
            update={"api": settings.api.model_copy(
                update={"channels": [], "subsystem": "sub1"}
            )}
        )
        app = create_app(settings)
        with TestClient(app):
            loaded = set(app.state.app_state.engines.keys())
        assert "ch-good" in loaded
        assert "ch-bad" not in loaded
