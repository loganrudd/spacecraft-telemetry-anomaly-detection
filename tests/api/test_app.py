"""Tests for api.app and api.logging_middleware."""

from __future__ import annotations

import time
import uuid

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from spacecraft_telemetry.api.app import create_app
from spacecraft_telemetry.api.logging_middleware import CorrelationIdMiddleware
from spacecraft_telemetry.core.config import load_settings

# ---------------------------------------------------------------------------
# Helpers for deferred background loading
# ---------------------------------------------------------------------------

def _wait_for_ready(app, timeout: float = 5.0) -> None:
    """Block until the background load task sets app.state.app_state."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if getattr(app.state, "app_state", None) is not None:
            return
        time.sleep(0.02)
    loading = getattr(app.state, "loading_state", None)
    if loading and loading.error:
        raise RuntimeError(f"loading failed: {loading.error}")
    raise TimeoutError("app_state not set within timeout")


def _wait_for_loading_done(app, timeout: float = 5.0) -> None:
    """Block until the background load task marks is_complete (success or failure)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        loading = getattr(app.state, "loading_state", None)
        if loading and loading.is_complete:
            return
        time.sleep(0.02)
    raise TimeoutError("loading_state.is_complete not set within timeout")

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

    def test_no_static_mount_when_static_dir_unset(self) -> None:
        """Default test config has no static_dir → no dashboard mount."""
        settings = load_settings("test")
        assert settings.api.static_dir is None
        app = create_app(settings)
        assert not any(getattr(r, "name", None) == "dashboard" for r in app.routes)

    def test_static_mount_serves_index_when_dir_exists(self, tmp_path) -> None:
        """A real static_dir gets mounted at / and serves index.html."""
        (tmp_path / "index.html").write_text("<!doctype html><title>dash</title>")
        settings = load_settings("test").model_copy(
            update={
                "api": load_settings("test").api.model_copy(
                    update={"static_dir": str(tmp_path)}
                )
            }
        )
        app = create_app(settings)
        # Mount registered with the documented name.
        assert any(getattr(r, "name", None) == "dashboard" for r in app.routes)

    def test_static_dir_missing_does_not_raise(self, tmp_path) -> None:
        """A configured-but-missing static_dir logs a warning, does not crash."""
        settings = load_settings("test").model_copy(
            update={
                "api": load_settings("test").api.model_copy(
                    update={"static_dir": str(tmp_path / "does-not-exist")}
                )
            }
        )
        app = create_app(settings)
        assert not any(getattr(r, "name", None) == "dashboard" for r in app.routes)


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
    def test_no_channels_results_in_degraded_health(self, tmp_path) -> None:
        """When no channel models are registered, background load records an error
        and /health returns 503 degraded (lifespan no longer raises)."""
        settings = load_settings("test")
        # Fresh empty DB — hermetic; no models registered.
        settings = settings.model_copy(
            update={
                "mlflow": settings.mlflow.model_copy(
                    update={"tracking_uri": f"sqlite:///{tmp_path}/empty.db"}
                )
            }
        )
        app = create_app(settings)
        with TestClient(app, raise_server_exceptions=False) as client:
            _wait_for_loading_done(app)
            assert app.state.loading_state.error is not None
            resp = client.get("/health")
        assert resp.status_code == 503
        assert resp.json()["status"] == "degraded"


# ---------------------------------------------------------------------------
# Lifespan branch coverage via stubs
# ---------------------------------------------------------------------------

_STUB_MAP = {"ch-a": "sub1", "ch-b": "sub2", "ch-c": "sub1"}


@pytest.fixture()
def _lifespan_patches(mocker):
    """Patch all external I/O in the lifespan to boot with no real deps."""
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

    mocker.patch("spacecraft_telemetry.api.app.configure_mlflow")
    mocker.patch(
        "spacecraft_telemetry.api.app.load_model_for_scoring",
        return_value=(stub_model, 10),
    )
    mocker.patch(
        "spacecraft_telemetry.api.app.load_scoring_params",
        return_value=stub_params,
    )
    # Replay series are no longer loaded in the lifespan (lazy per-stream now), so
    # there is nothing to patch for it here.
    # Default: behave as if the champion registry is unreachable, so resolution
    # falls back to the full channel list (the behaviour these branch tests were
    # written against). Tests exercising champion-gating override this return value.
    mocker.patch(
        "spacecraft_telemetry.api.app._resolve_champion_channels",
        return_value=None,
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
                update={"channels": ["ch-a", "ch-b"], "subsystems": ["sub1"]}
            )}
        )
        app = create_app(settings)
        with TestClient(app):
            _wait_for_ready(app)
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
                update={"channels": [], "subsystems": ["sub1"]}
            )}
        )
        app = create_app(settings)
        with TestClient(app):
            _wait_for_ready(app)
            assert set(app.state.app_state.engines.keys()) == {"ch-a", "ch-c"}

    def test_per_channel_error_is_skipped(self, mocker, _lifespan_patches) -> None:
        """A channel that fails to load is skipped; the service boots with the rest."""
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
                update={"channels": [], "subsystems": ["sub1"]}
            )}
        )
        app = create_app(settings)
        with TestClient(app) as client:
            _wait_for_ready(app)
            loaded = set(app.state.app_state.engines.keys())
            resp = client.get("/health")
        assert "ch-good" in loaded
        assert "ch-bad" not in loaded
        # Partial load must surface as degraded, not ok.
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "degraded"
        assert body["missing"] == ["ch-bad"]

    def test_whole_mission_resolves_from_champions(self, mocker, _lifespan_patches) -> None:
        """Whole-mission load list = the promoted-champion set, so channels_total
        counts only servable models and the frontend's ready==total gate completes
        (regression for the '62 / 76' stuck-loading bug)."""
        mocker.patch(
            "spacecraft_telemetry.api.app.load_channel_subsystem_map",
            return_value={"ch-a": "sub1", "ch-b": "sub1", "ch-unpromoted": "sub1"},
        )
        mocker.patch(
            "spacecraft_telemetry.api.app._resolve_champion_channels",
            return_value={"ch-a", "ch-b"},  # ch-unpromoted has no @champion
        )
        settings = load_settings("test")
        settings = settings.model_copy(
            update={"api": settings.api.model_copy(
                update={"channels": [], "subsystems": None}
            )}
        )
        app = create_app(settings)
        with TestClient(app):
            _wait_for_ready(app)
            assert set(app.state.app_state.engines.keys()) == {"ch-a", "ch-b"}
            assert app.state.loading_state.channels_total == 2
            assert app.state.loading_state.channels_ready == 2

    def test_explicit_channels_gated_by_champion(self, mocker, _lifespan_patches) -> None:
        """An explicitly requested channel with no @champion model is dropped, so
        channels_total stays equal to what can actually load."""
        mocker.patch(
            "spacecraft_telemetry.api.app.load_channel_subsystem_map",
            return_value=_STUB_MAP,
        )
        mocker.patch(
            "spacecraft_telemetry.api.app._resolve_champion_channels",
            return_value={"ch-a"},  # ch-b requested below but not promoted
        )
        settings = load_settings("test")
        settings = settings.model_copy(
            update={"api": settings.api.model_copy(
                update={"channels": ["ch-a", "ch-b"], "subsystems": None}
            )}
        )
        app = create_app(settings)
        with TestClient(app):
            _wait_for_ready(app)
            assert set(app.state.app_state.engines.keys()) == {"ch-a"}
            assert app.state.loading_state.channels_total == 1

    def test_registry_unavailable_falls_back_to_full_list(self, mocker, _lifespan_patches) -> None:
        """When the champion query fails (None), resolution falls back to the full
        channel list; the loader's require_champion gate is the serving backstop."""
        mocker.patch(
            "spacecraft_telemetry.api.app.load_channel_subsystem_map",
            return_value={"ch-a": "sub1", "ch-b": "sub1"},
        )
        mocker.patch(
            "spacecraft_telemetry.api.app._resolve_champion_channels",
            return_value=None,
        )
        settings = load_settings("test")
        settings = settings.model_copy(
            update={"api": settings.api.model_copy(
                update={"channels": [], "subsystems": None}
            )}
        )
        app = create_app(settings)
        with TestClient(app):
            _wait_for_ready(app)
            assert app.state.loading_state.channels_total == 2  # both attempted


class TestDriftMonitorPriming:
    """Lifespan must build drift_monitors from drift_references and prime them
    from the replay-slice tail, so the rolling window is already full on the
    first live/replay tick instead of taking window_size ticks to fill."""

    def test_drift_monitors_built_and_primed(self, mocker, _lifespan_patches) -> None:
        import numpy as np
        import pandas as pd

        from spacecraft_telemetry.evidently_monitoring.reference import (
            MONITORING_FEATURE_COLS,
        )

        mocker.patch(
            "spacecraft_telemetry.api.app.load_channel_subsystem_map",
            return_value=_STUB_MAP,
        )

        ref = pd.DataFrame({col: np.zeros(50) for col in MONITORING_FEATURE_COLS})

        async def _fake_drift_ref(ch, settings, sem, log):
            return ch, ref

        n = 200
        values = np.arange(n, dtype=np.float64)
        anom = np.zeros(n, dtype=bool)
        timestamps = pd.date_range("2020-01-01", periods=n, freq="30s").to_numpy()

        async def _fake_replay_slice(ch, settings, sem, log):
            return ch, (values, anom, timestamps)

        mocker.patch(
            "spacecraft_telemetry.api.app._load_drift_ref", side_effect=_fake_drift_ref
        )
        mocker.patch(
            "spacecraft_telemetry.api.app._load_replay_slice",
            side_effect=_fake_replay_slice,
        )

        settings = load_settings("test")
        settings = settings.model_copy(
            update={
                "api": settings.api.model_copy(
                    update={"channels": ["ch-a", "ch-b"], "subsystems": ["sub1"]}
                ),
                "drift": settings.drift.model_copy(update={"enabled": True}),
            }
        )
        app = create_app(settings)
        with TestClient(app):
            _wait_for_ready(app)
            state = app.state.app_state
            assert set(state.drift_monitors.keys()) == {"ch-a", "ch-b"}
            for monitor in state.drift_monitors.values():
                # Primed from the replay-slice tail -- window is already full,
                # not empty as it would be starting cold.
                assert len(monitor._window) == settings.drift.window_size

    def test_drift_disabled_leaves_monitors_empty(self, mocker, _lifespan_patches) -> None:
        mocker.patch(
            "spacecraft_telemetry.api.app.load_channel_subsystem_map",
            return_value=_STUB_MAP,
        )
        settings = load_settings("test")
        settings = settings.model_copy(
            update={"api": settings.api.model_copy(
                update={"channels": ["ch-a"], "subsystems": None}
            )}
        )
        app = create_app(settings)
        with TestClient(app):
            _wait_for_ready(app)
            assert app.state.app_state.drift_monitors == {}


class TestResolveChampionChannels:
    """Unit tests for the registry query that backs champion resolution."""

    def test_returns_only_channels_with_champion_alias(self, mocker) -> None:
        from types import SimpleNamespace

        from spacecraft_telemetry.api.app import _resolve_champion_channels
        from spacecraft_telemetry.mlflow_tracking.conventions import registered_model_name

        settings = load_settings("test")
        prefix = registered_model_name(settings.model.model_type, "ESA-Mission1", "")
        fake_client = mocker.Mock()
        fake_client.search_registered_models.return_value = [
            SimpleNamespace(name=f"{prefix}ch-a", aliases={"champion": "3"}),
            SimpleNamespace(name=f"{prefix}ch-b", aliases={}),  # registered, not promoted
            SimpleNamespace(name=f"{prefix}ch-c", aliases={"champion": "1", "candidate": "2"}),
        ]
        mocker.patch("spacecraft_telemetry.api.app.MlflowClient", return_value=fake_client)

        result = _resolve_champion_channels(settings, "ESA-Mission1", mocker.Mock())
        assert result == {"ch-a", "ch-c"}

    def test_returns_none_when_registry_query_fails(self, mocker) -> None:
        from spacecraft_telemetry.api.app import _resolve_champion_channels

        fake_client = mocker.Mock()
        fake_client.search_registered_models.side_effect = RuntimeError("mlflow down")
        mocker.patch("spacecraft_telemetry.api.app.MlflowClient", return_value=fake_client)

        settings = load_settings("test")
        assert _resolve_champion_channels(settings, "ESA-Mission1", mocker.Mock()) is None
