"""Tests for mlflow_tracking/registry.py.

latest_uri is a pure function — tested directly.

register_pytorch_model is tested with mock-based unit tests: mlflow.pytorch.log_model
is mocked (no PyTorch needed) and we assert it is called with the correct args.
MLflow's own log_model integration is not our responsibility to test here.

promote is tested against a real SQLite store.  We create model versions via
MlflowClient.create_model_version (which stores the source string directly and
does not validate the artifact path) rather than mlflow.register_model (which
in MLflow 3.x requires an actual artifact file at the source path).
"""

from __future__ import annotations

from contextlib import suppress
from unittest.mock import MagicMock, patch

import mlflow
import pytest

from spacecraft_telemetry.mlflow_tracking.registry import (
    CHAMPION_ALIAS,
    latest_uri,
    promote,
    register_pytorch_model,
)
from spacecraft_telemetry.mlflow_tracking.runs import open_run

_REGISTRY_CLIENT = (
    "spacecraft_telemetry.mlflow_tracking.registry"
    ".MlflowClient"
)


class TestLatestUri:
    def test_uses_champion_alias(self) -> None:
        assert latest_uri("telemanom-ESA-Mission1-channel_1") == (
            "models:/telemanom-ESA-Mission1-channel_1@champion"
        )

    def test_uri_uses_alias_format(self) -> None:
        uri = latest_uri("telemanom-ESA-Mission1-channel_1")
        assert uri.startswith("models:/")
        assert "@champion" in uri


class TestRegisterPytorchModel:
    """Unit tests: verify register_pytorch_model calls mlflow with correct args."""

    def test_calls_log_model_with_correct_params(self, mlflow_uri: str) -> None:
        fake_model = MagicMock()
        model_name = "telemanom-ESA-Mission1-channel_1"
        source_run_model_name = "channel_1"
        fake_run_id = "abc123"

        with (
            patch("spacecraft_telemetry.mlflow_tracking.registry.log_pytorch_model") as mock_log,
            patch(_REGISTRY_CLIENT) as mock_client_cls,
        ):
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.search_model_versions.return_value = [MagicMock()]

            register_pytorch_model(
                model=fake_model,
                name=model_name,
                run_id=fake_run_id,
                source_run_model_name=source_run_model_name,
            )

        mock_log.assert_called_once()
        kwargs = mock_log.call_args.kwargs
        assert kwargs["name"] == source_run_model_name
        assert kwargs["registered_model_name"] == model_name
        assert kwargs["pytorch_model"] is fake_model

    def test_returns_model_version_from_registry_query(self, mlflow_uri: str) -> None:
        fake_version = MagicMock()
        with patch("spacecraft_telemetry.mlflow_tracking.registry.log_pytorch_model"), \
             patch(_REGISTRY_CLIENT) as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.search_model_versions.return_value = [fake_version]

            result = register_pytorch_model(
                model=MagicMock(), name="my-model", run_id="run-xyz"
            )

        assert result is fake_version

    def test_returns_none_when_run_id_filter_empty(self, mlflow_uri: str) -> None:
        with patch("spacecraft_telemetry.mlflow_tracking.registry.log_pytorch_model"), \
             patch(_REGISTRY_CLIENT) as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.search_model_versions.return_value = []

            result = register_pytorch_model(
                model=MagicMock(), name="my-model", run_id="run-xyz"
            )

        assert result is None


def _create_version(name: str, run_id: str) -> None:
    """Helper: register a model version using the client API.

    Uses MlflowClient.create_model_version directly, which stores the source
    string without validating an artifact file (safe to call without logging
    an actual model artifact).
    """
    client = mlflow.MlflowClient()
    with suppress(Exception):
        client.create_registered_model(name)
    client.create_model_version(
        name=name,
        source=f"runs:/{run_id}/placeholder",
        run_id=run_id,
    )


class TestPromote:
    def test_promotes_latest_version(self, mlflow_uri: str) -> None:
        name = "telemanom-ESA-Mission1-channel_3"
        with open_run(experiment="exp", run_name="ch", tags={}) as run:
            assert run is not None
            _create_version(name, run.info.run_id)

        promote(name=name)

        client = mlflow.MlflowClient()
        mv = client.get_model_version_by_alias(name, CHAMPION_ALIAS)
        assert mv is not None

    def test_promotes_explicit_version(self, mlflow_uri: str) -> None:
        name = "telemanom-ESA-Mission1-channel_4"
        with open_run(experiment="exp", run_name="ch", tags={}) as run:
            assert run is not None
            _create_version(name, run.info.run_id)

        promote(name=name, version=1)

        client = mlflow.MlflowClient()
        mv = client.get_model_version_by_alias(name, CHAMPION_ALIAS)
        assert int(mv.version) == 1

    def test_resolves_to_latest_when_version_omitted(self, mlflow_uri: str) -> None:
        name = "telemanom-ESA-Mission1-channel_5"
        with open_run(experiment="exp", run_name="ch1", tags={}) as r1:
            assert r1 is not None
            _create_version(name, r1.info.run_id)
        with open_run(experiment="exp", run_name="ch2", tags={}) as r2:
            assert r2 is not None
            _create_version(name, r2.info.run_id)

        promote(name=name)

        client = mlflow.MlflowClient()
        mv = client.get_model_version_by_alias(name, CHAMPION_ALIAS)
        assert int(mv.version) == 2

    def test_raises_when_no_versions(self, mlflow_uri: str) -> None:
        with pytest.raises(ValueError, match="No versions found"):
            promote(name="telemanom-ESA-Mission1-nonexistent")


class TestDemote:
    def test_removes_champion_alias(self, mlflow_uri: str) -> None:
        from spacecraft_telemetry.mlflow_tracking.registry import demote

        name = "telemanom-ISS-P4000007"
        with open_run(experiment="exp", run_name="ch", tags={}) as run:
            assert run is not None
            _create_version(name, run.info.run_id)
        promote(name=name)
        client = mlflow.MlflowClient()
        assert client.get_model_version_by_alias(name, CHAMPION_ALIAS) is not None

        assert demote(name=name) is True

        # Alias is gone → resolving it now raises.
        with pytest.raises(Exception):  # noqa: B017  (MlflowException on missing alias)
            client.get_model_version_by_alias(name, CHAMPION_ALIAS)

    def test_noop_when_not_champion(self, mlflow_uri: str) -> None:
        """Demoting a version that was never promoted is a harmless no-op."""
        from spacecraft_telemetry.mlflow_tracking.registry import demote

        name = "telemanom-ISS-S6000008"
        with open_run(experiment="exp", run_name="ch", tags={}) as run:
            assert run is not None
            _create_version(name, run.info.run_id)

        # Registered, but never promoted → no alias to remove.
        assert demote(name=name) is False
