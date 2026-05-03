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

from unittest.mock import MagicMock, call, patch

import mlflow
import pytest

from spacecraft_telemetry.mlflow_tracking.registry import latest_uri, promote, register_pytorch_model
from spacecraft_telemetry.mlflow_tracking.runs import open_run


class TestLatestUri:
    def test_default_stage_is_production(self) -> None:
        assert latest_uri("telemanom-ESA-Mission1-channel_1") == (
            "models:/telemanom-ESA-Mission1-channel_1/Production"
        )

    def test_custom_stage(self) -> None:
        assert latest_uri("my-model", stage="Staging") == "models:/my-model/Staging"

    def test_uri_is_parseable(self) -> None:
        uri = latest_uri("telemanom-ESA-Mission1-channel_1")
        assert uri.startswith("models:/")
        assert "Production" in uri


class TestRegisterPytorchModel:
    """Unit tests: verify register_pytorch_model calls mlflow with correct args."""

    def test_calls_log_model_with_correct_params(self, mlflow_uri: str) -> None:
        fake_model = MagicMock()
        model_name = "telemanom-ESA-Mission1-channel_1"
        fake_run_id = "abc123"

        with patch("mlflow.pytorch.log_model") as mock_log, \
             patch("spacecraft_telemetry.mlflow_tracking.registry.mlflow.tracking.MlflowClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.search_model_versions.return_value = [MagicMock()]

            register_pytorch_model(model=fake_model, name=model_name, run_id=fake_run_id)

        mock_log.assert_called_once()
        kwargs = mock_log.call_args.kwargs
        assert kwargs["artifact_path"] == "model"
        assert kwargs["registered_model_name"] == model_name
        assert kwargs["pytorch_model"] is fake_model

    def test_returns_model_version_from_registry_query(self, mlflow_uri: str) -> None:
        fake_version = MagicMock()
        with patch("mlflow.pytorch.log_model"), \
             patch("spacecraft_telemetry.mlflow_tracking.registry.mlflow.tracking.MlflowClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.search_model_versions.return_value = [fake_version]

            result = register_pytorch_model(
                model=MagicMock(), name="my-model", run_id="run-xyz"
            )

        assert result is fake_version

    def test_falls_back_to_all_versions_when_run_id_filter_empty(
        self, mlflow_uri: str
    ) -> None:
        fallback_version = MagicMock()
        with patch("mlflow.pytorch.log_model"), \
             patch("spacecraft_telemetry.mlflow_tracking.registry.mlflow.tracking.MlflowClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            # First search (run_id filter) returns nothing; second (all) returns one.
            mock_client.search_model_versions.side_effect = [[], [fallback_version]]

            result = register_pytorch_model(
                model=MagicMock(), name="my-model", run_id="run-xyz"
            )

        assert result is fallback_version


def _create_version(name: str, run_id: str) -> None:
    """Helper: register a model version using the client API.

    Uses MlflowClient.create_model_version directly, which stores the source
    string without validating an artifact file (safe to call without logging
    an actual model artifact).
    """
    client = mlflow.tracking.MlflowClient()
    try:
        client.create_registered_model(name)
    except Exception:
        pass
    client.create_model_version(
        name=name,
        source=f"runs:/{run_id}/placeholder",
        run_id=run_id,
    )


class TestPromote:
    def test_promotes_to_production(self, mlflow_uri: str) -> None:
        name = "telemanom-ESA-Mission1-channel_3"
        with open_run(experiment="exp", run_name="ch", tags={}) as run:
            assert run is not None
            _create_version(name, run.info.run_id)

        promote(name=name, stage="Production")

        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(name, stages=["Production"])
        assert len(versions) == 1

    def test_promotes_explicit_version(self, mlflow_uri: str) -> None:
        name = "telemanom-ESA-Mission1-channel_4"
        with open_run(experiment="exp", run_name="ch", tags={}) as run:
            assert run is not None
            _create_version(name, run.info.run_id)

        promote(name=name, version=1, stage="Staging")

        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(name, stages=["Staging"])
        assert len(versions) == 1
        assert int(versions[0].version) == 1

    def test_resolves_to_latest_non_archived_when_version_omitted(
        self, mlflow_uri: str
    ) -> None:
        name = "telemanom-ESA-Mission1-channel_5"
        with open_run(experiment="exp", run_name="ch1", tags={}) as r1:
            assert r1 is not None
            _create_version(name, r1.info.run_id)
        with open_run(experiment="exp", run_name="ch2", tags={}) as r2:
            assert r2 is not None
            _create_version(name, r2.info.run_id)

        promote(name=name, stage="Production")  # no version specified

        client = mlflow.tracking.MlflowClient()
        prod = client.get_latest_versions(name, stages=["Production"])
        assert len(prod) == 1
        assert int(prod[0].version) == 2  # latest = 2

    def test_raises_when_no_versions(self, mlflow_uri: str) -> None:
        with pytest.raises(ValueError, match="No promotable versions"):
            promote(name="telemanom-ESA-Mission1-nonexistent", stage="Production")
