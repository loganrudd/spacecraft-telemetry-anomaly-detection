"""Integration tests for mlflow_tracking/runs.py.

Uses a per-test SQLite store (mlflow_uri fixture from conftest).
"""

from __future__ import annotations

import mlflow
import pytest

from spacecraft_telemetry.core.config import load_settings
from spacecraft_telemetry.mlflow_tracking.runs import (
    configure_mlflow,
    log_artifact_bytes,
    log_metrics_final,
    log_metrics_step,
    log_params,
    open_run,
)


class TestConfigureMlflow:
    def test_sets_tracking_uri(self, mlflow_uri: str) -> None:
        settings = load_settings("test").model_copy(
            update={
                "mlflow": load_settings("test").mlflow.model_copy(
                    update={"tracking_uri": mlflow_uri}
                )
            }
        )
        configure_mlflow(settings)
        assert mlflow.get_tracking_uri() == mlflow_uri

    def test_registry_uri_none_does_not_override(self, mlflow_uri: str) -> None:
        settings = load_settings("test").model_copy(
            update={
                "mlflow": load_settings("test").mlflow.model_copy(
                    update={"tracking_uri": mlflow_uri, "registry_uri": None}
                )
            }
        )
        configure_mlflow(settings)
        # No error; registry falls back to tracking_uri (MLflow default).


class TestOpenRun:
    def test_creates_experiment_and_run(self, mlflow_uri: str) -> None:
        with open_run(
            experiment="test-exp",
            run_name="run-a",
            tags={"model_type": "telemanom"},
        ) as run:
            assert run is not None
            assert run.info.run_name == "run-a"

        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name("test-exp")
        assert exp is not None
        runs = client.search_runs([exp.experiment_id])
        assert len(runs) == 1
        assert runs[0].data.tags["model_type"] == "telemanom"

    def test_run_is_finished_after_context_exits(self, mlflow_uri: str) -> None:
        with open_run(experiment="test-exp", run_name="r", tags={}) as run:
            assert run is not None
            run_id = run.info.run_id

        client = mlflow.tracking.MlflowClient()
        finished = client.get_run(run_id)
        assert finished.info.status == "FINISHED"

    def test_yields_none_on_start_failure(
        self, mlflow_uri: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _raise(**_: object) -> None:
            raise RuntimeError("forced")

        monkeypatch.setattr(mlflow, "start_run", _raise)

        body_executed = False
        with open_run(experiment="exp", run_name="r", tags={}) as run:
            body_executed = True
            assert run is None

        assert body_executed  # caller body always runs

    def test_nested_run(self, mlflow_uri: str) -> None:
        with open_run(experiment="exp", run_name="parent", tags={}) as parent:
            assert parent is not None
            with open_run(experiment="exp", run_name="child", tags={}, nested=True) as child:
                assert child is not None
                assert child.info.run_id != parent.info.run_id


class TestLogHelpers:
    def test_log_params_stores_values(self, mlflow_uri: str) -> None:
        with open_run(experiment="exp", run_name="r", tags={}):
            log_params({"lr": 0.001, "epochs": 35})

        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name("exp")
        assert exp is not None
        run = client.search_runs([exp.experiment_id])[0]
        assert run.data.params["lr"] == "0.001"
        assert run.data.params["epochs"] == "35"

    def test_log_metrics_step_records_step(self, mlflow_uri: str) -> None:
        with open_run(experiment="exp", run_name="r", tags={}):
            log_metrics_step({"train_loss": 0.5}, step=0)
            log_metrics_step({"train_loss": 0.3}, step=1)

        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name("exp")
        assert exp is not None
        run = client.search_runs([exp.experiment_id])[0]
        history = client.get_metric_history(run.info.run_id, "train_loss")
        assert len(history) == 2
        assert history[0].step == 0
        assert history[1].step == 1

    def test_log_metrics_final_stores_metric(self, mlflow_uri: str) -> None:
        with open_run(experiment="exp", run_name="r", tags={}):
            log_metrics_final({"best_val_loss": 0.12, "f0_5": 0.88})

        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name("exp")
        assert exp is not None
        run = client.search_runs([exp.experiment_id])[0]
        assert run.data.metrics["best_val_loss"] == pytest.approx(0.12)
        assert run.data.metrics["f0_5"] == pytest.approx(0.88)

    def test_log_params_noop_without_active_run(self, mlflow_uri: str) -> None:
        assert mlflow.active_run() is None
        log_params({"should": "not_crash"})  # must not raise

    def test_log_metrics_noop_without_active_run(self, mlflow_uri: str) -> None:
        assert mlflow.active_run() is None
        log_metrics_final({"loss": 0.5})  # must not raise

    def test_log_artifact_bytes_stores_file(self, mlflow_uri: str) -> None:
        with open_run(experiment="exp", run_name="r", tags={}):
            run_id = mlflow.active_run().info.run_id  # type: ignore[union-attr]
            log_artifact_bytes(b'{"key": "value"}', "configs/model_config.json")

        client = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(run_id, path="configs")
        names = [a.path for a in artifacts]
        assert any("model_config.json" in n for n in names)

    def test_log_artifact_bytes_noop_without_active_run(self, mlflow_uri: str) -> None:
        assert mlflow.active_run() is None
        log_artifact_bytes(b"data", "file.bin")  # must not raise
