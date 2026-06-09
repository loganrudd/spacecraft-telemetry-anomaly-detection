"""Integration tests for mlflow_tracking/runs.py.

Uses a per-test SQLite store (mlflow_uri fixture from conftest).
"""

from __future__ import annotations

import os
import time
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock

import mlflow
import pytest
from mlflow.tracking import MlflowClient

from spacecraft_telemetry.core.config import load_settings
from spacecraft_telemetry.mlflow_tracking.runs import (
    _install_id_token_auth,
    _parquet_stats,
    configure_mlflow,
    keep_mlflow_auth_fresh,
    log_artifact_bytes,
    log_input_dataset,
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

        client = MlflowClient()
        exp = client.get_experiment_by_name("test-exp")
        assert exp is not None
        runs = client.search_runs([exp.experiment_id])
        assert len(runs) == 1
        assert runs[0].data.tags["model_type"] == "telemanom"

    def test_run_is_finished_after_context_exits(self, mlflow_uri: str) -> None:
        with open_run(experiment="test-exp", run_name="r", tags={}) as run:
            assert run is not None
            run_id = run.info.run_id

        client = MlflowClient()
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

        client = MlflowClient()
        exp = client.get_experiment_by_name("exp")
        assert exp is not None
        run = client.search_runs([exp.experiment_id])[0]
        assert run.data.params["lr"] == "0.001"
        assert run.data.params["epochs"] == "35"

    def test_log_metrics_step_records_step(self, mlflow_uri: str) -> None:
        with open_run(experiment="exp", run_name="r", tags={}):
            log_metrics_step({"train_loss": 0.5}, step=0)
            log_metrics_step({"train_loss": 0.3}, step=1)

        client = MlflowClient()
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

        client = MlflowClient()
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

        client = MlflowClient()
        artifacts = client.list_artifacts(run_id, path="configs")
        names = [a.path for a in artifacts]
        assert any("model_config.json" in n for n in names)

    def test_log_artifact_bytes_noop_without_active_run(self, mlflow_uri: str) -> None:
        assert mlflow.active_run() is None
        log_artifact_bytes(b"data", "file.bin")  # must not raise


class TestParquetStats:
    def test_returns_zero_for_missing_path(self, tmp_path: Path) -> None:
        schema, num_rows, start, end = _parquet_stats(str(tmp_path / "nonexistent"))
        assert schema is None
        assert num_rows == 0
        assert start is None and end is None

    def test_returns_zero_for_directory_with_no_parquet(self, tmp_path: Path) -> None:
        (tmp_path / "readme.txt").write_text("hello")
        schema, num_rows, start, end = _parquet_stats(str(tmp_path))
        assert schema is None
        assert num_rows == 0

    def test_returns_correct_row_count_and_columns(self, tmp_path: Path) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        num_rows, num_cols = 120, 4
        table = pa.table({f"col_{i}": pa.array(range(num_rows)) for i in range(num_cols)})
        pq.write_table(table, tmp_path / "part.parquet")

        schema, n, start, end = _parquet_stats(str(tmp_path))
        assert n == num_rows
        assert schema is not None and len(schema.names) == num_cols
        # No telemetry_timestamp column → no date range.
        assert start is None and end is None

    def test_aggregates_rows_across_multiple_files(self, tmp_path: Path) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        for i in range(3):
            table = pa.table({"a": pa.array(range(10)), "b": pa.array(range(10))})
            pq.write_table(table, tmp_path / f"part_{i}.parquet")

        _schema, n, _start, _end = _parquet_stats(str(tmp_path))
        assert n == 30

    def test_extracts_timestamp_date_range(self, tmp_path: Path) -> None:
        import pandas as pd
        import pyarrow.parquet as pq

        ts = pd.date_range("2014-01-01", periods=100, freq="s", tz="UTC")
        df = pd.DataFrame({"telemetry_timestamp": ts, "value_normalized": range(100)})
        df.to_parquet(tmp_path / "part.parquet", index=False)

        _schema, n, start, end = _parquet_stats(str(tmp_path))
        assert n == 100
        assert start == ts[0].isoformat()
        assert end == ts[-1].isoformat()


class TestLogInputDataset:
    def test_noop_without_active_run(self, mlflow_uri: str) -> None:
        """log_input_dataset is a no-op when there is no active run."""
        assert mlflow.active_run() is None
        log_input_dataset(
            source="/data/processed/ESA-Mission1/train/mission_id=ESA-Mission1/channel_id=ch1",
            name="ESA-Mission1-ch1-train",
            digest="a" * 64,
            context="training",
        )  # must not raise

    def test_noop_when_digest_is_none(self, mlflow_uri: str) -> None:
        """log_input_dataset is a no-op when digest is None."""
        with open_run(experiment="exp", run_name="r", tags={}):
            log_input_dataset(
                source="/data/processed/ESA-Mission1/train",
                name="ESA-Mission1-ch1-train",
                digest=None,
                context="training",
            )  # must not raise; no dataset should be logged

        client = MlflowClient()
        exp = client.get_experiment_by_name("exp")
        assert exp is not None
        run = client.search_runs([exp.experiment_id])[0]
        assert run.inputs.dataset_inputs == []

    def test_records_dataset_in_active_run(self, mlflow_uri: str) -> None:
        """log_input_dataset stores name, digest, and context on the run."""
        fake_digest = "b" * 64
        with open_run(experiment="exp", run_name="r", tags={}) as run:
            assert run is not None
            run_id = run.info.run_id
            log_input_dataset(
                source="/data/processed/ESA-Mission1/train/mission_id=ESA-Mission1/channel_id=ch1",
                name="ESA-Mission1-ch1-train",
                digest=fake_digest,
                context="training",
            )

        client = MlflowClient()
        finished = client.get_run(run_id)
        inputs = finished.inputs.dataset_inputs
        assert len(inputs) == 1
        di = inputs[0]
        assert di.dataset.name == "ESA-Mission1-ch1-train"
        # digest is truncated to MLflow's 36-char column limit
        assert di.dataset.digest == fake_digest[:36]
        context_tags = [t for t in di.tags if t.key == "mlflow.data.context"]
        assert context_tags[0].value == "training"

    def test_records_evaluation_context(self, mlflow_uri: str) -> None:
        """context='evaluation' is stored correctly on a scoring run."""
        fake_digest = "c" * 64
        with open_run(experiment="exp", run_name="r", tags={}) as run:
            assert run is not None
            run_id = run.info.run_id
            log_input_dataset(
                source="/data/processed/ESA-Mission1/test/mission_id=ESA-Mission1/channel_id=ch1",
                name="ESA-Mission1-ch1-test",
                digest=fake_digest,
                context="evaluation",
            )

        client = MlflowClient()
        finished = client.get_run(run_id)
        di = finished.inputs.dataset_inputs[0]
        assert di.dataset.name == "ESA-Mission1-ch1-test"
        context_tags = [t for t in di.tags if t.key == "mlflow.data.context"]
        assert context_tags[0].value == "evaluation"


    def test_profile_shows_real_row_count_when_parquet_exists(
        self, mlflow_uri: str, tmp_path: Path
    ) -> None:
        """Profile shows actual row count (not 0) when a Parquet file is present."""
        import json

        import pyarrow as pa
        import pyarrow.parquet as pq

        import pandas as pd

        ts = pd.date_range("2014-01-01", periods=50, freq="s", tz="UTC")
        table = pa.table({
            "telemetry_timestamp": pa.array(ts),
            "value_normalized": pa.array(range(50)),
            "is_anomaly": pa.array([False] * 50),
        })
        pq.write_table(table, tmp_path / "part.parquet")
        fake_digest = "d" * 64

        with open_run(experiment="exp", run_name="r", tags={}) as run:
            assert run is not None
            run_id = run.info.run_id
            log_input_dataset(
                source=str(tmp_path),
                name="test-channel-train",
                digest=fake_digest,
                context="training",
            )

        client = MlflowClient()
        di = client.get_run(run_id).inputs.dataset_inputs[0]
        profile = json.loads(di.dataset.profile)
        assert profile["num_rows"] == 50
        # num_elements is intentionally dropped; start/end dates replace it.
        assert "num_elements" not in profile
        assert profile["start_date"] == ts[0].isoformat()
        assert profile["end_date"] == ts[-1].isoformat()


class TestInstallIdTokenAuth:
    @pytest.fixture(autouse=True)
    def _clear_cache(self) -> Generator[None, None, None]:
        from spacecraft_telemetry.mlflow_tracking import runs

        runs._token_cache.clear()
        yield
        runs._token_cache.clear()

    def test_noop_for_non_run_app_uri(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MLFLOW_TRACKING_TOKEN", raising=False)
        _install_id_token_auth("sqlite:///mlflow.db")
        assert "MLFLOW_TRACKING_TOKEN" not in os.environ

    def test_noop_for_http_non_run_app_uri(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MLFLOW_TRACKING_TOKEN", raising=False)
        _install_id_token_auth("http://localhost:5000")
        assert "MLFLOW_TRACKING_TOKEN" not in os.environ

    def test_sets_token_for_run_app_uri(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MLFLOW_TRACKING_TOKEN", raising=False)
        import google.auth.transport.requests as gtr
        import google.oauth2.id_token as gid

        monkeypatch.setattr(gtr, "Request", MagicMock())
        monkeypatch.setattr(gid, "fetch_id_token", lambda *_: "test-id-token")

        _install_id_token_auth("https://mlflow-xxxx-uc.a.run.app")

        assert os.environ.get("MLFLOW_TRACKING_TOKEN") == "test-id-token"

    def test_token_not_refetched_within_cache_window(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MLFLOW_TRACKING_TOKEN", raising=False)
        import google.auth.transport.requests as gtr
        import google.oauth2.id_token as gid

        monkeypatch.setattr(gtr, "Request", MagicMock())
        fetch_mock = MagicMock(return_value="cached-token")
        monkeypatch.setattr(gid, "fetch_id_token", fetch_mock)

        uri = "https://mlflow-xxxx-uc.a.run.app"
        _install_id_token_auth(uri)
        _install_id_token_auth(uri)

        fetch_mock.assert_called_once()

    def test_silently_skips_when_both_paths_fail(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import subprocess
        import sys

        monkeypatch.delenv("MLFLOW_TRACKING_TOKEN", raising=False)
        # Block google-auth (path 1) and gcloud subprocess (path 2) so the
        # function reaches the final warning+return without setting the env var.
        monkeypatch.setitem(sys.modules, "google.auth.transport.requests", None)
        monkeypatch.setattr(
            subprocess, "run", MagicMock(side_effect=FileNotFoundError("gcloud not found"))
        )

        _install_id_token_auth("https://mlflow-xxxx-uc.a.run.app")  # must not raise

        assert "MLFLOW_TRACKING_TOKEN" not in os.environ

    def test_gcloud_fallback_used_when_google_auth_fails(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import subprocess
        import sys
        from unittest.mock import MagicMock

        monkeypatch.delenv("MLFLOW_TRACKING_TOKEN", raising=False)
        monkeypatch.setitem(sys.modules, "google.auth.transport.requests", None)
        fake_result = MagicMock(returncode=0, stdout="gcloud-fallback-token\n", stderr="")
        monkeypatch.setattr(subprocess, "run", MagicMock(return_value=fake_result))

        _install_id_token_auth("https://mlflow-xxxx-uc.a.run.app")

        assert os.environ.get("MLFLOW_TRACKING_TOKEN") == "gcloud-fallback-token"


class TestKeepMlflowAuthFresh:
    """The background refresher used to keep a long Ray Tune sweep's token valid."""

    def test_refreshes_periodically_then_stops_cleanly(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # keep_mlflow_auth_fresh should call refresh_mlflow_auth on its interval
        # for the duration of the block, and stop on exit. Count delegations to
        # _install_id_token_auth (what refresh_mlflow_auth calls) instead of
        # hitting the metadata server.
        from spacecraft_telemetry.mlflow_tracking import runs

        calls: list[float] = []
        monkeypatch.setattr(
            runs, "_install_id_token_auth", lambda _uri: calls.append(time.monotonic())
        )

        with keep_mlflow_auth_fresh(interval_seconds=0.02):
            time.sleep(0.1)

        # ~20ms cadence over ~100ms fires several times; assert >=2 (timing-tolerant).
        assert len(calls) >= 2

        # The daemon must stop once the context exits — no further refreshes.
        n_at_exit = len(calls)
        time.sleep(0.06)
        assert len(calls) == n_at_exit

    def test_noop_thread_for_local_uri_exits_cleanly(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # For a local SQLite backend the refresh is a no-op; the manager must
        # still enter and exit without error and install no token. delenv first
        # so a token leaked by an earlier auth test doesn't mask the assertion
        # (production code sets MLFLOW_TRACKING_TOKEN directly, outside monkeypatch).
        from spacecraft_telemetry.mlflow_tracking import runs

        runs._token_cache.clear()
        monkeypatch.delenv("MLFLOW_TRACKING_TOKEN", raising=False)
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        with keep_mlflow_auth_fresh(interval_seconds=0.01):
            time.sleep(0.03)
        assert "MLFLOW_TRACKING_TOKEN" not in os.environ
