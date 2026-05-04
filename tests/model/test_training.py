"""Training loop integration tests — marked @pytest.mark.slow.

These run a real training pass on the CPU with tiny settings so they finish
in a few seconds. Excluded from the default CI run (pytest -m "not slow").
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

torch = pytest.importorskip("torch")

from datetime import UTC  # noqa: E402

import mlflow  # noqa: E402

from spacecraft_telemetry.core.config import Settings  # noqa: E402
from spacecraft_telemetry.model.training import (  # noqa: E402
    TrainingResult,
    train_channel,
)
from tests.model.conftest import _SERIES_FILE_SCHEMA, SeriesParquetFixture  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _override_settings(
    base: Settings,
    processed_dir: Path,
    artifacts_dir: Path,
    model_overrides: dict | None = None,
) -> Settings:
    """Return a Settings copy pointing at tmp_path directories."""
    model_updates: dict = {"artifacts_dir": artifacts_dir}
    if model_overrides:
        model_updates.update(model_overrides)
    return base.model_copy(
        update={
            "spark": base.spark.model_copy(
                update={"processed_data_dir": processed_dir}
            ),
            "model": base.model.model_copy(update=model_updates),
        }
    )


def _write_const_parquet(
    processed_dir: Path,
    mission: str,
    channel: str,
    n: int,
    window_size: int,
    value: float = 0.0,
) -> None:
    """Write per-timestep series Parquet where every row has the same constant value."""
    from datetime import datetime

    _BASE_DT = datetime(2000, 1, 1, tzinfo=UTC)
    timestamps = [
        pa.scalar(
            _BASE_DT.timestamp() + i * 90,
            type=pa.timestamp("s", tz="UTC"),
        ).cast(pa.timestamp("us", tz="UTC"))
        for i in range(n)
    ]
    table = pa.table(
        {
            "telemetry_timestamp": pa.array(timestamps, type=pa.timestamp("us", tz="UTC")),
            "value_normalized": pa.array([value] * n, type=pa.float32()),
            "segment_id": pa.array(np.zeros(n, dtype=np.int32)),
            "is_anomaly": pa.array([False] * n),
        },
        schema=_SERIES_FILE_SCHEMA,
    )
    split_dir = (
        processed_dir / mission / "train"
        / f"mission_id={mission}" / f"channel_id={channel}"
    )
    split_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, split_dir / "part.parquet")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_train_channel_runs_and_saves_artifacts(
    tiny_series_parquet: SeriesParquetFixture,
    mlflow_uri: str,
    tmp_path: Path,
) -> None:
    """train_channel returns a TrainingResult and logs model + train_log + norm_params to MLflow."""
    from spacecraft_telemetry.core.config import load_settings
    from spacecraft_telemetry.mlflow_tracking.conventions import experiment_name

    fx = tiny_series_parquet
    settings = _override_settings(
        load_settings("test").model_copy(
            update={"mlflow": load_settings("test").mlflow.model_copy(
                update={"tracking_uri": mlflow_uri}
            )}
        ),
        processed_dir=fx.processed_dir,
        artifacts_dir=tmp_path / "models",
    )

    result = train_channel(settings, fx.mission, fx.channel)

    assert isinstance(result, TrainingResult)
    assert result.epochs_run == len(result.train_losses)
    assert result.epochs_run == len(result.val_losses)
    assert 0 <= result.best_epoch < result.epochs_run

    # Verify that the MLflow run contains all expected artifacts.
    from mlflow.tracking import MlflowClient
    from spacecraft_telemetry.mlflow_tracking.conventions import (
        registered_model_name as _reg_name,
    )

    client = MlflowClient(tracking_uri=mlflow_uri)
    exp_name = experiment_name(settings.model.model_type, "training", fx.mission)
    exp = client.get_experiment_by_name(exp_name)
    assert exp is not None
    runs = client.search_runs([exp.experiment_id])
    assert len(runs) == 1
    artifacts = client.list_artifacts(runs[0].info.run_id)
    artifact_names = {a.path for a in artifacts}
    assert "normalization_params.json" in artifact_names, "norm params not logged to MLflow"
    assert "train_log.json" in artifact_names, "train_log.json not logged to MLflow"
    # In MLflow 3.x, log_model artifacts live in the models store, not run artifacts.
    # Verify the model was registered instead of checking list_artifacts.
    model_name = _reg_name(settings.model.model_type, fx.mission, fx.channel)
    versions = client.search_model_versions(f"name='{model_name}'")
    assert len(versions) >= 1, "mlflow.pytorch.log_model did not register a model version"


@pytest.mark.slow
def test_train_channel_result_structure_matches_epochs(
    tiny_series_parquet: SeriesParquetFixture,
    tmp_path: Path,
) -> None:
    """val_losses length must equal epochs_run; best_val_loss must be the min val loss."""
    from spacecraft_telemetry.core.config import load_settings

    fx = tiny_series_parquet
    settings = _override_settings(
        load_settings("test"),
        processed_dir=fx.processed_dir,
        artifacts_dir=tmp_path / "models",
    )

    result = train_channel(settings, fx.mission, fx.channel)

    assert len(result.val_losses) == result.epochs_run
    assert result.best_val_loss == min(result.val_losses)


@pytest.mark.slow
def test_early_stopping_triggers(tmp_path: Path) -> None:
    """With constant inputs/targets, the model converges in one epoch;
    patience=1 then stops training before completing all epochs."""
    from spacecraft_telemetry.core.config import load_settings

    mission, channel = "ESA-Mission1", "channel_1"
    processed_dir = tmp_path / "processed"
    window_size = 10

    # All-zero inputs and targets: after epoch 0 the model predicts ~0,
    # val_loss ≈ 0.  Epoch 1 does not improve strictly → patience=1 → stop.
    _write_const_parquet(
        processed_dir, mission, channel, n=30, window_size=window_size,
        value=0.0,
    )

    # Write normalization_params.json required by train_channel.
    import json as _json
    norm_file = processed_dir / mission / "normalization_params.json"
    norm_file.parent.mkdir(parents=True, exist_ok=True)
    norm_file.write_text(_json.dumps({channel: {"mean": 0.0, "std": 1.0}}))

    settings = _override_settings(
        load_settings("test"),
        processed_dir=processed_dir,
        artifacts_dir=tmp_path / "models",
        model_overrides={
            "epochs": 50,
            "early_stopping_patience": 1,
            "hidden_dim": 4,
            "batch_size": 8,
            "window_size": window_size,
        },
    )

    result = train_channel(settings, mission, channel)
    assert result.epochs_run < 50, (
        f"Expected early stopping to trigger before 50 epochs, got {result.epochs_run}"
    )


@pytest.mark.slow
def test_train_channel_creates_mlflow_run(
    tiny_series_parquet: SeriesParquetFixture,
    mlflow_uri: str,
    tmp_path: Path,
) -> None:
    """train_channel creates a run in the correct MLflow experiment with required tags."""
    from spacecraft_telemetry.core.config import load_settings
    from spacecraft_telemetry.mlflow_tracking.conventions import experiment_name

    fx = tiny_series_parquet
    settings = _override_settings(
        load_settings("test").model_copy(
            update={"mlflow": load_settings("test").mlflow.model_copy(
                update={"tracking_uri": mlflow_uri}
            )}
        ),
        processed_dir=fx.processed_dir,
        artifacts_dir=tmp_path / "models",
    )

    train_channel(settings, fx.mission, fx.channel)

    client = mlflow.tracking.MlflowClient()
    exp_name = experiment_name(settings.model.model_type, "training", fx.mission)
    exp = client.get_experiment_by_name(exp_name)
    assert exp is not None, f"experiment {exp_name!r} was not created"

    runs = client.search_runs([exp.experiment_id])
    assert len(runs) == 1
    tags = runs[0].data.tags
    assert tags["model_type"] == settings.model.model_type
    assert tags["mission_id"] == fx.mission
    assert tags["channel_id"] == fx.channel
    assert runs[0].data.metrics["best_val_loss"] is not None
    assert runs[0].data.params["hidden_dim"] == str(settings.model.hidden_dim)


@pytest.mark.slow
def test_train_channel_registers_model_version(
    tiny_series_parquet: SeriesParquetFixture,
    mlflow_uri: str,
    tmp_path: Path,
) -> None:
    """train_channel auto-registers a model version in the MLflow registry."""
    from spacecraft_telemetry.core.config import load_settings
    from spacecraft_telemetry.mlflow_tracking.conventions import registered_model_name

    fx = tiny_series_parquet
    settings = _override_settings(
        load_settings("test").model_copy(
            update={"mlflow": load_settings("test").mlflow.model_copy(
                update={"tracking_uri": mlflow_uri}
            )}
        ),
        processed_dir=fx.processed_dir,
        artifacts_dir=tmp_path / "models",
    )

    train_channel(settings, fx.mission, fx.channel)

    client = mlflow.tracking.MlflowClient()
    model_name = registered_model_name(settings.model.model_type, fx.mission, fx.channel)
    versions = list(client.search_model_versions(f"name='{model_name}'"))
    assert len(versions) >= 1, f"no registered versions found for {model_name!r}"
