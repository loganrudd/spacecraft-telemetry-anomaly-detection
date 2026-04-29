"""Training loop integration tests — marked @pytest.mark.slow.

These run a real training pass on the CPU with tiny settings so they finish
in a few seconds. Excluded from the default CI run (pytest -m "not slow").
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

torch = pytest.importorskip("torch")

from spacecraft_telemetry.core.config import ModelConfig, Settings
from spacecraft_telemetry.model.io import artifact_paths
from spacecraft_telemetry.model.training import TrainingResult, train_channel
from tests.model.conftest import WindowedParquetFixture, _WINDOW_FILE_SCHEMA


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
    target: float = 0.0,
) -> None:
    """Write a windowed Parquet where every window has the same constant value/target."""
    _BASE_DT = datetime(2000, 1, 1, tzinfo=timezone.utc)
    base_ts = pa.array([_BASE_DT] * n, type=pa.timestamp("us", tz="UTC"))
    table = pa.table(
        {
            "window_id": pa.array(np.arange(n, dtype=np.int64)),
            "segment_id": pa.array(np.zeros(n, dtype=np.int32)),
            "window_start_ts": base_ts,
            "window_end_ts": base_ts,
            "values": pa.array(
                [[value] * window_size] * n, type=pa.list_(pa.float32())
            ),
            "target": pa.array(np.full(n, target, dtype=np.float32)),
            "is_anomaly": pa.array([False] * n),
        },
        schema=_WINDOW_FILE_SCHEMA,
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
    tiny_windowed_parquet: WindowedParquetFixture,
    tmp_path: Path,
) -> None:
    """train_channel returns a TrainingResult and writes model.pt + train_log.json."""
    from spacecraft_telemetry.core.config import load_settings

    fx = tiny_windowed_parquet
    settings = _override_settings(
        load_settings("test"),
        processed_dir=fx.processed_dir,
        artifacts_dir=tmp_path / "models",
    )

    result = train_channel(settings, fx.mission, fx.channel)

    assert isinstance(result, TrainingResult)
    assert result.epochs_run == len(result.train_losses)
    assert result.epochs_run == len(result.val_losses)
    assert 0 <= result.best_epoch < result.epochs_run

    paths = artifact_paths(settings, fx.mission, fx.channel)
    assert Path(paths.model).exists(), "model.pt not written"
    assert Path(paths.config).exists(), "model_config.json not written"
    assert Path(paths.train_log).exists(), "train_log.json not written"


@pytest.mark.slow
def test_train_channel_result_structure_matches_epochs(
    tiny_windowed_parquet: WindowedParquetFixture,
    tmp_path: Path,
) -> None:
    """val_losses length must equal epochs_run; best_val_loss must be the min val loss."""
    from spacecraft_telemetry.core.config import load_settings

    fx = tiny_windowed_parquet
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
        value=0.0, target=0.0,
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
            "window_size": window_size,  # not a ModelConfig field — ignored
        },
    )
    # Remove window_size — it lives in SparkConfig, not ModelConfig
    settings = settings.model_copy(
        update={
            "spark": settings.spark.model_copy(update={"window_size": window_size}),
        }
    )

    result = train_channel(settings, mission, channel)
    assert result.epochs_run < 50, (
        f"Expected early stopping to trigger before 50 epochs, got {result.epochs_run}"
    )
