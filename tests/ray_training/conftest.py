"""Shared fixtures for Ray training tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from spacecraft_telemetry.core.config import Settings, load_settings

_MISSION = "ESA-Mission1"
_CHANNEL = "channel_1"

# PyArrow schema matching SERIES_SCHEMA from spark/schemas.py,
# minus Hive partition columns (encoded in directory names).
_SERIES_FILE_SCHEMA = pa.schema(
    [
        pa.field("telemetry_timestamp", pa.timestamp("us", tz="UTC")),
        pa.field("value_normalized", pa.float32()),
        pa.field("segment_id", pa.int32()),
        pa.field("is_anomaly", pa.bool_()),
    ]
)


def _write_series_split(
    base: Path,
    split: str,
    mission: str,
    channel: str,
    n_rows: int,
    n_anomaly_tail: int,
    segment_id: int,
) -> None:
    """Write a minimal per-timestep series Parquet partition."""
    import pandas as pd

    n_clean = n_rows - n_anomaly_tail
    timestamps = pd.date_range("2000-01-01", periods=n_rows, freq="s", tz="UTC")
    values = np.random.default_rng(0).standard_normal(n_rows).astype("float32")
    is_anomaly = [False] * n_clean + [True] * n_anomaly_tail

    partition_dir = base / split / f"mission_id={mission}" / f"channel_id={channel}"
    partition_dir.mkdir(parents=True, exist_ok=True)

    table = pa.table(
        {
            "telemetry_timestamp": pa.array(timestamps.astype("datetime64[us, UTC]")),
            "value_normalized": pa.array(values),
            "segment_id": pa.array([segment_id] * n_rows, type=pa.int32()),
            "is_anomaly": pa.array(is_anomaly),
        },
        schema=_SERIES_FILE_SCHEMA,
    )
    pq.write_table(table, partition_dir / "part.parquet")


@pytest.fixture(scope="session")
def ray_series_parquet(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    """Write minimal Parquet for one channel and return Settings pointing to it.

    Returns a test Settings object whose spark.processed_data_dir and
    model.artifacts_dir point into the tmp directory.  Keeps the fixture
    scoped to the session so the file is written once across all Ray tests.
    """
    base = tmp_path_factory.mktemp("ray_processed")

    test_cfg = load_settings("test")
    window = test_cfg.model.window_size
    horizon = test_cfg.model.prediction_horizon
    n_train = window + horizon + 5   # enough rows for at least one train window
    n_test = window + horizon + 5

    for split, n_rows, n_anom in [("train", n_train, 0), ("test", n_test, 3)]:
        _write_series_split(
            base / _MISSION,
            split,
            _MISSION,
            _CHANNEL,
            n_rows,
            n_anom,
            segment_id=0,
        )

    # Also write normalization_params.json (needed by score_channel).
    norm_dir = base / _MISSION
    norm_dir.mkdir(parents=True, exist_ok=True)
    import json
    (norm_dir / "normalization_params.json").write_text(
        json.dumps({_CHANNEL: {"mean": 0.0, "std": 1.0}})
    )

    artifacts_dir = tmp_path_factory.mktemp("ray_models")

    settings = test_cfg.model_copy(
        update={
            "spark": test_cfg.spark.model_copy(
                update={"processed_data_dir": str(base)}
            ),
            "model": test_cfg.model.model_copy(
                update={"artifacts_dir": str(artifacts_dir)}
            ),
        }
    )
    return settings


@pytest.fixture(scope="session")
def ray_local():
    """Start a Ray local cluster once for all Ray tests in the session.

    Yields control, then shuts Ray down after the session.
    """
    import sys, os
    import ray

    pythonpath = os.pathsep.join(p for p in sys.path if p)
    ray.init(
        num_cpus=2,
        ignore_reinit_error=True,
        runtime_env={"env_vars": {"PYTHONPATH": pythonpath}},
    )
    yield
    ray.shutdown()


@pytest.fixture(scope="session")
def ray_train_result(ray_local, ray_series_parquet: Settings) -> list:
    """Train channel_1 via Ray once; cache result for all train-assertion tests.

    Pays the Ray worker cold-start cost exactly once per session so that
    test_train_all_channels_ok and test_train_all_channels_max_channels_cap
    don't each spin up a fresh Ray task.
    """
    from spacecraft_telemetry.ray_training import train_all_channels

    return train_all_channels(ray_series_parquet, _MISSION, [_CHANNEL])


@pytest.fixture(scope="session")
def pretrained_channel(ray_series_parquet: Settings) -> Settings:
    """Train channel_1 once locally (no Ray) for reuse by all score tests.

    Session-scoped so the training run happens exactly once regardless of how
    many score tests depend on it.  Returns the same Settings object so tests
    can pass it directly to score_all_channels.
    """
    from spacecraft_telemetry.model.training import train_channel

    train_channel(ray_series_parquet, _MISSION, _CHANNEL)
    return ray_series_parquet

