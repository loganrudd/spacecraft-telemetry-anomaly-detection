"""End-to-end smoke test: ISS preprocessing → train_channel (no Ray, no GPU).

Uses the same synthetic-tick fixture pattern as tests/preprocess/test_iss_pipeline.py.
Window size comes from configs/test.yaml (model.window_size=10) so the fixture only
needs a handful of resampled buckets.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

torch = pytest.importorskip("torch")

from spacecraft_telemetry.core.config import Settings, load_settings  # noqa: E402
from spacecraft_telemetry.ingest.collector_io import RAW_TICK_SCHEMA  # noqa: E402
from spacecraft_telemetry.model.training import TrainingResult, train_channel  # noqa: E402
from spacecraft_telemetry.preprocess.pipeline import run_iss_preprocessing  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ISS_CHANNEL = "S1000003"
_ISS_MISSION = "ISS"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_iss_ticks(raw_root: Path, channel_id: str, n: int = 600) -> None:
    """Write one tick shard at 2 s cadence (n ticks → n*2 s ≈ grid_interval_s*n/15 buckets)."""
    base = pd.Timestamp("2026-06-01T00:00:00Z")
    rows = [
        {
            "telemetry_timestamp": base + pd.Timedelta(seconds=i * 2),
            "value": float(i % 100) * 0.1,
            "aos_timestamp": None,
        }
        for i in range(n)
    ]
    dest = raw_root / "ISS" / "ticks" / f"channel_id={channel_id}"
    dest.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=RAW_TICK_SCHEMA)
    pq.write_table(table, dest / "20260601T000000.parquet")


def _iss_settings(tmp_path: Path, mlflow_tracking_uri: str) -> Settings:
    """Build Settings pointing at tmp_path raw ticks, processed data, and MLflow."""
    base = load_settings("test")
    out_dir = tmp_path / "processed"
    out_dir.mkdir()
    raw_dir = tmp_path / "raw"

    return base.model_copy(
        update={
            "preprocess": base.preprocess.model_copy(
                update={"processed_data_dir": str(out_dir)}
            ),
            "collect": base.collect.model_copy(
                update={
                    "raw_ticks_dir": str(raw_dir),
                    "grid_interval_seconds": 30,
                }
            ),
            "model": base.model.model_copy(
                update={"artifacts_dir": str(tmp_path / "models")}
            ),
            "mlflow": base.mlflow.model_copy(
                update={"tracking_uri": mlflow_tracking_uri}
            ),
        }
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def iss_training_setup(tmp_path: Path, mlflow_uri: str) -> Settings:
    """Write raw ISS ticks, run preprocessing, return settings for train_channel."""
    # 600 ticks x 2 s = 1200 s -> ~40 grid buckets (30 s each).
    # With test window_size=10 and 80% train split: ~32 train buckets → ~22 windows.
    _write_iss_ticks(tmp_path / "raw", _ISS_CHANNEL, n=600)
    settings = _iss_settings(tmp_path, mlflow_uri)
    run_iss_preprocessing(settings, parallel=False)
    return settings


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_train_channel_iss_returns_result(iss_training_setup: Settings) -> None:
    """train_channel on ISS data returns a TrainingResult with at least 1 epoch."""
    result = train_channel(iss_training_setup, _ISS_MISSION, _ISS_CHANNEL)
    assert isinstance(result, TrainingResult)
    assert result.epochs_run >= 1
    assert result.best_val_loss is not None


@pytest.mark.slow
def test_train_channel_iss_registers_model(
    iss_training_setup: Settings, mlflow_uri: str
) -> None:
    """train_channel registers telemanom-ISS-S1000003 with subsystem=thermal tag."""
    from mlflow.tracking import MlflowClient

    from spacecraft_telemetry.mlflow_tracking.conventions import registered_model_name

    train_channel(iss_training_setup, _ISS_MISSION, _ISS_CHANNEL)

    client = MlflowClient(tracking_uri=mlflow_uri)
    model_name = registered_model_name(
        iss_training_setup.model.model_type, _ISS_MISSION, _ISS_CHANNEL
    )
    assert model_name == "telemanom-ISS-S1000003"

    versions = list(client.search_model_versions(f"name='{model_name}'"))
    assert len(versions) >= 1, f"no registered versions for {model_name!r}"

    # Verify the subsystem tag — Step 1 wrote channel_subsystems.json so
    # load_channel_subsystem_map returns {'S1000003': 'thermal'}.
    exp_name = f"telemanom-training-{_ISS_MISSION}"
    exp = client.get_experiment_by_name(exp_name)
    assert exp is not None, f"experiment {exp_name!r} not created"
    runs = client.search_runs([exp.experiment_id])
    assert len(runs) == 1
    tags = runs[0].data.tags
    assert tags.get("subsystem") == "thermal", (
        f"Expected subsystem='thermal', got {tags.get('subsystem')!r}. "
        "channel_subsystems.json must be written by run_iss_preprocessing (Step 1)."
    )
    assert tags.get("mission_id") == _ISS_MISSION
    assert tags.get("channel_id") == _ISS_CHANNEL
