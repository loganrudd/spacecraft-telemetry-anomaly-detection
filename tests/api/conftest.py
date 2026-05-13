"""Shared fixtures for the api test package."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

torch = pytest.importorskip("torch")

from fastapi import FastAPI  # noqa: E402

from spacecraft_telemetry.api.endpoints import router  # noqa: E402
from spacecraft_telemetry.api.inference import ChannelInferenceEngine  # noqa: E402
from spacecraft_telemetry.api.logging_middleware import CorrelationIdMiddleware  # noqa: E402
from spacecraft_telemetry.api.state import AppState  # noqa: E402
from spacecraft_telemetry.core.config import Settings, load_settings  # noqa: E402
from spacecraft_telemetry.model.io import ScoringParams  # noqa: E402

_MISSION = "test-mission"
_CHANNEL = "test-ch"
_SUBSYSTEM = "test-sub"

# PyArrow schema matching SERIES_SCHEMA (no Hive partition columns).
_SERIES_SCHEMA = pa.schema(
    [
        pa.field("telemetry_timestamp", pa.timestamp("us", tz="UTC")),
        pa.field("value_normalized", pa.float32()),
        pa.field("segment_id", pa.int32()),
        pa.field("is_anomaly", pa.bool_()),
    ]
)


def _write_series_parquet(
    base: Path,
    mission: str,
    channel: str,
    split: str,
    n_rows: int,
    n_anomaly_tail: int,
) -> None:
    """Write a tiny test-split Parquet partition under ``base``."""
    import pandas as pd

    n_clean = n_rows - n_anomaly_tail
    timestamps = pd.date_range("2000-01-01", periods=n_rows, freq="s", tz="UTC")
    rng = np.random.default_rng(42)
    values = rng.standard_normal(n_rows).astype("float32")
    is_anomaly = [False] * n_clean + [True] * n_anomaly_tail

    partition_dir = (
        base / mission / split / f"mission_id={mission}" / f"channel_id={channel}"
    )
    partition_dir.mkdir(parents=True, exist_ok=True)

    table = pa.table(
        {
            "telemetry_timestamp": pa.array(
                timestamps.astype("datetime64[us, UTC]")
            ),
            "value_normalized": pa.array(values),
            "segment_id": pa.array([0] * n_rows, type=pa.int32()),
            "is_anomaly": pa.array(is_anomaly),
        },
        schema=_SERIES_SCHEMA,
    )
    pq.write_table(table, partition_dir / "part.parquet")


class _ZeroModel(torch.nn.Module):
    """Stub LSTM that always predicts 0 — no training required."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], 1)


@pytest.fixture()
def test_settings() -> Settings:
    """Settings loaded from the test environment (test.yaml)."""
    return load_settings("test")


@pytest.fixture()
def api_parquet(tmp_path: Path) -> Path:
    """Write a tiny Parquet test-split and return the processed-data directory.

    100 rows total; last 20 are labeled as anomalies so that streaming tests
    can assert ``is_anomaly_true=True`` appears in the event stream.
    """
    _write_series_parquet(
        tmp_path,
        mission=_MISSION,
        channel=_CHANNEL,
        split="test",
        n_rows=100,
        n_anomaly_tail=20,
    )
    return tmp_path


@pytest.fixture()
def running_app(test_settings: Settings, api_parquet: Path) -> FastAPI:
    """FastAPI app with the router and pre-loaded AppState.

    Builds an ``AppState`` directly from a ``_ZeroModel`` engine and a tiny
    test Parquet — no MLflow training required.  The app has no lifespan
    handler so ``TestClient(running_app)`` can be used without a ``with``
    block; the state is already injected.
    """
    settings = test_settings.model_copy(
        update={
            "spark": test_settings.spark.model_copy(
                update={"processed_data_dir": api_parquet}
            )
        }
    )

    params = ScoringParams(
        threshold_window=settings.model.threshold_window,
        threshold_z=settings.model.threshold_z,
        error_smoothing_window=settings.model.error_smoothing_window,
        threshold_min_anomaly_len=settings.model.threshold_min_anomaly_len,
    )

    model: torch.nn.Module = _ZeroModel()
    model.eval()
    engine = ChannelInferenceEngine(
        mission=_MISSION,
        channel=_CHANNEL,
        model=model,  # type: ignore[arg-type]
        window_size=settings.model.window_size,
        params=params,
        device=torch.device("cpu"),
    )

    app = FastAPI()
    app.state.settings = settings
    app.state.app_state = AppState(
        settings=settings,
        mission=_MISSION,
        subsystem=_SUBSYSTEM,
        device=torch.device("cpu"),
        engines={_CHANNEL: engine},
        channel_subsystem_map={_CHANNEL: _SUBSYSTEM},
        startup_monotonic_ns=time.monotonic_ns(),
        mlflow_tracking_uri=settings.mlflow.tracking_uri,
    )
    app.add_middleware(CorrelationIdMiddleware)
    app.include_router(router)
    return app

