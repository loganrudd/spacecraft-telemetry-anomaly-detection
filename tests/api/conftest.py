"""Shared fixtures for the api test package."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

torch = pytest.importorskip("torch")

from types import MappingProxyType  # noqa: E402

from fastapi import FastAPI  # noqa: E402

from spacecraft_telemetry.api.endpoints import router  # noqa: E402
from spacecraft_telemetry.api.inference import ChannelInferenceEngine  # noqa: E402
from spacecraft_telemetry.api.logging_middleware import CorrelationIdMiddleware  # noqa: E402
from spacecraft_telemetry.api.state import AppState  # noqa: E402
from spacecraft_telemetry.core.config import Settings, load_settings  # noqa: E402
from spacecraft_telemetry.model.dataset import load_series_parquet  # noqa: E402
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

    values, _seg, anom, timestamps = load_series_parquet(
        api_parquet, _MISSION, _CHANNEL, "test"
    )

    app = FastAPI()
    app.state.settings = settings
    app.state.app_state = AppState(
        settings=settings,
        mission=_MISSION,
        subsystem=_SUBSYSTEM,
        device=torch.device("cpu"),
        engines=MappingProxyType({_CHANNEL: engine}),
        channel_subsystem_map=MappingProxyType({_CHANNEL: _SUBSYSTEM}),
        replay_data=MappingProxyType({_CHANNEL: (values, anom, timestamps)}),
        startup_monotonic_ns=time.monotonic_ns(),
        mlflow_tracking_uri=settings.mlflow.tracking_uri,
    )
    app.add_middleware(CorrelationIdMiddleware)
    app.include_router(router)
    return app


@pytest.fixture()
def running_app_with_spike(test_settings: Settings, tmp_path: Path) -> FastAPI:
    """FastAPI app wired to a Parquet containing a spike region.

    Layout (60 rows):
    - Rows 0-30 (31 rows): value = 0.0  -- warmup / nominal
    - Rows 31-59 (29 rows): value = 10.0 -- spike

    With _ZeroModel (always predicts 0), error_smoothing_window=5 (alpha=1/3),
    threshold_window=20, threshold_min_anomaly_len=2 (test.yaml defaults):
    - Threshold warmup ends after tick 30 (tick=window_size+threshold_window=30).
    - First spike tick (row 31): EWMA ≈ 3.33, threshold ≈ 0 → raw_flag=True.
    - Second spike tick (row 32): raw_flag=True → is_anomaly_predicted=True.

    The stream test uses this fixture to assert that the flag is ever True.
    """
    import pandas as pd

    n_zero, n_spike = 31, 29
    n_rows = n_zero + n_spike
    spike_values = np.concatenate(
        [np.zeros(n_zero, dtype=np.float32), np.full(n_spike, 10.0, dtype=np.float32)]
    )
    timestamps = pd.date_range("2000-01-01", periods=n_rows, freq="s", tz="UTC")
    is_anomaly = [False] * n_rows

    partition_dir = (
        tmp_path
        / _MISSION
        / "test"
        / f"mission_id={_MISSION}"
        / f"channel_id={_CHANNEL}"
    )
    partition_dir.mkdir(parents=True, exist_ok=True)
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table(
        {
            "telemetry_timestamp": pa.array(timestamps.astype("datetime64[us, UTC]")),
            "value_normalized": pa.array(spike_values),
            "segment_id": pa.array([0] * n_rows, type=pa.int32()),
            "is_anomaly": pa.array(is_anomaly),
        },
        schema=_SERIES_SCHEMA,
    )
    pq.write_table(table, partition_dir / "part.parquet")

    spike_parquet = tmp_path
    settings = test_settings.model_copy(
        update={
            "spark": test_settings.spark.model_copy(
                update={"processed_data_dir": spike_parquet}
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
    values_arr, _seg, anom_arr, ts_arr = load_series_parquet(
        spike_parquet, _MISSION, _CHANNEL, "test"
    )
    app = FastAPI()
    app.state.settings = settings
    app.state.app_state = AppState(
        settings=settings,
        mission=_MISSION,
        subsystem=_SUBSYSTEM,
        device=torch.device("cpu"),
        engines=MappingProxyType({_CHANNEL: engine}),
        channel_subsystem_map=MappingProxyType({_CHANNEL: _SUBSYSTEM}),
        replay_data=MappingProxyType({_CHANNEL: (values_arr, anom_arr, ts_arr)}),
        startup_monotonic_ns=time.monotonic_ns(),
        mlflow_tracking_uri=settings.mlflow.tracking_uri,
    )
    app.add_middleware(CorrelationIdMiddleware)
    app.include_router(router)
    return app


@pytest.fixture()
def running_app_empty(test_settings: Settings) -> FastAPI:
    """FastAPI app with an empty engines map — triggers 503 on /health.

    Constructs AppState directly with no loaded engines, which is the condition
    the degraded-health path requires. Uses a fixed startup timestamp so uptime
    is deterministic.
    """
    app = FastAPI()
    app.state.settings = test_settings
    app.state.app_state = AppState(
        settings=test_settings,
        mission=_MISSION,
        subsystem=_SUBSYSTEM,
        device=torch.device("cpu"),
        engines=MappingProxyType({}),
        channel_subsystem_map=MappingProxyType({}),
        replay_data=MappingProxyType({}),
        startup_monotonic_ns=time.monotonic_ns(),
        mlflow_tracking_uri=test_settings.mlflow.tracking_uri,
    )
    app.add_middleware(CorrelationIdMiddleware)
    app.include_router(router)
    return app

