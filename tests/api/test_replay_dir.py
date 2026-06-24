"""Tests for the replay_data_dir override in api.app._load_replay_slice.

Verifies that when settings.api.replay_data_dir is set, the replay slice is
loaded from that directory instead of preprocess.processed_data_dir.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from spacecraft_telemetry.core.config import load_settings

_SERIES_SCHEMA = pa.schema(
    [
        pa.field("telemetry_timestamp", pa.timestamp("us", tz="UTC")),
        pa.field("value_normalized", pa.float32()),
        pa.field("segment_id", pa.int32()),
        pa.field("is_anomaly", pa.bool_()),
    ]
)


def _write_parquet(base: Path, mission: str, channel: str, n_anomaly_tail: int) -> None:
    import pandas as pd

    n_rows = 50
    n_clean = n_rows - n_anomaly_tail
    timestamps = pd.date_range("2000-01-01", periods=n_rows, freq="s", tz="UTC")
    rng = np.random.default_rng(0)
    values = rng.standard_normal(n_rows).astype("float32")
    is_anomaly = [False] * n_clean + [True] * n_anomaly_tail

    part = base / mission / "test" / f"mission_id={mission}" / f"channel_id={channel}"
    part.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "telemetry_timestamp": pa.array(timestamps.astype("datetime64[us, UTC]")),
                "value_normalized": pa.array(values),
                "segment_id": pa.array([0] * n_rows, type=pa.int32()),
                "is_anomaly": pa.array(is_anomaly),
            },
            schema=_SERIES_SCHEMA,
        ),
        part / "part.parquet",
    )


class TestReplayDataDirConfig:
    def test_default_is_none(self) -> None:
        s = load_settings("test")
        assert s.api.replay_data_dir is None

    def test_override_via_model_copy(self, tmp_path: Path) -> None:
        s = load_settings("test")
        overridden = s.model_copy(
            update={"api": s.api.model_copy(update={"replay_data_dir": str(tmp_path)})}
        )
        assert overridden.api.replay_data_dir == str(tmp_path)
        # Original untouched
        assert s.api.replay_data_dir is None


class TestReplayDataDirLoadSlice:
    """_load_replay_slice should read from replay_data_dir when set."""

    def test_loads_from_replay_data_dir(self, tmp_path: Path) -> None:
        from spacecraft_telemetry.api.app import _load_replay_slice
        from spacecraft_telemetry.core.logging import setup_logging

        mission = "test-m"
        channel = "ch1"

        nominal_dir = tmp_path / "nominal"
        injected_dir = tmp_path / "injected"

        # nominal dir has zero anomalies, injected dir has 20
        _write_parquet(nominal_dir, mission, channel, n_anomaly_tail=0)
        _write_parquet(injected_dir, mission, channel, n_anomaly_tail=20)

        base_settings = load_settings("test")
        settings = base_settings.model_copy(
            update={
                "preprocess": base_settings.preprocess.model_copy(
                    update={"processed_data_dir": nominal_dir}
                ),
                "api": base_settings.api.model_copy(
                    update={
                        "mission": mission,
                        "replay_data_dir": str(injected_dir),
                        "replay_warmup_rows": 0,
                        "replay_max_rows": 50,
                    }
                ),
            }
        )
        setup_logging(settings.logging)

        import structlog

        log = structlog.get_logger("test")
        sem = asyncio.Semaphore(1)

        async def run() -> None:
            _, data = await _load_replay_slice(channel, settings, sem, log)
            assert data is not None, "replay slice should load successfully"
            _values, anom, _timestamps = data
            # Data from injected_dir has anomalies; nominal_dir has none.
            # The slice must contain at least one anomaly row (from injected_dir).
            assert anom.any(), (
                "replay slice loaded from nominal dir instead of replay_data_dir"
            )

        asyncio.run(run())

    def test_falls_back_to_processed_data_dir(self, tmp_path: Path) -> None:
        import structlog

        from spacecraft_telemetry.api.app import _load_replay_slice

        mission = "test-m"
        channel = "ch1"
        _write_parquet(tmp_path, mission, channel, n_anomaly_tail=10)

        base_settings = load_settings("test")
        settings = base_settings.model_copy(
            update={
                "preprocess": base_settings.preprocess.model_copy(
                    update={"processed_data_dir": tmp_path}
                ),
                "api": base_settings.api.model_copy(
                    update={
                        "mission": mission,
                        # replay_data_dir is None → fall back to processed_data_dir
                        "replay_warmup_rows": 0,
                        "replay_max_rows": 50,
                    }
                ),
            }
        )

        log = structlog.get_logger("test")
        sem = asyncio.Semaphore(1)

        async def run() -> None:
            _, data = await _load_replay_slice(channel, settings, sem, log)
            assert data is not None
            _values, anom, _timestamps = data
            assert anom.any(), "fallback dir has anomalies — slice should contain them"

        asyncio.run(run())
