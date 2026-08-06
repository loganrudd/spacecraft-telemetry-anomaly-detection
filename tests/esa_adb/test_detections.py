"""Tests for esa_adb.detections — flag reconstruction and mission-level OR-aggregation.

find_scoring_run / channel_detection_intervals / mission_detection_intervals
are exercised against a real (SQLite) MLflow backend and a tiny hand-built
series Parquet, without ever training or running inference — the whole point
of Plan 019 Stage A is that flags are reconstructible from a scoring run's
logged errors.npy/threshold.npy artifacts alone.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from spacecraft_telemetry.core.config import Settings, load_settings
from spacecraft_telemetry.esa_adb.detections import (
    channel_detection_intervals,
    find_scoring_run,
    mission_detection_intervals,
    per_channel_detection_intervals,
)
from spacecraft_telemetry.mlflow_tracking import (
    common_tags,
    experiment_name,
    log_artifact_bytes,
    log_params,
    open_run,
)
from spacecraft_telemetry.model.io import errors_to_bytes, threshold_to_bytes

_MISSION = "ESA-Mission1-Test"
_WINDOW_SIZE = 3
_PREDICTION_HORIZON = 1
_SPAN = _WINDOW_SIZE + _PREDICTION_HORIZON
_FREQ_S = 90

_SERIES_SCHEMA = pa.schema(
    [
        pa.field("telemetry_timestamp", pa.timestamp("us", tz="UTC")),
        pa.field("value_normalized", pa.float32()),
        pa.field("segment_id", pa.int32()),
        pa.field("is_anomaly", pa.bool_()),
    ]
)


def _write_test_series(processed_dir: Path, mission: str, channel: str, n_rows: int) -> None:
    base = datetime(2000, 1, 1, tzinfo=UTC)
    timestamps = [
        pa.scalar(base.timestamp() + i * _FREQ_S, type=pa.timestamp("s", tz="UTC")).cast(
            pa.timestamp("us", tz="UTC")
        )
        for i in range(n_rows)
    ]
    table = pa.table(
        {
            "telemetry_timestamp": pa.array(timestamps, type=pa.timestamp("us", tz="UTC")),
            "value_normalized": pa.array([0.0] * n_rows, type=pa.float32()),
            "segment_id": pa.array([0] * n_rows, type=pa.int32()),
            "is_anomaly": pa.array([False] * n_rows, type=pa.bool_()),
        },
        schema=_SERIES_SCHEMA,
    )
    partition_dir = (
        processed_dir / mission / "test" / f"mission_id={mission}" / f"channel_id={channel}"
    )
    partition_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, partition_dir / "part.parquet")


def _settings(processed_dir: Path, mlflow_uri: str) -> Settings:
    base_settings = load_settings("test")
    return base_settings.model_copy(
        update={
            "preprocess": base_settings.preprocess.model_copy(
                update={"processed_data_dir": str(processed_dir)}
            ),
            "model": base_settings.model.model_copy(
                update={"window_size": _WINDOW_SIZE, "prediction_horizon": _PREDICTION_HORIZON}
            ),
            "mlflow": base_settings.mlflow.model_copy(update={"tracking_uri": mlflow_uri}),
        }
    )


def _log_scoring_run(
    settings: Settings,
    mission: str,
    channel: str,
    *,
    smoothed: np.ndarray,
    threshold: np.ndarray,
    min_run_length: int,
    tuned: bool = False,
) -> str:
    """Log a scoring-run-shaped MLflow run without training or running inference.

    Mirrors exactly the artifacts/params/tags model.scoring.score_channel()
    itself logs (errors.npy, threshold.npy, threshold_min_anomaly_len param,
    channel_id tag, optional tuned_from_run tag) so detections.py exercises
    the real read path.
    """
    import mlflow

    exp = experiment_name(settings.model.model_type, "scoring", mission)
    extra = {"eval_split": "full_test"}
    if tuned:
        extra["tuned_from_run"] = "fake-hpo-run"
    tags = common_tags(
        model_type=settings.model.model_type,
        mission=mission,
        phase="scoring",
        channel=channel,
        extra=extra,
    )
    with open_run(experiment=exp, run_name=channel, tags=tags) as run:
        assert run is not None, "open_run failed to create a run against the test SQLite backend"
        log_params({"threshold_min_anomaly_len": min_run_length})
        log_artifact_bytes(errors_to_bytes(smoothed), "errors.npy")
        log_artifact_bytes(threshold_to_bytes(threshold), "threshold.npy")
        run_id = run.info.run_id
    client = mlflow.MlflowClient(tracking_uri=settings.mlflow.tracking_uri)
    assert client.get_run(run_id) is not None
    return str(run_id)


# ---------------------------------------------------------------------------
# find_scoring_run
# ---------------------------------------------------------------------------


class TestFindScoringRun:
    def test_raises_when_experiment_missing(self, mlflow_uri: str) -> None:
        with pytest.raises(RuntimeError, match="No MLflow experiment"):
            find_scoring_run("does-not-exist", "channel_41", mlflow_uri, tuned=False)

    def test_finds_untuned_run(self, tmp_path: Path, mlflow_uri: str) -> None:
        settings = _settings(tmp_path / "processed", mlflow_uri)
        run_id = _log_scoring_run(
            settings,
            _MISSION,
            "channel_41",
            smoothed=np.zeros(5),
            threshold=np.ones(5),
            min_run_length=1,
            tuned=False,
        )
        exp = experiment_name(settings.model.model_type, "scoring", _MISSION)
        found = find_scoring_run(exp, "channel_41", mlflow_uri, tuned=False)
        assert found == run_id

    def test_raises_when_only_untuned_exists_but_tuned_requested(
        self, tmp_path: Path, mlflow_uri: str
    ) -> None:
        settings = _settings(tmp_path / "processed", mlflow_uri)
        _log_scoring_run(
            settings,
            _MISSION,
            "channel_41",
            smoothed=np.zeros(5),
            threshold=np.ones(5),
            min_run_length=1,
            tuned=False,
        )
        exp = experiment_name(settings.model.model_type, "scoring", _MISSION)
        with pytest.raises(RuntimeError, match="tuned scoring run"):
            find_scoring_run(exp, "channel_41", mlflow_uri, tuned=True)

    def test_finds_tuned_run_over_untuned(self, tmp_path: Path, mlflow_uri: str) -> None:
        settings = _settings(tmp_path / "processed", mlflow_uri)
        _log_scoring_run(
            settings,
            _MISSION,
            "channel_41",
            smoothed=np.zeros(5),
            threshold=np.ones(5),
            min_run_length=1,
            tuned=False,
        )
        tuned_run_id = _log_scoring_run(
            settings,
            _MISSION,
            "channel_41",
            smoothed=np.zeros(5),
            threshold=np.ones(5),
            min_run_length=1,
            tuned=True,
        )
        exp = experiment_name(settings.model.model_type, "scoring", _MISSION)
        assert find_scoring_run(exp, "channel_41", mlflow_uri, tuned=True) == tuned_run_id


# ---------------------------------------------------------------------------
# channel_detection_intervals
# ---------------------------------------------------------------------------


class TestChannelDetectionIntervals:
    def test_reconstructs_flag_interval_from_logged_run(
        self, tmp_path: Path, mlflow_uri: str
    ) -> None:
        processed_dir = tmp_path / "processed"
        n_rows = 10  # -> M = n_rows - span + 1 = 7 windows
        _write_test_series(processed_dir, _MISSION, "channel_41", n_rows)
        settings = _settings(processed_dir, mlflow_uri)

        # Flags true at window indices 3,4 (smoothed > threshold), run length 2.
        smoothed = np.array([0, 0, 0, 5, 5, 0, 0], dtype=np.float64)
        threshold = np.ones(7, dtype=np.float64)
        run_id = _log_scoring_run(
            settings,
            _MISSION,
            "channel_41",
            smoothed=smoothed,
            threshold=threshold,
            min_run_length=2,
        )

        intervals = channel_detection_intervals(settings, _MISSION, "channel_41", run_id)

        assert len(intervals) == 1
        start, end = intervals[0]
        # target_timestamps[i] = base + (i + span - 1) * FREQ_S; span=4.
        # Intervals are tz-aware UTC (see _flags_to_intervals docstring) —
        # the underlying Parquet timestamps are naive numpy datetime64 but
        # represent UTC instants.
        base = pd.Timestamp("2000-01-01T00:00:00", tz="UTC")
        expected_start = base + pd.Timedelta((3 + _SPAN - 1) * _FREQ_S, "s")
        expected_end = (
            base + pd.Timedelta((4 + _SPAN - 1) * _FREQ_S, "s") + pd.Timedelta(_FREQ_S, "s")
        )
        assert start == expected_start
        assert end == expected_end

    def test_no_flags_returns_empty_list(self, tmp_path: Path, mlflow_uri: str) -> None:
        processed_dir = tmp_path / "processed"
        _write_test_series(processed_dir, _MISSION, "channel_41", n_rows=10)
        settings = _settings(processed_dir, mlflow_uri)

        run_id = _log_scoring_run(
            settings,
            _MISSION,
            "channel_41",
            smoothed=np.zeros(7),
            threshold=np.ones(7),
            min_run_length=1,
        )
        assert channel_detection_intervals(settings, _MISSION, "channel_41", run_id) == []

    def test_window_count_mismatch_raises(self, tmp_path: Path, mlflow_uri: str) -> None:
        processed_dir = tmp_path / "processed"
        _write_test_series(processed_dir, _MISSION, "channel_41", n_rows=10)  # M=7
        settings = _settings(processed_dir, mlflow_uri)

        # Logged errors.npy has the wrong length (5, not 7).
        run_id = _log_scoring_run(
            settings,
            _MISSION,
            "channel_41",
            smoothed=np.zeros(5),
            threshold=np.ones(5),
            min_run_length=1,
        )
        with pytest.raises(ValueError, match="Window count mismatch"):
            channel_detection_intervals(settings, _MISSION, "channel_41", run_id)


# ---------------------------------------------------------------------------
# per_channel_detection_intervals
# ---------------------------------------------------------------------------


class TestPerChannelDetectionIntervals:
    def test_keeps_channel_identity_separate(self, tmp_path: Path, mlflow_uri: str) -> None:
        processed_dir = tmp_path / "processed"
        _write_test_series(processed_dir, _MISSION, "channel_41", n_rows=10)
        _write_test_series(processed_dir, _MISSION, "channel_42", n_rows=10)
        settings = _settings(processed_dir, mlflow_uri)

        smoothed_41 = np.array([5, 0, 0, 0, 0, 0, 0], dtype=np.float64)
        smoothed_42 = np.array([0, 0, 0, 0, 0, 5, 0], dtype=np.float64)
        threshold = np.ones(7, dtype=np.float64)
        _log_scoring_run(
            settings,
            _MISSION,
            "channel_41",
            smoothed=smoothed_41,
            threshold=threshold,
            min_run_length=1,
        )
        _log_scoring_run(
            settings,
            _MISSION,
            "channel_42",
            smoothed=smoothed_42,
            threshold=threshold,
            min_run_length=1,
        )

        result = per_channel_detection_intervals(
            settings, _MISSION, ["channel_41", "channel_42"], tuned=False
        )
        assert set(result.keys()) == {"channel_41", "channel_42"}
        assert len(result["channel_41"]) == 1
        assert len(result["channel_42"]) == 1
        assert result["channel_41"] != result["channel_42"]


# ---------------------------------------------------------------------------
# mission_detection_intervals
# ---------------------------------------------------------------------------


class TestMissionDetectionIntervals:
    def test_unions_across_channels(self, tmp_path: Path, mlflow_uri: str) -> None:
        processed_dir = tmp_path / "processed"
        _write_test_series(processed_dir, _MISSION, "channel_41", n_rows=10)
        _write_test_series(processed_dir, _MISSION, "channel_42", n_rows=10)
        settings = _settings(processed_dir, mlflow_uri)

        # channel_41 flags at window 0; channel_42 flags at window 5 — disjoint.
        smoothed_41 = np.array([5, 0, 0, 0, 0, 0, 0], dtype=np.float64)
        smoothed_42 = np.array([0, 0, 0, 0, 0, 5, 0], dtype=np.float64)
        threshold = np.ones(7, dtype=np.float64)
        _log_scoring_run(
            settings,
            _MISSION,
            "channel_41",
            smoothed=smoothed_41,
            threshold=threshold,
            min_run_length=1,
        )
        _log_scoring_run(
            settings,
            _MISSION,
            "channel_42",
            smoothed=smoothed_42,
            threshold=threshold,
            min_run_length=1,
        )

        intervals = mission_detection_intervals(
            settings, _MISSION, ["channel_41", "channel_42"], tuned=False
        )
        assert len(intervals) == 2, "disjoint per-channel flags must not merge"

    def test_missing_channel_run_raises(self, tmp_path: Path, mlflow_uri: str) -> None:
        processed_dir = tmp_path / "processed"
        _write_test_series(processed_dir, _MISSION, "channel_41", n_rows=10)
        settings = _settings(processed_dir, mlflow_uri)

        _log_scoring_run(
            settings,
            _MISSION,
            "channel_41",
            smoothed=np.zeros(7),
            threshold=np.ones(7),
            min_run_length=1,
        )
        with pytest.raises(RuntimeError, match="channel_42"):
            mission_detection_intervals(
                settings, _MISSION, ["channel_41", "channel_42"], tuned=False
            )
