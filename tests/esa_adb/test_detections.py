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
from typing import ClassVar

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from spacecraft_telemetry.core.config import Settings, load_settings
from spacecraft_telemetry.esa_adb.detections import (
    channel_detection_intervals,
    channel_detection_intervals_from_spec,
    find_scoring_run,
    find_scoring_run_and_artifacts,
    find_scoring_run_id,
    mission_detection_intervals,
    mission_intervals_from_per_channel,
    per_channel_detection_intervals,
)
from spacecraft_telemetry.esa_adb.offline import OfflineRunMap, RunSpec
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
            find_scoring_run_id("does-not-exist", "channel_41", mlflow_uri, tuned=False)

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
        found = find_scoring_run_id(exp, "channel_41", mlflow_uri, tuned=False)
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
            find_scoring_run_id(exp, "channel_41", mlflow_uri, tuned=True)

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
        assert find_scoring_run_id(exp, "channel_41", mlflow_uri, tuned=True) == tuned_run_id

    def test_returns_the_full_run_with_tags_not_just_the_id(
        self, tmp_path: Path, mlflow_uri: str
    ) -> None:
        """docs/reviews/022, item A2: find_scoring_run now returns the Run
        (search_runs already fully populates it, tags included) rather than
        discarding everything but the id — callers that need tags (e.g.
        tuned_provenance) no longer pay a second client.get_run() round trip
        for a run this function already had in hand."""
        settings = _settings(tmp_path / "processed", mlflow_uri)
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
        run = find_scoring_run(exp, "channel_41", mlflow_uri, tuned=True)
        assert str(run.info.run_id) == tuned_run_id
        assert "tuned_from_run" in run.data.tags


class TestTunedRunClassification:
    """A run scored with grid-selected params must NOT read as the untuned
    baseline. score_channel writes `tuned_from_run` only when given an HPO run
    id; a config from scripts/threshold_ceiling.py has provenance but no such
    run, so it carries `tuned_source` instead. If only the former counted, that
    run would be filed as the protocol-matched Hundman-defaults row the ESA-ADB
    report compares against the paper — a tuned config masquerading as untuned.
    """

    def _run_with(self, tags: dict[str, str]) -> object:
        class _Run:
            def __init__(self, t: dict[str, str]) -> None:
                class _D:
                    def __init__(self, tt: dict[str, str]) -> None:
                        self.tags = tt

                self.data = _D(t)

        return _Run(tags)

    def test_tuned_from_run_counts_as_tuned(self) -> None:
        from spacecraft_telemetry.esa_adb.detections import _is_tuned_run

        assert _is_tuned_run(self._run_with({"tuned_from_run": "abc"})) is True

    def test_tuned_source_counts_as_tuned(self) -> None:
        from spacecraft_telemetry.esa_adb.detections import _is_tuned_run

        assert _is_tuned_run(self._run_with({"tuned_source": "grid"})) is True

    def test_neither_tag_is_untuned(self) -> None:
        from spacecraft_telemetry.esa_adb.detections import _is_tuned_run

        assert _is_tuned_run(self._run_with({"eval_split": "full_test"})) is False

    def test_grid_tuned_run_is_not_returned_as_the_untuned_baseline(
        self, tmp_path: Path, mlflow_uri: str
    ) -> None:
        """End-to-end: a genuine untuned run and a NEWER grid-tuned run coexist.
        Asking for untuned must return the older genuine baseline, not the
        newer grid run — recency must not override the tuned/untuned split."""
        import mlflow

        settings = _settings(tmp_path / "processed", mlflow_uri)
        baseline_id = _log_scoring_run(
            settings, _MISSION, "channel_41",
            smoothed=np.zeros(5), threshold=np.ones(5), min_run_length=1, tuned=False,
        )
        # A later grid-tuned run, tagged the way runner.py now tags one.
        exp = experiment_name(settings.model.model_type, "scoring", _MISSION)
        tags = common_tags(
            model_type=settings.model.model_type, mission=_MISSION, phase="scoring",
            channel="channel_41",
            extra={"eval_split": "full_test", "tuned_source": "threshold_ceiling grid"},
        )
        with open_run(experiment=exp, run_name="channel_41", tags=tags) as run:
            assert run is not None
            log_params({"threshold_min_anomaly_len": 1})
            log_artifact_bytes(errors_to_bytes(np.zeros(5)), "errors.npy")
            log_artifact_bytes(threshold_to_bytes(np.ones(5)), "threshold.npy")
            grid_id = run.info.run_id
        assert mlflow.MlflowClient(tracking_uri=mlflow_uri).get_run(grid_id) is not None

        assert find_scoring_run_id(exp, "channel_41", mlflow_uri, tuned=False) == baseline_id
        assert find_scoring_run_id(exp, "channel_41", mlflow_uri, tuned=True) == grid_id


def _log_multivariate_scoring_run(
    settings: Settings,
    mission: str,
    subsystem: str,
    channels: list[str],
    *,
    smoothed_by_channel: dict[str, np.ndarray],
    threshold_by_channel: dict[str, np.ndarray],
    min_run_length: int,
    tuned: bool = False,
) -> str:
    """Log a run shaped exactly like model.scoring.score_channel's multivariate
    output: NO channel_id tag, a subsystem tag plus a comma-joined `channels`
    tag, and per-channel arrays nested at errors/{ch}.npy, threshold/{ch}.npy.
    """
    import mlflow

    exp = experiment_name(settings.model.model_type, "scoring", mission)
    extra = {"eval_split": "full_test", "channels": ",".join(channels)}
    if tuned:
        extra["tuned_from_run"] = "fake-hpo-run"
    tags = common_tags(
        model_type=settings.model.model_type,
        mission=mission,
        phase="scoring",
        channel=None,  # the defining property: no channel_id on a joint run
        subsystem=subsystem,
        extra=extra,
    )
    with open_run(experiment=exp, run_name=subsystem, tags=tags) as run:
        assert run is not None
        log_params({"threshold_min_anomaly_len": min_run_length})
        for ch in channels:
            log_artifact_bytes(errors_to_bytes(smoothed_by_channel[ch]), f"errors/{ch}.npy")
            log_artifact_bytes(
                threshold_to_bytes(threshold_by_channel[ch]), f"threshold/{ch}.npy"
            )
        run_id = run.info.run_id
    client = mlflow.MlflowClient(tracking_uri=settings.mlflow.tracking_uri)
    assert client.get_run(run_id) is not None
    return str(run_id)


class TestFindScoringRunAndArtifacts:
    """docs/plans/021: `esa-adb report` must locate a channel scored inside a
    multivariate subsystem group. Such a run carries no channel_id tag, so the
    univariate lookup can never match it — without the fallback the whole
    mission-level report is unobtainable for a multivariate arm."""

    _CHANNELS: ClassVar[list[str]] = ["channel_41", "channel_42"]

    def _mv_settings(self, tmp_path: Path, mlflow_uri: str) -> Settings:
        """Settings whose subsystem map resolves both channels to subsystem_5."""
        import json

        processed = tmp_path / "processed"
        meta = processed / _MISSION / "metadata"
        meta.mkdir(parents=True, exist_ok=True)
        (meta / "channel_subsystems.json").write_text(
            json.dumps({ch: "subsystem_5" for ch in self._CHANNELS})
        )
        return _settings(processed, mlflow_uri)

    def test_univariate_lookup_still_wins_and_uses_root_artifacts(
        self, tmp_path: Path, mlflow_uri: str
    ) -> None:
        """Null-default: a univariate run must resolve exactly as before."""
        settings = self._mv_settings(tmp_path, mlflow_uri)
        run_id = _log_scoring_run(
            settings, _MISSION, "channel_41",
            smoothed=np.zeros(5), threshold=np.ones(5), min_run_length=1, tuned=False,
        )
        exp = experiment_name(settings.model.model_type, "scoring", _MISSION)
        got = find_scoring_run_and_artifacts(
            settings, _MISSION, exp, "channel_41", tuned=False
        )
        assert got == (run_id, "errors.npy", "threshold.npy")

    def test_falls_back_to_multivariate_run_with_nested_artifacts(
        self, tmp_path: Path, mlflow_uri: str
    ) -> None:
        settings = self._mv_settings(tmp_path, mlflow_uri)
        run_id = _log_multivariate_scoring_run(
            settings, _MISSION, "subsystem_5", self._CHANNELS,
            smoothed_by_channel={c: np.zeros(5) for c in self._CHANNELS},
            threshold_by_channel={c: np.ones(5) for c in self._CHANNELS},
            min_run_length=1,
        )
        exp = experiment_name(settings.model.model_type, "scoring", _MISSION)
        for ch in self._CHANNELS:
            assert find_scoring_run_and_artifacts(
                settings, _MISSION, exp, ch, tuned=False
            ) == (run_id, f"errors/{ch}.npy", f"threshold/{ch}.npy")

    def test_rejects_run_whose_channels_tag_excludes_the_channel(
        self, tmp_path: Path, mlflow_uri: str
    ) -> None:
        """A shared subsystem name is not proof of membership — a narrower
        group could reuse it. Must not silently attribute another group's
        errors array to this channel."""
        settings = self._mv_settings(tmp_path, mlflow_uri)
        _log_multivariate_scoring_run(
            settings, _MISSION, "subsystem_5", ["channel_41"],  # 42 NOT a member
            smoothed_by_channel={"channel_41": np.zeros(5)},
            threshold_by_channel={"channel_41": np.ones(5)},
            min_run_length=1,
        )
        exp = experiment_name(settings.model.model_type, "scoring", _MISSION)
        with pytest.raises(RuntimeError, match="multivariate scoring run"):
            find_scoring_run_and_artifacts(settings, _MISSION, exp, "channel_42", tuned=False)

    def test_respects_tuned_flag(self, tmp_path: Path, mlflow_uri: str) -> None:
        settings = self._mv_settings(tmp_path, mlflow_uri)
        _log_multivariate_scoring_run(
            settings, _MISSION, "subsystem_5", self._CHANNELS,
            smoothed_by_channel={c: np.zeros(5) for c in self._CHANNELS},
            threshold_by_channel={c: np.ones(5) for c in self._CHANNELS},
            min_run_length=1, tuned=False,
        )
        exp = experiment_name(settings.model.model_type, "scoring", _MISSION)
        with pytest.raises(RuntimeError):
            find_scoring_run_and_artifacts(settings, _MISSION, exp, "channel_41", tuned=True)

        tuned_id = _log_multivariate_scoring_run(
            settings, _MISSION, "subsystem_5", self._CHANNELS,
            smoothed_by_channel={c: np.zeros(5) for c in self._CHANNELS},
            threshold_by_channel={c: np.ones(5) for c in self._CHANNELS},
            min_run_length=1, tuned=True,
        )
        assert find_scoring_run_and_artifacts(
            settings, _MISSION, exp, "channel_41", tuned=True
        )[0] == tuned_id

    def test_error_names_both_searches(self, tmp_path: Path, mlflow_uri: str) -> None:
        """When neither lookup finds anything, the message must say so — a
        univariate-only error would name the wrong remedy on a multivariate
        mission."""
        settings = self._mv_settings(tmp_path, mlflow_uri)
        _log_scoring_run(
            settings, _MISSION, "channel_41",
            smoothed=np.zeros(5), threshold=np.ones(5), min_run_length=1, tuned=False,
        )
        exp = experiment_name(settings.model.model_type, "scoring", _MISSION)
        with pytest.raises(RuntimeError) as exc:
            find_scoring_run_and_artifacts(settings, _MISSION, exp, "channel_99", tuned=False)
        msg = str(exc.value)
        assert "channel_id" in msg and "multivariate scoring run" in msg


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
        """The stale-run guard: a mismatched errors.npy length must raise, not
        silently misalign flags to the wrong timestamps. This is the only guard
        standing between a stale run (re-preprocessed but not re-scored) and
        corrupted detection intervals — the message must carry both counts so
        the mismatch is diagnosable without re-deriving it from the arrays.
        """
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
        with pytest.raises(ValueError, match="Window count mismatch") as exc_info:
            channel_detection_intervals(settings, _MISSION, "channel_41", run_id)
        message = str(exc_info.value)
        assert "5 windows" in message, "message must carry the errors.npy window count"
        assert "yields 7" in message, "message must carry the current test-partition window count"


# ---------------------------------------------------------------------------
# channel_detection_intervals_from_spec (offline path — no MLflow server)
# ---------------------------------------------------------------------------


def _write_run_spec_files(
    runs_dir: Path, run_id: str, smoothed: np.ndarray, threshold: np.ndarray
) -> RunSpec:
    """Write errors.npy / threshold.npy to disk and return the matching RunSpec.

    Mirrors what scripts/stage_adb_offline.sh downloads from GCS, but writes
    directly via model.io's own serialisers so no MLflow run is involved.
    """
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "errors.npy").write_bytes(errors_to_bytes(smoothed))
    (run_dir / "threshold.npy").write_bytes(threshold_to_bytes(threshold))
    return RunSpec(
        run_id=run_id,
        errors_path=str(run_dir / "errors.npy"),
        threshold_path=str(run_dir / "threshold.npy"),
        threshold_min_anomaly_len=2,
    )


class TestChannelDetectionIntervalsFromSpec:
    def test_matches_mlflow_path_result(self, tmp_path: Path, mlflow_uri: str) -> None:
        """The offline path must reconstruct identical intervals to the MLflow path."""
        processed_dir = tmp_path / "processed"
        n_rows = 10  # -> M = 7 windows
        _write_test_series(processed_dir, _MISSION, "channel_41", n_rows)
        settings = _settings(processed_dir, mlflow_uri)

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
        online_intervals = channel_detection_intervals(settings, _MISSION, "channel_41", run_id)

        spec = _write_run_spec_files(tmp_path / "runs", "offline-run", smoothed, threshold)
        offline_intervals = channel_detection_intervals_from_spec(
            settings, _MISSION, "channel_41", spec
        )

        assert offline_intervals == online_intervals
        assert len(offline_intervals) == 1

    def test_no_flags_returns_empty_list(self, tmp_path: Path) -> None:
        processed_dir = tmp_path / "processed"
        _write_test_series(processed_dir, _MISSION, "channel_41", n_rows=10)
        settings = _settings(processed_dir, "sqlite:///unused.db")

        spec = _write_run_spec_files(tmp_path / "runs", "offline-empty", np.zeros(7), np.ones(7))
        assert channel_detection_intervals_from_spec(settings, _MISSION, "channel_41", spec) == []

    def test_window_count_mismatch_raises(self, tmp_path: Path) -> None:
        """Offline counterpart of the stale-run guard — same message contract."""
        processed_dir = tmp_path / "processed"
        _write_test_series(processed_dir, _MISSION, "channel_41", n_rows=10)  # M=7
        settings = _settings(processed_dir, "sqlite:///unused.db")

        spec = _write_run_spec_files(
            tmp_path / "runs", "offline-mismatch", np.zeros(5), np.ones(5)
        )
        with pytest.raises(ValueError, match="Window count mismatch") as exc_info:
            channel_detection_intervals_from_spec(settings, _MISSION, "channel_41", spec)
        message = str(exc_info.value)
        assert "5 windows" in message, "message must carry the errors.npy window count"
        assert "yields 7" in message, "message must carry the current test-partition window count"

    def test_missing_artifact_raises_file_not_found(self, tmp_path: Path) -> None:
        processed_dir = tmp_path / "processed"
        _write_test_series(processed_dir, _MISSION, "channel_41", n_rows=10)
        settings = _settings(processed_dir, "sqlite:///unused.db")

        spec = RunSpec(
            run_id="missing",
            errors_path=str(tmp_path / "runs" / "missing" / "errors.npy"),
            threshold_path=str(tmp_path / "runs" / "missing" / "threshold.npy"),
            threshold_min_anomaly_len=2,
        )
        with pytest.raises(FileNotFoundError):
            channel_detection_intervals_from_spec(settings, _MISSION, "channel_41", spec)


# ---------------------------------------------------------------------------
# per_channel_detection_intervals / mission_detection_intervals with run_map
# ---------------------------------------------------------------------------


class TestOfflineRunMapIntegration:
    def test_per_channel_uses_run_map_without_mlflow(self, tmp_path: Path) -> None:
        """With a run_map, no MLflow server is contacted at all (bogus tracking_uri)."""
        processed_dir = tmp_path / "processed"
        _write_test_series(processed_dir, _MISSION, "channel_41", n_rows=10)
        settings = _settings(processed_dir, "sqlite:///this/path/does/not/exist.db")

        smoothed = np.array([0, 0, 0, 5, 5, 0, 0], dtype=np.float64)
        threshold = np.ones(7, dtype=np.float64)
        spec = _write_run_spec_files(tmp_path / "runs", "run-a", smoothed, threshold)
        run_map = OfflineRunMap(mission=_MISSION, baseline={"channel_41": spec}, tuned={})

        result = per_channel_detection_intervals(
            settings, _MISSION, ["channel_41"], tuned=False, run_map=run_map
        )
        assert len(result["channel_41"]) == 1

    def test_mission_detection_intervals_unions_via_run_map(self, tmp_path: Path) -> None:
        processed_dir = tmp_path / "processed"
        _write_test_series(processed_dir, _MISSION, "channel_41", n_rows=10)
        _write_test_series(processed_dir, _MISSION, "channel_42", n_rows=10)
        settings = _settings(processed_dir, "sqlite:///this/path/does/not/exist.db")

        # _write_run_spec_files hardcodes threshold_min_anomaly_len=2, so each
        # spike must span >=2 consecutive windows to survive flag_anomalies.
        threshold = np.ones(7, dtype=np.float64)
        spec_41 = _write_run_spec_files(
            tmp_path / "runs",
            "run-41",
            np.array([5, 5, 0, 0, 0, 0, 0], dtype=np.float64),
            threshold,
        )
        spec_42 = _write_run_spec_files(
            tmp_path / "runs",
            "run-42",
            np.array([0, 0, 0, 0, 5, 5, 0], dtype=np.float64),
            threshold,
        )
        run_map = OfflineRunMap(
            mission=_MISSION,
            baseline={"channel_41": spec_41, "channel_42": spec_42},
            tuned={},
        )

        result = mission_detection_intervals(
            settings, _MISSION, ["channel_41", "channel_42"], tuned=False, run_map=run_map
        )
        assert len(result) == 2

    def test_missing_channel_in_run_map_raises_key_error(self, tmp_path: Path) -> None:
        processed_dir = tmp_path / "processed"
        _write_test_series(processed_dir, _MISSION, "channel_41", n_rows=10)
        settings = _settings(processed_dir, "sqlite:///this/path/does/not/exist.db")

        run_map = OfflineRunMap(mission=_MISSION, baseline={}, tuned={})
        with pytest.raises(KeyError, match="channel_41"):
            per_channel_detection_intervals(
                settings, _MISSION, ["channel_41"], tuned=False, run_map=run_map
            )


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

    def test_configure_mlflow_failure_is_logged_not_silently_swallowed(
        self,
        tmp_path: Path,
        mlflow_uri: str,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A configure_mlflow failure (e.g. GCP ID-token fetch) must be visible.

        Regression for the observed 2026-08-15 case: fetch_id_token_failed
        passed silently and the run only worked because ambient gcloud
        credentials happened to cover it. The failure must not vanish — and
        since this branch is where MLflow IS required, the call must still
        fail loudly downstream (find_scoring_run) rather than being masked.

        Checks capsys rather than structlog.testing.capture_logs(): this
        module's logger is realized (and cached, per core/logging.py's
        cache_logger_on_first_use=True) well before this test runs inside the
        full suite, which makes capture_logs() miss the event. structlog
        emits to stdout regardless (see test_iss_io.py's precedent).
        """
        processed_dir = tmp_path / "processed"
        _write_test_series(processed_dir, _MISSION, "channel_41", n_rows=10)
        settings = _settings(processed_dir, mlflow_uri)

        def _raise(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("fetch_id_token_failed")

        monkeypatch.setattr(
            "spacecraft_telemetry.esa_adb.detections.configure_mlflow", _raise
        )

        with pytest.raises(RuntimeError, match="No MLflow experiment"):
            per_channel_detection_intervals(settings, _MISSION, ["channel_41"], tuned=False)

        stdout = capsys.readouterr().out
        assert "esa_adb.detections.configure_mlflow_failed" in stdout
        assert "warning" in stdout
        assert "fetch_id_token_failed" in stdout


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


# ---------------------------------------------------------------------------
# mission_intervals_from_per_channel
# ---------------------------------------------------------------------------


class TestMissionIntervalsFromPerChannel:
    """Pure interval-algebra tests — no MLflow/parquet I/O.

    esa_adb.report.build_report composes from an already-fetched
    per_channel_detection_intervals() dict via this function instead of
    calling mission_detection_intervals() (which would re-fetch every
    channel's artifacts a second time — see A1 in docs/reviews/019).
    """

    def _ts(self, *minutes: int) -> list[pd.Timestamp]:
        base = pd.Timestamp("2000-01-01T00:00:00Z")
        return [base + pd.Timedelta(minutes=m) for m in minutes]

    def test_unions_disjoint_channels(self) -> None:
        t0, t1, t2, t3 = self._ts(0, 1, 2, 3)
        per_channel = {"channel_41": [(t0, t1)], "channel_42": [(t2, t3)]}
        result = mission_intervals_from_per_channel(per_channel)
        assert result == [(t0, t1), (t2, t3)]

    def test_merges_overlapping_channels(self) -> None:
        t0, t1, t2, t3 = self._ts(0, 1, 2, 3)
        per_channel = {"channel_41": [(t0, t2)], "channel_42": [(t1, t3)]}
        result = mission_intervals_from_per_channel(per_channel)
        assert result == [(t0, t3)]

    def test_empty_dict_returns_empty_list(self) -> None:
        assert mission_intervals_from_per_channel({}) == []

    def test_matches_mission_detection_intervals(
        self, tmp_path: Path, mlflow_uri: str
    ) -> None:
        """Composing from an already-fetched dict must equal the fetch-again wrapper."""
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

        channels = ["channel_41", "channel_42"]
        per_channel = per_channel_detection_intervals(settings, _MISSION, channels, tuned=False)
        composed = mission_intervals_from_per_channel(per_channel)
        fetched_again = mission_detection_intervals(settings, _MISSION, channels, tuned=False)
        assert composed == fetched_again


class TestMultivariateRunLookupUsesGroupKey:
    """The mission report resolves a multivariate run by the key it was SCORED
    under, which since docs/plans/023 stage .4 is the channel GROUP.

    model/scoring.py tags a multivariate run ``subsystem = <the key passed as
    `channel`>`` — a group id like ``subsystem_6_g09``, not ``subsystem_6``.
    Querying the subsystem matches nothing, every channel resolves to "not
    scored", and the report silently drops its multivariate detections while
    still producing a plausible-looking table. Asserting only "a run was
    found" would pass against that bug, so this pins the tag value queried.
    """

    class _Run:
        def __init__(self, tags: dict[str, str]) -> None:
            class _D:
                def __init__(self, t: dict[str, str]) -> None:
                    self.tags = t

            self.data = _D(tags)
            self.info = type("I", (), {"run_id": "mv-run"})()

    def _patch(self, monkeypatch: pytest.MonkeyPatch, group_map: dict[str, str]) -> list[str]:
        import sys
        import types as _types

        from spacecraft_telemetry.esa_adb import detections as _det

        queried: list[str] = []
        run = self._Run({"channels": "channel_47,channel_48,channel_49", "tuned_source": "grid"})

        class _Client:
            def __init__(self, *a: object, **k: object) -> None: ...
            def get_experiment_by_name(self, _n: str) -> object:
                return type("E", (), {"experiment_id": "1"})()
            def search_runs(self, _ids: object, filter_string: str = "", **_k: object) -> list:
                queried.append(filter_string)
                return [run]

        monkeypatch.setitem(
            sys.modules, "mlflow", _types.SimpleNamespace(MlflowClient=_Client)
        )
        monkeypatch.setattr(_det, "load_channel_group_map", lambda *_a, **_k: group_map)
        return queried

    def test_queries_the_group_key_not_the_subsystem(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from spacecraft_telemetry.esa_adb.detections import _find_multivariate_scoring_run

        queried = self._patch(monkeypatch, {"channel_47": "subsystem_6_g09"})
        settings = load_settings("test")
        found = _find_multivariate_scoring_run(
            settings, "ESA-Mission1", "exp", "channel_47", tuned=True
        )
        assert found is not None
        assert queried == ["tags.subsystem = 'subsystem_6_g09'"], queried

    def test_channel_absent_from_the_group_map_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from spacecraft_telemetry.esa_adb.detections import _find_multivariate_scoring_run

        self._patch(monkeypatch, {})
        settings = load_settings("test")
        assert (
            _find_multivariate_scoring_run(
                settings, "ESA-Mission1", "exp", "channel_47", tuned=True
            )
            is None
        )
