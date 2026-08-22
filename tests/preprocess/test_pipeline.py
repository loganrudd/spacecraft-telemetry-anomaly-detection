"""Integration tests for preprocess/pipeline.py — run_preprocessing (sequential mode).

All tests use parallel=False to avoid Ray initialisation cost. The parallel (Ray)
path is not tested here — it is covered by test_parity.py (marked @pytest.mark.slow).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from spacecraft_telemetry.preprocess.pipeline import run_preprocessing

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_partition(base: Path, mission: str, split: str, channel: str) -> pd.DataFrame:
    """Read a channel partition and return a sorted DataFrame."""
    partition_dir = base / mission / split / f"mission_id={mission}" / f"channel_id={channel}"
    files = sorted(partition_dir.glob("*.parquet"))
    assert files, f"No parquet files in {partition_dir}"
    tables = [pq.read_table(f, partitioning=None) for f in files]
    import pyarrow as pa
    table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
    return table.to_pandas().sort_values("telemetry_timestamp").reset_index(drop=True)


# ---------------------------------------------------------------------------
# run_preprocessing — happy-path
# ---------------------------------------------------------------------------


class TestRunPreprocessingHappyPath:
    def test_returns_summary_dict(self, pipeline_result) -> None:
        assert isinstance(pipeline_result.summary, dict)

    def test_summary_has_expected_keys(self, pipeline_result) -> None:
        assert set(pipeline_result.summary.keys()) == {
            "channels_processed", "rows_in", "train_rows", "test_rows"
        }

    def test_channels_processed_equals_one(self, pipeline_result) -> None:
        assert pipeline_result.summary["channels_processed"] == 1

    def test_row_counts_add_up(self, pipeline_result) -> None:
        s = pipeline_result.summary
        assert s["train_rows"] + s["test_rows"] == s["rows_in"]

    def test_train_split_directory_created(self, pipeline_result) -> None:
        assert (pipeline_result.out_dir / "ESA-Mission1" / "train").is_dir()

    def test_test_split_directory_created(self, pipeline_result) -> None:
        assert (pipeline_result.out_dir / "ESA-Mission1" / "test").is_dir()

    def test_partition_dirs_created(self, pipeline_result) -> None:
        out = pipeline_result.out_dir
        ch = "channel_id=channel_1"
        assert (out / "ESA-Mission1" / "train" / "mission_id=ESA-Mission1" / ch).is_dir()
        assert (out / "ESA-Mission1" / "test" / "mission_id=ESA-Mission1" / ch).is_dir()

    def test_normalization_params_json_written(self, pipeline_result) -> None:
        params_path = pipeline_result.out_dir / "ESA-Mission1" / "normalization_params.json"
        assert params_path.exists()

    def test_normalization_params_has_channel_key(self, pipeline_result) -> None:
        params_path = pipeline_result.out_dir / "ESA-Mission1" / "normalization_params.json"
        params = json.loads(params_path.read_text())
        assert "channel_1" in params

    def test_normalization_params_contains_mean_and_std(self, pipeline_result) -> None:
        params_path = pipeline_result.out_dir / "ESA-Mission1" / "normalization_params.json"
        params = json.loads(params_path.read_text())
        ch = params["channel_1"]
        assert "mean" in ch and "std" in ch
        assert isinstance(ch["mean"], float)
        assert isinstance(ch["std"], float)


# ---------------------------------------------------------------------------
# run_preprocessing — output data quality
# ---------------------------------------------------------------------------


class TestRunPreprocessingOutputData:
    def test_train_rows_sorted_by_timestamp(self, pipeline_result) -> None:
        train_df = _read_partition(pipeline_result.out_dir, "ESA-Mission1", "train", "channel_1")
        diffs = train_df["telemetry_timestamp"].diff().iloc[1:]
        assert (diffs >= pd.Timedelta(0)).all()

    def test_train_before_test_temporally(self, pipeline_result) -> None:
        out = pipeline_result.out_dir
        train_df = _read_partition(out, "ESA-Mission1", "train", "channel_1")
        test_df = _read_partition(out, "ESA-Mission1", "test", "channel_1")
        assert train_df["telemetry_timestamp"].max() < test_df["telemetry_timestamp"].min()

    def test_value_normalized_is_float32(self, pipeline_result) -> None:
        train_df = _read_partition(pipeline_result.out_dir, "ESA-Mission1", "train", "channel_1")
        assert train_df["value_normalized"].dtype == np.float32

    def test_segment_id_is_int32(self, pipeline_result) -> None:
        train_df = _read_partition(pipeline_result.out_dir, "ESA-Mission1", "train", "channel_1")
        assert train_df["segment_id"].dtype == np.int32

    def test_is_anomaly_present(self, pipeline_result) -> None:
        train_df = _read_partition(pipeline_result.out_dir, "ESA-Mission1", "train", "channel_1")
        assert "is_anomaly" in train_df.columns

    def test_some_rows_are_anomalous(self, pipeline_result) -> None:
        # Labels use half-open intervals [start, end): rows 10-13, 40-43, 70-73
        # (4 rows per segment x 3 segments = 12 total).  All fall in train
        # (train_fraction=0.8 on 100 rows, cutoff after row 79).
        out = pipeline_result.out_dir
        train_df = _read_partition(out, "ESA-Mission1", "train", "channel_1")
        test_df = _read_partition(out, "ESA-Mission1", "test", "channel_1")
        combined = pd.concat([train_df, test_df]).reset_index(drop=True)
        assert combined["is_anomaly"].sum() == 12
        anomalous_positions = combined.index[combined["is_anomaly"]].tolist()
        assert anomalous_positions == [10, 11, 12, 13, 40, 41, 42, 43, 70, 71, 72, 73]


# ---------------------------------------------------------------------------
# run_preprocessing — idempotency and edge cases
# ---------------------------------------------------------------------------


class TestRunPreprocessingEdgeCases:
    def test_rerun_is_idempotent(self, settings) -> None:
        run_preprocessing(settings, "ESA-Mission1", parallel=False)
        summary1 = run_preprocessing(settings, "ESA-Mission1", parallel=False)
        summary2 = run_preprocessing(settings, "ESA-Mission1", parallel=False)
        assert summary1 == summary2

    def test_rerun_does_not_duplicate_parquet_files(self, settings) -> None:
        run_preprocessing(settings, "ESA-Mission1", parallel=False)
        run_preprocessing(settings, "ESA-Mission1", parallel=False)
        out = Path(str(settings.preprocess.processed_data_dir))
        partition = (
            out / "ESA-Mission1" / "train"
            / "mission_id=ESA-Mission1" / "channel_id=channel_1"
        )
        assert len(list(partition.glob("*.parquet"))) == 1

    def test_explicit_channel_list(self, settings) -> None:
        summary = run_preprocessing(
            settings, "ESA-Mission1", channels=["channel_1"], parallel=False
        )
        assert summary["channels_processed"] == 1

    def test_missing_channel_raises(self, settings) -> None:
        with pytest.raises(FileNotFoundError, match="channel_99"):
            run_preprocessing(
                settings, "ESA-Mission1", channels=["channel_99"], parallel=False
            )

    def test_missing_labels_csv_produces_all_nominal(
        self, pipeline_input_dir: Path, tmp_path: Path
    ) -> None:
        from spacecraft_telemetry.core.config import DataConfig, PreprocessingConfig, Settings

        # Remove the labels file.
        labels_file = pipeline_input_dir / "ESA-Mission1" / "labels.csv"
        labels_file.unlink()

        out_dir = tmp_path / "out_nolabels"
        out_dir.mkdir()
        settings = Settings(
            data=DataConfig(sample_data_dir=pipeline_input_dir),
            preprocess=PreprocessingConfig(processed_data_dir=out_dir, train_fraction=0.8),
        )
        run_preprocessing(settings, "ESA-Mission1", parallel=False)

        train_df = _read_partition(out_dir, "ESA-Mission1", "train", "channel_1")
        test_df = _read_partition(out_dir, "ESA-Mission1", "test", "channel_1")
        assert not pd.concat([train_df, test_df])["is_anomaly"].any()


# ---------------------------------------------------------------------------
# ESA common time grid — docs/plans/023 stage .3
# ---------------------------------------------------------------------------


def _write_phase_offset_input(base: Path) -> Path:
    """Two channels on the same 30s cadence, 16.1s out of phase.

    This is the measured ESA blocker in miniature (docs/plans/023: channel_70 /
    channel_71 share not one timestamp across 2.9M rows each). Returns the input
    root that run_preprocessing expects.
    """
    from spacecraft_telemetry.preprocess.io import read_channel  # noqa: F401  (schema ref)

    mission = "ESA-Mission1"
    channels_dir = base / "input" / mission / "channels"
    channels_dir.mkdir(parents=True)
    start = pd.Timestamp("2000-01-01", tz="UTC")
    for name, phase_s in (("channel_70", 0.0), ("channel_71", 16.1)):
        idx = pd.DatetimeIndex(
            [start + pd.Timedelta(seconds=phase_s + 30.0 * i) for i in range(400)],
            name="datetime",
        ).as_unit("us")
        values = pd.array([float(i % 17) * 0.5 for i in range(400)], dtype="float32")
        pd.DataFrame({name: values}, index=idx).to_parquet(channels_dir / f"{name}.parquet")
    return base / "input"


def _epoch_seconds(ts: pd.Series) -> pd.Series:
    """Whole seconds since the epoch — precision-agnostic (us on disk, ns in pandas)."""
    return (ts - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()


def _settings_with_grid(input_dir: Path, out_dir: Path, **preprocess_kwargs):
    from spacecraft_telemetry.core.config import DataConfig, PreprocessingConfig, Settings

    return Settings(
        data=DataConfig(sample_data_dir=input_dir),
        preprocess=PreprocessingConfig(
            processed_data_dir=out_dir, train_fraction=0.8, **preprocess_kwargs
        ),
    )


class TestRunPreprocessingTimeGrid:
    def test_no_grid_leaves_native_timestamps(self, settings) -> None:
        # grid_interval_seconds defaults to None — today's behaviour, byte for
        # byte. The 90s input cadence must survive untouched.
        run_preprocessing(settings, "ESA-Mission1", parallel=False)
        out = Path(str(settings.preprocess.processed_data_dir))
        train = _read_partition(out, "ESA-Mission1", "train", "channel_1")
        diffs = train["telemetry_timestamp"].diff().iloc[1:].dt.total_seconds()
        assert (diffs == 90.0).all()

    def test_grid_snaps_timestamps_to_the_grid(
        self, pipeline_input_dir: Path, tmp_path: Path
    ) -> None:
        out_dir = tmp_path / "out_grid"
        out_dir.mkdir()
        s = _settings_with_grid(pipeline_input_dir, out_dir, grid_interval_seconds=300)
        run_preprocessing(s, "ESA-Mission1", parallel=False)

        train = _read_partition(out_dir, "ESA-Mission1", "train", "channel_1")
        assert (_epoch_seconds(train["telemetry_timestamp"]) % 300 == 0).all()

    def test_grid_preserves_the_on_disk_column_contract(
        self, pipeline_input_dir: Path, tmp_path: Path
    ) -> None:
        # Stage .3 changes the CONTENTS of the Parquet, never its schema —
        # .claude/rules/preprocess.md's downstream contract.
        out_dir = tmp_path / "out_grid_schema"
        out_dir.mkdir()
        s = _settings_with_grid(pipeline_input_dir, out_dir, grid_interval_seconds=300)
        run_preprocessing(s, "ESA-Mission1", parallel=False)

        train = _read_partition(out_dir, "ESA-Mission1", "train", "channel_1")
        assert list(train.columns) == [
            "telemetry_timestamp", "value_normalized", "segment_id", "is_anomaly",
        ]

    def test_grid_writes_normalization_params(
        self, pipeline_input_dir: Path, tmp_path: Path
    ) -> None:
        out_dir = tmp_path / "out_grid_params"
        out_dir.mkdir()
        s = _settings_with_grid(pipeline_input_dir, out_dir, grid_interval_seconds=300)
        run_preprocessing(s, "ESA-Mission1", parallel=False)

        params = json.loads((out_dir / "ESA-Mission1" / "normalization_params.json").read_text())
        assert set(params["channel_1"]) == {"mean", "std"}

    def test_per_channel_override_wins(self, tmp_path: Path) -> None:
        input_dir = _write_phase_offset_input(tmp_path)
        out_dir = tmp_path / "out_override"
        out_dir.mkdir()
        s = _settings_with_grid(
            input_dir,
            out_dir,
            grid_interval_seconds=30,
            channel_grid_interval_seconds={"channel_71": 300},
        )
        run_preprocessing(s, "ESA-Mission1", parallel=False)

        a = _read_partition(out_dir, "ESA-Mission1", "train", "channel_70")
        b = _read_partition(out_dir, "ESA-Mission1", "train", "channel_71")
        assert (_epoch_seconds(a["telemetry_timestamp"]) % 30 == 0).all()
        assert (_epoch_seconds(b["telemetry_timestamp"]) % 300 == 0).all()

    def test_phase_offset_channels_share_no_timestamps_without_a_grid(
        self, tmp_path: Path
    ) -> None:
        # The "before" half of the result: this is what blocks multivariate
        # grouping today.
        input_dir = _write_phase_offset_input(tmp_path)
        out_dir = tmp_path / "out_native"
        out_dir.mkdir()
        s = _settings_with_grid(input_dir, out_dir)
        run_preprocessing(s, "ESA-Mission1", parallel=False)

        a = _read_partition(out_dir, "ESA-Mission1", "train", "channel_70")
        b = _read_partition(out_dir, "ESA-Mission1", "train", "channel_71")
        common = pd.DatetimeIndex(a["telemetry_timestamp"]).intersection(
            pd.DatetimeIndex(b["telemetry_timestamp"])
        )
        assert len(common) == 0

    def test_grid_makes_phase_offset_channels_jointly_alignable(
        self, tmp_path: Path
    ) -> None:
        # The "after" half, measured through the real pipeline rather than a
        # timestamp simulation — stage .3's whole point.
        input_dir = _write_phase_offset_input(tmp_path)
        out_dir = tmp_path / "out_gridded"
        out_dir.mkdir()
        s = _settings_with_grid(input_dir, out_dir, grid_interval_seconds=30)
        run_preprocessing(s, "ESA-Mission1", parallel=False)

        a = _read_partition(out_dir, "ESA-Mission1", "train", "channel_70")
        b = _read_partition(out_dir, "ESA-Mission1", "train", "channel_71")
        common = pd.DatetimeIndex(a["telemetry_timestamp"]).intersection(
            pd.DatetimeIndex(b["telemetry_timestamp"])
        )
        assert len(common) >= min(len(a), len(b)) - 1


class TestSmallTaskCpus:
    """Packing headroom for the Ray fan-out — docs/plans/023 stage .3.

    Resampling adds a full bucketing pass while the native frame is still live,
    which measured at ~2.1GB peak per "small" ESA channel against ~950MB
    without it. At 4-per-node on 6GiB workers that OOM-killed tasks and
    eventually a whole node, and the symptom was missing channels in the output
    tree rather than a raised error — hence a pinned unit test.
    """

    def test_no_grid_packs_four_per_node(self) -> None:
        from spacecraft_telemetry.core.config import PreprocessingConfig
        from spacecraft_telemetry.preprocess.pipeline import small_task_cpus

        assert small_task_cpus(PreprocessingConfig(), ["channel_1", "channel_2"]) == 1

    def test_mission_wide_grid_halves_the_packing(self) -> None:
        from spacecraft_telemetry.core.config import PreprocessingConfig
        from spacecraft_telemetry.preprocess.pipeline import small_task_cpus

        cfg = PreprocessingConfig(grid_interval_seconds=30)
        assert small_task_cpus(cfg, ["channel_1", "channel_2"]) == 2

    def test_a_single_gridded_channel_is_enough_to_halve_it(self) -> None:
        # The reservation is per-task and uniform, so one resampled channel in
        # the batch is enough to need the headroom.
        from spacecraft_telemetry.core.config import PreprocessingConfig
        from spacecraft_telemetry.preprocess.pipeline import small_task_cpus

        cfg = PreprocessingConfig(channel_grid_interval_seconds={"channel_2": 30})
        assert small_task_cpus(cfg, ["channel_1", "channel_2"]) == 2

    def test_overrides_for_other_channels_do_not_apply(self) -> None:
        from spacecraft_telemetry.core.config import PreprocessingConfig
        from spacecraft_telemetry.preprocess.pipeline import small_task_cpus

        cfg = PreprocessingConfig(channel_grid_interval_seconds={"channel_99": 30})
        assert small_task_cpus(cfg, ["channel_1", "channel_2"]) == 1
