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
    def test_returns_summary_dict(self, settings) -> None:
        summary = run_preprocessing(settings, "ESA-Mission1", parallel=False)
        assert isinstance(summary, dict)

    def test_summary_has_expected_keys(self, settings) -> None:
        summary = run_preprocessing(settings, "ESA-Mission1", parallel=False)
        assert set(summary.keys()) == {
            "channels_processed", "rows_in", "train_rows", "test_rows"
        }

    def test_channels_processed_equals_one(self, settings) -> None:
        summary = run_preprocessing(settings, "ESA-Mission1", parallel=False)
        assert summary["channels_processed"] == 1

    def test_row_counts_add_up(self, settings) -> None:
        summary = run_preprocessing(settings, "ESA-Mission1", parallel=False)
        assert summary["train_rows"] + summary["test_rows"] == summary["rows_in"]

    def test_train_split_directory_created(self, settings) -> None:
        run_preprocessing(settings, "ESA-Mission1", parallel=False)
        train_dir = (
            Path(str(settings.preprocess.processed_data_dir))
            / "ESA-Mission1" / "train"
        )
        assert train_dir.is_dir()

    def test_test_split_directory_created(self, settings) -> None:
        run_preprocessing(settings, "ESA-Mission1", parallel=False)
        test_dir = (
            Path(str(settings.preprocess.processed_data_dir))
            / "ESA-Mission1" / "test"
        )
        assert test_dir.is_dir()

    def test_partition_dirs_created(self, settings) -> None:
        run_preprocessing(settings, "ESA-Mission1", parallel=False)
        out = Path(str(settings.preprocess.processed_data_dir))
        train_partition = (
            out / "ESA-Mission1" / "train"
            / "mission_id=ESA-Mission1" / "channel_id=channel_1"
        )
        test_partition = (
            out / "ESA-Mission1" / "test"
            / "mission_id=ESA-Mission1" / "channel_id=channel_1"
        )
        assert train_partition.is_dir()
        assert test_partition.is_dir()

    def test_normalization_params_json_written(self, settings) -> None:
        run_preprocessing(settings, "ESA-Mission1", parallel=False)
        params_path = (
            Path(str(settings.preprocess.processed_data_dir))
            / "ESA-Mission1" / "normalization_params.json"
        )
        assert params_path.exists()

    def test_normalization_params_has_channel_key(self, settings) -> None:
        run_preprocessing(settings, "ESA-Mission1", parallel=False)
        params_path = (
            Path(str(settings.preprocess.processed_data_dir))
            / "ESA-Mission1" / "normalization_params.json"
        )
        params = json.loads(params_path.read_text())
        assert "channel_1" in params

    def test_normalization_params_contains_mean_and_std(self, settings) -> None:
        run_preprocessing(settings, "ESA-Mission1", parallel=False)
        params_path = (
            Path(str(settings.preprocess.processed_data_dir))
            / "ESA-Mission1" / "normalization_params.json"
        )
        params = json.loads(params_path.read_text())
        ch = params["channel_1"]
        assert "mean" in ch and "std" in ch
        assert isinstance(ch["mean"], float)
        assert isinstance(ch["std"], float)


# ---------------------------------------------------------------------------
# run_preprocessing — output data quality
# ---------------------------------------------------------------------------


class TestRunPreprocessingOutputData:
    def test_train_rows_sorted_by_timestamp(self, settings) -> None:
        run_preprocessing(settings, "ESA-Mission1", parallel=False)
        out = Path(str(settings.preprocess.processed_data_dir))
        train_df = _read_partition(out, "ESA-Mission1", "train", "channel_1")
        diffs = train_df["telemetry_timestamp"].diff().iloc[1:]
        assert (diffs >= pd.Timedelta(0)).all()

    def test_train_before_test_temporally(self, settings) -> None:
        run_preprocessing(settings, "ESA-Mission1", parallel=False)
        out = Path(str(settings.preprocess.processed_data_dir))
        train_df = _read_partition(out, "ESA-Mission1", "train", "channel_1")
        test_df = _read_partition(out, "ESA-Mission1", "test", "channel_1")
        assert train_df["telemetry_timestamp"].max() < test_df["telemetry_timestamp"].min()

    def test_value_normalized_is_float32(self, settings) -> None:
        run_preprocessing(settings, "ESA-Mission1", parallel=False)
        out = Path(str(settings.preprocess.processed_data_dir))
        train_df = _read_partition(out, "ESA-Mission1", "train", "channel_1")
        assert train_df["value_normalized"].dtype == np.float32

    def test_segment_id_is_int32(self, settings) -> None:
        run_preprocessing(settings, "ESA-Mission1", parallel=False)
        out = Path(str(settings.preprocess.processed_data_dir))
        train_df = _read_partition(out, "ESA-Mission1", "train", "channel_1")
        assert train_df["segment_id"].dtype == np.int32

    def test_is_anomaly_present(self, settings) -> None:
        run_preprocessing(settings, "ESA-Mission1", parallel=False)
        out = Path(str(settings.preprocess.processed_data_dir))
        train_df = _read_partition(out, "ESA-Mission1", "train", "channel_1")
        assert "is_anomaly" in train_df.columns

    def test_some_rows_are_anomalous(self, settings) -> None:
        # Labels use half-open intervals [start, end): rows 10-13, 40-43, 70-73
        # (4 rows per segment x 3 segments = 12 total).  All fall in train
        # (train_fraction=0.8 on 100 rows, cutoff after row 79).
        run_preprocessing(settings, "ESA-Mission1", parallel=False)
        out = Path(str(settings.preprocess.processed_data_dir))
        train_df = _read_partition(out, "ESA-Mission1", "train", "channel_1")
        test_df = _read_partition(out, "ESA-Mission1", "test", "channel_1")
        combined = pd.concat([train_df, test_df]).reset_index(drop=True)
        assert combined["is_anomaly"].sum() == 12
        # Verify anomalous rows fall at the expected offsets within the combined series.
        anomalous_positions = combined.index[combined["is_anomaly"]].tolist()
        expected = [10, 11, 12, 13, 40, 41, 42, 43, 70, 71, 72, 73]
        assert anomalous_positions == expected


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
