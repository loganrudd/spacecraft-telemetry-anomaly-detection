"""Parity test: parallel (Ray) preprocessing must match sequential (pandas) output.

Marked @pytest.mark.slow — excluded from the default push CI run, included in
PR-to-main CI where Ray cold-start cost is amortised across the suite.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from spacecraft_telemetry.preprocess.pipeline import run_preprocessing


def _read_sorted(base: Path, mission: str, split: str, channel: str) -> pd.DataFrame:
    """Read a channel partition and return a timestamp-sorted DataFrame."""
    partition_dir = base / mission / split / f"mission_id={mission}" / f"channel_id={channel}"
    files = sorted(partition_dir.glob("*.parquet"))
    assert files, f"No parquet files in {partition_dir}"
    tables = [pq.read_table(f, partitioning=None) for f in files]
    table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
    return table.to_pandas().sort_values("telemetry_timestamp").reset_index(drop=True)


@pytest.mark.slow
def test_parallel_matches_sequential(settings, ray_local, tmp_path) -> None:
    """parallel=True and parallel=False must produce identical summaries and on-disk data."""
    from spacecraft_telemetry.core.config import PreprocessingConfig, Settings

    # Sequential run uses the function-scoped `settings` fixture (fresh output dir).
    seq_summary = run_preprocessing(settings, "ESA-Mission1", parallel=False)
    seq_out = Path(str(settings.preprocess.processed_data_dir))

    # Parallel run gets its own output dir so files don't collide.
    par_out = tmp_path / "par_output"
    par_out.mkdir()
    par_settings = Settings(
        data=settings.data,
        preprocess=PreprocessingConfig(
            processed_data_dir=par_out,
            train_fraction=settings.preprocess.train_fraction,
            feature_windows=list(settings.preprocess.feature_windows),
        ),
    )
    par_summary = run_preprocessing(par_settings, "ESA-Mission1", parallel=True)

    # Summaries must be identical.
    assert seq_summary == par_summary

    # On-disk Parquet must match for both splits.
    for split in ("train", "test"):
        seq_df = _read_sorted(seq_out, "ESA-Mission1", split, "channel_1")
        par_df = _read_sorted(par_out, "ESA-Mission1", split, "channel_1")
        pd.testing.assert_frame_equal(seq_df, par_df, check_like=False)

    # Normalization params must match.
    seq_params = json.loads(
        (seq_out / "ESA-Mission1" / "normalization_params.json").read_text()
    )
    par_params = json.loads(
        (par_out / "ESA-Mission1" / "normalization_params.json").read_text()
    )
    assert seq_params == par_params


@pytest.mark.slow
def test_parallel_matches_sequential_with_grid(pipeline_input_dir, ray_local, tmp_path) -> None:
    """docs/reviews/023-channel-time-grid.md T1: grid+Ray was untested by
    construction — the settings.preprocess.model_copy(...) reconstruction
    above rebuilds a fresh PreprocessingConfig from named fields only, so a
    grid_interval_seconds set on the input settings never reached the
    parallel run at all. This test carries the grid fields through both arms
    from the start rather than reconstructing settings mid-test.
    """
    from spacecraft_telemetry.core.config import DataConfig, PreprocessingConfig, Settings

    def _grid_settings(out_dir: Path) -> Settings:
        return Settings(
            data=DataConfig(sample_data_dir=pipeline_input_dir),
            preprocess=PreprocessingConfig(
                processed_data_dir=out_dir, train_fraction=0.8, grid_interval_seconds=30,
            ),
        )

    seq_out = tmp_path / "seq_output"
    seq_out.mkdir()
    seq_summary = run_preprocessing(_grid_settings(seq_out), "ESA-Mission1", parallel=False)

    par_out = tmp_path / "par_output"
    par_out.mkdir()
    par_summary = run_preprocessing(_grid_settings(par_out), "ESA-Mission1", parallel=True)

    assert seq_summary == par_summary

    for split in ("train", "test"):
        seq_df = _read_sorted(seq_out, "ESA-Mission1", split, "channel_1")
        par_df = _read_sorted(par_out, "ESA-Mission1", split, "channel_1")
        pd.testing.assert_frame_equal(seq_df, par_df, check_like=False)
        # Confirm the grid actually applied on both arms, not just that they
        # agree with each other (they'd also agree if both silently ignored it).
        epoch = seq_df["telemetry_timestamp"] - pd.Timestamp("1970-01-01", tz="UTC")
        epoch_s = epoch.dt.total_seconds()
        assert (epoch_s % 30 == 0).all()

    seq_params = json.loads((seq_out / "ESA-Mission1" / "normalization_params.json").read_text())
    par_params = json.loads((par_out / "ESA-Mission1" / "normalization_params.json").read_text())
    assert seq_params == par_params
