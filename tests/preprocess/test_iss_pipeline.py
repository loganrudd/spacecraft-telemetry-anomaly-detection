"""Integration tests for run_iss_preprocessing (preprocess/pipeline.py).

Uses synthetic ISS tick shards written to tmp_path.  All runs use
parallel=False to avoid Ray cold-start cost in CI.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from spacecraft_telemetry.core.config import Settings
from spacecraft_telemetry.preprocess.pipeline import run_iss_preprocessing
from spacecraft_telemetry.preprocess.schemas import ISS_SERIES_FILE_SCHEMA

# ---------------------------------------------------------------------------
# Schema shared with collector_io
# ---------------------------------------------------------------------------

RAW_TICK_SCHEMA = pa.schema(
    [
        pa.field("telemetry_timestamp", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("value", pa.float32(), nullable=False),
        pa.field("aos_timestamp", pa.float64(), nullable=True),
    ]
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_tick_shard(dest_dir: Path, filename: str, rows: list[dict]) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=RAW_TICK_SCHEMA)
    pq.write_table(table, str(dest_dir / filename))


def _write_iss_channel(
    raw_root: Path,
    channel_id: str,
    n: int = 1200,
    interval_s: float = 2.0,
    start: str = "2026-06-01T00:00:00Z",
    gap_after_row: int | None = None,
    gap_duration_s: float = 91.0,
) -> None:
    """Write synthetic tick shards for one ISS channel.

    Args:
        raw_root:      Root directory (becomes {raw_root}/ISS/ticks/channel_id=X/)
        channel_id:    ISS PUI
        n:             Total number of ticks
        interval_s:    Cadence between ticks
        start:         Start timestamp
        gap_after_row: If given, insert a gap of gap_duration_s after this row index
        gap_duration_s: Duration of the gap in seconds
    """
    base = pd.Timestamp(start, tz="UTC")
    rows: list[dict] = []
    elapsed = 0.0
    for i in range(n):
        if gap_after_row is not None and i == gap_after_row:
            elapsed += gap_duration_s
        rows.append(
            {
                "telemetry_timestamp": base + pd.Timedelta(seconds=elapsed),
                "value": float(i % 100) * 0.1,
                "aos_timestamp": None,
            }
        )
        elapsed += interval_s

    channel_dir = raw_root / "ISS" / "ticks" / f"channel_id={channel_id}"
    _write_tick_shard(channel_dir, "20260601T000000.parquet", rows)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def raw_root(tmp_path: Path) -> Path:
    """Write 3 channels at 2s cadence over 40 minutes.

    All channels share a simultaneous 91s gap at row 600 to simulate a real
    ISS LOS event (all channels go silent at the same time).  The LOS mask
    requires cross-channel silence, so per-channel gaps don't trigger it.
    """
    for ch in ["S1000003", "P4000001", "USLAB000018"]:
        _write_iss_channel(
            tmp_path,
            ch,
            n=1200,
            interval_s=2.0,
            gap_after_row=600,
            gap_duration_s=91.0,
        )
    return tmp_path


@pytest.fixture()
def settings_iss(tmp_path: Path, raw_root: Path) -> Settings:
    out_dir = tmp_path / "processed"
    out_dir.mkdir()
    s = Settings(
        _env_file=None,  # type: ignore[call-arg]
    )
    s = s.model_copy(
        update={
            "preprocess": s.preprocess.model_copy(
                update={"processed_data_dir": str(out_dir)}
            ),
            "collect": s.collect.model_copy(
                update={
                    "raw_ticks_dir": str(raw_root),
                    "grid_interval_seconds": 30,
                }
            ),
        }
    )
    return s


# ---------------------------------------------------------------------------
# Helper to find partition Parquet files
# ---------------------------------------------------------------------------


def _find_parts(out_dir: Path, split: str, channel_id: str) -> list[Path]:
    pattern = f"ISS/{split}/mission_id=ISS/channel_id={channel_id}/part.parquet"
    return list((out_dir / pattern).parent.glob("part.parquet"))


def _read_part(out_dir: Path, split: str, channel_id: str) -> pd.DataFrame:
    parts = _find_parts(out_dir, split, channel_id)
    assert len(parts) == 1, f"Expected 1 part file, got {len(parts)}"
    return pq.read_table(str(parts[0]), partitioning=None).to_pandas()


# ---------------------------------------------------------------------------
# TestRunIssPreprocessingE2E
# ---------------------------------------------------------------------------


class TestRunIssPreprocessingE2E:
    def test_returns_summary_dict(
        self, settings_iss: Settings, raw_root: Path
    ) -> None:
        summary = run_iss_preprocessing(settings_iss, parallel=False)
        assert set(summary.keys()) == {
            "channels_processed",
            "rows_in",
            "train_rows",
            "test_rows",
        }
        assert summary["channels_processed"] == 3

    def test_partition_dirs_created(
        self, settings_iss: Settings, raw_root: Path
    ) -> None:
        run_iss_preprocessing(settings_iss, parallel=False)
        out_dir = Path(settings_iss.preprocess.processed_data_dir)
        for ch in ["S1000003", "P4000001", "USLAB000018"]:
            for split in ["train", "test"]:
                parts = _find_parts(out_dir, split, ch)
                assert len(parts) == 1, f"Missing part for {ch}/{split}"

    def test_normalization_params_written(
        self, settings_iss: Settings, raw_root: Path
    ) -> None:
        run_iss_preprocessing(settings_iss, parallel=False)
        params_path = (
            Path(settings_iss.preprocess.processed_data_dir)
            / "ISS"
            / "normalization_params.json"
        )
        assert params_path.exists()
        params = json.loads(params_path.read_text())
        assert set(params.keys()) == {"S1000003", "P4000001", "USLAB000018"}
        for ch_params in params.values():
            assert "mean" in ch_params
            assert "std" in ch_params

    def test_channels_txt_written(
        self, settings_iss: Settings, raw_root: Path
    ) -> None:
        run_iss_preprocessing(settings_iss, parallel=False)
        channels_txt = (
            Path(settings_iss.preprocess.processed_data_dir) / "ISS" / "channels.txt"
        )
        assert channels_txt.exists()
        lines = [ln for ln in channels_txt.read_text().splitlines() if ln]
        assert set(lines) == {"S1000003", "P4000001", "USLAB000018"}

    def test_output_has_is_los_column(
        self, settings_iss: Settings, raw_root: Path
    ) -> None:
        run_iss_preprocessing(settings_iss, parallel=False)
        out_dir = Path(settings_iss.preprocess.processed_data_dir)
        df = _read_part(out_dir, "train", "S1000003")
        assert "is_los" in df.columns
        assert df["is_los"].dtype == bool

    def test_is_los_true_over_gap_false_elsewhere(
        self, settings_iss: Settings, raw_root: Path
    ) -> None:
        # The simultaneous 91s gap → some rows should be is_los==True (over the
        # LOS window + smear) and the majority should be False.
        run_iss_preprocessing(settings_iss, parallel=False)
        out_dir = Path(settings_iss.preprocess.processed_data_dir)
        train = _read_part(out_dir, "train", "S1000003")
        test = _read_part(out_dir, "test", "S1000003")
        df = pd.concat([train, test], ignore_index=True)
        assert df["is_los"].any(), "Expected some is_los==True rows over the gap"
        assert not df["is_los"].all(), "Expected some is_los==False rows outside the gap"

    def test_segment_boundary_at_first_los_row(
        self, settings_iss: Settings, raw_root: Path
    ) -> None:
        # The first is_los==True row must be in a different segment than the
        # row immediately before it, confirming the LOS bump (not detect_gaps)
        # created the boundary.
        run_iss_preprocessing(settings_iss, parallel=False)
        out_dir = Path(settings_iss.preprocess.processed_data_dir)
        train = _read_part(out_dir, "train", "S1000003")
        test = _read_part(out_dir, "test", "S1000003")
        df = pd.concat([train, test], ignore_index=True).sort_values("telemetry_timestamp")
        los_rows = df[df["is_los"]]
        if los_rows.empty:
            return  # vacuously satisfied if no LOS
        first_los_pos = df.index.get_loc(los_rows.index[0])
        if first_los_pos == 0:
            return  # no preceding row to compare
        pre_seg = df.iloc[first_los_pos - 1]["segment_id"]
        los_seg = df.iloc[first_los_pos]["segment_id"]
        assert los_seg != pre_seg, "segment_id must change at the first is_los==True row"

    def test_is_anomaly_all_false(
        self, settings_iss: Settings, raw_root: Path
    ) -> None:
        run_iss_preprocessing(settings_iss, parallel=False)
        out_dir = Path(settings_iss.preprocess.processed_data_dir)
        for split in ["train", "test"]:
            df = _read_part(out_dir, split, "S1000003")
            assert not df["is_anomaly"].any(), f"is_anomaly must be all-False for ISS ({split})"

    def test_segment_id_increments_at_los(
        self, settings_iss: Settings, raw_root: Path
    ) -> None:
        # All 3 channels have a simultaneous 91s gap → LOS mask fires → pipeline
        # bumps segment_id at the first LOS-transition row so the LSTM never
        # creates windows spanning the LOS boundary.
        run_iss_preprocessing(settings_iss, parallel=False)
        out_dir = Path(settings_iss.preprocess.processed_data_dir)
        train = _read_part(out_dir, "train", "S1000003")
        test = _read_part(out_dir, "test", "S1000003")
        df = pd.concat([train, test], ignore_index=True)
        assert df["segment_id"].max() >= 1, "Expected at least 2 segments after LOS"

    def test_segment_id_increments_at_los_onset_and_recovery(
        self, settings_iss: Settings, raw_root: Path
    ) -> None:
        # XOR semantics: segment_id must change at both the first is_los==True row
        # (onset) and the first row after recovery (is_los==False after True).
        # This verifies the bump is not onset-only, which would leave LOS rows in
        # the same segment as the post-recovery nominal rows.
        run_iss_preprocessing(settings_iss, parallel=False)
        out_dir = Path(settings_iss.preprocess.processed_data_dir)
        train = _read_part(out_dir, "train", "S1000003")
        test = _read_part(out_dir, "test", "S1000003")
        df = pd.concat([train, test], ignore_index=True).sort_values("telemetry_timestamp")

        # Locate the LOS region and check segments on each side.
        los_rows = df[df["is_los"]]
        if los_rows.empty:
            return  # no LOS in output → test is vacuously satisfied
        first_los_idx = los_rows.index[0]
        last_los_idx = los_rows.index[-1]
        pre_los_seg = (
            df.loc[:first_los_idx - 1, "segment_id"].iloc[-1] if first_los_idx > 0 else None
        )
        los_seg = df.loc[first_los_idx, "segment_id"]
        post_los_rows = df.loc[last_los_idx + 1:]
        if pre_los_seg is not None:
            assert los_seg != pre_los_seg, "segment_id must change at LOS onset"
        if not post_los_rows.empty:
            post_los_seg = post_los_rows["segment_id"].iloc[0]
            assert post_los_seg != los_seg, "segment_id must change at LOS recovery"

    def test_context_items_not_in_output(
        self, settings_iss: Settings, raw_root: Path
    ) -> None:
        # Write context item directories alongside telemetry.
        for ctx in ["TIME_000001", "USLAB000086"]:
            ctx_dir = raw_root / "ISS" / "ticks" / f"channel_id={ctx}"
            ctx_dir.mkdir(parents=True, exist_ok=True)
            rows = [
                {
                    "telemetry_timestamp": pd.Timestamp("2026-06-01T00:00:00Z"),
                    "value": 0.0,
                    "aos_timestamp": None,
                }
            ]
            pq.write_table(
                pa.Table.from_pylist(rows, schema=RAW_TICK_SCHEMA),
                str(ctx_dir / "shard.parquet"),
            )
        run_iss_preprocessing(settings_iss, parallel=False)
        out_dir = Path(settings_iss.preprocess.processed_data_dir)
        for ctx in ["TIME_000001", "USLAB000086"]:
            parts = _find_parts(out_dir, "train", ctx)
            assert len(parts) == 0, f"Context item {ctx} must not be preprocessed"

    def test_rerun_is_idempotent(
        self, settings_iss: Settings, raw_root: Path
    ) -> None:
        summary1 = run_iss_preprocessing(settings_iss, parallel=False)
        summary2 = run_iss_preprocessing(settings_iss, parallel=False)
        assert summary1 == summary2
        out_dir = Path(settings_iss.preprocess.processed_data_dir)
        # No duplicate part files after re-run.
        for ch in ["S1000003", "P4000001", "USLAB000018"]:
            for split in ["train", "test"]:
                parts = _find_parts(out_dir, split, ch)
                assert len(parts) == 1

    def test_output_schema_matches_iss_schema(
        self, settings_iss: Settings, raw_root: Path
    ) -> None:
        run_iss_preprocessing(settings_iss, parallel=False)
        out_dir = Path(settings_iss.preprocess.processed_data_dir)
        parts = _find_parts(out_dir, "train", "S1000003")
        table = pq.read_table(str(parts[0]), partitioning=None)
        assert table.schema.equals(ISS_SERIES_FILE_SCHEMA)

    def test_downstream_4col_projection_works(
        self, settings_iss: Settings, raw_root: Path
    ) -> None:
        # Downstream ESA consumers project 4 columns; is_los must be silently ignored.
        run_iss_preprocessing(settings_iss, parallel=False)
        out_dir = Path(settings_iss.preprocess.processed_data_dir)
        parts = _find_parts(out_dir, "train", "S1000003")
        cols = ["telemetry_timestamp", "value_normalized", "segment_id", "is_anomaly"]
        df = pq.read_table(str(parts[0]), columns=cols, partitioning=None).to_pandas()
        assert list(df.columns) == cols

    def test_explicit_channel_list(
        self, settings_iss: Settings, raw_root: Path
    ) -> None:
        summary = run_iss_preprocessing(
            settings_iss, channels=["S1000003"], parallel=False
        )
        assert summary["channels_processed"] == 1
        out_dir = Path(settings_iss.preprocess.processed_data_dir)
        assert len(_find_parts(out_dir, "train", "S1000003")) == 1
        assert len(_find_parts(out_dir, "train", "P4000001")) == 0

    def test_train_temporally_before_test(
        self, settings_iss: Settings, raw_root: Path
    ) -> None:
        run_iss_preprocessing(settings_iss, parallel=False)
        out_dir = Path(settings_iss.preprocess.processed_data_dir)
        train = _read_part(out_dir, "train", "P4000001")
        test = _read_part(out_dir, "test", "P4000001")
        assert train["telemetry_timestamp"].max() < test["telemetry_timestamp"].min()

    def test_no_file_not_found_when_dir_missing(
        self, settings_iss: Settings
    ) -> None:
        s = settings_iss.model_copy(
            update={
                "collect": settings_iss.collect.model_copy(
                    update={"raw_ticks_dir": "/nonexistent/path"}
                )
            }
        )
        with pytest.raises(FileNotFoundError):
            run_iss_preprocessing(s, parallel=False)


# ---------------------------------------------------------------------------
# Slow tests (parallel Ray path) — excluded from default CI run
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestRunIssPreprocessingParallel:
    """Verifies the parallel Ray path produces the same output as sequential.

    Covers _run_iss_parallel, ray.put(los_mask), and settings absolutization.
    Run explicitly: pytest tests/preprocess/test_iss_pipeline.py -m slow
    """

    def test_parallel_matches_sequential(
        self, settings_iss: Settings, raw_root: Path, ray_local: None
    ) -> None:
        seq_summary = run_iss_preprocessing(settings_iss, parallel=False)

        # parallel run needs a fresh output dir to avoid collision.
        from pathlib import Path as _Path

        par_out = _Path(settings_iss.preprocess.processed_data_dir).parent / "processed_par"
        par_out.mkdir()
        par_settings = settings_iss.model_copy(
            update={
                "preprocess": settings_iss.preprocess.model_copy(
                    update={"processed_data_dir": str(par_out)}
                )
            }
        )
        par_summary = run_iss_preprocessing(par_settings, parallel=True)

        assert seq_summary["channels_processed"] == par_summary["channels_processed"]
        assert seq_summary["train_rows"] == par_summary["train_rows"]
        assert seq_summary["test_rows"] == par_summary["test_rows"]

        # Output files must exist for each channel in both runs.
        seq_dir = _Path(settings_iss.preprocess.processed_data_dir)
        for ch in ["S1000003", "P4000001", "USLAB000018"]:
            for split in ["train", "test"]:
                assert len(_find_parts(seq_dir, split, ch)) == 1
                assert len(_find_parts(par_out, split, ch)) == 1
