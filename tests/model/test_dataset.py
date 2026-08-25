"""Tests for model.dataset — load_series_parquet, _build_window_index,
WindowedSequenceDataset, make_dataloaders, make_test_dataloader,
window_target_timestamps."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

torch = pytest.importorskip("torch")

from spacecraft_telemetry.model.dataset import (  # noqa: E402
    WindowedSequenceDataset,
    _align_multi_channel,
    _build_window_index,
    _joint_segment_ids,
    _resolve_channel_group,
    _window_any_anomalous,
    load_multichannel_series_parquet,
    load_series_metadata,
    load_series_parquet,
    make_dataloaders,
    make_test_dataloader,
    window_target_timestamps,
    window_target_timestamps_from_metadata,
)
from tests.model.conftest import SeriesParquetFixture  # noqa: E402

# ---------------------------------------------------------------------------
# load_series_parquet
# ---------------------------------------------------------------------------


def test_load_series_returns_expected_shapes(
    tiny_series_parquet: SeriesParquetFixture,
) -> None:
    fx = tiny_series_parquet
    values, seg_ids, is_anomaly, timestamps = load_series_parquet(
        fx.processed_dir, fx.mission, fx.channel, "train"
    )
    n_rows = sum((60, 30, 10))  # _SEG_SIZES_TRAIN
    assert values.shape == (n_rows,)
    assert seg_ids.shape == (n_rows,)
    assert is_anomaly.shape == (n_rows,)
    assert timestamps.shape == (n_rows,)


def test_load_series_dtypes(tiny_series_parquet: SeriesParquetFixture) -> None:
    fx = tiny_series_parquet
    values, seg_ids, is_anomaly, _ = load_series_parquet(
        fx.processed_dir, fx.mission, fx.channel, "train"
    )
    assert values.dtype == np.float32
    assert seg_ids.dtype == np.int32
    assert is_anomaly.dtype == bool


def test_load_series_sorted_by_timestamp(
    tiny_series_parquet: SeriesParquetFixture,
) -> None:
    """Rows must be in ascending timestamp order after loading."""
    fx = tiny_series_parquet
    _, _, _, timestamps = load_series_parquet(
        fx.processed_dir, fx.mission, fx.channel, "train"
    )
    diffs = np.diff(pd.DatetimeIndex(timestamps).asi8)
    assert (diffs >= 0).all(), "Timestamps are not sorted ascending"


def test_load_series_missing_channel_raises(
    tiny_series_parquet: SeriesParquetFixture,
) -> None:
    fx = tiny_series_parquet
    with pytest.raises(FileNotFoundError, match="channel_id=nonexistent"):
        load_series_parquet(fx.processed_dir, fx.mission, "nonexistent", "train")


def test_load_series_empty_dir_raises(tmp_path: Path) -> None:
    empty_dir = (
        tmp_path / "processed" / "ESA-Mission1" / "train"
        / "mission_id=ESA-Mission1" / "channel_id=channel_1"
    )
    empty_dir.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match=r"no \.parquet files"):
        load_series_parquet(
            tmp_path / "processed", "ESA-Mission1", "channel_1", "train"
        )


# ---------------------------------------------------------------------------
# load_series_metadata
# ---------------------------------------------------------------------------


def test_load_series_metadata_matches_load_series_parquet(
    tiny_series_parquet: SeriesParquetFixture,
) -> None:
    """The lighter loader must return identical segment_ids/is_anomaly/timestamps."""
    fx = tiny_series_parquet
    _, expected_seg_ids, expected_is_anomaly, expected_timestamps = load_series_parquet(
        fx.processed_dir, fx.mission, fx.channel, "train"
    )
    seg_ids, is_anomaly, timestamps = load_series_metadata(
        fx.processed_dir, fx.mission, fx.channel, "train"
    )
    np.testing.assert_array_equal(seg_ids, expected_seg_ids)
    np.testing.assert_array_equal(is_anomaly, expected_is_anomaly)
    np.testing.assert_array_equal(timestamps, expected_timestamps)


def test_load_series_metadata_dtypes(tiny_series_parquet: SeriesParquetFixture) -> None:
    fx = tiny_series_parquet
    seg_ids, is_anomaly, _ = load_series_metadata(
        fx.processed_dir, fx.mission, fx.channel, "train"
    )
    assert seg_ids.dtype == np.int32
    assert is_anomaly.dtype == bool


def test_load_series_metadata_missing_channel_raises(
    tiny_series_parquet: SeriesParquetFixture,
) -> None:
    fx = tiny_series_parquet
    with pytest.raises(FileNotFoundError, match="channel_id=nonexistent"):
        load_series_metadata(fx.processed_dir, fx.mission, "nonexistent", "train")


# ---------------------------------------------------------------------------
# _build_window_index
# ---------------------------------------------------------------------------


def test_window_index_basic_count() -> None:
    """N=60 rows, W=10, H=1, no anomalies → 50 valid indices."""
    n = 60
    seg_ids = np.zeros(n, dtype=np.int32)
    is_anomaly = np.zeros(n, dtype=bool)
    idx = _build_window_index(seg_ids, is_anomaly, 10, 1, skip_anomalous_windows=False)
    assert len(idx) == 50


def test_window_index_skips_segment_boundaries() -> None:
    """Segments of size W-1 produce zero windows; W+H produces one."""
    W, H = 10, 1
    # Two segments: seg 0 = W-1 rows (too short), seg 1 = W+H rows (exactly 1)
    seg_ids = np.array([0] * (W - 1) + [1] * (W + H), dtype=np.int32)
    is_anomaly = np.zeros(len(seg_ids), dtype=bool)
    idx = _build_window_index(seg_ids, is_anomaly, W, H, skip_anomalous_windows=False)
    assert len(idx) == 1
    # The single valid start is right after the first segment.
    assert idx[0] == W - 1


def test_window_index_no_cross_boundary_windows() -> None:
    """No returned index should have its span straddle two segments."""
    W, H = 5, 1
    # seg 0: 8 rows, seg 1: 8 rows (interleaved in time)
    seg_ids = np.array([0] * 8 + [1] * 8, dtype=np.int32)
    is_anomaly = np.zeros(16, dtype=bool)
    idx = _build_window_index(seg_ids, is_anomaly, W, H, skip_anomalous_windows=False)
    span = W + H
    for s in idx:
        assert seg_ids[s] == seg_ids[s + span - 1], f"Boundary crossing at s={s}"


def test_window_index_skips_anomalous_windows_when_requested() -> None:
    """With skip_anomalous_windows=True, windows touching anomalous steps are dropped."""
    W, H = 3, 1
    n = 10
    seg_ids = np.zeros(n, dtype=np.int32)
    is_anomaly = np.zeros(n, dtype=bool)
    # Make row 5 anomalous — it falls inside windows with starts 2, 3, 4, 5.
    is_anomaly[5] = True

    idx_skip = _build_window_index(
        seg_ids, is_anomaly, W, H, skip_anomalous_windows=True
    )
    idx_keep = _build_window_index(
        seg_ids, is_anomaly, W, H, skip_anomalous_windows=False
    )

    assert len(idx_keep) == 7   # N - span + 1 = 10 - 4 + 1 = 7
    # N=10, span=W+H=4, valid starts = 10-4+1 = 7; 4 of them overlap row 5.
    # Starts that include row 5: span=[s, s+4), row 5 in range iff s <= 5 < s+4
    # i.e. s in {2, 3, 4, 5} → 4 starts dropped.
    assert len(idx_keep) == 7
    assert len(idx_skip) == 3
    # Remaining starts: 0, 1, 6.
    assert set(idx_skip.tolist()) == {0, 1, 6}


def test_window_index_empty_when_too_short() -> None:
    """Series shorter than span returns empty index."""
    W, H = 10, 1
    seg_ids = np.zeros(5, dtype=np.int32)  # 5 < W + H = 11
    is_anomaly = np.zeros(5, dtype=bool)
    idx = _build_window_index(seg_ids, is_anomaly, W, H, skip_anomalous_windows=False)
    assert len(idx) == 0


# ---------------------------------------------------------------------------
# WindowedSequenceDataset
# ---------------------------------------------------------------------------


def test_dataset_len(tiny_series_parquet: SeriesParquetFixture) -> None:
    fx = tiny_series_parquet
    n = 20
    values = np.arange(n, dtype=np.float32)
    seg_ids = np.zeros(n, dtype=np.int32)
    is_anomaly = np.zeros(n, dtype=bool)
    idx = _build_window_index(seg_ids, is_anomaly, fx.window_size, fx.prediction_horizon, False)
    ds = WindowedSequenceDataset(values, idx, fx.window_size, fx.prediction_horizon)
    assert len(ds) == len(idx)


def test_dataset_item_shapes(tiny_series_parquet: SeriesParquetFixture) -> None:
    fx = tiny_series_parquet
    W, H = fx.window_size, fx.prediction_horizon
    n = W + H + 5
    values = np.arange(n, dtype=np.float32)
    seg_ids = np.zeros(n, dtype=np.int32)
    is_anomaly = np.zeros(n, dtype=bool)
    idx = _build_window_index(seg_ids, is_anomaly, W, H, False)
    ds = WindowedSequenceDataset(values, idx, W, H)
    x, y = ds[0]
    assert x.shape == (W, 1), f"Expected ({W}, 1), got {x.shape}"
    assert y.shape == torch.Size([])  # scalar


def test_dataset_values_correct() -> None:
    """x and y must match the expected slices from the values array."""
    W, H = 3, 1
    values = np.arange(10, dtype=np.float32)
    seg_ids = np.zeros(10, dtype=np.int32)
    is_anomaly = np.zeros(10, dtype=bool)
    idx = _build_window_index(seg_ids, is_anomaly, W, H, False)

    ds = WindowedSequenceDataset(values, idx, W, H)
    # start index 0 → x = values[0:3] = [0,1,2], y = values[3] = 3.0
    x, y = ds[0]
    assert x.squeeze(-1).tolist() == pytest.approx([0.0, 1.0, 2.0])
    assert y.item() == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# make_dataloaders
# ---------------------------------------------------------------------------


def test_dataloader_yields_correct_batch_shape(
    tiny_series_parquet: SeriesParquetFixture,
) -> None:
    """First batch x must have shape (B, W, 1)."""
    from spacecraft_telemetry.core.config import Settings

    fx = tiny_series_parquet
    settings = Settings(
        model={
            "window_size": fx.window_size,
            "prediction_horizon": fx.prediction_horizon,
            "batch_size": 8,
        },
        preprocess={"processed_data_dir": str(fx.processed_dir)},
    )
    train_loader, _ = make_dataloaders(settings, fx.mission, fx.channel)
    x_batch, _ = next(iter(train_loader))
    assert x_batch.shape == (8, fx.window_size, 1)


def test_val_split_is_temporal_not_random(
    tiny_series_parquet: SeriesParquetFixture,
) -> None:
    """Val set must be the contiguous tail of the valid window index."""
    from spacecraft_telemetry.core.config import Settings

    fx = tiny_series_parquet
    settings = Settings(
        model={"window_size": fx.window_size, "prediction_horizon": fx.prediction_horizon,
               "val_fraction": 0.2, "batch_size": 4},
        preprocess={"processed_data_dir": str(fx.processed_dir)},
    )
    n_val = max(1, int(fx.n_train_windows * 0.2))
    n_train_expected = fx.n_train_windows - n_val

    train_loader, val_loader = make_dataloaders(settings, fx.mission, fx.channel)
    assert len(train_loader.dataset) == n_train_expected  # type: ignore[arg-type]
    assert len(val_loader.dataset) == n_val               # type: ignore[arg-type]


def test_val_loader_is_deterministic(
    tiny_series_parquet: SeriesParquetFixture,
) -> None:
    """Two passes over the val loader must yield identical order — shuffle=False."""
    from spacecraft_telemetry.core.config import Settings

    fx = tiny_series_parquet
    settings = Settings(
        model={
            "window_size": fx.window_size,
            "prediction_horizon": fx.prediction_horizon,
            "batch_size": 4,
        },
        preprocess={"processed_data_dir": str(fx.processed_dir)},
    )
    _, val_loader = make_dataloaders(settings, fx.mission, fx.channel)

    val_pass1 = torch.cat([y for _, y in val_loader]).numpy()
    val_pass2 = torch.cat([y for _, y in val_loader]).numpy()
    np.testing.assert_array_equal(val_pass1, val_pass2)


def test_make_dataloaders_skips_anomalous_train_windows(
    tiny_series_parquet: SeriesParquetFixture,
) -> None:
    """Train split has no anomalies, so all valid windows are included."""
    from spacecraft_telemetry.core.config import Settings

    fx = tiny_series_parquet
    settings = Settings(
        model={"window_size": fx.window_size, "prediction_horizon": fx.prediction_horizon},
        preprocess={"processed_data_dir": str(fx.processed_dir)},
    )
    train_loader, val_loader = make_dataloaders(settings, fx.mission, fx.channel)
    total = len(train_loader.dataset) + len(val_loader.dataset)  # type: ignore[arg-type]
    assert total == fx.n_train_windows


# ---------------------------------------------------------------------------
# make_test_dataloader
# ---------------------------------------------------------------------------


def test_make_test_dataloader_returns_correct_window_count(
    tiny_series_parquet: SeriesParquetFixture,
) -> None:
    from spacecraft_telemetry.core.config import Settings

    fx = tiny_series_parquet
    settings = Settings(
        model={"window_size": fx.window_size, "prediction_horizon": fx.prediction_horizon},
        preprocess={"processed_data_dir": str(fx.processed_dir)},
    )
    loader, target_timestamps, window_is_anomaly = make_test_dataloader(
        settings, fx.mission, fx.channel
    )
    assert len(loader.dataset) == fx.n_test_windows  # type: ignore[arg-type]
    assert len(target_timestamps) == fx.n_test_windows
    assert len(window_is_anomaly) == fx.n_test_windows


def test_make_test_dataloader_target_timestamps_monotone(
    tiny_series_parquet: SeriesParquetFixture,
) -> None:
    """target_timestamps must be non-decreasing (windows are in time order)."""
    from spacecraft_telemetry.core.config import Settings

    fx = tiny_series_parquet
    settings = Settings(
        model={"window_size": fx.window_size, "prediction_horizon": fx.prediction_horizon},
        preprocess={"processed_data_dir": str(fx.processed_dir)},
    )
    _, target_timestamps, _ = make_test_dataloader(settings, fx.mission, fx.channel)
    diffs = np.diff(pd.DatetimeIndex(target_timestamps).asi8)
    assert (diffs >= 0).all(), "target_timestamps are not non-decreasing"


def test_make_test_dataloader_anomaly_flags_match_tail(
    tiny_series_parquet: SeriesParquetFixture,
) -> None:
    """Windows overlapping the last 5 anomalous rows must be flagged."""
    from spacecraft_telemetry.core.config import Settings

    fx = tiny_series_parquet
    settings = Settings(
        model={"window_size": fx.window_size, "prediction_horizon": fx.prediction_horizon},
        preprocess={"processed_data_dir": str(fx.processed_dir)},
    )
    _, _, window_is_anomaly = make_test_dataloader(settings, fx.mission, fx.channel)

    # At least some windows must be flagged (those whose span touches the anomalous tail).
    assert window_is_anomaly.any(), "Expected some anomalous windows in test split"
    # Nominal windows at the start must NOT be flagged.
    # First window (start=0) covers rows 0..span-1 — all nominal since anomaly
    # starts at row (N_TEST_ROWS - ANOMALY_ROWS) = 25, and span=11.
    assert not window_is_anomaly[0], "First window should not be anomalous"


# ---------------------------------------------------------------------------
# window_target_timestamps
# ---------------------------------------------------------------------------


def test_window_target_timestamps_matches_make_test_dataloader(
    tiny_series_parquet: SeriesParquetFixture,
) -> None:
    """The torch-free helper must reproduce make_test_dataloader's timestamps exactly.

    This is the property esa_adb.detections relies on: reconstructing
    detection intervals from a logged errors.npy array (indexed the same way
    as make_test_dataloader's target_timestamps) without re-running inference.
    """
    from spacecraft_telemetry.core.config import Settings

    fx = tiny_series_parquet
    settings = Settings(
        model={"window_size": fx.window_size, "prediction_horizon": fx.prediction_horizon},
        preprocess={"processed_data_dir": str(fx.processed_dir)},
    )
    _, expected_timestamps, _ = make_test_dataloader(settings, fx.mission, fx.channel)
    actual_timestamps = window_target_timestamps(settings, fx.mission, fx.channel)

    np.testing.assert_array_equal(actual_timestamps, expected_timestamps)


def test_window_target_timestamps_returns_correct_count(
    tiny_series_parquet: SeriesParquetFixture,
) -> None:
    from spacecraft_telemetry.core.config import Settings

    fx = tiny_series_parquet
    settings = Settings(
        model={"window_size": fx.window_size, "prediction_horizon": fx.prediction_horizon},
        preprocess={"processed_data_dir": str(fx.processed_dir)},
    )
    result = window_target_timestamps(settings, fx.mission, fx.channel)
    assert len(result) == fx.n_test_windows


def test_window_target_timestamps_from_metadata_matches_disk_read(
    tiny_series_parquet: SeriesParquetFixture,
) -> None:
    """The preloaded-arrays path must reproduce window_target_timestamps exactly.

    Regression for docs/plans/019 P2/P3: esa_adb.report.build_report preloads
    (segment_ids, is_anomaly, timestamps) once per channel via
    load_series_metadata() and reuses them across mission_timeline,
    hpo_cutoff, and detection reconstruction instead of re-reading the
    partition each time — this must be a pure refactor with no behaviour
    change.
    """
    from spacecraft_telemetry.core.config import Settings

    fx = tiny_series_parquet
    settings = Settings(
        model={"window_size": fx.window_size, "prediction_horizon": fx.prediction_horizon},
        preprocess={"processed_data_dir": str(fx.processed_dir)},
    )
    expected = window_target_timestamps(settings, fx.mission, fx.channel)

    segment_ids, is_anomaly, timestamps = load_series_metadata(
        fx.processed_dir, fx.mission, fx.channel, "test"
    )
    actual = window_target_timestamps_from_metadata(settings, segment_ids, is_anomaly, timestamps)

    np.testing.assert_array_equal(actual, expected)


# ---------------------------------------------------------------------------
# Multivariate (docs/plans/021-multivariate-telemanom.md)
# ---------------------------------------------------------------------------


def _write_synthetic_channel(
    processed_dir: Path,
    mission: str,
    channel: str,
    split: str,
    n_rows: int,
    value_offset: float,
    anomaly_tail: int = 0,
    seg_sizes: tuple[int, ...] | None = None,
) -> None:
    """Write one channel's per-timestep series Parquet for multi-channel tests.

    All channels share the same 90s-cadence timestamp grid starting at a
    fixed epoch, so tests fully control alignment; ``value_offset`` keeps
    each channel's values distinguishable after stacking into (N, C).
    """
    seg_sizes = seg_sizes or (n_rows,)
    assert sum(seg_sizes) == n_rows
    base = datetime(2000, 1, 1, tzinfo=UTC).timestamp()
    timestamps = [
        pa.scalar(base + i * 90, type=pa.timestamp("s", tz="UTC")).cast(
            pa.timestamp("us", tz="UTC")
        )
        for i in range(n_rows)
    ]
    values = [float(i) + value_offset for i in range(n_rows)]
    seg_ids = [seg for seg, size in enumerate(seg_sizes) for _ in range(size)]
    is_anomaly = [False] * n_rows
    for i in range(n_rows - anomaly_tail, n_rows):
        is_anomaly[i] = True

    table = pa.table(
        {
            "telemetry_timestamp": pa.array(timestamps, type=pa.timestamp("us", tz="UTC")),
            "value_normalized": pa.array(values, type=pa.float32()),
            "segment_id": pa.array(seg_ids, type=pa.int32()),
            "is_anomaly": pa.array(is_anomaly, type=pa.bool_()),
        }
    )
    part_dir = processed_dir / mission / split / f"mission_id={mission}" / f"channel_id={channel}"
    part_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, part_dir / "part.parquet")


# --- _resolve_channel_group ---


def test_resolve_channel_group_none_defaults_to_single_channel() -> None:
    from spacecraft_telemetry.core.config import Settings

    settings = Settings()
    assert _resolve_channel_group(settings, "channel_41") == ["channel_41"]


def test_resolve_channel_group_uses_input_channels_when_set() -> None:
    from spacecraft_telemetry.core.config import Settings

    settings = Settings(
        model={
            "input_channels": ["channel_41", "channel_42"],
            "target_channels": ["channel_41", "channel_42"],
        }
    )
    assert _resolve_channel_group(settings, "subsystem_1") == ["channel_41", "channel_42"]


# --- _joint_segment_ids ---


def test_joint_segment_ids_boundary_from_any_channel() -> None:
    # channel A boundary at t=3; channel B boundary at t=2 -> joint has both.
    seg = np.array([[0, 0], [0, 0], [0, 1], [1, 1], [1, 1]], dtype=np.int32)
    joint = _joint_segment_ids(seg)
    np.testing.assert_array_equal(joint, [0, 0, 1, 2, 2])


def test_joint_segment_ids_empty_and_single_row() -> None:
    assert _joint_segment_ids(np.empty((0, 2), dtype=np.int32)).shape == (0,)
    np.testing.assert_array_equal(_joint_segment_ids(np.array([[3, 3]], dtype=np.int32)), [0])


# --- _align_multi_channel ---


def test_align_multi_channel_full_overlap() -> None:
    ts = np.array([0, 60, 120, 180], dtype="datetime64[s]")
    ch_a = (
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        np.zeros(4, dtype=np.int32),
        np.zeros(4, dtype=bool),
        ts,
    )
    ch_b = (
        np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32),
        np.zeros(4, dtype=np.int32),
        np.array([False, True, False, False]),
        ts,
    )
    values, seg, is_anom, out_ts = _align_multi_channel([ch_a, ch_b], ["a", "b"])
    assert values.shape == (4, 2)
    np.testing.assert_array_equal(values[:, 0], [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_equal(values[:, 1], [10.0, 20.0, 30.0, 40.0])
    np.testing.assert_array_equal(is_anom[:, 1], [False, True, False, False])
    np.testing.assert_array_equal(seg, [0, 0, 0, 0])
    np.testing.assert_array_equal(pd.DatetimeIndex(out_ts), pd.DatetimeIndex(ts))


def test_align_multi_channel_partial_overlap_intersects() -> None:
    """Alignment loss: a channel missing rows shrinks the joined series,
    and the surviving rows must still map to the correct source values."""
    ts_a = np.array([0, 60, 120, 180], dtype="datetime64[s]")
    ts_b = np.array([60, 120, 180, 240], dtype="datetime64[s]")  # missing t=0, extra t=240
    ch_a = (
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        np.zeros(4, dtype=np.int32),
        np.zeros(4, dtype=bool),
        ts_a,
    )
    ch_b = (
        np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32),
        np.zeros(4, dtype=np.int32),
        np.zeros(4, dtype=bool),
        ts_b,
    )
    values, _seg, _is_anom, out_ts = _align_multi_channel([ch_a, ch_b], ["a", "b"])
    assert len(out_ts) == 3
    np.testing.assert_array_equal(values[:, 0], [2.0, 3.0, 4.0])
    np.testing.assert_array_equal(values[:, 1], [10.0, 20.0, 30.0])


def test_align_multi_channel_zero_overlap_raises() -> None:
    ts_a = np.array([0, 60], dtype="datetime64[s]")
    ts_b = np.array([1000, 1060], dtype="datetime64[s]")
    zeros_f32, zeros_i32, zeros_bool = (
        np.zeros(2, dtype=np.float32),
        np.zeros(2, dtype=np.int32),
        np.zeros(2, dtype=bool),
    )
    ch_a = (zeros_f32, zeros_i32, zeros_bool, ts_a)
    ch_b = (zeros_f32, zeros_i32, zeros_bool, ts_b)
    with pytest.raises(ValueError, match="No overlapping timestamps"):
        _align_multi_channel([ch_a, ch_b], ["a", "b"])


def test_align_multi_channel_none_values_returns_none_and_skips_values_array() -> None:
    """docs/reviews/023-channel-time-grid.md P2: a metadata-only alignment
    (every element's values is None) must not materialise an (N, C) values
    array — segment_ids/is_anomaly/timestamps are unaffected."""
    ts = np.array([0, 60, 120, 180], dtype="datetime64[s]")
    ch_a = (None, np.zeros(4, dtype=np.int32), np.zeros(4, dtype=bool), ts)
    ch_b = (None, np.zeros(4, dtype=np.int32), np.array([False, True, False, False]), ts)
    values, seg, is_anom, out_ts = _align_multi_channel([ch_a, ch_b], ["a", "b"])
    assert values is None
    np.testing.assert_array_equal(is_anom[:, 1], [False, True, False, False])
    np.testing.assert_array_equal(seg, [0, 0, 0, 0])
    assert len(out_ts) == 4


def test_align_multi_channel_releases_each_channel_as_its_column_is_copied() -> None:
    """P2: per_channel is consumed — each element is set to None once its
    column is filled, so peak memory isn't every input channel plus the
    aligned output arrays held simultaneously."""
    ts = np.array([0, 60, 120, 180], dtype="datetime64[s]")
    ch_a = (
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32), np.zeros(4, dtype=np.int32),
        np.zeros(4, dtype=bool), ts,
    )
    ch_b = (
        np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32), np.zeros(4, dtype=np.int32),
        np.zeros(4, dtype=bool), ts,
    )
    per_channel = [ch_a, ch_b]
    _align_multi_channel(per_channel, ["a", "b"])
    assert per_channel == [None, None]


def test_load_window_labels_from_metadata_matches_the_reading_variant(
    tmp_path: Path,
) -> None:
    """The preload variant must be indistinguishable from the re-reading one.

    docs/reviews/021 item D1: threshold_ceiling.py read each channel's test
    partition four times because the labels call had no preload counterpart.
    Adding one is only safe if it computes exactly the same thing — this pins
    that, since a divergence would misalign errors against labels rather than
    fail.
    """
    from spacecraft_telemetry.core.config import load_settings
    from spacecraft_telemetry.model.dataset import (
        load_series_metadata,
        load_window_labels,
        load_window_labels_from_metadata,
    )

    mission, channel = "ESA-Mission1", "channel_1"
    processed_dir = tmp_path / "processed"
    n = 60
    base = datetime(2000, 1, 1, tzinfo=UTC).timestamp()
    ts = [
        pa.scalar(base + i * 90, type=pa.timestamp("s", tz="UTC")).cast(
            pa.timestamp("us", tz="UTC")
        )
        for i in range(n)
    ]
    # Two segments and a mid-series anomaly run, so the window index is not
    # trivially "every start" and the OR actually has something to find.
    segments = np.array([0] * 35 + [1] * 25, dtype=np.int32)
    flags = [False] * 20 + [True] * 6 + [False] * 34
    table = pa.table({
        "telemetry_timestamp": pa.array(ts, type=pa.timestamp("us", tz="UTC")),
        "value_normalized": pa.array([float(i % 5) for i in range(n)], type=pa.float32()),
        "segment_id": pa.array(segments),
        "is_anomaly": pa.array(flags),
    })
    part = (
        processed_dir / mission / "test" / f"mission_id={mission}" / f"channel_id={channel}"
    )
    part.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, part / "part.parquet")

    base_settings = load_settings("test")
    settings = base_settings.model_copy(update={
        "preprocess": base_settings.preprocess.model_copy(
            update={"processed_data_dir": processed_dir}
        ),
        "model": base_settings.model.model_copy(
            update={"window_size": 5, "forecast_steps": 3}
        ),
    })

    segment_ids, is_anomaly, _timestamps = load_series_metadata(
        processed_dir, mission, channel, "test"
    )
    expected = load_window_labels(settings, mission, channel)
    actual = load_window_labels_from_metadata(settings, segment_ids, is_anomaly)

    assert len(expected) > 0
    np.testing.assert_array_equal(actual, expected)


class TestChannelKeyPairing:
    """docs/reviews/021 item A2 — `channel` means two different things.

    A real channel_id univariately; a group KEY (subsystem name) when
    input_channels is set. The typed fix was deliberately declined (~8 modules,
    immediately before DC-VAE reuses that boundary); this cheap assert catches
    the one confusion the duality actually enables, which would otherwise
    register a whole-group model under a real channel's name and pollute the
    univariate registry with no shape error anywhere.
    """

    @staticmethod
    def _cfg(**kwargs: Any) -> Any:
        from spacecraft_telemetry.core.config import ModelConfig

        return ModelConfig(**kwargs)

    def test_group_key_is_accepted(self) -> None:
        from spacecraft_telemetry.model.dataset import check_channel_key_pairing

        group = ["channel_41", "channel_42"]
        check_channel_key_pairing(
            self._cfg(input_channels=group, target_channels=group), "subsystem_5"
        )

    def test_group_member_as_key_raises(self) -> None:
        from spacecraft_telemetry.model.dataset import check_channel_key_pairing

        group = ["channel_41", "channel_42"]
        with pytest.raises(ValueError, match="is a MEMBER of input_channels"):
            check_channel_key_pairing(
                self._cfg(input_channels=group, target_channels=group), "channel_41"
            )

    def test_univariate_path_is_unaffected(self) -> None:
        """With input_channels unset, `channel` IS a real channel and must
        always pass — the guard must not touch the pre-021 path."""
        from spacecraft_telemetry.model.dataset import check_channel_key_pairing

        check_channel_key_pairing(self._cfg(), "channel_41")


class TestConcurrentChannelLoadOrder:
    """docs/reviews/021 item D2 — channel loads run concurrently now.

    ``_align_multi_channel`` indexes ``channels`` by POSITION, so a result list
    ordered by completion instead of submission would attribute every channel's
    values to a different channel: identical shapes, no exception, every metric
    silently wrong. These pin submission order under deliberately adversarial
    completion order.
    """

    def test_results_follow_input_order_not_completion_order(self) -> None:
        import time

        from spacecraft_telemetry.model.dataset import _load_channels_ordered

        channels = [f"channel_{i}" for i in range(6)]

        def _slow_first(channel: str) -> str:
            # The FIRST channel is slowest, so completion order is the exact
            # reverse of submission order — the worst case for the bug.
            time.sleep(0.05 * (len(channels) - channels.index(channel)))
            return channel

        assert _load_channels_ordered(_slow_first, channels) == channels

    def test_single_channel_skips_the_pool(self) -> None:
        """The univariate path must take on no thread-pool behaviour."""
        import threading

        from spacecraft_telemetry.model.dataset import _load_channels_ordered

        calling_thread: list[int] = []

        def _record(channel: str) -> str:
            calling_thread.append(threading.get_ident())
            return channel

        assert _load_channels_ordered(_record, ["channel_1"]) == ["channel_1"]
        assert calling_thread == [threading.get_ident()]

    def test_a_failing_channel_propagates(self) -> None:
        """A missing partition must still raise, not be swallowed by the pool."""
        from spacecraft_telemetry.model.dataset import _load_channels_ordered

        def _boom(channel: str) -> str:
            if channel == "channel_2":
                raise FileNotFoundError(channel)
            return channel

        with pytest.raises(FileNotFoundError, match="channel_2"):
            _load_channels_ordered(_boom, ["channel_1", "channel_2", "channel_3"])


def test_multichannel_columns_match_requested_order_under_concurrency(
    tmp_path: Path,
) -> None:
    """End-to-end: column i of the loaded array is channels[i]'s data.

    Each channel is written with a distinct constant value, so a mis-ordered
    result is detectable by value rather than only by shape. The requested
    order is deliberately NOT the on-disk/sorted order.
    """
    from spacecraft_telemetry.model.dataset import load_multichannel_series_parquet

    mission = "ESA-Mission1"
    processed_dir = tmp_path / "processed"
    n = 12
    base = datetime(2000, 1, 1, tzinfo=UTC).timestamp()
    ts = [
        pa.scalar(base + i * 90, type=pa.timestamp("s", tz="UTC")).cast(
            pa.timestamp("us", tz="UTC")
        )
        for i in range(n)
    ]
    written = [f"channel_{i}" for i in range(6)]
    for marker, channel in enumerate(written):
        table = pa.table({
            "telemetry_timestamp": pa.array(ts, type=pa.timestamp("us", tz="UTC")),
            "value_normalized": pa.array([float(marker)] * n, type=pa.float32()),
            "segment_id": pa.array(np.zeros(n, dtype=np.int32)),
            "is_anomaly": pa.array([marker == 3] * n),
        })
        part = (
            processed_dir / mission / "test"
            / f"mission_id={mission}" / f"channel_id={channel}"
        )
        part.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, part / "part.parquet")

    requested = ["channel_4", "channel_0", "channel_5", "channel_2", "channel_3", "channel_1"]
    values, _seg, is_anom, _out_ts = load_multichannel_series_parquet(
        processed_dir, mission, requested, "test"
    )

    assert values.shape == (n, len(requested))
    for i, channel in enumerate(requested):
        expected_marker = float(written.index(channel))
        np.testing.assert_array_equal(
            values[:, i],
            np.full(n, expected_marker, dtype=np.float32),
            err_msg=f"column {i} should hold {channel}'s values",
        )
    # The per-channel anomaly flag must follow the same positional mapping.
    assert is_anom[:, requested.index("channel_3")].all()
    assert not is_anom[:, requested.index("channel_0")].any()


class TestMultivariateGroupSizeGuard:
    """docs/reviews/021 item A4 — nothing bounded the group size.

    _align_multi_channel materialises all C channels densely; a whole mission
    passed as one group would need ~13 GB. An OOM on a spot worker is an
    expensive way to learn that, so the group size is checked up front.
    """

    @staticmethod
    def _channels(n: int, rows: int = 3) -> tuple[list[Any], list[str]]:
        ts = (np.arange(rows, dtype="int64") * 60).astype("datetime64[s]")
        one = (
            np.zeros(rows, dtype=np.float32),
            np.zeros(rows, dtype=np.int32),
            np.zeros(rows, dtype=bool),
            ts,
        )
        return [one] * n, [f"channel_{i}" for i in range(n)]

    @staticmethod
    def _budget(n: int, rows: int) -> int:
        from spacecraft_telemetry.model.dataset import _BYTES_PER_ALIGNED_ROW_PER_CHANNEL

        return n * rows * _BYTES_PER_ALIGNED_ROW_PER_CHANNEL

    def test_group_at_the_budget_is_accepted(self) -> None:
        """The boundary is inclusive — a group exactly at the limit must not
        be refused, or the documented maximum is off by one."""
        per_channel, names = self._channels(4)
        values, _seg, _is_anom, _ts = _align_multi_channel(
            per_channel, names, max_bytes=self._budget(4, 3)
        )
        assert values.shape == (3, 4)

    def test_group_above_the_budget_raises_with_the_estimate_and_the_remedy(self) -> None:
        per_channel, names = self._channels(4)
        with pytest.raises(ValueError, match="would materialise") as exc:
            _align_multi_channel(per_channel, names, max_bytes=self._budget(4, 3) - 1)
        # The message must name the remedy, not just the rule.
        assert "Split the group" in str(exc.value)

    def test_budget_follows_row_count_not_just_channel_count(self) -> None:
        """The reason this guard stopped being a channel count (plan 023 .3).

        Same number of channels, 100x the rows: the first fits and the second
        does not. A count limit cannot express that, which is how ESA's
        41-channel subsystem_6 came to be refused at ~0.44 GB gridded while a
        group nine times heavier was waved through.
        """
        budget = self._budget(41, 3)
        narrow, names = self._channels(41, rows=3)
        _align_multi_channel(narrow, names, max_bytes=budget)  # fits exactly

        wide, names = self._channels(41, rows=300)
        with pytest.raises(ValueError, match="would materialise"):
            _align_multi_channel(wide, names, max_bytes=budget)

    def test_no_count_limit_applies_by_default(self) -> None:
        """41 channels is ESA-Mission1's real subsystem_6 on the 30 s grid, and
        it is above the retired 32-channel constant. It must be accepted."""
        per_channel, names = self._channels(41)
        values, _seg, _is_anom, _ts = _align_multi_channel(per_channel, names)
        assert values.shape == (3, 41)

    def test_limit_is_overridable_for_a_deliberate_large_group(self) -> None:
        per_channel, names = self._channels(3)
        with pytest.raises(ValueError, match="above the max_channels=2"):
            _align_multi_channel(per_channel, names, max_channels=2)

    def test_six_channel_path_is_untouched(self) -> None:
        """The size plan 021 actually runs — pinned so tightening the limit
        later cannot silently break the published configuration."""
        per_channel, names = self._channels(6)
        values, _seg, _is_anom, _ts = _align_multi_channel(per_channel, names)
        assert values.shape == (3, 6)


class TestEstimateAlignBytes:
    """Pins the calibration _MAX_MULTIVARIATE_BYTES's comment claims.

    Every rejection test above passes an explicit max_bytes, so the SHIPPED
    default constant itself could drift to e.g. 2**60 and the suite would
    stay green (docs/reviews/023-channel-time-grid.md finding T2). These two
    assert against the shipped constant directly, using the comment's own
    ~100-channel native-vs-gridded numbers.
    """

    def test_a_100_channel_native_mission_is_rejected(self) -> None:
        from spacecraft_telemetry.model.dataset import (
            _MAX_MULTIVARIATE_BYTES,
            _estimate_align_bytes,
        )

        assert _estimate_align_bytes(100, 7_700_000) > _MAX_MULTIVARIATE_BYTES

    def test_the_same_mission_gridded_is_accepted(self) -> None:
        from spacecraft_telemetry.model.dataset import (
            _MAX_MULTIVARIATE_BYTES,
            _estimate_align_bytes,
        )

        assert _estimate_align_bytes(100, 680_000) <= _MAX_MULTIVARIATE_BYTES


def test_load_multichannel_series_parquet_missing_channel_raises(tmp_path: Path) -> None:
    mission = "ESA-Mission1"
    processed_dir = tmp_path / "processed"
    _write_synthetic_channel(processed_dir, mission, "channel_1", "train", 10, value_offset=0.0)
    with pytest.raises(FileNotFoundError, match="channel_id=channel_2"):
        load_multichannel_series_parquet(
            processed_dir, mission, ["channel_1", "channel_2"], "train"
        )


# --- _window_any_anomalous ---


def test_window_any_anomalous_1d_matches_prefix_sum_definition() -> None:
    is_anom = np.array([False, False, True, False, False])
    indices = np.array([0, 1, 2], dtype=np.int32)
    result = _window_any_anomalous(is_anom, indices, span=2)
    np.testing.assert_array_equal(result, [False, True, True])


def test_window_any_anomalous_2d_is_independent_per_channel() -> None:
    is_anom = np.array([[False, True], [False, False], [True, False]])
    indices = np.array([0, 1], dtype=np.int32)
    result = _window_any_anomalous(is_anom, indices, span=2)
    assert result.shape == (2, 2)
    np.testing.assert_array_equal(result, [[False, True], [True, False]])


# --- WindowedSequenceDataset (multivariate branch) ---


def test_windowed_dataset_multivariate_item_shapes() -> None:
    W, H, C = 3, 1, 2
    n = W + H + 5
    values = np.stack(
        [np.arange(n, dtype=np.float32), np.arange(n, dtype=np.float32) * 10], axis=1
    )
    seg_ids = np.zeros(n, dtype=np.int32)
    is_anomaly = np.zeros(n, dtype=bool)
    idx = _build_window_index(seg_ids, is_anomaly, W, H, False)
    ds = WindowedSequenceDataset(values, idx, W, H)
    x, y = ds[0]
    assert x.shape == (W, C)
    assert y.shape == (C,)


def test_windowed_dataset_multivariate_values_correct() -> None:
    W, H = 3, 1
    values = np.stack(
        [np.arange(10, dtype=np.float32), np.arange(10, dtype=np.float32) + 100], axis=1
    )
    seg_ids = np.zeros(10, dtype=np.int32)
    is_anomaly = np.zeros(10, dtype=bool)
    idx = _build_window_index(seg_ids, is_anomaly, W, H, False)
    ds = WindowedSequenceDataset(values, idx, W, H)
    x, y = ds[0]
    np.testing.assert_allclose(x.numpy(), [[0, 100], [1, 101], [2, 102]])
    np.testing.assert_allclose(y.numpy(), [3, 103])


def test_windowed_dataset_rejects_bad_ndim() -> None:
    values = np.zeros((4, 2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="1-D or 2-D"):
        WindowedSequenceDataset(values, np.array([0], dtype=np.int32), 2, 1)


# --- make_dataloaders / make_test_dataloader with input_channels ---


def test_make_dataloaders_multichannel_shapes_and_window_count(tmp_path: Path) -> None:
    from spacecraft_telemetry.core.config import Settings

    mission = "ESA-Mission1"
    processed_dir = tmp_path / "processed"
    W, H = 5, 1
    n_rows = 20
    _write_synthetic_channel(
        processed_dir, mission, "channel_1", "train", n_rows, value_offset=0.0
    )
    _write_synthetic_channel(
        processed_dir, mission, "channel_2", "train", n_rows, value_offset=100.0
    )

    settings = Settings(
        model={
            "window_size": W,
            "prediction_horizon": H,
            "input_channels": ["channel_1", "channel_2"],
            "target_channels": ["channel_1", "channel_2"],
            "batch_size": 4,
        },
        preprocess={"processed_data_dir": str(processed_dir)},
    )
    train_loader, val_loader = make_dataloaders(settings, mission, "subsystem_1")
    x_batch, y_batch = next(iter(train_loader))
    assert x_batch.shape[1:] == (W, 2)
    assert y_batch.shape[1:] == (2,)

    span = W + H
    expected_windows = n_rows - span + 1
    total = len(train_loader.dataset) + len(val_loader.dataset)  # type: ignore[arg-type]
    assert total == expected_windows


def test_make_test_dataloader_multichannel_per_channel_anomaly_flags(tmp_path: Path) -> None:
    from spacecraft_telemetry.core.config import Settings

    mission = "ESA-Mission1"
    processed_dir = tmp_path / "processed"
    W, H = 3, 1
    n_rows = 10
    _write_synthetic_channel(
        processed_dir, mission, "channel_1", "test", n_rows, value_offset=0.0, anomaly_tail=0
    )
    _write_synthetic_channel(
        processed_dir, mission, "channel_2", "test", n_rows, value_offset=100.0, anomaly_tail=2
    )

    settings = Settings(
        model={
            "window_size": W,
            "prediction_horizon": H,
            "input_channels": ["channel_1", "channel_2"],
            "target_channels": ["channel_1", "channel_2"],
        },
        preprocess={"processed_data_dir": str(processed_dir)},
    )
    _loader, _target_timestamps, window_is_anomaly = make_test_dataloader(
        settings, mission, "subsystem_1"
    )
    assert window_is_anomaly.shape[1] == 2
    assert not window_is_anomaly[:, 0].any()  # channel_1 is clean
    assert window_is_anomaly[:, 1].any()  # channel_2 has an anomalous tail
    assert not window_is_anomaly[0, 1]  # first window doesn't touch the tail


class TestForecastSteps:
    """Multi-step horizon (docs/plans/021, stage 021.7b).

    The span, the target-timestamp attribution, and the per-window anomaly OR
    must all agree about how many timesteps a window consumes — a one-off
    disagreement misaligns errors against labels silently rather than raising.
    """

    def test_span_and_offset_reduce_to_pre_021_7_at_f1(self) -> None:
        from spacecraft_telemetry.core.config import Settings
        from spacecraft_telemetry.model.dataset import first_target_offset, window_span

        cfg = Settings(model={"window_size": 250, "prediction_horizon": 1}).model
        assert window_span(cfg) == 250 + 1
        assert first_target_offset(cfg) == 250 + 1 - 1

    def test_span_grows_with_horizon(self) -> None:
        from spacecraft_telemetry.core.config import Settings
        from spacecraft_telemetry.model.dataset import first_target_offset, window_span

        cfg = Settings(model={
            "window_size": 250, "prediction_horizon": 1, "forecast_steps": 10
        }).model
        assert window_span(cfg) == 250 + 1 + 10 - 1
        # The first target does NOT move — only the span extends past it.
        assert first_target_offset(cfg) == 250

    def test_longer_horizon_yields_fewer_windows(self) -> None:
        """A longer horizon consumes more timesteps, so fewer windows fit."""
        n, W = 60, 10
        seg = np.zeros(n, dtype=np.int32)
        anom = np.zeros(n, dtype=bool)
        one = _build_window_index(seg, anom, W, 1, False, forecast_steps=1)
        ten = _build_window_index(seg, anom, W, 1, False, forecast_steps=10)
        assert len(one) == n - W  # 50
        assert len(ten) == len(one) - 9

    def test_multivariate_target_is_channel_major(self) -> None:
        """y[i] must be channel i's trajectory, matching TelemanomLSTM's
        (B, C, H) output so the MSE is elementwise with no transpose."""
        W, F, C = 3, 4, 2
        n = 30
        values = np.stack(
            [np.arange(n, dtype=np.float32), np.arange(n, dtype=np.float32) + 100], axis=1
        )
        idx = _build_window_index(
            np.zeros(n, dtype=np.int32), np.zeros(n, dtype=bool), W, 1, False, forecast_steps=F
        )
        ds = WindowedSequenceDataset(values, idx, W, 1, F)
        x, y = ds[0]
        assert x.shape == (W, C)
        assert y.shape == (C, F)
        t0 = 0 + W + 1 - 1
        np.testing.assert_allclose(y[0].numpy(), values[t0 : t0 + F, 0])
        np.testing.assert_allclose(y[1].numpy(), values[t0 : t0 + F, 1])

    def test_univariate_multi_step_target_shape(self) -> None:
        W, F = 3, 4
        n = 30
        values = np.arange(n, dtype=np.float32)
        idx = _build_window_index(
            np.zeros(n, dtype=np.int32), np.zeros(n, dtype=bool), W, 1, False, forecast_steps=F
        )
        ds = WindowedSequenceDataset(values, idx, W, 1, F)
        x, y = ds[0]
        assert x.shape == (W, 1)
        assert y.shape == (1, F)
        np.testing.assert_allclose(y[0].numpy(), values[W : W + F])

    def test_f1_shapes_are_unchanged(self) -> None:
        """The whole point of branching at F==1: no trailing axis appears."""
        W = 3
        n = 20
        idx = _build_window_index(
            np.zeros(n, dtype=np.int32), np.zeros(n, dtype=bool), W, 1, False
        )
        uni = WindowedSequenceDataset(np.arange(n, dtype=np.float32), idx, W, 1, 1)
        assert uni[0][1].shape == torch.Size([])
        multi_vals = np.stack([np.arange(n, dtype=np.float32)] * 2, axis=1)
        multi = WindowedSequenceDataset(multi_vals, idx, W, 1, 1)
        assert multi[0][1].shape == (2,)

    def test_targets_never_read_past_the_array(self) -> None:
        """The last valid window's final target must be the last element —
        an off-by-one in the span would silently read garbage or crash."""
        W, F = 5, 6
        n = 40
        values = np.arange(n, dtype=np.float32)
        idx = _build_window_index(
            np.zeros(n, dtype=np.int32), np.zeros(n, dtype=bool), W, 1, False, forecast_steps=F
        )
        ds = WindowedSequenceDataset(values, idx, W, 1, F)
        _x, y = ds[len(ds) - 1]
        assert float(y[0, -1]) == float(values[-1])

    def test_end_to_end_dataloader_shapes(self, tmp_path: Path) -> None:
        from spacecraft_telemetry.core.config import Settings

        mission = "ESA-Mission1"
        processed_dir = tmp_path / "processed"
        chans = ["channel_1", "channel_2"]
        for i, ch in enumerate(chans):
            _write_synthetic_channel(
                processed_dir, mission, ch, "train", 40, value_offset=float(i) * 100
            )
        settings = Settings(
            model={
                "window_size": 5, "prediction_horizon": 1, "forecast_steps": 4,
                "input_channels": chans, "target_channels": chans, "batch_size": 3,
            },
            preprocess={"processed_data_dir": str(processed_dir)},
        )
        train_loader, _ = make_dataloaders(settings, mission, "subsystem_1")
        x, y = next(iter(train_loader))
        assert x.shape[1:] == (5, 2)
        assert y.shape[1:] == (2, 4)


def test_input_channels_single_item_matches_default_univariate_path(
    tiny_series_parquet: SeriesParquetFixture,
) -> None:
    """input_channels=[channel] must reproduce input_channels=None exactly —
    the operative form of 'C==1 reproduces the univariate path exactly.'"""
    from spacecraft_telemetry.core.config import Settings

    fx = tiny_series_parquet
    model_base = {"window_size": fx.window_size, "prediction_horizon": fx.prediction_horizon}
    preprocess = {"processed_data_dir": str(fx.processed_dir)}

    settings_default = Settings(model=model_base, preprocess=preprocess)
    settings_explicit = Settings(
        model={**model_base, "input_channels": [fx.channel], "target_channels": [fx.channel]},
        preprocess=preprocess,
    )

    train_a, val_a = make_dataloaders(settings_default, fx.mission, fx.channel)
    train_b, val_b = make_dataloaders(settings_explicit, fx.mission, fx.channel)
    assert len(train_a.dataset) == len(train_b.dataset)  # type: ignore[arg-type]
    assert len(val_a.dataset) == len(val_b.dataset)  # type: ignore[arg-type]

    # shuffle=True on the train loader, so compare via the deterministic val loader.
    xa_val = torch.cat([x for x, _ in val_a])
    xb_val = torch.cat([x for x, _ in val_b])
    ya_val = torch.cat([y for _, y in val_a])
    yb_val = torch.cat([y for _, y in val_b])
    np.testing.assert_array_equal(xa_val.numpy(), xb_val.numpy())
    np.testing.assert_array_equal(ya_val.numpy(), yb_val.numpy())

    loader_a, ts_a, wa = make_test_dataloader(settings_default, fx.mission, fx.channel)
    loader_b, ts_b, wb = make_test_dataloader(settings_explicit, fx.mission, fx.channel)
    np.testing.assert_array_equal(ts_a, ts_b)
    np.testing.assert_array_equal(wa, wb)
    xa2 = torch.cat([x for x, _ in loader_a])
    xb2 = torch.cat([x for x, _ in loader_b])
    np.testing.assert_array_equal(xa2.numpy(), xb2.numpy())
