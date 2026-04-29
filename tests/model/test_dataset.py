"""Tests for model.dataset — _load_series_parquet, _build_window_index,
WindowedSequenceDataset, make_dataloaders, make_test_dataloader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from spacecraft_telemetry.model.dataset import (
    WindowedSequenceDataset,
    _build_window_index,
    _load_series_parquet,
    make_dataloaders,
    make_test_dataloader,
)
from tests.model.conftest import SeriesParquetFixture, _SERIES_FILE_SCHEMA


# ---------------------------------------------------------------------------
# _load_series_parquet
# ---------------------------------------------------------------------------


def test_load_series_returns_expected_shapes(
    tiny_series_parquet: SeriesParquetFixture,
) -> None:
    fx = tiny_series_parquet
    values, seg_ids, is_anomaly, timestamps = _load_series_parquet(
        fx.processed_dir, fx.mission, fx.channel, "train"
    )
    n_rows = sum((60, 30, 10))  # _SEG_SIZES_TRAIN
    assert values.shape == (n_rows,)
    assert seg_ids.shape == (n_rows,)
    assert is_anomaly.shape == (n_rows,)
    assert timestamps.shape == (n_rows,)


def test_load_series_dtypes(tiny_series_parquet: SeriesParquetFixture) -> None:
    fx = tiny_series_parquet
    values, seg_ids, is_anomaly, _ = _load_series_parquet(
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
    _, _, _, timestamps = _load_series_parquet(
        fx.processed_dir, fx.mission, fx.channel, "train"
    )
    diffs = np.diff(pd.DatetimeIndex(timestamps).asi8)
    assert (diffs >= 0).all(), "Timestamps are not sorted ascending"


def test_load_series_missing_channel_raises(
    tiny_series_parquet: SeriesParquetFixture,
) -> None:
    fx = tiny_series_parquet
    with pytest.raises(FileNotFoundError, match="channel_id=nonexistent"):
        _load_series_parquet(fx.processed_dir, fx.mission, "nonexistent", "train")


def test_load_series_empty_dir_raises(tmp_path: Path) -> None:
    empty_dir = (
        tmp_path / "processed" / "ESA-Mission1" / "train"
        / "mission_id=ESA-Mission1" / "channel_id=channel_1"
    )
    empty_dir.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="no .parquet files"):
        _load_series_parquet(
            tmp_path / "processed", "ESA-Mission1", "channel_1", "train"
        )


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
        model={"window_size": fx.window_size, "prediction_horizon": fx.prediction_horizon, "batch_size": 8},
        spark={"processed_data_dir": str(fx.processed_dir)},
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
        spark={"processed_data_dir": str(fx.processed_dir)},
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
        model={"window_size": fx.window_size, "prediction_horizon": fx.prediction_horizon, "batch_size": 4},
        spark={"processed_data_dir": str(fx.processed_dir)},
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
        spark={"processed_data_dir": str(fx.processed_dir)},
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
        spark={"processed_data_dir": str(fx.processed_dir)},
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
        spark={"processed_data_dir": str(fx.processed_dir)},
    )
    _, target_timestamps, _ = make_test_dataloader(settings, fx.mission, fx.channel)
    diffs = np.diff(pd.DatetimeIndex(target_timestamps).asi8)
    assert (diffs >= 0).all(), "target_timestamps are not non-decreasing"


def test_make_test_dataloader_anomaly_flags_match_tail(
    tiny_series_parquet: SeriesParquetFixture,
) -> None:
    """Windows overlapping the last 5 anomalous rows must be flagged."""
    from spacecraft_telemetry.core.config import Settings
    from tests.model.conftest import _ANOMALY_ROWS

    fx = tiny_series_parquet
    settings = Settings(
        model={"window_size": fx.window_size, "prediction_horizon": fx.prediction_horizon},
        spark={"processed_data_dir": str(fx.processed_dir)},
    )
    _, _, window_is_anomaly = make_test_dataloader(settings, fx.mission, fx.channel)

    # At least some windows must be flagged (those whose span touches the anomalous tail).
    assert window_is_anomaly.any(), "Expected some anomalous windows in test split"
    # Nominal windows at the start must NOT be flagged.
    W, H = fx.window_size, fx.prediction_horizon
    span = W + H
    # First window (start=0) covers rows 0..span-1 — all nominal since anomaly
    # starts at row (N_TEST_ROWS - ANOMALY_ROWS) = 25, and span=11.
    assert not window_is_anomaly[0], "First window should not be anomalous"
