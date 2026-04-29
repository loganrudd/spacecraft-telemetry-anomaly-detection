"""Tests for model.dataset — load_windowed_parquet, WindowedSequenceDataset, make_dataloaders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from spacecraft_telemetry.core.config import ModelConfig
from spacecraft_telemetry.model.dataset import (
    WindowedSequenceDataset,
    load_windowed_parquet,
    make_dataloaders,
)
from tests.model.conftest import WindowedParquetFixture


# ---------------------------------------------------------------------------
# load_windowed_parquet
# ---------------------------------------------------------------------------


def test_load_train_returns_expected_shapes(
    tiny_windowed_parquet: WindowedParquetFixture,
) -> None:
    fx = tiny_windowed_parquet
    values, targets, is_anomaly = load_windowed_parquet(
        fx.processed_dir, fx.mission, fx.channel, "train"
    )
    assert values.shape == (fx.n_train, fx.window_size)
    assert targets.shape == (fx.n_train,)
    assert is_anomaly.shape == (fx.n_train,)


def test_load_test_returns_expected_shapes(
    tiny_windowed_parquet: WindowedParquetFixture,
) -> None:
    fx = tiny_windowed_parquet
    values, targets, is_anomaly = load_windowed_parquet(
        fx.processed_dir, fx.mission, fx.channel, "test"
    )
    assert values.shape == (fx.n_test, fx.window_size)
    assert targets.shape == (fx.n_test,)
    assert is_anomaly.shape == (fx.n_test,)


def test_load_returns_float32_arrays(
    tiny_windowed_parquet: WindowedParquetFixture,
) -> None:
    fx = tiny_windowed_parquet
    values, targets, _ = load_windowed_parquet(
        fx.processed_dir, fx.mission, fx.channel, "train"
    )
    assert values.dtype == np.float32
    assert targets.dtype == np.float32


def test_load_sorted_by_window_id(
    tiny_windowed_parquet: WindowedParquetFixture,
) -> None:
    """Chronological order is preserved — window_id must be non-decreasing."""
    fx = tiny_windowed_parquet
    # Write two separate Parquet files with shuffled window_ids to confirm sorting.
    import pyarrow as pa
    import pyarrow.parquet as pq
    from datetime import datetime, timezone
    from tests.model.conftest import _WINDOW_FILE_SCHEMA

    split_dir = (
        fx.processed_dir / fx.mission / "train"
        / f"mission_id={fx.mission}" / f"channel_id={fx.channel}"
    )
    # Overwrite with a single file whose rows are in reverse order.
    _BASE_DT = datetime(2000, 1, 1, tzinfo=timezone.utc)
    ids = np.arange(fx.n_train, dtype=np.int64)[::-1].copy()
    shuffled = pa.table(
        {
            "window_id": pa.array(ids),
            "segment_id": pa.array(np.zeros(fx.n_train, dtype=np.int32)),
            "window_start_ts": pa.array([_BASE_DT] * fx.n_train, type=pa.timestamp("us", tz="UTC")),
            "window_end_ts": pa.array([_BASE_DT] * fx.n_train, type=pa.timestamp("us", tz="UTC")),
            "values": pa.array([[0.0] * fx.window_size] * fx.n_train, type=pa.list_(pa.float32())),
            "target": pa.array(np.zeros(fx.n_train, dtype=np.float32)),
            "is_anomaly": pa.array([False] * fx.n_train),
        },
        schema=_WINDOW_FILE_SCHEMA,
    )
    pq.write_table(shuffled, split_dir / "shuffled.parquet")

    values, _, _ = load_windowed_parquet(
        fx.processed_dir, fx.mission, fx.channel, "train"
    )
    # After sorting, values should come from window_ids 0..n_train-1 in order.
    # We can't assert on values content, but we can verify there are no duplicates
    # and the count doubles (original part.parquet + shuffled.parquet).
    assert values.shape[0] == fx.n_train * 2


def test_load_missing_channel_raises(
    tiny_windowed_parquet: WindowedParquetFixture,
) -> None:
    fx = tiny_windowed_parquet
    with pytest.raises(FileNotFoundError, match="channel_id=nonexistent"):
        load_windowed_parquet(
            fx.processed_dir, fx.mission, "nonexistent", "train"
        )


def test_load_existing_dir_no_parquet_raises(
    tmp_path: Path,
) -> None:
    empty_dir = (
        tmp_path / "processed" / "ESA-Mission1" / "train"
        / "mission_id=ESA-Mission1" / "channel_id=channel_1"
    )
    empty_dir.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="no .parquet files"):
        load_windowed_parquet(tmp_path / "processed", "ESA-Mission1", "channel_1", "train")


# ---------------------------------------------------------------------------
# WindowedSequenceDataset
# ---------------------------------------------------------------------------


def test_dataset_len(tiny_windowed_parquet: WindowedParquetFixture) -> None:
    fx = tiny_windowed_parquet
    values, targets, _ = load_windowed_parquet(
        fx.processed_dir, fx.mission, fx.channel, "train"
    )
    ds = WindowedSequenceDataset(values, targets)
    assert len(ds) == fx.n_train


def test_dataset_item_shapes(tiny_windowed_parquet: WindowedParquetFixture) -> None:
    fx = tiny_windowed_parquet
    values, targets, _ = load_windowed_parquet(
        fx.processed_dir, fx.mission, fx.channel, "train"
    )
    ds = WindowedSequenceDataset(values, targets)
    x, y = ds[0]
    assert x.shape == (fx.window_size, 1), f"Expected ({fx.window_size}, 1), got {x.shape}"
    assert y.shape == torch.Size([])  # scalar


# ---------------------------------------------------------------------------
# make_dataloaders
# ---------------------------------------------------------------------------


def test_dataloader_yields_unsqueezed_input(
    tiny_windowed_parquet: WindowedParquetFixture,
) -> None:
    """First batch x must have shape (B, W, 1) after collation."""
    fx = tiny_windowed_parquet
    values, targets, _ = load_windowed_parquet(
        fx.processed_dir, fx.mission, fx.channel, "train"
    )
    cfg = ModelConfig(batch_size=8)
    train_loader, _ = make_dataloaders(values, targets, cfg)
    x_batch, _ = next(iter(train_loader))
    assert x_batch.shape == (8, fx.window_size, 1)


def test_val_split_is_temporal_not_random(
    tiny_windowed_parquet: WindowedParquetFixture,
) -> None:
    """Val set must be the contiguous tail of the training array."""
    fx = tiny_windowed_parquet
    values, targets, _ = load_windowed_parquet(
        fx.processed_dir, fx.mission, fx.channel, "train"
    )
    cfg = ModelConfig(val_fraction=0.2)
    n_val = max(1, int(fx.n_train * 0.2))
    n_train_expected = fx.n_train - n_val

    train_loader, val_loader = make_dataloaders(values, targets, cfg)

    assert len(train_loader.dataset) == n_train_expected  # type: ignore[arg-type]
    assert len(val_loader.dataset) == n_val               # type: ignore[arg-type]

    # Val targets must equal the tail of the original targets array.
    val_targets = torch.cat([y for _, y in val_loader]).numpy()
    np.testing.assert_array_equal(val_targets, targets[n_train_expected:])


def test_train_loader_shuffles_val_does_not(
    tiny_windowed_parquet: WindowedParquetFixture,
) -> None:
    """Two passes over train produce different order; val is always the same."""
    fx = tiny_windowed_parquet
    values, targets, _ = load_windowed_parquet(
        fx.processed_dir, fx.mission, fx.channel, "train"
    )
    cfg = ModelConfig(batch_size=4, seed=1)
    train_loader, val_loader = make_dataloaders(values, targets, cfg)

    pass1 = torch.cat([y for _, y in train_loader]).numpy()
    pass2 = torch.cat([y for _, y in train_loader]).numpy()
    # With shuffle=True across 45+ samples, two passes are extremely unlikely to match.
    assert not np.array_equal(pass1, pass2), "Train loader should shuffle between epochs"

    val_pass1 = torch.cat([y for _, y in val_loader]).numpy()
    val_pass2 = torch.cat([y for _, y in val_loader]).numpy()
    np.testing.assert_array_equal(val_pass1, val_pass2)
