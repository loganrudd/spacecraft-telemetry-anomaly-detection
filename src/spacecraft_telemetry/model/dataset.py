"""Dataset utilities for Telemanom LSTM training.

Reads per-timestep series Parquet (written by the Spark preprocessing pipeline)
via PyArrow — no Spark dependency at training time.  Constructs LSTM windows
on-the-fly in the DataLoader, avoiding the 250× disk inflation of pre-materialized
window arrays.

Public API:
    make_dataloaders(settings, mission, channel)     -> (train_loader, val_loader)
    make_test_dataloader(settings, mission, channel) -> (loader, target_timestamps,
                                                         window_is_anomaly)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset as _TorchDataset

from spacecraft_telemetry.core.config import Settings


def _load_series_parquet(
    processed_dir: Path,
    mission: str,
    channel: str,
    split: Literal["train", "test"],
) -> tuple[
    np.ndarray[Any, np.dtype[np.float32]],
    np.ndarray[Any, np.dtype[np.int32]],
    np.ndarray[Any, np.dtype[np.bool_]],
    np.ndarray[Any, Any],
]:
    """Read per-timestep series for one channel partition.

    Reads from the Hive-partitioned layout:
        {processed_dir}/{mission}/{split}/mission_id={mission}/channel_id={channel}/*.parquet

    Returns:
        values:      (N,) float32  — normalized values, sorted by timestamp
        segment_ids: (N,) int32    — segment ID per timestep (for boundary detection)
        is_anomaly:  (N,) bool     — per-timestep anomaly flag
        timestamps:  (N,) datetime64[ns] — telemetry timestamp per timestep

    Raises:
        FileNotFoundError: If the partition directory doesn't exist or has no
            Parquet files.
    """
    partition_dir = (
        processed_dir / mission / split
        / f"mission_id={mission}" / f"channel_id={channel}"
    )
    if not partition_dir.exists():
        raise FileNotFoundError(
            f"No series Parquet found for mission={mission!r} channel={channel!r} "
            f"split={split!r}. Expected directory: {partition_dir}"
        )

    parquet_files = sorted(partition_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"Directory exists but contains no .parquet files: {partition_dir}"
        )

    tables = [
        pq.read_table(
            f,
            columns=["telemetry_timestamp", "value_normalized", "segment_id", "is_anomaly"],
        )
        for f in parquet_files
    ]
    table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
    df = table.to_pandas().sort_values("telemetry_timestamp").reset_index(drop=True)

    values = df["value_normalized"].to_numpy(dtype=np.float32)   # (N,)
    segment_ids = df["segment_id"].to_numpy(dtype=np.int32)       # (N,)
    is_anomaly = df["is_anomaly"].to_numpy(dtype=bool)            # (N,)
    timestamps = df["telemetry_timestamp"].to_numpy()             # (N,) datetime64[ns]

    return values, segment_ids, is_anomaly, timestamps


def _build_window_index(
    segment_ids: np.ndarray[Any, np.dtype[np.int32]],
    is_anomaly: np.ndarray[Any, np.dtype[np.bool_]],
    window_size: int,
    prediction_horizon: int,
    skip_anomalous_windows: bool,
) -> "np.ndarray[Any, Any]":
    """Return int32 array of valid window start indices.

    A start index ``s`` is valid iff:

    1. ``segment_ids[s : s + window_size + prediction_horizon]`` is all one
       value — the window plus target step don't span a segment gap.
    2. If ``skip_anomalous_windows``: no timestep in that span is anomalous.

    Segment IDs are assigned in ascending temporal order by the Spark
    preprocessing pipeline, so ``segment_ids[s] == segment_ids[s + span - 1]``
    is a correct and O(N)-vectorisable boundary check.

    Args:
        segment_ids:            (N,) int32 — segment ID per timestep.
        is_anomaly:             (N,) bool  — per-timestep anomaly flag.
        window_size:            Number of input timesteps (W).
        prediction_horizon:     Steps from the window end to the target (H).
                                Target index = s + W + H - 1.
        skip_anomalous_windows: Skip windows that contain any anomalous step.

    Returns:
        int32 array of valid start indices, length 0 when none qualify.
    """
    n = len(segment_ids)
    span = window_size + prediction_horizon  # total positions consumed per window

    if n < span:
        return np.empty(0, dtype=np.int32)

    starts = np.arange(n - span + 1, dtype=np.int32)

    # Boundary check: first and last positions in span must share a segment.
    no_gap = segment_ids[starts] == segment_ids[starts + span - 1]

    if skip_anomalous_windows:
        # Prefix-sum for O(1) range anomaly queries.
        cumsum = np.empty(n + 1, dtype=np.int64)
        cumsum[0] = 0
        np.cumsum(is_anomaly, out=cumsum[1:])
        window_has_anomaly = (cumsum[starts + span] - cumsum[starts]) > 0
        return starts[no_gap & ~window_has_anomaly].astype(np.int32)  # type: ignore[no-any-return]

    return starts[no_gap].astype(np.int32)  # type: ignore[no-any-return]


class WindowedSequenceDataset(_TorchDataset):  # type: ignore[type-arg]
    """Index-based sliding-window Dataset over a per-timestep series.

    Stores the full values tensor once and slices windows lazily in
    ``__getitem__``, so memory use is O(N) not O(N × W).

    Each item: (x, y) where:
        x: (W, 1) float32 tensor — window values, unsqueezed for LSTM input
        y: ()     float32 tensor — target value at index s + W + H - 1
    """

    def __init__(
        self,
        values: np.ndarray[Any, np.dtype[np.float32]],
        start_indices: np.ndarray[Any, np.dtype[np.int32]],
        window_size: int,
        prediction_horizon: int,
    ) -> None:
        super().__init__()
        # Contiguous tensor for fast slice in __getitem__.
        self._values = torch.from_numpy(np.ascontiguousarray(values))  # (N,) float32
        self._starts = start_indices                                    # (M,) int32
        self._W = window_size
        self._H = prediction_horizon

    def __len__(self) -> int:
        return len(self._starts)

    def __getitem__(self, idx: int) -> "tuple[torch.Tensor, torch.Tensor]":
        s = int(self._starts[idx])
        x = self._values[s : s + self._W].unsqueeze(-1)   # (W, 1)
        y = self._values[s + self._W + self._H - 1]       # scalar
        return x, y


def make_dataloaders(
    settings: Settings,
    mission: str,
    channel: str,
) -> "tuple[DataLoader[tuple[torch.Tensor, torch.Tensor]], DataLoader[tuple[torch.Tensor, torch.Tensor]]]":
    """Build train/val DataLoaders from per-timestep series Parquet.

    Reads the train-split Parquet, builds a window index (skipping windows
    that cross segment boundaries or contain any anomalous timestep), then
    partitions into a temporal train/val split.

    Val split: last ``val_fraction`` of valid windows (contiguous tail).
    Train DataLoader shuffles; val DataLoader does not.
    ``num_workers=0`` on MPS (macOS); cloud.yaml sets ``num_workers=4``.
    ``pin_memory`` is enabled automatically when CUDA is available.

    Args:
        settings: Fully resolved Settings.
        mission:  Mission name, e.g. ``"ESA-Mission1"``.
        channel:  Channel ID, e.g. ``"channel_1"``.
    """
    cfg = settings.model
    values, segment_ids, is_anomaly, _ = _load_series_parquet(
        settings.spark.processed_data_dir, mission, channel, "train"
    )
    all_indices = _build_window_index(
        segment_ids, is_anomaly, cfg.window_size, cfg.prediction_horizon,
        skip_anomalous_windows=True,
    )

    n = len(all_indices)
    if n < 2:
        raise ValueError(
            f"Too few valid windows ({n}) for mission={mission!r} channel={channel!r}. "
            "Need at least 2 (1 train + 1 val). "
            "Increase the time series length or decrease window_size."
        )
    n_val = max(1, int(n * cfg.val_fraction))
    n_train = n - n_val

    train_ds = WindowedSequenceDataset(
        values, all_indices[:n_train], cfg.window_size, cfg.prediction_horizon
    )
    val_ds = WindowedSequenceDataset(
        values, all_indices[n_train:], cfg.window_size, cfg.prediction_horizon
    )

    _pin = torch.cuda.is_available()
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=_pin,
    )
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=_pin,
    )
    return train_loader, val_loader


def make_test_dataloader(
    settings: Settings,
    mission: str,
    channel: str,
) -> "tuple[DataLoader[tuple[torch.Tensor, torch.Tensor]], np.ndarray[Any, Any], np.ndarray[Any, np.dtype[np.bool_]]]":
    """Build a DataLoader for the test split, with aligned per-window metadata.

    Unlike the train DataLoader, anomalous windows are **not** skipped —
    evaluation requires all windows, including those overlapping anomalies.

    Returns:
        loader:            DataLoader over all valid (cross-segment-free) windows.
        target_timestamps: (M,) datetime64[ns] — timestamp at each window's
                           target position (index s + W + H - 1).
        window_is_anomaly: (M,) bool — True iff any step in [s, s+W+H) is
                           anomalous (``any(...)`` semantics; matches Phase 4
                           window-overlap definition for metric continuity).

    Args:
        settings: Fully resolved Settings.
        mission:  Mission name, e.g. ``"ESA-Mission1"``.
        channel:  Channel ID, e.g. ``"channel_1"``.
    """
    cfg = settings.model
    values, segment_ids, is_anomaly, timestamps = _load_series_parquet(
        settings.spark.processed_data_dir, mission, channel, "test"
    )
    indices = _build_window_index(
        segment_ids, is_anomaly, cfg.window_size, cfg.prediction_horizon,
        skip_anomalous_windows=False,
    )

    span = cfg.window_size + cfg.prediction_horizon
    target_timestamps = timestamps[indices + span - 1]

    # Window-level is_anomaly via prefix sum — any() over [s, s+span).
    cumsum = np.empty(len(is_anomaly) + 1, dtype=np.int64)
    cumsum[0] = 0
    np.cumsum(is_anomaly, out=cumsum[1:])
    window_is_anomaly = (cumsum[indices + span] - cumsum[indices]) > 0

    ds = WindowedSequenceDataset(values, indices, cfg.window_size, cfg.prediction_horizon)
    _pin = torch.cuda.is_available()
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        ds,
        batch_size=cfg.inference_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=_pin,
    )
    return loader, target_timestamps, window_is_anomaly
