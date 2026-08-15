"""Dataset utilities for Telemanom LSTM training.

Reads per-timestep series Parquet (written by the preprocessing pipeline)
via PyArrow.  Constructs LSTM windows on-the-fly in the DataLoader, avoiding
the 250x disk inflation of pre-materialized window arrays.

Public API:
    make_dataloaders(settings, mission, channel)     -> (train_loader, val_loader)
    make_test_dataloader(settings, mission, channel) -> (loader, target_timestamps,
                                                         window_is_anomaly)
    load_window_labels(settings, mission, channel)   -> window_is_anomaly (no torch)
    window_target_timestamps(settings, mission, channel) -> target_timestamps (no torch)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as _TorchDataset
from upath import UPath

from spacecraft_telemetry.core.config import Settings
from spacecraft_telemetry.core.paths import to_upath


def _read_partition_table(
    processed_dir: Path | UPath | str,
    mission: str,
    channel: str,
    split: Literal["train", "test"],
    columns: list[str],
) -> pa.Table:
    """Read + concat + timestamp-sort one channel partition's Parquet files.

    Shared by load_series_parquet and load_series_metadata so both read the
    identical partition-discovery/sort logic and differ only in which
    columns PyArrow actually parses.

    Raises:
        FileNotFoundError: If the partition directory doesn't exist or has no
            Parquet files.
    """
    partition_dir = (
        to_upath(processed_dir) / mission / split
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

    tables = [pq.read_table(str(f), columns=columns) for f in parquet_files]
    table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
    return table.sort_by("telemetry_timestamp")


def load_series_parquet(
    processed_dir: Path | UPath | str,
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
    table = _read_partition_table(
        processed_dir, mission, channel, split,
        columns=["telemetry_timestamp", "value_normalized", "segment_id", "is_anomaly"],
    )

    values = table.column("value_normalized").to_numpy(zero_copy_only=False).astype(np.float32)
    segment_ids = table.column("segment_id").to_numpy(zero_copy_only=False).astype(np.int32)
    is_anomaly = table.column("is_anomaly").to_numpy(zero_copy_only=False).astype(bool)
    timestamps = table.column("telemetry_timestamp").to_numpy(zero_copy_only=False)

    return values, segment_ids, is_anomaly, timestamps


def load_series_metadata(
    processed_dir: Path | UPath | str,
    mission: str,
    channel: str,
    split: Literal["train", "test"],
) -> tuple[
    np.ndarray[Any, np.dtype[np.int32]],
    np.ndarray[Any, np.dtype[np.bool_]],
    np.ndarray[Any, Any],
]:
    """Read only the small per-timestep columns needed for windowing/timeline math.

    Skips ``value_normalized`` — the largest column on disk and the one
    column none of esa_adb.timeline / esa_adb.report / esa_adb.detections
    ever reads. Same partition as load_series_parquet, just fewer columns
    parsed — use this wherever only segment_ids/is_anomaly/timestamps are
    needed (window-index building, target-timestamp derivation, observed
    timeline construction).

    Returns:
        segment_ids: (N,) int32    — segment ID per timestep
        is_anomaly:  (N,) bool     — per-timestep anomaly flag
        timestamps:  (N,) datetime64[ns] — telemetry timestamp per timestep

    Raises:
        FileNotFoundError: Same as load_series_parquet.
    """
    table = _read_partition_table(
        processed_dir, mission, channel, split,
        columns=["telemetry_timestamp", "segment_id", "is_anomaly"],
    )

    segment_ids = table.column("segment_id").to_numpy(zero_copy_only=False).astype(np.int32)
    is_anomaly = table.column("is_anomaly").to_numpy(zero_copy_only=False).astype(bool)
    timestamps = table.column("telemetry_timestamp").to_numpy(zero_copy_only=False)

    return segment_ids, is_anomaly, timestamps


def _build_window_index(
    segment_ids: np.ndarray[Any, np.dtype[np.int32]],
    is_anomaly: np.ndarray[Any, np.dtype[np.bool_]],
    window_size: int,
    prediction_horizon: int,
    skip_anomalous_windows: bool,
) -> np.ndarray[Any, Any]:
    """Return int32 array of valid window start indices.

    A start index ``s`` is valid iff:

    1. ``segment_ids[s : s + window_size + prediction_horizon]`` is all one
       value — the window plus target step don't span a segment gap.
    2. If ``skip_anomalous_windows``: no timestep in that span is anomalous.

    Segment IDs are assigned in ascending temporal order by the preprocessing
    pipeline, so ``segment_ids[s] == segment_ids[s + span - 1]``
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
    ``__getitem__``, so memory use is O(N) not O(N x W).

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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        s = int(self._starts[idx])
        x = self._values[s : s + self._W].unsqueeze(-1)   # (W, 1)
        y = self._values[s + self._W + self._H - 1]       # scalar
        return x, y


def make_dataloaders(
    settings: Settings,
    mission: str,
    channel: str,
) -> tuple[
    DataLoader[tuple[torch.Tensor, torch.Tensor]],
    DataLoader[tuple[torch.Tensor, torch.Tensor]],
]:
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
    values, segment_ids, is_anomaly, _ = load_series_parquet(
        settings.preprocess.processed_data_dir, mission, channel, "train"
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
) -> tuple[
    DataLoader[tuple[torch.Tensor, torch.Tensor]],
    np.ndarray[Any, Any],
    np.ndarray[Any, np.dtype[np.bool_]],
]:
    """Build a DataLoader for the test split, with aligned per-window metadata.

    Unlike the train DataLoader, anomalous windows are **not** skipped —
    evaluation requires all windows, including those overlapping anomalies.

    Returns:
        loader:            DataLoader over all valid (cross-segment-free) windows.
        target_timestamps: (M,) datetime64[ns] — timestamp at each window's
                           target position (index s + W + H - 1).
        window_is_anomaly: (M,) bool — True iff any step in [s, s+W+H) is
                           anomalous (``any(...)`` semantics; matches Phase 3
                           window-overlap definition for metric continuity).

    Args:
        settings: Fully resolved Settings.
        mission:  Mission name, e.g. ``"ESA-Mission1"``.
        channel:  Channel ID, e.g. ``"channel_1"``.
    """
    cfg = settings.model
    values, segment_ids, is_anomaly, timestamps = load_series_parquet(
        settings.preprocess.processed_data_dir, mission, channel, "test"
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


def load_window_labels(
    settings: Settings,
    mission: str,
    channel: str,
) -> np.ndarray[Any, np.dtype[np.bool_]]:
    """Return per-window anomaly labels for the test split without building a DataLoader.

    Uses the same windowing logic as make_test_dataloader() but has no torch
    dependency — suitable for import in numpy-only Ray Tune trial functions
    (Phase 5).

    Returns:
        (M,) bool array — True iff any timestep in the window+horizon span
        is anomalous (any() semantics; matches make_test_dataloader).

    Args:
        settings: Fully resolved Settings.
        mission:  Mission name, e.g. ``"ESA-Mission1"``.
        channel:  Channel ID, e.g. ``"channel_1"``.
    """
    cfg = settings.model
    _, segment_ids, is_anomaly, _ = load_series_parquet(
        settings.preprocess.processed_data_dir, mission, channel, "test"
    )
    indices = _build_window_index(
        segment_ids, is_anomaly, cfg.window_size, cfg.prediction_horizon,
        skip_anomalous_windows=False,
    )
    span = cfg.window_size + cfg.prediction_horizon
    cumsum = np.empty(len(is_anomaly) + 1, dtype=np.int64)
    cumsum[0] = 0
    np.cumsum(is_anomaly, out=cumsum[1:])
    result: np.ndarray[Any, np.dtype[np.bool_]] = (
        (cumsum[indices + span] - cumsum[indices]) > 0
    )
    return result


def window_target_timestamps_from_metadata(
    settings: Settings,
    segment_ids: np.ndarray[Any, np.dtype[np.int32]],
    is_anomaly: np.ndarray[Any, np.dtype[np.bool_]],
    timestamps: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    """Pure computation of window_target_timestamps given preloaded metadata arrays.

    Split out of window_target_timestamps so a caller that already holds a
    channel's (segment_ids, is_anomaly, timestamps) — loaded once via
    load_series_metadata() — can reuse it across multiple calls (timeline,
    HPO cutoff, untuned + tuned detection reconstruction) instead of
    re-reading the parquet partition every time (docs/plans/019 P2/P3).
    """
    cfg = settings.model
    indices = _build_window_index(
        segment_ids, is_anomaly, cfg.window_size, cfg.prediction_horizon,
        skip_anomalous_windows=False,
    )
    span = cfg.window_size + cfg.prediction_horizon
    result: np.ndarray[Any, Any] = timestamps[indices + span - 1]
    return result


def window_target_timestamps(
    settings: Settings,
    mission: str,
    channel: str,
) -> np.ndarray[Any, Any]:
    """Return per-window target timestamps for the test split without a DataLoader.

    Uses the same windowing logic as make_test_dataloader() but has no torch
    dependency — lets esa_adb.detections re-derive detection intervals from a
    scoring run's logged errors.npy/threshold.npy without re-running inference
    (window start indices are deterministic given window_size and
    prediction_horizon, so this reproduces make_test_dataloader's
    target_timestamps exactly).

    Reads only segment_ids/is_anomaly/timestamps (load_series_metadata, not
    load_series_parquet) — this function never uses the values column. A
    caller making several calls for the same channel (e.g. build_report)
    should instead preload once via load_series_metadata() and call
    window_target_timestamps_from_metadata() directly.

    Returns:
        (M,) datetime64[ns] — timestamp at each window's target position
        (index s + W + H - 1), aligned 1:1 with load_window_labels() and with
        the errors.npy/threshold.npy arrays score_channel() logs.

    Args:
        settings: Fully resolved Settings.
        mission:  Mission name, e.g. ``"ESA-Mission1"``.
        channel:  Channel ID, e.g. ``"channel_1"``.
    """
    segment_ids, is_anomaly, timestamps = load_series_metadata(
        settings.preprocess.processed_data_dir, mission, channel, "test"
    )
    return window_target_timestamps_from_metadata(settings, segment_ids, is_anomaly, timestamps)
