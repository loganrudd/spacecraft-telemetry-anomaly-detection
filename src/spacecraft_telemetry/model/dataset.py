"""Dataset utilities for Telemanom LSTM training.

Reads windowed Parquet (written by the Spark preprocessing pipeline) via PyArrow —
no Spark dependency at training time.

Wraps arrays in a torch Dataset and produces train/val DataLoaders.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from spacecraft_telemetry.core.config import ModelConfig

if TYPE_CHECKING:
    import torch
    from torch.utils.data import DataLoader, Dataset


def load_windowed_parquet(
    processed_dir: Path,
    mission: str,
    channel: str,
    split: Literal["train", "test"],
) -> tuple[np.ndarray[tuple[int, int], np.dtype[np.float32]], np.ndarray[tuple[int], np.dtype[np.float32]], np.ndarray[tuple[int], np.dtype[np.bool_]]]:
    """Load windowed Parquet written by the Spark preprocessing pipeline.

    Reads from the Hive-partitioned layout:
        {processed_dir}/{mission}/{split}/mission_id={mission}/channel_id={channel}/*.parquet

    Partition columns (mission_id, channel_id) are encoded in directory names,
    not in the Parquet files themselves — this is standard Spark behaviour.

    Returns:
        values:     (N, W) float32 array sorted by window_id (chronological order)
        targets:    (N,)   float32 array of one-step-ahead targets
        is_anomaly: (N,)   bool array of anomaly flags

    Raises:
        FileNotFoundError: If the partition directory doesn't exist or has no Parquet files.
    """
    partition_dir = (
        processed_dir / mission / split
        / f"mission_id={mission}" / f"channel_id={channel}"
    )

    if not partition_dir.exists():
        raise FileNotFoundError(
            f"No windowed Parquet found for mission={mission!r} channel={channel!r} "
            f"split={split!r}. Expected directory: {partition_dir}"
        )

    parquet_files = sorted(partition_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"Directory exists but contains no .parquet files: {partition_dir}"
        )

    tables = [
        pq.read_table(f, columns=["window_id", "values", "target", "is_anomaly"])
        for f in parquet_files
    ]
    table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]

    df = table.to_pandas().sort_values("window_id").reset_index(drop=True)

    values = np.array(df["values"].tolist(), dtype=np.float32)       # (N, W)
    targets = df["target"].to_numpy(dtype=np.float32)                 # (N,)
    is_anomaly = df["is_anomaly"].to_numpy(dtype=bool)                # (N,)

    return values, targets, is_anomaly


class WindowedSequenceDataset:
    """Wraps windowed numpy arrays for use with a PyTorch DataLoader.

    Implements __len__ and __getitem__ so torch DataLoader accepts it without
    requiring Dataset as a base class (duck typing). This keeps the module
    importable without a PyTorch install.

    Each item: (x, y) where:
        x: (W, 1) float32 tensor  — window values, unsqueezed for LSTM input
        y: ()     float32 tensor  — one-step-ahead target (scalar)
    """

    def __init__(self, values: np.ndarray[Any, np.dtype[np.float32]], targets: np.ndarray[Any, np.dtype[np.float32]]) -> None:
        import torch
        self._values = torch.from_numpy(values)    # (N, W)
        self._targets = torch.from_numpy(targets)  # (N,)

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, idx: int) -> "tuple[torch.Tensor, torch.Tensor]":
        x = self._values[idx].unsqueeze(-1)  # (W,) → (W, 1)
        y = self._targets[idx]               # scalar
        return x, y


def make_dataloaders(
    values: np.ndarray[Any, np.dtype[np.float32]],
    targets: np.ndarray[Any, np.dtype[np.float32]],
    model_config: ModelConfig,
) -> "tuple[DataLoader[tuple[torch.Tensor, torch.Tensor]], DataLoader[tuple[torch.Tensor, torch.Tensor]]]":
    """Split values/targets into train/val DataLoaders.

    Val split: the last val_fraction of windows (time-ordered, contiguous tail).
    Train DataLoader shuffles; val DataLoader does not.
    num_workers=0 on MPS (macOS); cloud.yaml sets num_workers=4 for CUDA nodes.
    pin_memory is enabled automatically when CUDA is available.
    """
    import torch
    from torch.utils.data import DataLoader

    n = len(values)
    if n < 2:
        raise ValueError(
            f"Too few windows ({n}) to create a train/val split. "
            "Need at least 2 windows (1 train + 1 val). "
            "Increase the time series length or decrease --window-size."
        )
    n_val = max(1, int(n * model_config.val_fraction))
    n_train = n - n_val

    train_ds = WindowedSequenceDataset(values[:n_train], targets[:n_train])
    val_ds = WindowedSequenceDataset(values[n_train:], targets[n_train:])

    # pin_memory speeds up host→GPU transfers; only effective with CUDA.
    # num_workers=0 is required on MPS (macOS); cloud.yaml sets num_workers=4.
    _pin = torch.cuda.is_available()
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        train_ds,
        batch_size=model_config.batch_size,
        shuffle=True,
        num_workers=model_config.num_workers,
        pin_memory=_pin,
    )
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        val_ds,
        batch_size=model_config.batch_size,
        shuffle=False,
        num_workers=model_config.num_workers,
        pin_memory=_pin,
    )
    return train_loader, val_loader
