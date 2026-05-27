"""Pandas preprocessing transforms: null handling, gap detection, normalization,
train/test splitting, and anomaly labeling.

Each function is a pure DataFrame → DataFrame transform (or DataFrame → (DataFrame, dict)).
All transforms receive a single-channel DataFrame — the @ray.remote fan-out in
pipeline.py handles cross-channel parallelism, so Window.partitionBy is unnecessary.

Functional parity with spark/transforms.py is verified by tests/preprocess/test_parity.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from spacecraft_telemetry.core.logging import get_logger

log = get_logger(__name__)


def handle_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill null values, dropping any leading nulls.

    Nulls are filled using the last non-null value in timestamp order.
    Rows with no prior non-null value (leading nulls) are dropped because
    there is no valid value to carry forward.

    Args:
        df: DataFrame with 'value' and 'telemetry_timestamp' columns.

    Returns:
        DataFrame with no null values in 'value'.
    """
    if not df["value"].isna().any():
        log.info("handle_nulls.skipped")
        return df

    df = df.copy()
    df = df.sort_values("telemetry_timestamp")
    df["value"] = df["value"].ffill()
    df = df.dropna(subset=["value"]).reset_index(drop=True)

    if df.empty:
        log.warning("handle_nulls.all_rows_dropped")
    else:
        log.info("handle_nulls")

    return df


def detect_gaps(df: pd.DataFrame, gap_multiplier: float = 3.0) -> pd.DataFrame:
    """Detect time gaps and assign contiguous segment IDs.

    A gap is an interval between consecutive rows exceeding gap_multiplier *
    the median sampling interval for that channel. The first row is never a gap.

    Adds two columns:
    - is_gap (bool): True on the first row after a gap.
    - segment_id (int32): 0-based, increments at each gap boundary.

    Args:
        df:            DataFrame with 'telemetry_timestamp' column, sorted by time.
        gap_multiplier: Factor above median interval that triggers a gap flag.

    Returns:
        DataFrame with 'is_gap' and 'segment_id' columns added.
    """
    df = df.sort_values("telemetry_timestamp").reset_index(drop=True)

    intervals = df["telemetry_timestamp"].diff().dt.total_seconds()
    # Exclude the first row (NaN interval) from median calculation.
    median_interval = float(intervals.iloc[1:].median())

    threshold = gap_multiplier * median_interval
    is_gap = (intervals > threshold).fillna(False)

    df["is_gap"] = is_gap
    df["segment_id"] = is_gap.cumsum().astype("int32")

    log.info("detect_gaps", gap_multiplier=gap_multiplier, median_interval_s=median_interval)
    return df


def normalize(
    df: pd.DataFrame,
    method: str = "z-score",
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Add a value_normalized column using per-channel z-score normalization.

    Computes mean and std from the DataFrame, then adds:
        value_normalized = (value - mean) / std

    Channels with std=0 (constant signal) are normalized to 0.0.

    The returned params dict must be persisted as normalization_params.json —
    these values are required at inference time (FastAPI) to apply the identical
    transform to incoming telemetry.

    Args:
        df:     DataFrame with 'value' and 'channel_id' columns.
        method: Normalization method — currently only "z-score" is supported.

    Returns:
        (normalized_df, params) where params = {channel_id: {"mean": ..., "std": ...}}.
    """
    channel_id = str(df["channel_id"].iloc[0])

    mean = float(df["value"].mean())
    # ddof=1 matches Spark's STDDEV_SAMP (sample standard deviation).
    std = float(df["value"].std(ddof=1))

    if std == 0.0 or np.isnan(std):
        df["value_normalized"] = np.float32(0.0)
        std = 0.0
    else:
        df["value_normalized"] = ((df["value"] - mean) / std).astype("float32")

    params: dict[str, dict[str, float]] = {channel_id: {"mean": mean, "std": std}}

    log.info("normalize", method=method, channel_id=channel_id, mean=mean, std=std)
    return df, params


def temporal_train_test_split(
    df: pd.DataFrame,
    train_fraction: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into train and test sets using a timestamp cutoff.

    The cutoff is:
        min_ts + train_fraction * (max_ts - min_ts)

    Rows at or before the cutoff → train; rows after → test. This is a temporal
    (non-random) split matching the Spark implementation's per-channel semantics.

    Args:
        df:             DataFrame with 'telemetry_timestamp' column.
        train_fraction: Fraction of the time range assigned to training.

    Returns:
        (train_df, test_df)
    """
    ts = df["telemetry_timestamp"]
    min_ts = ts.min()
    max_ts = ts.max()
    cutoff = min_ts + train_fraction * (max_ts - min_ts)

    train = df[ts <= cutoff].reset_index(drop=True)
    test = df[ts > cutoff].reset_index(drop=True)

    log.info(
        "temporal_train_test_split",
        train_fraction=train_fraction,
        train_rows=len(train),
        test_rows=len(test),
    )
    return train, test


def label_timesteps(df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    """Add per-timestep is_anomaly boolean column.

    A timestep is anomalous iff it falls inside any [start_time, end_time)
    half-open label segment for the same channel_id.

    Channels absent from labels_df are treated as fully nominal (all False).

    Args:
        df:         DataFrame with 'telemetry_timestamp' and 'channel_id' columns.
        labels_df:  Labels DataFrame with 'channel_id', 'start_time', 'end_time'.

    Returns:
        DataFrame with 'is_anomaly' (bool) column added.
    """
    channel_id = str(df["channel_id"].iloc[0])

    channel_labels = labels_df[labels_df["channel_id"] == channel_id]

    if channel_labels.empty:
        df["is_anomaly"] = False
        log.info("label_timesteps", channel_id=channel_id, n_labels=0)
        return df

    ts = df["telemetry_timestamp"]
    is_anomaly = pd.Series(False, index=df.index)

    for _, row in channel_labels.iterrows():
        # Half-open interval: start inclusive, end exclusive — matches Spark semantics.
        in_interval = (ts >= row["start_time"]) & (ts < row["end_time"])
        is_anomaly = is_anomaly | in_interval

    df["is_anomaly"] = is_anomaly

    log.info(
        "label_timesteps",
        channel_id=channel_id,
        n_labels=len(channel_labels),
        n_anomalous=int(is_anomaly.sum()),
    )
    return df
