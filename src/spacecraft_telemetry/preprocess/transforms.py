"""Pandas preprocessing transforms: null handling, gap detection, normalization,
train/test splitting, and anomaly labeling.

Each function is a pure DataFrame → DataFrame transform (or DataFrame → (DataFrame, dict)).
All transforms receive a single-channel DataFrame — the @ray.remote fan-out in
pipeline.py handles cross-channel parallelism, so no per-channel grouping is needed.

Parity between the parallel (Ray) and sequential (pandas) code paths is verified
by tests/preprocess/test_parity.py.

ISS-specific transforms
-----------------------
resample_to_grid    — bin raw irregular ticks onto a regular time grid
compute_los_mask    — cross-channel Loss-of-Signal detection
augment_with_los    — merge is_los flag into a resampled channel DataFrame

All three are pure pandas with no ISS-specific imports so the Phase 17 live pump
can import them from this module without dragging in ingest or collector code.
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
    # ddof=1 → sample standard deviation (matches normalize_value reference impl).
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
    train_lookback: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into train and test sets using a timestamp cutoff.

    The cutoff is:
        min_ts + train_fraction * (max_ts - min_ts)

    Rows at or before the cutoff → train; rows after → test. The test window
    is always the full 20% tail — it is never affected by train_lookback.

    When train_lookback is set (a pandas offset alias, e.g. "730D"), only the
    most recent `train_lookback` of data before the cutoff is kept for training.
    This caps stale history without touching the evaluation window.

    Args:
        df:             DataFrame with 'telemetry_timestamp' column.
        train_fraction: Fraction of the time range assigned to training.
        train_lookback: Optional pandas offset alias capping how far back
                        training data reaches (e.g. "730D" for 2 years).

    Returns:
        (train_df, test_df)
    """
    ts = df["telemetry_timestamp"]
    min_ts = ts.min()
    max_ts = ts.max()
    cutoff = min_ts + train_fraction * (max_ts - min_ts)

    train = df[ts <= cutoff]
    if train_lookback is not None:
        lookback_start = cutoff - pd.Timedelta(train_lookback)
        train = train[train["telemetry_timestamp"] >= lookback_start]

    train = train.reset_index(drop=True)
    test = df[ts > cutoff].reset_index(drop=True)

    log.info(
        "temporal_train_test_split",
        train_fraction=train_fraction,
        train_lookback=train_lookback,
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
        # Half-open interval: start inclusive, end exclusive.
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


# ---------------------------------------------------------------------------
# ISS-specific transforms
# ---------------------------------------------------------------------------


def resample_to_grid(
    ticks_df: pd.DataFrame,
    channel_id: str,
    mission_id: str,
    grid_interval_seconds: int = 30,
) -> pd.DataFrame:
    """Resample raw irregular ISS ticks to a regular time grid.

    Takes the raw tick DataFrame produced by ``read_iss_ticks`` (which has
    event-driven, variable-cadence rows) and bins it onto a uniform grid by:
      1. Setting ``telemetry_timestamp`` as a DatetimeIndex.
      2. Resampling with mean aggregation per bucket.
      3. Forward-filling sparse buckets (P4000001 power-voltage has p90=40s
         between ticks, so occasional 30s buckets receive no ticks and are
         filled from the previous value).

    The output column contract is identical to ``read_channel()`` for ESA:
        telemetry_timestamp  datetime64[us, UTC]
        value                float32
        channel_id           str
        mission_id           str

    This means all downstream transforms (handle_nulls, detect_gaps, normalize,
    label_timesteps, temporal_train_test_split) accept the output unchanged.

    Args:
        ticks_df:              Raw tick DataFrame with columns
                               [telemetry_timestamp, value, aos_timestamp].
        channel_id:            ISS PUI string (e.g. "S1000003").
        mission_id:            Mission name (always "ISS" in practice).
        grid_interval_seconds: Grid step in seconds (default 30).

    Returns:
        DataFrame with columns [telemetry_timestamp, value, channel_id, mission_id].
    """
    df = ticks_df[["telemetry_timestamp", "value"]].copy()
    df = df.set_index("telemetry_timestamp").sort_index()
    rule = f"{grid_interval_seconds}s"
    resampled = df["value"].resample(rule).mean().ffill()
    result = resampled.reset_index()
    result.columns = pd.Index(["telemetry_timestamp", "value"])
    result["value"] = result["value"].astype("float32")
    result["channel_id"] = channel_id
    result["mission_id"] = mission_id

    log.info(
        "resample_to_grid",
        channel_id=channel_id,
        grid_interval_s=grid_interval_seconds,
        raw_rows=len(ticks_df),
        grid_rows=len(result),
    )
    return result[["telemetry_timestamp", "value", "channel_id", "mission_id"]]


def compute_los_mask(
    all_ticks_df: pd.DataFrame,
    grid_interval_seconds: int = 30,
) -> pd.Series:
    """Derive a boolean Loss-of-Signal mask from a cross-channel tick archive.

    A 30-second grid bucket is marked as LOS when NO telemetry channel has any
    tick in that bucket.  The mask is then expanded by one bucket on each side
    to account for TDRS handover smear (a real LOS onset/recovery may straddle
    a bucket boundary).

    The caller is responsible for excluding context items (TIME_000001,
    USLAB000086) from ``all_ticks_df`` — they are not telemetry and their
    absence would incorrectly suppress LOS detection.

    TIME_000001 is intentionally excluded as a sole LOS criterion because a
    1-hour dry-run showed it can stall ~5 min while all other channels keep
    updating (feed-side clock pause, not signal loss).

    Args:
        all_ticks_df:          DataFrame with columns
                               [telemetry_timestamp, channel_id] covering
                               ALL subscribed telemetry channels.
        grid_interval_seconds: Grid step in seconds (must match resample_to_grid).

    Returns:
        pd.Series of dtype bool, indexed by a DatetimeIndex of all 30-second
        grid buckets spanning the archive's time range.  True = LOS.
    """
    if all_ticks_df.empty:
        return pd.Series(dtype=bool, name="is_los")

    ts = all_ticks_df["telemetry_timestamp"]
    freq = f"{grid_interval_seconds}s"

    grid_start = ts.min().floor(freq)
    grid_end = ts.max().ceil(freq)
    grid_index = pd.date_range(start=grid_start, end=grid_end, freq=freq, tz="UTC")

    # Floor each tick timestamp to its bucket.
    bucket_ts = ts.dt.floor(freq)
    occupied = set(bucket_ts)

    los_raw = pd.Series(
        [t not in occupied for t in grid_index],
        index=grid_index,
        dtype=bool,
        name="is_los",
    )

    # Expand by one bucket on each side (TDRS handover smear).
    los_expanded = (
        los_raw
        | los_raw.shift(1, fill_value=False)
        | los_raw.shift(-1, fill_value=False)
    )
    los_expanded.name = "is_los"

    n_los = int(los_expanded.sum())
    log.info(
        "compute_los_mask",
        grid_interval_s=grid_interval_seconds,
        total_buckets=len(grid_index),
        los_buckets=n_los,
    )
    return los_expanded


def augment_with_los(
    resampled_df: pd.DataFrame,
    los_mask: pd.Series,
) -> pd.DataFrame:
    """Add an ``is_los`` boolean column to a resampled channel DataFrame.

    Joins the pre-computed LOS mask onto the resampled data by timestamp.
    Buckets that appear in the resampled DataFrame but not in the mask
    (e.g. because the mask was computed from a different time range) default
    to False — they are treated as nominal.

    Args:
        resampled_df: Output of ``resample_to_grid`` with column
                      ``telemetry_timestamp``.
        los_mask:     Boolean Series indexed by grid timestamps (output of
                      ``compute_los_mask``).

    Returns:
        resampled_df with an additional ``is_los`` (bool) column.
    """
    df = resampled_df.copy()
    los_df = los_mask.rename("is_los").reset_index()
    los_df.columns = pd.Index(["telemetry_timestamp", "is_los"])
    df = df.merge(los_df, on="telemetry_timestamp", how="left")
    df["is_los"] = df["is_los"].fillna(False).astype(bool)

    log.info(
        "augment_with_los",
        rows=len(df),
        los_rows=int(df["is_los"].sum()),
    )
    return df
