"""Reference profile building for Evidently drift monitoring (Phase 7).

A reference profile is the train-split feature DataFrame for one channel.
It is built once (after Spark preprocessing) and persisted to Parquet so that
each ``drift batch`` invocation can load it without re-reading all train data.

Public API
----------
MONITORING_FEATURE_COLS : list[str]
    Canonical 14-column list consumed by Evidently's ColumnMapping.
compute_feature_dataframe(series_df, settings) -> pd.DataFrame
    Adds rolling feature columns to a raw-series DataFrame in Pandas.
build_reference_profile(settings, mission, channel) -> pd.DataFrame
    Reads train Parquet → compute_feature_dataframe → sample cap.
reference_profile_path(settings, mission, channel) -> Path
    Canonical on-disk path for a channel's reference.parquet.
save_reference_profile(df, path) -> None
load_reference_profile(path) -> pd.DataFrame
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from spacecraft_telemetry.core.config import Settings
from spacecraft_telemetry.features.definitions import _DEFAULT_WINDOWS, FEATURE_DEFINITIONS

# ---------------------------------------------------------------------------
# Monitoring column set
# ---------------------------------------------------------------------------
# value_normalized + all engineered feature columns, in the order that Evidently
# receives them.  Column names and window sizes are derived from FEATURE_DEFINITIONS
# (defaults: windows [10, 50, 100]) and MUST agree with settings.spark.feature_windows
# at monitoring time.  Changing feature_windows without updating FEATURE_DEFINITIONS
# will cause a KeyError in compute_feature_dataframe.

MONITORING_FEATURE_COLS: list[str] = ["value_normalized"] + [
    fd.name for fd in FEATURE_DEFINITIONS
]

# Rolling windows that MONITORING_FEATURE_COLS was built from.  Sourced directly
# from _DEFAULT_WINDOWS in features/definitions.py — the two share one definition.
# compute_feature_dataframe validates settings.spark.feature_windows against this.
_MONITORING_WINDOWS: list[int] = sorted(_DEFAULT_WINDOWS)


def compute_feature_dataframe(
    series_df: pd.DataFrame,
    settings: Settings,
) -> pd.DataFrame:
    """Add rolling feature columns via pd.DataFrame.rolling().  Drops NaN warmup rows.

    Window sizes come from ``settings.spark.feature_windows``.  For the output
    columns to match ``MONITORING_FEATURE_COLS`` exactly, the window list must
    equal ``FEATURE_DEFINITIONS``'s default windows ``[10, 50, 100]``.

    Args:
        series_df: DataFrame with at least a ``value_normalized`` column.
                   ``telemetry_timestamp`` (UTC-aware) is used for rate_of_change
                   when present; falls back to a simple value diff otherwise.
        settings: Runtime settings; ``settings.spark.feature_windows`` drives
                  which window sizes are computed.

    Returns:
        DataFrame with columns ``MONITORING_FEATURE_COLS``, NaN warmup rows
        dropped, index reset to 0-based sequential integers.
    """
    windows = list(settings.spark.feature_windows)
    if sorted(windows) != _MONITORING_WINDOWS:
        raise ValueError(
            f"settings.spark.feature_windows {sorted(windows)} does not match the "
            f"windows used to build MONITORING_FEATURE_COLS {_MONITORING_WINDOWS}. "
            "Update FEATURE_DEFINITIONS or set feature_windows to "
            f"{_MONITORING_WINDOWS} in your config."
        )
    df = series_df.copy()

    for w in windows:
        rolled = df["value_normalized"].rolling(window=w)
        df[f"rolling_mean_{w}"] = rolled.mean()
        df[f"rolling_std_{w}"] = rolled.std(ddof=1)
        df[f"rolling_min_{w}"] = rolled.min()
        df[f"rolling_max_{w}"] = rolled.max()

    # rate_of_change: Δvalue / Δtime_seconds
    if "telemetry_timestamp" in df.columns:
        ts = pd.to_datetime(df["telemetry_timestamp"], utc=True)
        dt_s = ts.diff().dt.total_seconds().replace(0.0, float("nan"))
        dv = df["value_normalized"].diff()
        df["rate_of_change"] = dv / dt_s
    else:
        # Fallback: assume unit time steps (useful for testing without timestamps)
        df["rate_of_change"] = df["value_normalized"].diff()

    # Drop NaN warmup rows (first max(window)-1 from rolling, first row from diff)
    # and return only the monitoring columns.
    return df[MONITORING_FEATURE_COLS].dropna().reset_index(drop=True)


def _load_channel_series(
    settings: Settings,
    mission: str,
    channel: str,
    split: str,
) -> pd.DataFrame:
    """Read the tail of a channel's raw series Parquet for monitoring.

    Path: ``{processed_data_dir}/{mission}/{split}/mission_id={M}/channel_id={C}/``

    Only ``telemetry_timestamp`` and ``value_normalized`` are loaded (half the
    I/O vs the full SERIES_SCHEMA).  After sorting, only the last
    ``reference_sample_rows + max(feature_windows)`` rows are retained before
    returning — this is the minimum contiguous prefix needed to produce
    ``reference_sample_rows`` valid rolling-feature rows after warmup.  For a
    500K-row channel this reduces the downstream rolling-feature compute from
    O(500K x W) to O(5100 x W).

    The "recent tail" interpretation is intentional: the most recent training
    rows are the relevant reference distribution when current data arrives today.

    Args:
        settings: Runtime settings.
        mission:  Mission identifier, e.g. ``"ESA-Mission1"``.
        channel:  Channel identifier, e.g. ``"channel_1"``.
        split:    ``"train"`` or ``"test"``.

    Returns:
        DataFrame with ``telemetry_timestamp`` and ``value_normalized``, sorted
        by timestamp, index reset to 0-based. At most
        ``reference_sample_rows + max(feature_windows)`` rows.

    Raises:
        FileNotFoundError: If the partition directory does not exist.
    """
    partition_dir = (
        Path(settings.spark.processed_data_dir)
        / mission
        / split
        / f"mission_id={mission}"
        / f"channel_id={channel}"
    )
    if not partition_dir.exists():
        raise FileNotFoundError(
            f"No {split} series Parquet for mission={mission!r} channel={channel!r}. "
            f"Expected: {partition_dir}"
        )
    table = pq.read_table(
        str(partition_dir),
        columns=["telemetry_timestamp", "value_normalized"],
    )
    df = table.to_pandas().sort_values("telemetry_timestamp")

    # Tail-slice before rolling: only contiguous rows needed for output.
    max_window = max(settings.spark.feature_windows)
    rows_needed = settings.monitoring.reference_sample_rows + max_window
    if len(df) > rows_needed:
        df = df.tail(rows_needed)

    return df.reset_index(drop=True)


def build_reference_profile(
    settings: Settings,
    mission: str,
    channel: str,
) -> pd.DataFrame:
    """Build a reference feature DataFrame for the given channel.

    Reads the train-split Parquet for ``(mission, channel)`` from the
    Hive-partitioned directory layout written by the Spark pipeline::

        {processed_data_dir}/{mission}/train/mission_id={M}/channel_id={C}/

    Applies ``compute_feature_dataframe`` and samples down to at most
    ``settings.monitoring.reference_sample_rows`` rows (reproducible via
    ``random_state=42``).

    Args:
        settings: Runtime settings (paths, window sizes, sample cap).
        mission:  Mission identifier, e.g. ``"ESA-Mission1"``.
        channel:  Channel identifier, e.g. ``"channel_1"``.

    Returns:
        DataFrame with columns ``MONITORING_FEATURE_COLS``, at most
        ``settings.monitoring.reference_sample_rows`` rows.

    Raises:
        FileNotFoundError: If the channel partition directory does not exist.
    """
    df = _load_channel_series(settings, mission, channel, "train")
    df = compute_feature_dataframe(df, settings)
    n = settings.monitoring.reference_sample_rows
    if len(df) > n:
        df = df.sample(n=n, random_state=42).reset_index(drop=True)
    return df


def build_current_profile(
    settings: Settings,
    mission: str,
    channel: str,
) -> pd.DataFrame:
    """Build a current feature DataFrame from the test split for drift comparison.

    Mirrors ``build_reference_profile`` but reads the test split.  Called by
    ``drift batch`` to produce the "current" side of the Evidently report.

    Args:
        settings: Runtime settings (paths, window sizes, sample cap).
        mission:  Mission identifier, e.g. ``"ESA-Mission1"``.
        channel:  Channel identifier, e.g. ``"channel_1"``.

    Returns:
        DataFrame with columns ``MONITORING_FEATURE_COLS``, at most
        ``settings.monitoring.reference_sample_rows`` rows.

    Raises:
        FileNotFoundError: If the test partition directory does not exist.
    """
    df = _load_channel_series(settings, mission, channel, "test")
    df = compute_feature_dataframe(df, settings)
    n = settings.monitoring.reference_sample_rows
    if len(df) > n:
        df = df.sample(n=n, random_state=42).reset_index(drop=True)
    return df


def reference_profile_path(
    settings: Settings,
    mission: str,
    channel: str,
) -> Path:
    """Return the on-disk path for a channel's reference profile Parquet.

    Path: ``{reference_profiles_dir}/{mission}/{channel}/reference.parquet``
    """
    return (
        Path(settings.monitoring.reference_profiles_dir)
        / mission
        / channel
        / "reference.parquet"
    )


def save_reference_profile(df: pd.DataFrame, path: Path) -> None:
    """Persist a reference profile DataFrame to Parquet.

    Creates parent directories as needed.

    Args:
        df:   DataFrame with columns ``MONITORING_FEATURE_COLS``.
        path: Destination file path (use :func:`reference_profile_path` for the
              canonical location).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def load_reference_profile(path: Path) -> pd.DataFrame:
    """Load a previously saved reference profile from Parquet.

    Args:
        path: Path returned by :func:`reference_profile_path`.

    Returns:
        DataFrame with columns ``MONITORING_FEATURE_COLS``.

    Raises:
        FileNotFoundError: If the profile has not been built yet.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Reference profile not found at {path}. "
            "Run `spacecraft-telemetry drift batch` first to build it."
        )
    return pd.read_parquet(path)
