"""Pandas/PyArrow I/O: read raw channel Parquet + labels CSV, write processed output.

All reads produce DataFrames with a fixed column contract; all writes produce the
Hive-partitioned layout:
    {output_path}/mission_id={M}/channel_id={C}/part.parquet

This layout is consumed by model/dataset.py, ray_fanout/runner.py, and
evidently_monitoring/reference.py — none of those files change.

ISS functions
-------------
read_iss_ticks      — concatenate hourly tick shards for one channel
discover_iss_channels — list telemetry PUIs available in the raw-tick archive
read_all_iss_ticks_for_los — load timestamps from all channels for LOS detection
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from upath import UPath

from spacecraft_telemetry.core.logging import get_logger
from spacecraft_telemetry.core.paths import to_upath
from spacecraft_telemetry.preprocess.schemas import SERIES_FILE_SCHEMA

log = get_logger(__name__)


def read_channel(path: UPath, channel_id: str, mission_id: str) -> pd.DataFrame:
    """Read a raw channel Parquet file and return a standardised DataFrame.

    Raw files written by ingest/sample.py have:
      - DatetimeIndex named "datetime" (UTC microseconds)
      - One float32 column named after the channel (e.g. "channel_1")

    Returns a DataFrame with columns:
        telemetry_timestamp  datetime64[us, UTC]
        value                float32  (nullable — cleaned in transforms step)
        channel_id           str
        mission_id           str

    Raises:
        ValueError: If either expected column is absent from the file.
    """
    df = pd.read_parquet(str(path))

    # Promote DatetimeIndex to a regular column named "datetime".
    if isinstance(df.index, pd.DatetimeIndex):
        index_name = df.index.name or "datetime"
        df = df.reset_index()
        if index_name != "datetime":
            df = df.rename(columns={index_name: "datetime"})

    if "datetime" not in df.columns:
        raise ValueError(
            f"Expected timestamp column 'datetime' not found in {path}. "
            f"Available columns: {df.columns.tolist()}"
        )
    if channel_id not in df.columns:
        raise ValueError(
            f"Expected value column {channel_id!r} not found in {path}. "
            f"Available columns: {df.columns.tolist()}"
        )

    df = df.rename(columns={"datetime": "telemetry_timestamp", channel_id: "value"})
    if not pd.api.types.is_numeric_dtype(df["value"]):
        raise TypeError(
            f"channel {channel_id!r} has non-numeric dtype {df['value'].dtype} "
            f"(categorical) — skipping"
        )
    df["value"] = df["value"].astype("float32")
    # from_codes avoids building a 32M-element Python list (~1.9GB transient) before
    # converting to Categorical. int8 codes array is 32MB; the list approach was ~1.9GB.
    _codes = np.zeros(len(df), dtype=np.int8)
    df["channel_id"] = pd.Categorical.from_codes(_codes, categories=pd.Index([channel_id]))
    df["mission_id"] = pd.Categorical.from_codes(_codes, categories=pd.Index([mission_id]))

    # Ensure UTC-aware timestamps (ingest writes UTC; guard against tz-naive files).
    ts = df["telemetry_timestamp"]
    if ts.dt.tz is None:
        df["telemetry_timestamp"] = ts.dt.tz_localize("UTC")
    elif str(ts.dt.tz) != "UTC":
        df["telemetry_timestamp"] = ts.dt.tz_convert("UTC")

    log.info("read_channel", path=str(path), channel_id=channel_id, mission_id=mission_id)
    return df[["telemetry_timestamp", "value", "channel_id", "mission_id"]]


def read_labels(path: UPath) -> pd.DataFrame:
    """Read a labels CSV and return a standardised DataFrame.

    Raw CSV has columns: ID, Channel, StartTime, EndTime
    StartTime/EndTime are ISO 8601 strings (e.g. "2000-01-01T00:15:00Z").
    Sub-second precision (e.g. "...15.429Z") is stripped before parsing.

    Returns a DataFrame with columns:
        anomaly_id   str
        channel_id   str
        start_time   datetime64[us, UTC]
        end_time     datetime64[us, UTC]
    """
    df = pd.read_csv(str(path))
    df = df.rename(columns={"ID": "anomaly_id", "Channel": "channel_id"})

    def _parse_ts(series: pd.Series) -> pd.Series:
        # Strip optional sub-second component (.NNN) before the trailing Z.
        normalized = series.str.replace(r"\.\d+Z$", "Z", regex=True)
        return pd.to_datetime(normalized, format="%Y-%m-%dT%H:%M:%SZ", utc=True)

    df["start_time"] = _parse_ts(df["StartTime"])
    df["end_time"] = _parse_ts(df["EndTime"])

    log.info("read_labels", path=str(path), n_labels=len(df))
    return df[["anomaly_id", "channel_id", "start_time", "end_time"]]


def write_series(df: pd.DataFrame, output_path: UPath) -> None:
    """Write per-timestep series data to a Hive partition directory.

    Each call writes exactly one channel's data. The partition path is derived
    from the mission_id and channel_id values already present in the DataFrame:
        {output_path}/mission_id={M}/channel_id={C}/part.parquet

    The output file contains only the four non-partition columns defined in
    SERIES_FILE_SCHEMA (telemetry_timestamp, value_normalized, segment_id,
    is_anomaly). Partition column values are encoded in the directory names,
    not stored in the file — matching the convention used by PyArrow's
    read_table and the downstream model/dataset.py reader.

    Args:
        df:          DataFrame with all SERIES_SCHEMA_COLS present.
        output_path: Root output directory for this split
                     (e.g. data/processed/ESA-Mission1/train/).
    """
    mission_id = str(df["mission_id"].iloc[0])
    channel_id = str(df["channel_id"].iloc[0])

    partition_dir = to_upath(output_path) / f"mission_id={mission_id}" / f"channel_id={channel_id}"
    if not str(partition_dir).startswith("gs://"):
        partition_dir.mkdir(parents=True, exist_ok=True)

    out_df = (
        df[["telemetry_timestamp", "value_normalized", "segment_id", "is_anomaly"]]
        .sort_values("telemetry_timestamp")
        .reset_index(drop=True)
    )

    table = pa.Table.from_pandas(out_df, schema=SERIES_FILE_SCHEMA, preserve_index=False)
    pq.write_table(table, str(partition_dir / "part.parquet"))

    log.info(
        "write_series",
        output_path=str(output_path),
        mission_id=mission_id,
        channel_id=channel_id,
        rows=len(out_df),
    )


# ---------------------------------------------------------------------------
# ISS raw-tick read functions
# ---------------------------------------------------------------------------


def read_iss_ticks(
    raw_ticks_dir: str | UPath,
    channel_id: str,
) -> pd.DataFrame:
    """Read all hourly tick shards for one ISS channel and return a single DataFrame.

    The Phase 12 collector writes shards to:
        {raw_ticks_dir}/ISS/ticks/channel_id={channel_id}/{YYYYMMDDTHHMMSS}.parquet

    Shard filenames are lexicographically ordered by timestamp so glob + sort
    gives chronological order without parsing the filename.

    Args:
        raw_ticks_dir: Root directory for raw tick data (local path or gs:// URI).
        channel_id:    ISS PUI (e.g. "S1000003", "USLAB000018").

    Returns:
        DataFrame matching RAW_TICK_SCHEMA columns:
            telemetry_timestamp  datetime64[us, UTC]
            value                float32
            aos_timestamp        float64 (nullable)
        Rows are sorted by telemetry_timestamp.

    Raises:
        FileNotFoundError: If the channel directory does not exist or contains
                           no Parquet shards.
    """
    ticks_dir = to_upath(raw_ticks_dir) / "ISS" / "ticks" / f"channel_id={channel_id}"
    if not ticks_dir.exists():
        raise FileNotFoundError(f"ISS tick directory not found: {ticks_dir}")

    shards = sorted(ticks_dir.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No Parquet shards in {ticks_dir}")

    # partitioning=None prevents PyArrow 18+ from injecting a channel_id column
    # from the Hive directory name, which would create a duplicate column conflict.
    tables = [pq.read_table(str(s), partitioning=None) for s in shards]
    df = pa.concat_tables(tables).to_pandas()
    df = df.sort_values("telemetry_timestamp").reset_index(drop=True)

    log.info(
        "read_iss_ticks",
        channel_id=channel_id,
        n_shards=len(shards),
        rows=len(df),
    )
    return df


def discover_iss_channels(
    raw_ticks_dir: str | UPath,
    exclude: list[str] | None = None,
) -> list[str]:
    """List ISS telemetry channel IDs available in the raw-tick archive.

    Globs for ``channel_id=*`` directories under
    ``{raw_ticks_dir}/ISS/ticks/`` and extracts the PUI from each directory
    name.  Context items (TIME_000001, USLAB000086) are excluded by default
    because they are not telemetry channels and must not be preprocessed.

    Args:
        raw_ticks_dir: Root directory for raw tick data.
        exclude:       Channel IDs to exclude.  Defaults to CONTEXT_ITEMS
                       from ``ingest.iss_channels``.

    Returns:
        Sorted list of channel IDs (PUIs).
    """
    from spacecraft_telemetry.ingest.iss_channels import CONTEXT_ITEMS

    if exclude is None:
        exclude = CONTEXT_ITEMS

    ticks_root = to_upath(raw_ticks_dir) / "ISS" / "ticks"
    channel_dirs = sorted(ticks_root.glob("channel_id=*"))
    channels = [d.name.removeprefix("channel_id=") for d in channel_dirs]
    channels = [c for c in channels if c not in exclude]

    log.info("discover_iss_channels", n_found=len(channels), excluded=exclude)
    return channels


def read_all_iss_ticks_for_los(
    raw_ticks_dir: str | UPath,
    channel_ids: list[str],
) -> pd.DataFrame:
    """Load tick timestamps from all channels for cross-channel LOS detection.

    Reads only the ``telemetry_timestamp`` and ``channel_id`` columns from
    each channel's shards — ``value`` and ``aos_timestamp`` are dropped
    immediately to minimise memory overhead.  The result is the minimal
    DataFrame required by ``compute_los_mask``.

    Args:
        raw_ticks_dir: Root directory for raw tick data.
        channel_ids:   Telemetry channel IDs to include (no context items).

    Returns:
        DataFrame with columns [telemetry_timestamp (UTC), channel_id (str)].
    """
    frames: list[pd.DataFrame] = []
    for ch in channel_ids:
        ticks = read_iss_ticks(raw_ticks_dir, ch)
        # Keep only the two columns needed for LOS detection.
        mini = ticks[["telemetry_timestamp"]].copy()
        mini["channel_id"] = ch
        frames.append(mini)

    if not frames:
        return pd.DataFrame(columns=["telemetry_timestamp", "channel_id"])

    result = pd.concat(frames, ignore_index=True)
    log.info(
        "read_all_iss_ticks_for_los",
        n_channels=len(channel_ids),
        total_rows=len(result),
    )
    return result
