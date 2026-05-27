"""Pandas/PyArrow I/O: read raw channel Parquet + labels CSV, write processed output.

Replaces spark/io.py. All reads produce DataFrames with the same column contracts
as before; all writes produce the same Hive-partitioned layout:
    {output_path}/mission_id={M}/channel_id={C}/part.parquet

This layout is consumed by model/dataset.py, ray_training/runner.py, and
evidently_monitoring/reference.py — none of those files change.
"""

from __future__ import annotations

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
    df["channel_id"] = channel_id
    df["mission_id"] = mission_id

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
