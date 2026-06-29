"""Compute empirical LOS-duration statistics from the raw-tick archive.

Used by LivePump to surface an honest ``expected_resume_in_s`` estimate in the
LOS status event rather than a hardcoded constant.  Phrased as a historical
median, not a countdown promise.

Usage::

    stats = compute_los_stats(raw_ticks_dir, mission="ISS", grid_interval_seconds=30)
    if stats:
        log.info("los_stats.loaded", median_s=stats.median_s, p90_s=stats.p90_s)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from upath import UPath

from spacecraft_telemetry.core.logging import get_logger
from spacecraft_telemetry.ingest.iss_channels import ISS_CHANNELS
from spacecraft_telemetry.preprocess.transforms import compute_los_mask

log = get_logger("api.live.los_stats")

# Minimum number of confirmed LOS events needed before the estimate is
# considered reliable.  Below this threshold None is returned so the banner
# shows no ETA rather than a noise-dominated estimate.
_MIN_LOS_EVENTS = 3


@dataclass(frozen=True)
class LosStats:
    """Empirical LOS-duration statistics derived from the raw-tick archive."""

    median_s: float
    p90_s: float
    n_events: int


def compute_los_stats(
    raw_ticks_dir: Path | UPath | str,
    mission: str = "ISS",
    grid_interval_seconds: int = 30,
) -> LosStats | None:
    """Load the raw-tick archive and compute LOS gap duration statistics.

    Reads the most recent shard files from ``{raw_ticks_dir}/{mission}/``
    for all telemetry channels (excluding context items), builds the cross-
    channel gap mask via ``compute_los_mask``, and derives median / p90 gap
    duration.

    Returns ``None`` when:
    - The archive directory is missing or empty.
    - Fewer than ``_MIN_LOS_EVENTS`` LOS events are found (insufficient data).
    - Any I/O error occurs (GCS unavailable, corrupt Parquet).

    Args:
        raw_ticks_dir:         Root directory of the raw-tick archive.
                               Supports ``gs://`` URIs via UPath / gcsfs.
        mission:               Mission name used as a subdirectory under
                               ``raw_ticks_dir``.
        grid_interval_seconds: Grid resolution in seconds (must match the
                               pump's resampler).

    Returns:
        ``LosStats`` with median and p90 LOS duration in seconds, or ``None``.
    """
    # Raw tick shards are stored as Hive partitions:
    # {raw_ticks_dir}/ISS/ticks/channel_id={PUI}/{timestamp}.parquet
    # channel_id is in the directory name, not in the file columns.
    ticks_root = UPath(raw_ticks_dir) / mission / "ticks"
    telemetry_channels = set(ISS_CHANNELS.keys())
    frames: list[pd.DataFrame] = []

    for channel_id in telemetry_channels:
        channel_dir = ticks_root / f"channel_id={channel_id}"
        try:
            parquet_files = list(channel_dir.glob("*.parquet"))
        except Exception as exc:
            log.debug("los_stats.channel_glob_failed", channel_id=channel_id, error=str(exc))
            continue
        for path in sorted(parquet_files):
            try:
                df = pd.read_parquet(path, columns=["telemetry_timestamp"])
                df["channel_id"] = channel_id
                if not df.empty:
                    frames.append(df)
            except Exception as exc:
                log.warning("los_stats.read_failed", path=str(path), error=str(exc))
                continue

    if not frames:
        log.info("los_stats.no_telemetry_data", ticks_root=str(ticks_root))
        return None

    all_ticks = pd.concat(frames, ignore_index=True)
    if all_ticks.empty:
        return None

    # Ensure UTC-aware timestamps.
    ts_col = all_ticks["telemetry_timestamp"]
    if ts_col.dt.tz is None:
        all_ticks["telemetry_timestamp"] = ts_col.dt.tz_localize("UTC")

    # compute_los_mask expects column "telemetry_timestamp" — satisfied above.
    los_mask = compute_los_mask(all_ticks, grid_interval_seconds=grid_interval_seconds)

    if los_mask.empty or not los_mask.any():
        log.info("los_stats.no_los_found")
        return None

    # Measure the duration of each contiguous LOS run.
    durations_s = _measure_los_runs(los_mask, grid_interval_seconds)

    if len(durations_s) < _MIN_LOS_EVENTS:
        log.info(
            "los_stats.insufficient_events",
            n_events=len(durations_s),
            required=_MIN_LOS_EVENTS,
        )
        return None

    arr = np.array(durations_s, dtype=float)
    median_s = float(np.median(arr))
    p90_s = float(np.percentile(arr, 90))

    log.info(
        "los_stats.computed",
        n_events=len(durations_s),
        median_s=round(median_s),
        p90_s=round(p90_s),
    )
    return LosStats(median_s=median_s, p90_s=p90_s, n_events=len(durations_s))


def _measure_los_runs(
    los_mask: pd.Series,
    grid_interval_seconds: int,
) -> list[float]:
    """Extract the duration in seconds of each contiguous True run in los_mask."""
    durations: list[float] = []
    in_run = False
    run_len = 0
    for val in los_mask:
        if val:
            in_run = True
            run_len += 1
        elif in_run:
            durations.append(float(run_len * grid_interval_seconds))
            in_run = False
            run_len = 0
    if in_run and run_len:
        durations.append(float(run_len * grid_interval_seconds))
    return durations
