"""Observed (per-segment) evaluation timeline for the ESA-ADB report.

The TNR_t term in esa_adb.metrics.corrected_event_wise needs a "nominal
nanoseconds" denominator — the amount of real, non-gap evaluated time. Using
a single [min(timestamp), max(timestamp)] envelope would count LOS/segment
gaps as nominal time, inflating TNR_t and flattering precision. This module
builds the timeline from contiguous same-segment_id runs in the processed
test partition instead (see docs/plans/019, Open Question 1).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from spacecraft_telemetry.esa_adb.intervals import union
from spacecraft_telemetry.model.dataset import load_series_parquet

if TYPE_CHECKING:
    from spacecraft_telemetry.core.config import Settings
    from spacecraft_telemetry.esa_adb.intervals import Interval


def channel_timeline(settings: Settings, mission: str, channel: str) -> list[Interval]:
    """Observed time span for one channel's test partition, split at segment gaps.

    Each contiguous run of rows sharing the same segment_id becomes one
    interval [first_timestamp, last_timestamp] — different segments (LOS
    gaps, detect_gaps() boundaries) are NOT bridged into one interval.

    ``timestamps`` is a tz-naive numpy datetime64 array (PyArrow's to_numpy()
    drops the tz label from the UTC-typed Parquet column even though the
    instants are UTC) — localized to UTC here so the result is comparable to
    esa_adb.events' tz-aware pd.Timestamp intervals (read_labels parses with
    utc=True).
    """
    _, segment_ids, _, timestamps = load_series_parquet(
        settings.preprocess.processed_data_dir, mission, channel, "test"
    )
    if len(timestamps) == 0:
        return []

    intervals: list[Interval] = []
    seg_start = 0
    n = len(segment_ids)
    for i in range(1, n + 1):
        if i == n or segment_ids[i] != segment_ids[seg_start]:
            intervals.append(
                (
                    pd.Timestamp(timestamps[seg_start]).tz_localize("UTC"),
                    pd.Timestamp(timestamps[i - 1]).tz_localize("UTC"),
                )
            )
            seg_start = i
    return intervals


def mission_timeline(settings: Settings, mission: str, channels: list[str]) -> list[Interval]:
    """Union of channel_timeline() across ``channels``."""
    result: list[Interval] = []
    for channel in channels:
        result = union(result, channel_timeline(settings, mission, channel))
    return result
