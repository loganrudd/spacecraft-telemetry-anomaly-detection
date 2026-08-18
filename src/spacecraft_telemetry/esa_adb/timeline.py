"""Observed (per-segment) evaluation timeline for the ESA-ADB report.

The TNR_t term in esa_adb.metrics.corrected_event_wise needs a "nominal
nanoseconds" denominator — the amount of real, non-gap evaluated time. Using
a single [min(timestamp), max(timestamp)] envelope would count LOS/segment
gaps as nominal time, inflating TNR_t and flattering precision. This module
builds the timeline from contiguous same-segment_id runs in the processed
test partition instead (see docs/plans/019, Open Question 1).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from spacecraft_telemetry.esa_adb.intervals import union
from spacecraft_telemetry.model.dataset import load_series_metadata

if TYPE_CHECKING:
    import numpy as np

    from spacecraft_telemetry.core.config import Settings
    from spacecraft_telemetry.esa_adb.intervals import Interval


def channel_timeline_from_metadata(
    segment_ids: np.ndarray[Any, np.dtype[np.int32]],
    timestamps: np.ndarray[Any, Any],
) -> list[Interval]:
    """Pure computation of channel_timeline given preloaded metadata arrays.

    Split out of channel_timeline so a caller that already holds a channel's
    (segment_ids, timestamps) — loaded once via load_series_metadata() — can
    reuse it instead of re-reading the parquet partition (docs/plans/019 P2/P3).

    ``timestamps`` is a tz-naive numpy datetime64 array (PyArrow's to_numpy()
    drops the tz label from the UTC-typed Parquet column even though the
    instants are UTC) — localized to UTC here so the result is comparable to
    esa_adb.events' tz-aware pd.Timestamp intervals (read_labels parses with
    utc=True).
    """
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


def channel_timeline(settings: Settings, mission: str, channel: str) -> list[Interval]:
    """Observed time span for one channel's test partition, split at segment gaps.

    Each contiguous run of rows sharing the same segment_id becomes one
    interval [first_timestamp, last_timestamp] — different segments (LOS
    gaps, detect_gaps() boundaries) are NOT bridged into one interval.

    Reads only segment_ids/timestamps (load_series_metadata, not
    load_series_parquet) — this function never uses is_anomaly or values. A
    caller making several calls for the same channel (e.g. build_report)
    should instead preload once via load_series_metadata() and call
    channel_timeline_from_metadata() directly.
    """
    segment_ids, _, timestamps = load_series_metadata(
        settings.preprocess.processed_data_dir, mission, channel, "test",
        variant=settings.variant,
    )
    return channel_timeline_from_metadata(segment_ids, timestamps)


def mission_timeline(
    settings: Settings,
    mission: str,
    channels: list[str],
    *,
    metadata_by_channel: dict[
        str, tuple[np.ndarray[Any, np.dtype[np.int32]], np.ndarray[Any, Any], np.ndarray[Any, Any]]
    ]
    | None = None,
) -> list[Interval]:
    """Union of channel_timeline() across ``channels``.

    Args:
        metadata_by_channel: When given, reuse each channel's preloaded
            (segment_ids, is_anomaly, timestamps) — the load_series_metadata()
            shape — instead of re-reading the parquet partition (see
            esa_adb.report.build_report, docs/plans/019 P2/P3). is_anomaly is
            accepted but unused here; the shared shape lets one preload serve
            mission_timeline, _hpo_cutoff, and detection reconstruction alike.
    """
    result: list[Interval] = []
    for channel in channels:
        if metadata_by_channel is not None:
            segment_ids, _, timestamps = metadata_by_channel[channel]
            channel_result = channel_timeline_from_metadata(segment_ids, timestamps)
        else:
            channel_result = channel_timeline(settings, mission, channel)
        result = union(result, channel_result)
    return result
