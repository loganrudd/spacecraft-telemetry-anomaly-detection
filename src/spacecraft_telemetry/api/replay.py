"""Async replay generator for the SSE telemetry stream.

Reads the test-split Parquet for one channel and yields (timestamp, value,
is_anomaly) tuples at a speed-multiplied wall-clock rate.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from upath import UPath

from spacecraft_telemetry.model.dataset import load_series_parquet

ReplayData = tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]


def _anomaly_slice(
    anom: np.ndarray[Any, Any],
    warmup_rows: int,
    max_rows: int,
) -> slice:
    """Return a slice of at most ``max_rows`` rows starting ``warmup_rows``
    before the first contiguous anomaly run.

    The warmup ensures the dynamic threshold is populated before the first
    anomaly arrives in the stream, so the detector actually fires during the
    demo rather than being cold. Falls back to the first ``max_rows`` rows
    when the channel has no is_anomaly=True rows.
    """
    hits = np.where(anom)[0]
    if hits.size == 0:
        return slice(0, max_rows)
    start = max(0, int(hits[0]) - warmup_rows)
    return slice(start, start + max_rows)


async def replay_channel(
    processed_dir: Path | UPath | str,
    mission: str,
    channel: str,
    speed: float,
    tick_interval_seconds: float,
    cached_data: ReplayData | None = None,
    warmup_rows: int = 500,
    max_rows: int = 3000,
) -> AsyncGenerator[tuple[datetime, float, bool], None]:
    """Async generator yielding ``(timestamp, value, is_anomaly)`` per tick.

    Reads the test-split Parquet for ``channel``, slices to a short anomaly-dense
    window (``max_rows`` rows starting ``warmup_rows`` before the first labeled
    anomaly), and emits one row per ``tick_interval_seconds / speed`` seconds.

    The slice keeps Cloud Run instance up-time to ~30s at 100x instead of
    hours for a full 9M-row test series, and ensures the anomaly appears early
    in the demo rather than after a long nominal stretch.  Pass ``max_rows=0``
    to disable slicing and replay the full test set.

    Args:
        processed_dir:         Path to the preprocessed Parquet tree.
        mission:               Mission identifier (e.g. ``"ESA-Mission1"``).
        channel:               Channel identifier (e.g. ``"channel_12"``).
        speed:                 Replay speed multiplier (must be > 0).
        tick_interval_seconds: Nominal wall-clock interval between ticks.
        cached_data:           Pre-loaded (values, anom, timestamps) arrays;
                               skips the Parquet read when provided (tests /
                               startup pre-cache).
        warmup_rows:           Rows before the first anomaly to include so
                               the dynamic threshold is warm on arrival.
        max_rows:              Total rows to emit. 0 = no limit (full test set).
    """
    if cached_data is not None:
        values, anom, timestamps = cached_data
    else:
        values, _seg, anom, timestamps = await asyncio.to_thread(
            load_series_parquet, processed_dir, mission, channel, "test"
        )

    if max_rows > 0:
        sl = _anomaly_slice(anom, warmup_rows, max_rows)
        values = values[sl]
        anom = anom[sl]
        timestamps = timestamps[sl]

    delay = tick_interval_seconds / speed
    for ts, v, a in zip(timestamps, values, anom, strict=False):
        yield pd.Timestamp(ts).to_pydatetime(), float(v), bool(a)
        await asyncio.sleep(delay)
