"""Async replay generator for the SSE telemetry stream.

Reads the test-split Parquet for one channel and yields (timestamp, value,
is_anomaly_true) tuples at a speed-multiplied wall-clock rate.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from datetime import datetime
from pathlib import Path

from spacecraft_telemetry.model.dataset import load_series_parquet


async def replay_channel(
    processed_dir: Path,
    mission: str,
    channel: str,
    speed: float,
    tick_interval_seconds: float,
) -> AsyncGenerator[tuple[datetime, float, bool], None]:
    """Async generator yielding ``(timestamp, value, is_anomaly_true)`` per tick.

    Reads all rows from the test-split Parquet for ``channel`` and emits them
    one at a time, sleeping ``tick_interval_seconds / speed`` between ticks to
    simulate real-time replay at the requested speed multiplier.

    Args:
        processed_dir:         Path to the Spark-preprocessed Parquet tree.
        mission:               Mission identifier (e.g. ``"ESA-Mission1"``).
        channel:               Channel identifier (e.g. ``"A-1"``).
        speed:                 Replay speed multiplier (must be > 0).
        tick_interval_seconds: Nominal wall-clock interval between ticks in the
                               source data (e.g. ``1.0`` for 1 Hz telemetry).
    """
    values, _seg, anom, timestamps = load_series_parquet(
        processed_dir, mission, channel, "test"
    )
    delay = tick_interval_seconds / speed
    for ts, v, a in zip(timestamps, values, anom, strict=False):
        yield datetime.fromisoformat(str(ts)), float(v), bool(a)
        await asyncio.sleep(delay)
