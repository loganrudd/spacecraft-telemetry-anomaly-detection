"""Tests for api.replay — async generator and timestamp conversion.

Regression guard for the numpy.datetime64 / pd.Timestamp dual-path bug:
timezone-naive datetime64[ns] parquet columns yield numpy.datetime64 scalars
from to_numpy(), while timezone-aware columns yield pd.Timestamp objects.
Both must produce Python datetime objects without raising.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spacecraft_telemetry.api.replay import replay_channel


async def _collect(cached_data) -> list[tuple[datetime, float, bool]]:
    events = []
    async for ts, v, a in replay_channel(
        processed_dir=Path("/unused"),
        mission="m",
        channel="c",
        speed=1_000_000.0,
        tick_interval_seconds=1.0,
        cached_data=cached_data,
    ):
        events.append((ts, v, a))
    return events


_VALUES = np.array([0.1, 0.2, 0.3], dtype=np.float32)
_ANOM = np.array([False, True, False], dtype=bool)


class TestReplayChannelTimestampConversion:
    async def test_timezone_naive_datetime64_yields_python_datetime(self) -> None:
        """Regression: numpy.datetime64 must not raise AttributeError.

        load_series_parquet returns timezone-naive datetime64[ns] when the
        parquet column has no timezone metadata.
        pd.Timestamp(ts) must bridge the gap — ts.to_pydatetime() would fail here.
        """
        timestamps = np.array(
            ["2000-01-01", "2000-01-02", "2000-01-03"], dtype="datetime64[ns]"
        )
        events = await _collect((_VALUES, _ANOM, timestamps))
        assert len(events) == 3
        for ts, _, _ in events:
            assert isinstance(ts, datetime), f"Expected datetime, got {type(ts)}"

    async def test_timezone_aware_pandas_timestamps_yields_python_datetime(self) -> None:
        """Timezone-aware pd.Timestamp objects (test fixture path) must also work."""
        timestamps = np.array(
            [
                pd.Timestamp("2000-01-01", tz="UTC"),
                pd.Timestamp("2000-01-02", tz="UTC"),
                pd.Timestamp("2000-01-03", tz="UTC"),
            ],
            dtype=object,
        )
        events = await _collect((_VALUES, _ANOM, timestamps))
        assert len(events) == 3
        for ts, _, _ in events:
            assert isinstance(ts, datetime), f"Expected datetime, got {type(ts)}"

    async def test_values_and_anomaly_flags_pass_through_correctly(self) -> None:
        timestamps = np.array(
            ["2000-01-01", "2000-01-02", "2000-01-03"], dtype="datetime64[ns]"
        )
        events = await _collect((_VALUES, _ANOM, timestamps))
        vals = [v for _, v, _ in events]
        anoms = [a for _, _, a in events]
        assert vals == pytest.approx([0.1, 0.2, 0.3], rel=1e-5)
        assert anoms == [False, True, False]
