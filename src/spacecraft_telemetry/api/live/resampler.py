"""Incremental 30-second grid resampler for the live ISS telemetry pump.

OnlineGridResampler is the stateful, tick-by-tick equivalent of the batch
resample_to_grid() transform (preprocess/transforms.py).  It produces bucket
means that are byte-identical to the pandas resample().mean().ffill() output
when compared at float32 precision.

Bucket boundaries are aligned to the start of the UTC day (midnight), matching
pandas' default origin='start_day' resample behaviour.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta


class OnlineGridResampler:
    """Incremental mean + ffill resampler onto a fixed time grid.

    Accepts irregular (timestamp, value) ticks and emits (bucket_timestamp,
    mean_value) pairs for each completed grid bucket, forward-filling any
    empty buckets from the previous non-empty value.

    A bucket is "completed" when a tick arrives for a strictly later bucket,
    matching pandas resample() semantics where a bucket only appears in the
    output once the next bucket has started.

    Usage::

        resampler = OnlineGridResampler(grid_interval_seconds=30)
        for ts, raw in ticks:
            for bucket_ts, mean_val in resampler.push(ts, raw):
                engine.step(normalize(channel, mean_val), ...)
        # At stream end or flush interval:
        for bucket_ts, mean_val in resampler.flush():
            engine.step(normalize(channel, mean_val), ...)
    """

    def __init__(self, grid_interval_seconds: int = 30) -> None:
        self._interval = grid_interval_seconds
        self._current_bucket: datetime | None = None
        self._bucket_values: list[float] = []
        self._last_mean: float | None = None

    @staticmethod
    def floor_to_grid(ts: datetime, interval_s: int) -> datetime:
        """Floor *ts* to the nearest grid boundary aligned to UTC midnight.

        Matches pandas resample(origin='start_day') bucket labelling exactly:
        buckets are counted in whole multiples of *interval_s* from midnight.
        """
        ts_utc = ts.astimezone(UTC)
        day_start = ts_utc.replace(hour=0, minute=0, second=0, microsecond=0)
        secs = (ts_utc - day_start).total_seconds()
        floored = math.floor(secs / interval_s) * interval_s
        return day_start + timedelta(seconds=floored)

    def push(self, ts: datetime, value: float) -> list[tuple[datetime, float]]:
        """Push one tick. Returns closed (bucket_ts, mean_value) pairs.

        Returns:
            - Empty list: tick belongs to the currently open bucket.
            - One item: current bucket just closed, no gap to the new bucket.
            - Multiple items: current bucket closed plus gap buckets ffilled.
        """
        bucket = self.floor_to_grid(ts, self._interval)
        results: list[tuple[datetime, float]] = []

        if self._current_bucket is None:
            self._current_bucket = bucket
            self._bucket_values = [value]
            return results

        if bucket == self._current_bucket:
            self._bucket_values.append(value)
            return results

        # Close the current bucket.
        mean_val = sum(self._bucket_values) / len(self._bucket_values)
        results.append((self._current_bucket, mean_val))
        self._last_mean = mean_val

        # Forward-fill any gap buckets between the closed and the new one.
        next_bucket = self._current_bucket + timedelta(seconds=self._interval)
        while next_bucket < bucket:
            # _last_mean is guaranteed non-None here (just assigned above).
            results.append((next_bucket, self._last_mean))
            next_bucket += timedelta(seconds=self._interval)

        # Open the new bucket.
        self._current_bucket = bucket
        self._bucket_values = [value]
        return results

    def flush(self) -> list[tuple[datetime, float]]:
        """Close and return the current open bucket (if any).

        Call at stream end or on a periodic flush deadline.  After flush()
        the resampler is reset and ready to accumulate new buckets.
        """
        if not self._bucket_values or self._current_bucket is None:
            return []
        mean_val = sum(self._bucket_values) / len(self._bucket_values)
        result = [(self._current_bucket, mean_val)]
        self._last_mean = mean_val
        self._bucket_values = []
        self._current_bucket = None
        return result
