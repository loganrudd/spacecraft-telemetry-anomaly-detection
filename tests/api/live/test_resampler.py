"""Tests for OnlineGridResampler.

Key assertions:
1. Single tick per bucket: mean == value.
2. Multiple ticks per bucket: mean aggregation.
3. Gap buckets: forward-filled from last non-empty bucket.
4. Flush: returns the open bucket.
5. Equivalence with batch resample_to_grid at float32 precision.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from spacecraft_telemetry.api.live.resampler import OnlineGridResampler
from spacecraft_telemetry.preprocess.transforms import resample_to_grid

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

INTERVAL = 30  # seconds


def _ts(day_secs: float) -> datetime:
    """UTC datetime at *day_secs* seconds past midnight on 2000-01-01."""
    return datetime(2000, 1, 1, tzinfo=UTC) + timedelta(seconds=day_secs)


def _run_online(ticks: list[tuple[datetime, float]]) -> list[tuple[datetime, float]]:
    """Push all ticks through the resampler and flush at the end."""
    r = OnlineGridResampler(INTERVAL)
    results: list[tuple[datetime, float]] = []
    for ts, v in ticks:
        results.extend(r.push(ts, v))
    results.extend(r.flush())
    return results


def _run_batch(ticks: list[tuple[datetime, float]]) -> list[tuple[datetime, float]]:
    """Run batch resample_to_grid on the same ticks, return (ts, value) pairs."""
    df = pd.DataFrame(
        {
            "telemetry_timestamp": pd.to_datetime([ts for ts, _ in ticks], utc=True),
            "value": pd.array([v for _, v in ticks], dtype="float32"),
            "aos_timestamp": pd.array([None] * len(ticks), dtype=object),
        }
    )
    # resample_to_grid expects telemetry_timestamp as datetime64[us, UTC]
    df["telemetry_timestamp"] = df["telemetry_timestamp"].astype("datetime64[us, UTC]")
    result_df = resample_to_grid(
        df, channel_id="test-ch", mission_id="test", grid_interval_seconds=INTERVAL
    )
    return [
        (row.telemetry_timestamp.to_pydatetime(), float(row.value))
        for row in result_df.itertuples()
    ]


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_single_tick_per_bucket_mean_equals_value() -> None:
    """One tick in a bucket: mean == value."""
    ticks = [(_ts(0), 1.0), (_ts(30), 2.0), (_ts(60), 3.0)]
    results = _run_online(ticks)
    assert len(results) == 3
    values = [v for _, v in results]
    assert values == pytest.approx([1.0, 2.0, 3.0])


def test_multiple_ticks_per_bucket_mean_aggregation() -> None:
    """Multiple ticks in one bucket: result is arithmetic mean."""
    ticks = [
        (_ts(5), 10.0),
        (_ts(15), 20.0),
        (_ts(25), 30.0),
        # Next bucket forces the previous to close
        (_ts(35), 99.0),
    ]
    results = _run_online(ticks)
    # Bucket 0 (t=0): mean(10, 20, 30) = 20.0; bucket 30 (t=30): mean(99) = 99.0
    assert len(results) == 2
    assert results[0][1] == pytest.approx(20.0)
    assert results[1][1] == pytest.approx(99.0)


def test_gap_buckets_forward_filled() -> None:
    """Gap buckets between two ticks are ffilled from the last known mean."""
    ticks = [
        (_ts(5), 7.0),
        (_ts(95), 3.0),  # jumps over buckets at t=30 and t=60
    ]
    results = _run_online(ticks)
    # bucket 0 (t=0): 7.0; bucket 30: ffill=7.0; bucket 60: ffill=7.0; bucket 90: 3.0
    assert len(results) == 4
    ts_labels = [ts for ts, _ in results]
    assert ts_labels[0] == _ts(0)
    assert ts_labels[1] == _ts(30)
    assert ts_labels[2] == _ts(60)
    assert ts_labels[3] == _ts(90)
    vals = [v for _, v in results]
    assert vals == pytest.approx([7.0, 7.0, 7.0, 3.0])


def test_flush_returns_open_bucket() -> None:
    """flush() closes and returns the currently open bucket."""
    r = OnlineGridResampler(INTERVAL)
    list(r.push(_ts(5), 42.0))  # opens bucket at t=0, returns nothing
    flushed = r.flush()
    assert len(flushed) == 1
    assert flushed[0] == (_ts(0), pytest.approx(42.0))


def test_flush_on_empty_resampler_returns_empty() -> None:
    """flush() on a fresh resampler with no ticks returns []."""
    r = OnlineGridResampler(INTERVAL)
    assert r.flush() == []


def test_floor_to_grid_aligns_to_midnight() -> None:
    """floor_to_grid floors to the correct 30-second boundary from midnight."""
    ts = _ts(65.9)  # 1 min 5.9 sec → bucket at t=60
    bucket = OnlineGridResampler.floor_to_grid(ts, INTERVAL)
    assert bucket == _ts(60)


def test_floor_to_grid_exact_boundary() -> None:
    """A timestamp exactly on a boundary maps to that boundary."""
    ts = _ts(60.0)
    bucket = OnlineGridResampler.floor_to_grid(ts, INTERVAL)
    assert bucket == _ts(60)


def test_bucket_timestamps_are_utc_aware() -> None:
    """All emitted bucket timestamps are UTC-aware datetime objects."""
    results = _run_online([(_ts(5), 1.0), (_ts(35), 2.0)])
    for ts, _ in results:
        assert ts.tzinfo is not None
        assert ts.utcoffset() == timedelta(0)


# ---------------------------------------------------------------------------
# Equivalence with batch resample_to_grid
# ---------------------------------------------------------------------------


def _build_irregular_ticks(
    n_buckets: int, ticks_per_bucket: int, seed: int = 0
) -> list[tuple[datetime, float]]:
    """Generate *n_buckets* buckets of *ticks_per_bucket* irregular ticks."""
    rng = np.random.default_rng(seed)
    ticks: list[tuple[datetime, float]] = []
    for bucket_idx in range(n_buckets):
        bucket_start = bucket_idx * INTERVAL
        # Random offsets within the bucket (0 to INTERVAL-1 seconds)
        offsets = sorted(rng.uniform(0, INTERVAL - 1, ticks_per_bucket).tolist())
        for offset in offsets:
            ts = _ts(bucket_start + offset)
            value = float(rng.standard_normal())
            ticks.append((ts, value))
    return ticks


@pytest.mark.parametrize(
    "n_buckets,ticks_per_bucket",
    [
        (5, 1),
        (10, 3),
        (8, 1),
        (6, 5),
    ],
)
def test_online_matches_batch_resample(n_buckets: int, ticks_per_bucket: int) -> None:
    """Online resampler output is numerically identical to batch resample_to_grid.

    Comparison is at float32 precision since the batch transform converts to
    float32 before returning.
    """
    ticks = _build_irregular_ticks(n_buckets, ticks_per_bucket)
    online = _run_online(ticks)
    batch = _run_batch(ticks)

    assert len(online) == len(batch), f"length mismatch: online={len(online)}, batch={len(batch)}"

    for i, ((o_ts, o_val), (b_ts, b_val)) in enumerate(zip(online, batch, strict=False)):
        # Timestamps must match exactly.
        assert o_ts == b_ts, f"bucket {i}: timestamp mismatch {o_ts} vs {b_ts}"
        # Values compared at float32 precision.  The batch path converts inputs to
        # float32 before averaging; the online path stays in float64 throughout.
        # This causes single-ULP differences (~1e-7) on multi-tick buckets, so we
        # allow atol=1e-6 (~8x float32 machine epsilon) while still catching any
        # real algorithmic divergence.
        np.testing.assert_allclose(
            np.float32(o_val),
            np.float32(b_val),
            rtol=0,
            atol=1e-6,
            err_msg=f"bucket {i}: value mismatch {o_val} vs {b_val}",
        )


def test_online_matches_batch_with_gap_buckets() -> None:
    """ffill in online resampler matches pandas ffill for gap buckets."""
    # Three populated buckets with a two-bucket gap in the middle.
    ticks = [
        (_ts(10), 5.0),  # bucket 0
        (_ts(100), 9.0),  # bucket 90 (gap buckets at 30 and 60 must ffill)
        (_ts(130), 1.0),  # bucket 120
    ]
    online = _run_online(ticks)
    batch = _run_batch(ticks)

    assert len(online) == len(batch)
    for i, ((o_ts, o_val), (b_ts, b_val)) in enumerate(zip(online, batch, strict=False)):
        assert o_ts == b_ts, f"bucket {i}: timestamp mismatch"
        np.testing.assert_allclose(
            np.float32(o_val),
            np.float32(b_val),
            rtol=0,
            atol=1e-6,
            err_msg=f"bucket {i}: ffill value mismatch",
        )
