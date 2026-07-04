"""Tests for api.drift.RollingDriftMonitor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacecraft_telemetry.api.drift import DriftSnapshot, FeatureDrift, RollingDriftMonitor
from spacecraft_telemetry.evidently_monitoring.reference import REALTIME_FEATURE_COLS

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_N_FEATURES = len(REALTIME_FEATURE_COLS)  # 2: value_normalized + rate_of_change
_WINDOW_SIZE = 64
_TICK_INTERVAL = 10


def _make_reference(n_rows: int = 1200, seed: int = 0) -> pd.DataFrame:
    """Build a synthetic reference DataFrame matching MONITORING_FEATURE_COLS.

    The monitor subsets to REALTIME_FEATURE_COLS internally, so all 14 columns
    must be present here (as they would be in a real reference.parquet).
    """
    rng = np.random.default_rng(seed)
    v = pd.Series(rng.standard_normal(n_rows).astype(float))
    df = pd.DataFrame({"value_normalized": v})
    for w in (10, 50, 100):
        df[f"rolling_mean_{w}"] = v.rolling(w, min_periods=1).mean()
        df[f"rolling_std_{w}"]  = v.rolling(w, min_periods=2).std().fillna(0.0)
        df[f"rolling_min_{w}"]  = v.rolling(w, min_periods=1).min()
        df[f"rolling_max_{w}"]  = v.rolling(w, min_periods=1).max()
    df["rate_of_change"] = v.diff().fillna(0.0)
    return df


def _make_monitor(
    reference: pd.DataFrame | None = None,
    window_size: int = _WINDOW_SIZE,
    tick_interval: int = _TICK_INTERVAL,
    rate_interval_seconds: float = 1.0,
    confirm_windows: int = 1,
) -> RollingDriftMonitor:
    ref = reference if reference is not None else _make_reference()
    return RollingDriftMonitor(
        channel="test-ch",
        reference=ref,
        window_size=window_size,
        tick_interval=tick_interval,
        feature_drift_threshold=0.10,
        channel_drift_threshold=0.30,
        rate_interval_seconds=rate_interval_seconds,
        confirm_windows=confirm_windows,
    )


def _nominal_row(reference: pd.DataFrame, rng: np.random.Generator) -> dict[str, float]:
    """Sample value_normalized from the reference distribution (no drift).

    Only value_normalized is pushed — rolling features are recomputed by
    _add_rolling_features inside _compute_drift.
    """
    idx = int(rng.integers(0, len(reference)))
    return {"value_normalized": float(reference.iloc[idx]["value_normalized"])}


def _drifted_row(reference: pd.DataFrame) -> dict[str, float]:
    """Return value_normalized shifted far from the reference mean (+5 std)."""
    ref_mean = float(reference["value_normalized"].mean())
    ref_std = max(float(reference["value_normalized"].std()), 1e-6)
    return {"value_normalized": ref_mean + 5.0 * ref_std}


# ---------------------------------------------------------------------------
# Unit tests — no Evidently I/O
# ---------------------------------------------------------------------------


class TestPushAndWindow:
    def test_push_buffers_up_to_window_size(self) -> None:
        monitor = _make_monitor(window_size=_WINDOW_SIZE)
        rng = np.random.default_rng(1)
        ref = _make_reference()
        for _ in range(300):
            monitor.push(_nominal_row(ref, rng))
        # deque maxlen caps the buffer
        assert len(monitor._window) == _WINDOW_SIZE

    def test_push_increments_tick_count(self) -> None:
        monitor = _make_monitor()
        ref = _make_reference()
        rng = np.random.default_rng(2)
        for _ in range(5):
            monitor.push(_nominal_row(ref, rng))
        assert monitor._tick_count == 5

    def test_missing_keys_produce_nan_not_error(self) -> None:
        monitor = _make_monitor()
        monitor.push({})  # all keys missing → NaN row — must not raise
        assert len(monitor._window) == 1


class TestRateIntervalScaling:
    """rate_of_change must match the reference profile's Δvalue/Δt_seconds units.

    evidently_monitoring/reference.py builds rate_of_change as a per-second rate.
    The live monitor only receives value_normalized per tick with no timestamps,
    so it must divide the per-tick diff by rate_interval_seconds to reproduce
    the same units — otherwise the feature is off by a constant factor and
    reads as permanent drift regardless of the actual data (see drift.py
    _add_rolling_features).
    """

    def test_default_interval_is_legacy_per_tick(self) -> None:
        monitor = _make_monitor(rate_interval_seconds=1.0)
        df = pd.DataFrame({"value_normalized": [0.0, 2.0, 4.0, 6.0]})
        out = monitor._add_rolling_features(df)
        assert out["rate_of_change"].iloc[1:].tolist() == pytest.approx([2.0, 2.0, 2.0])

    def test_interval_divides_per_tick_delta(self) -> None:
        monitor = _make_monitor(rate_interval_seconds=30.0)
        df = pd.DataFrame({"value_normalized": [0.0, 3.0, 6.0, 9.0]})
        out = monitor._add_rolling_features(df)
        # Per-tick delta is 3.0; divided by the 30s grid interval -> 0.1/s.
        assert out["rate_of_change"].iloc[1:].tolist() == pytest.approx([0.1, 0.1, 0.1])

    def test_mismatched_interval_would_inflate_score_by_constant_factor(self) -> None:
        # Regression guard for the actual bug: pushing the SAME data through two
        # monitors that differ only in rate_interval_seconds must produce rate
        # values that differ by exactly that factor -- proving the fix is a pure
        # unit correction, not a change in what's being measured.
        legacy = _make_monitor(rate_interval_seconds=1.0)
        grid = _make_monitor(rate_interval_seconds=30.0)
        df = pd.DataFrame({"value_normalized": [1.0, 4.0, 2.0, 5.0, 3.0]})
        legacy_roc = legacy._add_rolling_features(df.copy())["rate_of_change"]
        grid_roc = grid._add_rolling_features(df.copy())["rate_of_change"]
        ratio = (legacy_roc / grid_roc).dropna()
        assert ratio.tolist() == pytest.approx([30.0] * len(ratio))


class TestShouldRun:
    def test_false_before_window_full(self) -> None:
        monitor = _make_monitor(window_size=_WINDOW_SIZE, tick_interval=_TICK_INTERVAL)
        ref = _make_reference()
        rng = np.random.default_rng(3)
        for _ in range(_WINDOW_SIZE - 1):
            monitor.push(_nominal_row(ref, rng))
        # Window not yet full — should_run must be False regardless of tick count.
        assert not monitor.should_run()

    def test_true_on_interval_boundary_when_full(self) -> None:
        monitor = _make_monitor(window_size=_WINDOW_SIZE, tick_interval=_TICK_INTERVAL)
        ref = _make_reference()
        rng = np.random.default_rng(4)
        # Fill the window exactly — tick_count == window_size.
        # should_run is True only when tick_count % tick_interval == 0.
        hits = []
        for _ in range(_WINDOW_SIZE + _TICK_INTERVAL * 3):
            monitor.push(_nominal_row(ref, rng))
            if monitor.should_run():
                hits.append(monitor._tick_count)
        # Every hit must be at a tick_interval boundary.
        assert all(h % _TICK_INTERVAL == 0 for h in hits)
        # At least one hit must have occurred.
        assert len(hits) >= 1

    def test_false_between_intervals(self) -> None:
        monitor = _make_monitor(window_size=_WINDOW_SIZE, tick_interval=_TICK_INTERVAL)
        ref = _make_reference()
        rng = np.random.default_rng(5)
        # Fill window and advance past the first boundary.
        for _ in range(_WINDOW_SIZE + 1):
            monitor.push(_nominal_row(ref, rng))
        # Tick count is now window_size + 1; next boundary is window_size + (interval - 1).
        assert not monitor.should_run()


class TestPriming:
    def test_should_run_shortly_after_priming(self) -> None:
        """Priming with window_size seed values must not require a full refill.

        app.py primes each monitor from the tail of its replay slice before
        any live tick arrives (mirrors engine priming). should_run() must
        therefore fire within one tick_interval of the first live tick, not
        after window_size + tick_interval more ticks as it would starting
        from an empty window.
        """
        monitor = _make_monitor(window_size=_WINDOW_SIZE, tick_interval=_TICK_INTERVAL)
        ref = _make_reference()
        rng = np.random.default_rng(7)
        for _ in range(_WINDOW_SIZE):
            monitor.push(_nominal_row(ref, rng))
        fired_within = None
        for i in range(1, _TICK_INTERVAL + 1):
            monitor.push(_nominal_row(ref, rng))
            if monitor.should_run():
                fired_within = i
                break
        assert fired_within is not None
        assert fired_within <= _TICK_INTERVAL


class TestConfirmWindows:
    """K-consecutive confirmation on RollingDriftMonitor._compute_drift.

    Drives ``_compute_drift`` directly with crafted ``current`` frames so the
    alert sequence is deterministic, rather than depending on scipy noise
    across successive ``run()`` calls.
    """

    def _current_df(self, values: list[float]) -> pd.DataFrame:
        return pd.DataFrame({"value_normalized": values})

    def test_k1_fires_immediately(self) -> None:
        ref = _make_reference()
        monitor = _make_monitor(reference=ref, confirm_windows=1)
        drifted = [_drifted_row(ref)["value_normalized"] for _ in range(_WINDOW_SIZE)]
        snapshot = monitor._compute_drift(self._current_df(drifted))
        assert snapshot.drifted is True

    def test_k3_requires_three_consecutive_alerts(self) -> None:
        ref = _make_reference()
        monitor = _make_monitor(reference=ref, confirm_windows=3)
        drifted = [_drifted_row(ref)["value_normalized"] for _ in range(_WINDOW_SIZE)]
        s1 = monitor._compute_drift(self._current_df(drifted))
        assert s1.drifted is False
        s2 = monitor._compute_drift(self._current_df(drifted))
        assert s2.drifted is False
        s3 = monitor._compute_drift(self._current_df(drifted))
        assert s3.drifted is True

    def test_non_alert_resets_counter(self) -> None:
        # Reliably nominal Wasserstein estimates need a larger sample than
        # _WINDOW_SIZE=64 (see test_run_returns_snapshot_when_nominal, which
        # uses the same 1200/256 shapes) -- independently-drawn nominal rows
        # at small N show enough sampling noise in rate_of_change to spuriously
        # alert.
        ref = _make_reference(n_rows=1200, seed=30)
        monitor = _make_monitor(reference=ref, confirm_windows=3)
        drifted = [_drifted_row(ref)["value_normalized"] for _ in range(256)]
        rng = np.random.default_rng(99)
        nominal = [_nominal_row(ref, rng)["value_normalized"] for _ in range(256)]

        monitor._compute_drift(self._current_df(drifted))
        monitor._compute_drift(self._current_df(drifted))
        # A non-alerting run resets the counter to 0.
        reset_snapshot = monitor._compute_drift(self._current_df(nominal))
        assert reset_snapshot.drifted is False
        # Counter restarts at 1, not 3 -- one more drifted run is not enough.
        s = monitor._compute_drift(self._current_df(drifted))
        assert s.drifted is False

    def test_percent_drifted_unchanged_by_confirm_windows(self) -> None:
        ref = _make_reference()
        drifted = [_drifted_row(ref)["value_normalized"] for _ in range(_WINDOW_SIZE)]
        m1 = _make_monitor(reference=ref, confirm_windows=1)
        m3 = _make_monitor(reference=ref, confirm_windows=3)
        s1 = m1._compute_drift(self._current_df(drifted))
        s3 = m3._compute_drift(self._current_df(drifted))
        assert s1.percent_drifted == pytest.approx(s3.percent_drifted)
        assert s1.drifted is True
        assert s3.drifted is False


# ---------------------------------------------------------------------------
# Async tests — exercise Evidently (marked slow)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_returns_none_before_window_full() -> None:
    monitor = _make_monitor(window_size=_WINDOW_SIZE)
    ref = _make_reference()
    rng = np.random.default_rng(6)
    for _ in range(_WINDOW_SIZE // 2):
        monitor.push(_nominal_row(ref, rng))
    result = await monitor.run()
    assert result is None


@pytest.mark.asyncio
@pytest.mark.slow
async def test_run_returns_snapshot_when_drifted() -> None:
    # Use prod-realistic reference size so Wasserstein estimates are stable.
    ref = _make_reference(n_rows=1200, seed=10)
    monitor = _make_monitor(reference=ref, window_size=256, tick_interval=_TICK_INTERVAL)
    # Push heavily shifted rows — all features displaced by 5 std from reference mean.
    for _ in range(256):
        monitor.push(_drifted_row(ref))
    snapshot = await monitor.run()
    assert isinstance(snapshot, DriftSnapshot)
    assert snapshot.channel == "test-ch"
    assert 0.0 <= snapshot.percent_drifted <= 1.0
    assert snapshot.drifted is True
    assert len(snapshot.features) == _N_FEATURES
    assert all(isinstance(f, FeatureDrift) for f in snapshot.features)
    # At least some features must be individually flagged.
    assert any(f.drifted for f in snapshot.features)


@pytest.mark.asyncio
@pytest.mark.slow
async def test_run_returns_snapshot_when_nominal() -> None:
    # Use prod-realistic shapes: ref=1200, current window=256.
    # At these sizes the normalized Wasserstein distance for truly nominal
    # N(0,1) data is reliably well below the 0.10 threshold.
    ref = _make_reference(n_rows=1200, seed=20)
    monitor = _make_monitor(reference=ref, window_size=256, tick_interval=_TICK_INTERVAL)
    rng = np.random.default_rng(21)
    for _ in range(256):
        monitor.push(_nominal_row(ref, rng))
    snapshot = await monitor.run()
    assert isinstance(snapshot, DriftSnapshot)
    assert snapshot.drifted is False
    assert snapshot.percent_drifted < 0.30


def test_monitor_tracks_realtime_feature_cols() -> None:
    """Reference is subsetted to REALTIME_FEATURE_COLS on construction.

    If a future REALTIME_FEATURE_COLS change adds or removes a column, this test
    will catch the mismatch before it silently changes what the monitor computes.
    """
    monitor = _make_monitor()
    assert list(monitor._reference.columns) == list(REALTIME_FEATURE_COLS)


@pytest.mark.asyncio
@pytest.mark.slow
async def test_full_nan_window_does_not_raise() -> None:
    """Pushing a full window of {} rows (all-NaN) must not raise during run().

    All-NaN value_normalized → diff() stays NaN → dropna() gives empty array
    → score=0, drifted=False for every column. Monitor must survive and return
    a zero-score snapshot rather than raise or return None.
    """
    monitor = _make_monitor(window_size=_WINDOW_SIZE, tick_interval=_WINDOW_SIZE)
    for _ in range(_WINDOW_SIZE):
        monitor.push({})  # all keys missing → all NaN
    snapshot = await monitor.run()
    # Evidently raises ValueError on all-NaN columns; _compute_drift catches it
    # and returns a zero-score, non-drifted snapshot so the pump task survives.
    assert isinstance(snapshot, DriftSnapshot)
    assert len(snapshot.features) == _N_FEATURES
    assert snapshot.drifted is False
    assert snapshot.percent_drifted == pytest.approx(0.0)
    for f in snapshot.features:
        assert f.score == pytest.approx(0.0)
        assert f.drifted is False
