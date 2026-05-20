"""Tests for api.drift.RollingDriftMonitor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacecraft_telemetry.api.drift import DriftSnapshot, FeatureDrift, RollingDriftMonitor
from spacecraft_telemetry.evidently_monitoring.reference import MONITORING_FEATURE_COLS

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_N_FEATURES = len(MONITORING_FEATURE_COLS)
_WINDOW_SIZE = 64
_TICK_INTERVAL = 10


def _make_reference(n_rows: int = 500, seed: int = 0) -> pd.DataFrame:
    """Build a synthetic reference DataFrame matching MONITORING_FEATURE_COLS."""
    rng = np.random.default_rng(seed)
    data = {col: rng.standard_normal(n_rows).astype(float) for col in MONITORING_FEATURE_COLS}
    return pd.DataFrame(data)


def _make_monitor(
    reference: pd.DataFrame | None = None,
    window_size: int = _WINDOW_SIZE,
    tick_interval: int = _TICK_INTERVAL,
) -> RollingDriftMonitor:
    ref = reference if reference is not None else _make_reference()
    return RollingDriftMonitor(
        channel="test-ch",
        reference=ref,
        window_size=window_size,
        tick_interval=tick_interval,
        feature_drift_threshold=0.05,
        channel_drift_threshold=0.30,
    )


def _nominal_row(reference: pd.DataFrame, rng: np.random.Generator) -> dict[str, float]:
    """Sample one row from the reference distribution (no drift)."""
    idx = int(rng.integers(0, len(reference)))
    return {col: float(reference.iloc[idx][col]) for col in MONITORING_FEATURE_COLS}


def _drifted_row(reference: pd.DataFrame) -> dict[str, float]:
    """Return a row shifted far from the reference mean (+5 std)."""
    ref_mean = reference[MONITORING_FEATURE_COLS].mean()
    ref_std = reference[MONITORING_FEATURE_COLS].std().clip(lower=1e-6)
    return {col: float(ref_mean[col] + 5.0 * ref_std[col]) for col in MONITORING_FEATURE_COLS}


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
    ref = _make_reference(n_rows=500, seed=10)
    monitor = _make_monitor(reference=ref, window_size=_WINDOW_SIZE, tick_interval=_TICK_INTERVAL)
    # Push heavily shifted rows — all features displaced by 5 std from reference mean.
    for _ in range(_WINDOW_SIZE):
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
    ref = _make_reference(n_rows=500, seed=20)
    monitor = _make_monitor(reference=ref, window_size=_WINDOW_SIZE, tick_interval=_TICK_INTERVAL)
    rng = np.random.default_rng(21)
    for _ in range(_WINDOW_SIZE):
        monitor.push(_nominal_row(ref, rng))
    snapshot = await monitor.run()
    assert isinstance(snapshot, DriftSnapshot)
    assert snapshot.drifted is False
    assert snapshot.percent_drifted < 0.30
