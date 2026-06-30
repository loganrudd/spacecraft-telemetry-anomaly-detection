"""Tests for ChannelInferenceEngine.prime().

Key assertions
--------------
1. After prime(window_size values) the next step() returns a non-None prediction.
2. prime + step produces identical output to a cold engine that stepped
   through all those values (state equivalence).
3. prime with fewer than window_size values does not produce a prediction on the
   next step (still in warmup).
4. prime resets EWMA / threshold state so a primed engine behaves identically
   to a fresh one, regardless of prior step() calls.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

torch = pytest.importorskip("torch")

from spacecraft_telemetry.api.inference import ChannelInferenceEngine  # noqa: E402
from spacecraft_telemetry.model.io import ScoringParams  # noqa: E402

# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------

W = 5       # window_size — small so tests are fast
TW = 8      # threshold_window
K = 3       # threshold_min_anomaly_len
SPAN = 4    # error_smoothing_window
Z = 1.0     # threshold_z

_PARAMS = ScoringParams(
    threshold_window=TW,
    threshold_z=Z,
    error_smoothing_window=SPAN,
    threshold_min_anomaly_len=K,
)

_TS = datetime(2000, 1, 1, tzinfo=UTC)
_DT = timedelta(seconds=30)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _ZeroModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], 1)


def _make_engine() -> ChannelInferenceEngine:
    m = _ZeroModel()
    m.eval()
    return ChannelInferenceEngine(
        mission="test",
        channel="test-ch",
        model=m,  # type: ignore[arg-type]
        window_size=W,
        params=_PARAMS,
        device=torch.device("cpu"),
    )


def _step_n(engine: ChannelInferenceEngine, values: list[float]) -> None:
    """Advance engine through *values* and discard results."""
    for i, v in enumerate(values):
        engine.step(v, _TS + i * _DT, False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_prime_full_window_enables_prediction() -> None:
    """prime() with window_size values causes the next step() to predict."""
    engine = _make_engine()
    engine.prime([0.0] * W)
    event = engine.step(0.0, _TS, False)
    assert event.prediction is not None, "expected non-None prediction after full prime"


def test_prime_partial_window_stays_in_warmup() -> None:
    """prime() with W-2 values: one step() gives W-1 total, still in warmup."""
    engine = _make_engine()
    engine.prime([0.0] * (W - 2))  # two short; after one step still W-1 total
    event = engine.step(0.0, _TS, False)
    assert event.prediction is None, "expected None prediction (insufficient prime)"


def test_prime_matches_cold_engine_step() -> None:
    """prime(history) + step(v) == cold engine.step through same history + v.

    A primed engine starts with the last window_size values pre-loaded.
    Stepping it with one new value must produce the same output as a cold
    engine that stepped through the entire sequence from scratch.
    """
    history = [float(i % 5) for i in range(W + TW + K + 5)]
    new_value = 3.7
    ts = _TS + len(history) * _DT

    # Cold engine — step through history then add new_value.
    cold = _make_engine()
    _step_n(cold, history)
    expected = cold.step(new_value, ts, False)

    # Primed engine — pre-loaded with last W values of history, then step.
    primed = _make_engine()
    primed.prime(history)
    actual = primed.step(new_value, ts, False)

    assert actual.prediction == pytest.approx(expected.prediction), "prediction mismatch"
    assert actual.value_normalized == pytest.approx(expected.value_normalized)


def test_prime_resets_ewma_regardless_of_prior_state() -> None:
    """prime() resets EWMA so a dirty engine behaves identically to a fresh one.

    A primed engine that previously stepped through a spike must give the
    same output as a fresh primed engine.
    """
    history = [0.0] * (W + TW)

    # Dirty engine: stepped through a spike to dirty the EWMA state.
    dirty = _make_engine()
    _step_n(dirty, [100.0] * (W + TW + K))  # spike — elevated _s_prev
    dirty.prime(history)

    # Fresh engine: just primed.
    fresh = _make_engine()
    fresh.prime(history)

    new_value = 1.5
    ts = _TS + len(history) * _DT
    e_dirty = dirty.step(new_value, ts, False)
    e_fresh = fresh.step(new_value, ts, False)

    assert e_dirty.prediction == pytest.approx(e_fresh.prediction), (
        "prediction differs after prime on dirty vs fresh engine"
    )
    assert e_dirty.smoothed_error == pytest.approx(e_fresh.smoothed_error), (
        "smoothed_error differs after prime on dirty vs fresh engine"
    )


def test_prime_more_than_window_size_uses_last_w_values() -> None:
    """prime() with more values than window_size uses only the last window_size."""
    history_long = [float(i) for i in range(W + 10)]
    history_tail = history_long[-W:]

    engine_long = _make_engine()
    engine_long.prime(history_long)

    engine_tail = _make_engine()
    engine_tail.prime(history_tail)

    new_value = 0.5
    ts = _TS
    e_long = engine_long.step(new_value, ts, False)
    e_tail = engine_tail.step(new_value, ts, False)

    assert e_long.prediction == pytest.approx(e_tail.prediction), (
        "prime with long history should behave like prime with just the tail"
    )
