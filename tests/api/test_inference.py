"""Contract tests for api.inference.ChannelInferenceEngine.

The primary assertion is that the online engine produces smoothed_error and
threshold values that are byte-identical (to floating-point precision) to the
offline pipeline (smooth_errors / dynamic_threshold in model/scoring.py) past
warmup.

Series design
-------------
500 ticks.  Zero-prediction stub model (always returns 0).
Values: 0.0 for ticks 0-199, 100.0 for ticks 200-219, 0.0 for ticks 220-499.
With small W/Tw the spike region is well past warmup, giving clean assertions.

Parameters (small to keep tests fast)
--------------------------------------
  W   = window_size              = 5
  Tw  = threshold_window         = 8
  K   = threshold_min_anomaly_len = 3
  span = error_smoothing_window  = 4
  z   = threshold_z              = 1.0
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from spacecraft_telemetry.api.inference import (  # noqa: E402
    ChannelInferenceEngine,
    TelemetryEvent,
)
from spacecraft_telemetry.model.io import ScoringParams  # noqa: E402
from spacecraft_telemetry.model.scoring import (  # noqa: E402
    dynamic_threshold,
    smooth_errors,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

W = 5        # window_size
TW = 8       # threshold_window
K = 3        # threshold_min_anomaly_len
SPAN = 4     # error_smoothing_window
Z = 1.0      # threshold_z

N = 500        # total ticks
SPIKE_START = 200
SPIKE_END = 220  # exclusive — ticks 200..219 are the anomaly

# Derived warmup boundaries (0-indexed tick positions in events list)
# First prediction fires when len(window_buf) == W, i.e. at tick W-1.
PRED_START = W - 1          # first tick with prediction (= first smoothed error)
THRESH_START = PRED_START + TW  # first tick with non-None threshold

_PARAMS = ScoringParams(
    threshold_window=TW,
    threshold_z=Z,
    error_smoothing_window=SPAN,
    threshold_min_anomaly_len=K,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _ZeroModel(torch.nn.Module):
    """Stub that always predicts 0 regardless of input."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], 1)


def _make_series(n: int = N) -> np.ndarray:
    values = np.zeros(n, dtype=np.float32)
    values[SPIKE_START:SPIKE_END] = 100.0
    return values


def _make_timestamps(n: int = N) -> list[datetime]:
    base = datetime(2000, 1, 1, tzinfo=UTC)
    return [base + timedelta(seconds=i) for i in range(n)]


def _build_engine(model: torch.nn.Module | None = None) -> ChannelInferenceEngine:
    m = model or _ZeroModel()
    m.eval()
    return ChannelInferenceEngine(
        mission="test-mission",
        channel="test-channel",
        model=m,  # type: ignore[arg-type]
        window_size=W,
        params=_PARAMS,
        device=torch.device("cpu"),
    )


def _run_engine(
    engine: ChannelInferenceEngine,
    values: np.ndarray,
    timestamps: list[datetime],
    is_anomaly: np.ndarray | None = None,
) -> list[TelemetryEvent]:
    if is_anomaly is None:
        is_anomaly = np.zeros(len(values), dtype=bool)
    return [
        engine.step(float(v), ts, bool(a))
        for v, ts, a in zip(values, timestamps, is_anomaly, strict=False)
    ]


def _offline_pipeline(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run offline smooth_errors + dynamic_threshold on a zero-model residual.

    Returns (smoothed, threshold) each of shape (N - W,) aligned so that
    index 0 corresponds to the first post-model-warmup prediction.
    """
    # Zero model: e_t = |values[t] - 0| = |values[t]|
    # The offline pipeline operates on window residuals, but since the model
    # always predicts 0, e_t = |target_t| regardless of window alignment.
    # We align to post-warmup: first prediction is for window 0..W-1, target = values[W-1].
    targets = values[W - 1 :]  # shape (N - W + 1,)  — last element of each window
    errors = targets - 0.0  # zero model
    smoothed = smooth_errors(errors, span=SPAN)
    threshold = dynamic_threshold(smoothed, window=TW, z=Z)
    return smoothed, threshold


# ---------------------------------------------------------------------------
# Warmup behaviour
# ---------------------------------------------------------------------------


def test_first_w_events_have_none_prediction() -> None:
    """First W-1 ticks must have prediction=None (model warmup).

    The first prediction fires when len(window_buf) == W, which happens at
    tick index W-1 (0-indexed), so ticks 0..W-2 are in warmup.
    """
    engine = _build_engine()
    values = _make_series()
    timestamps = _make_timestamps()
    events = _run_engine(engine, values, timestamps)

    for i, event in enumerate(events[:PRED_START]):
        assert event.prediction is None, f"tick {i}: expected None prediction"
        assert event.residual is None, f"tick {i}: expected None residual"
        assert event.smoothed_error is None, f"tick {i}: expected None smoothed_error"
        assert event.threshold is None, f"tick {i}: expected None threshold"
        assert event.is_anomaly_predicted is False, f"tick {i}: expected False flag"


def test_next_tw_events_have_none_threshold() -> None:
    """Ticks PRED_START through THRESH_START-1 have non-None smoothed_error but None threshold.

    There are exactly TW such ticks (the threshold warmup phase).
    """
    engine = _build_engine()
    values = _make_series()
    timestamps = _make_timestamps()
    events = _run_engine(engine, values, timestamps)

    for i, event in enumerate(events[PRED_START:THRESH_START]):
        tick = PRED_START + i
        assert event.prediction is not None, f"tick {tick}: expected non-None prediction"
        assert event.smoothed_error is not None, f"tick {tick}: expected non-None smoothed_error"
        assert event.threshold is None, f"tick {tick}: expected None threshold (TW warmup)"
        assert event.is_anomaly_predicted is False, f"tick {tick}: expected False flag (warmup)"


def test_post_warmup_events_have_all_fields() -> None:
    """All fields are non-None from tick THRESH_START (= W+TW-1) onwards."""
    engine = _build_engine()
    values = _make_series()
    timestamps = _make_timestamps()
    events = _run_engine(engine, values, timestamps)

    for i, event in enumerate(events[THRESH_START:]):
        tick = THRESH_START + i
        assert event.prediction is not None, f"tick {tick}: expected non-None prediction"
        assert event.smoothed_error is not None, f"tick {tick}: expected non-None smoothed_error"
        assert event.threshold is not None, f"tick {tick}: expected non-None threshold"


# ---------------------------------------------------------------------------
# Numerical correctness: smoothed_error matches offline pipeline
# ---------------------------------------------------------------------------


def test_online_smoothed_error_matches_offline() -> None:
    """smoothed_error values must match offline smooth_errors to 1e-9 atol.

    online events[PRED_START:] align with offline_smoothed[0:] — both have
    N - W + 1 elements, where PRED_START = W - 1.
    """
    engine = _build_engine()
    values = _make_series()
    timestamps = _make_timestamps()
    events = _run_engine(engine, values, timestamps)

    offline_smoothed, _ = _offline_pipeline(values)

    online_smoothed = np.array(
        [e.smoothed_error for e in events[PRED_START:]], dtype=np.float64
    )
    np.testing.assert_allclose(
        online_smoothed, offline_smoothed, atol=1e-9,
        err_msg="online smoothed_error diverges from offline smooth_errors",
    )


def test_online_threshold_matches_offline() -> None:
    """threshold values must match offline dynamic_threshold past full warmup.

    First non-None threshold is at online event THRESH_START (= W+TW-1),
    aligning with offline_threshold[TW] — both have N - W - TW + 1 elements.
    """
    engine = _build_engine()
    values = _make_series()
    timestamps = _make_timestamps()
    events = _run_engine(engine, values, timestamps)

    _, offline_threshold = _offline_pipeline(values)

    online_threshold = np.array(
        [e.threshold for e in events[THRESH_START:]], dtype=np.float64
    )
    # atol=1e-5: pandas rolling variance (Welford's algorithm) vs numpy's naive
    # std accumulate different floating-point rounding; 1e-5 is tight enough
    # to confirm correctness while tolerating algorithm-level FP differences.
    np.testing.assert_allclose(
        online_threshold, offline_threshold[TW:], atol=1e-5,
        err_msg="online threshold diverges from offline dynamic_threshold",
    )


# ---------------------------------------------------------------------------
# Anomaly flag: K-trailing property
# ---------------------------------------------------------------------------


def _trailing_flag(raw: np.ndarray, k: int) -> np.ndarray:
    """Compute K-trailing anomaly flags: True at t iff raw[t-k+1:t+1] all True."""
    result = np.zeros(len(raw), dtype=bool)
    for t in range(k - 1, len(raw)):
        if np.all(raw[t - k + 1 : t + 1]):
            result[t] = True
    return result


def test_online_anomaly_flag_matches_trailing_edge() -> None:
    """is_anomaly_predicted must equal K-trailing flag over offline raw flags.

    The online flag is True at tick t iff the last K raw flags (s_t > thresh)
    are all True, which is a trailing-edge variant of flag_anomalies with a
    (K-1)-tick leading lag vs the offline result.
    """
    engine = _build_engine()
    values = _make_series()
    timestamps = _make_timestamps()
    events = _run_engine(engine, values, timestamps)

    offline_smoothed, offline_threshold = _offline_pipeline(values)

    # Compute offline raw flags (no min-run-length filter).
    offline_raw = offline_smoothed > offline_threshold  # shape (N - W + 1,)

    # Trailing-edge flags over offline raw (matches online K-trailing logic).
    expected = _trailing_flag(offline_raw, K)  # shape (N - W + 1,)

    # Online flags aligned to post-model-warmup region (events[PRED_START:]).
    online_flags = np.array(
        [e.is_anomaly_predicted for e in events[PRED_START:]], dtype=bool
    )

    # Compare from TW onwards (past threshold warmup; both are False before).
    np.testing.assert_array_equal(
        online_flags[TW:], expected[TW:],
        err_msg="online is_anomaly_predicted diverges from K-trailing offline flags",
    )


def test_anomaly_detected_in_spike_region() -> None:
    """The spike region must trigger at least K consecutive True raw flags."""
    engine = _build_engine()
    values = _make_series()
    timestamps = _make_timestamps()
    events = _run_engine(engine, values, timestamps)

    # Spike is at original ticks 200-219; offset W for post-warmup alignment.
    spike_events = events[SPIKE_START + W - 1 : SPIKE_END + W - 1]
    any_anomaly = any(e.is_anomaly_predicted for e in spike_events)
    assert any_anomaly, "expected at least one anomaly_predicted=True in spike region"


# ---------------------------------------------------------------------------
# Event structure
# ---------------------------------------------------------------------------


def test_step_returns_telemetry_event() -> None:
    """step() must return a TelemetryEvent Pydantic model."""
    engine = _build_engine()
    ts = datetime(2000, 1, 1, tzinfo=UTC)
    for _ in range(W + TW):
        event = engine.step(0.0, ts, False)
    assert isinstance(event, TelemetryEvent)


def test_event_mission_channel_propagated() -> None:
    """mission and channel fields must match the engine constructor args."""
    engine = ChannelInferenceEngine(
        mission="ESA-Mission1",
        channel="channel_42",
        model=_ZeroModel(),  # type: ignore[arg-type]
        window_size=W,
        params=_PARAMS,
        device=torch.device("cpu"),
    )
    ts = datetime(2000, 1, 1, tzinfo=UTC)
    event = engine.step(0.0, ts, False)
    assert event.mission == "ESA-Mission1"
    assert event.channel == "channel_42"


def test_event_value_normalized_propagated() -> None:
    """value_normalized must equal the input value passed to step()."""
    engine = _build_engine()
    ts = datetime(2000, 1, 1, tzinfo=UTC)
    event = engine.step(3.14, ts, False)
    assert event.value_normalized == pytest.approx(3.14)


def test_event_is_anomaly_true_propagated() -> None:
    """is_anomaly_true must reflect the label passed to step()."""
    engine = _build_engine()
    ts = datetime(2000, 1, 1, tzinfo=UTC)
    assert engine.step(0.0, ts, True).is_anomaly_true is True
    assert engine.step(0.0, ts, False).is_anomaly_true is False


def test_event_serialisable_to_json() -> None:
    """TelemetryEvent must serialise to JSON via Pydantic model_dump_json()."""
    engine = _build_engine()
    ts = datetime(2000, 1, 1, tzinfo=UTC)
    event = engine.step(0.0, ts, False)
    json_str = event.model_dump_json()
    assert "test-mission" in json_str
    assert "test-channel" in json_str
