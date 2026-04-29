"""Tests for model.scoring — pure-numpy threshold functions (no torch required)."""

from __future__ import annotations

import numpy as np
import pytest

from spacecraft_telemetry.model.scoring import (
    dynamic_threshold,
    evaluate,
    flag_anomalies,
    smooth_errors,
)

# ---------------------------------------------------------------------------
# smooth_errors
# ---------------------------------------------------------------------------


def test_smooth_errors_is_ewma() -> None:
    """Verify against the recursive EWMA formula with span=2 (alpha=2/3).

    alpha = 2 / (span + 1) = 2/3
    s[0] = 1
    s[1] = 2/3*2 + 1/3*1  = 5/3
    s[2] = 2/3*3 + 1/3*5/3 = 23/9
    s[3] = 2/3*4 + 1/3*23/9 = 95/27
    s[4] = 2/3*5 + 1/3*95/27 = 365/81
    """
    errors = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = smooth_errors(errors, span=2)
    expected = np.array([1, 5 / 3, 23 / 9, 95 / 27, 365 / 81])
    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_smooth_errors_uses_absolute_values() -> None:
    """Negative errors should produce the same result as their absolute counterparts."""
    pos = smooth_errors(np.array([1.0, 2.0, 3.0]), span=3)
    neg = smooth_errors(np.array([-1.0, -2.0, -3.0]), span=3)
    np.testing.assert_allclose(pos, neg)


def test_smooth_errors_single_element() -> None:
    result = smooth_errors(np.array([5.0]), span=10)
    assert result.shape == (1,)
    np.testing.assert_allclose(result[0], 5.0)


# ---------------------------------------------------------------------------
# dynamic_threshold
# ---------------------------------------------------------------------------


def test_dynamic_threshold_first_entry_is_inf() -> None:
    """Position 0 has no history — threshold must be inf so it never flags."""
    smoothed = np.ones(10, dtype=np.float64)
    thresh = dynamic_threshold(smoothed, window=5, z=3.0)
    assert thresh[0] == np.inf, f"Expected inf at index 0, got {thresh[0]}"


def test_dynamic_threshold_no_nan_after_warmup() -> None:
    """No NaN values anywhere in the output (warmup uses min_periods=1)."""
    smoothed = np.random.default_rng(0).standard_normal(50)
    thresh = dynamic_threshold(smoothed, window=10, z=3.0)
    assert not np.any(np.isnan(thresh)), "Unexpected NaN in threshold array"


def test_dynamic_threshold_shape_matches_input() -> None:
    smoothed = np.ones(20)
    thresh = dynamic_threshold(smoothed, window=5, z=2.0)
    assert thresh.shape == smoothed.shape


def test_dynamic_threshold_steady_state_uses_previous_window() -> None:
    """After warmup, threshold at t must be based on smoothed[t-window : t], not t."""
    # Constant background then a spike at position 15; threshold at 15 should
    # be computed from positions 5..14 (constant background), so the spike
    # should NOT contaminate its own threshold.
    smoothed = np.zeros(30, dtype=np.float64)
    smoothed[15] = 100.0  # large spike
    thresh = dynamic_threshold(smoothed, window=10, z=3.0)
    # threshold[15] is based on smoothed[5:15] which are all 0 → threshold ≈ 0
    assert thresh[15] < 1.0, (
        f"threshold[15]={thresh[15]:.4f} should reflect background (0), not the spike"
    )


# ---------------------------------------------------------------------------
# flag_anomalies
# ---------------------------------------------------------------------------


def test_flag_anomalies_drops_short_runs() -> None:
    # Run of 1 at index 2; run of 5 at indices 5-9.
    smoothed = np.array([0, 0, 2, 0, 0, 2, 2, 2, 2, 2], dtype=np.float64)
    threshold = np.ones(10, dtype=np.float64)
    flags = flag_anomalies(smoothed, threshold, min_run_length=3)

    assert not flags[2], "Run of length 1 should be dropped"
    assert np.all(flags[5:10]), "Run of length 5 should be kept"
    assert not np.any(flags[:2]), "Background should not be flagged"
    assert not flags[3], "Background should not be flagged"
    assert not flags[4], "Background should not be flagged"


def test_flag_anomalies_keeps_exact_min_length_run() -> None:
    # Run of exactly min_run_length=3 at indices 1-3.
    smoothed = np.array([0, 2, 2, 2, 0], dtype=np.float64)
    threshold = np.ones(5)
    flags = flag_anomalies(smoothed, threshold, min_run_length=3)
    assert np.all(flags[1:4])
    assert not flags[0]
    assert not flags[4]


def test_flag_anomalies_all_below_threshold() -> None:
    smoothed = np.zeros(10)
    threshold = np.ones(10)
    flags = flag_anomalies(smoothed, threshold, min_run_length=1)
    assert not np.any(flags)


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


def test_evaluate_known_confusion() -> None:
    """Hand-computed TP/FP/FN: TP=2, FP=1, FN=1 → precision=recall=2/3."""
    true = np.array([True, True, False, False, True, False])
    pred = np.array([True, False, True, False, True, False])
    # TP=2 (idx 0,4), FP=1 (idx 2), FN=1 (idx 1)
    metrics = evaluate(true, pred)

    assert abs(metrics["precision"] - 2 / 3) < 1e-9
    assert abs(metrics["recall"] - 2 / 3) < 1e-9
    assert abs(metrics["f1"] - 2 / 3) < 1e-9

    # F0.5 with beta=0.5, beta²=0.25, p=r=2/3:
    # = 1.25 * (2/3)² / (0.25*(2/3) + 2/3) = 1.25*(4/9) / (5/6) = (5/9)*(6/5) = 2/3
    assert abs(metrics["f0_5"] - 2 / 3) < 1e-9


def test_evaluate_all_correct() -> None:
    true = np.array([True, True, False, False])
    metrics = evaluate(true, true)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["f0_5"] == 1.0


def test_evaluate_no_positive_predictions() -> None:
    """No predictions → precision=0, recall=0, f-scores=0 (no ZeroDivision)."""
    true = np.array([True, False, True])
    pred = np.array([False, False, False])
    metrics = evaluate(true, pred)
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0
    assert metrics["f0_5"] == 0.0


def test_evaluate_no_actual_positives() -> None:
    """All nominal ground truth → recall=0 (no ZeroDivision)."""
    true = np.array([False, False, False])
    pred = np.array([True, False, True])
    metrics = evaluate(true, pred)
    assert metrics["recall"] == 0.0


# ---------------------------------------------------------------------------
# Integration: dynamic_threshold detects an injected spike
# ---------------------------------------------------------------------------


def test_dynamic_threshold_detects_injected_spike() -> None:
    """A 3-point spike of 20× background std must be flagged after warm-up."""
    rng = np.random.default_rng(42)
    n = 200
    background_std = 0.1
    errors = rng.normal(0, background_std, n)

    # Inject spike at indices 100–102 (well past warm-up window of 30).
    errors[100:103] = 20 * background_std

    smoothed = smooth_errors(errors, span=10)
    threshold = dynamic_threshold(smoothed, window=30, z=2.0)
    flags = flag_anomalies(smoothed, threshold, min_run_length=1)

    assert np.any(flags[99:104]), (
        "Injected spike at indices 100–102 was not detected by flag_anomalies"
    )
