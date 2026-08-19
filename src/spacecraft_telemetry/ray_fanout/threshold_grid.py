"""Post-hoc (threshold_z, min_error_value) grid sweep over saved error arrays.

Answers "how good could this arm get if its thresholds were tuned perfectly?"
without a GPU, a re-tune, or a forward pass — the scoring parameters are applied
*after* inference, so a saved ``errors.npy`` is enough to re-derive flags for any
(z, floor) pair.

Why this exists (docs/plans/021, stage 021.5b): plan 021's first result credited
multivariate with +0.061 mean segF0.5 over the univariate arm, but sweeping both
arms here showed the gap collapses to ~+0.020 at their respective ceilings —
most of the apparent architecture win was the univariate arm's HPO landing on a
worse operating point, inside a search space that contained a better one. A
comparison between two architectures is only meaningful once both are tuned to
comparable quality, and this makes that check cheap enough to be routine.

Also useful for the converse question (docs/plans/021 Validation): whether a
tuned value sitting near a search-space bound was genuinely truncated. Proximity
to a bound proves nothing; sweeping past it settles it for free.

**Scope limit — read before trusting a ceiling.** ``error_smoothing_window`` is
baked into the saved smoothed array and therefore CANNOT be varied here. A
ceiling from this module is conditional on the smoothing the run was scored
with; only a real re-tune equalizes that axis.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from spacecraft_telemetry.model.scoring import evaluate_overlap, flag_anomalies

# A (z, floor) pair and the mean objective it achieved.
GridPoint = tuple[float, float]


def precompute_threshold_terms(
    smoothed: np.ndarray[Any, Any], window: int
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Split :func:`model.scoring.dynamic_threshold` into its z-independent parts.

    ``dynamic_threshold`` computes ``(rolling_mean + z*rolling_std).shift(1)``,
    and ``shift`` is linear, so the two rolling terms can be computed ONCE and
    recombined for any z. That turns an N-point grid from N rolling passes over
    a 7.6M-element array into one — the difference between a sweep that takes
    minutes and one that takes an hour.

    Returns (shifted_mean, shifted_std); recombine via
    :func:`threshold_from_terms`, which reproduces ``dynamic_threshold``
    exactly (pinned by test).
    """
    s = pd.Series(np.asarray(smoothed, dtype=np.float64))
    rolling_mean = s.rolling(window, min_periods=1).mean()
    # ddof=0 and the fillna BEFORE shifting, matching dynamic_threshold's order.
    rolling_std = s.rolling(window, min_periods=1).std(ddof=0).fillna(0.0)
    return (
        rolling_mean.shift(1).to_numpy(dtype=np.float64),
        rolling_std.shift(1).to_numpy(dtype=np.float64),
    )


def threshold_from_terms(
    shifted_mean: np.ndarray[Any, Any], shifted_std: np.ndarray[Any, Any], z: float
) -> np.ndarray[Any, Any]:
    """Rebuild a dynamic threshold for one z from precomputed rolling terms.

    Position 0 has no history, so both terms are NaN there and the threshold is
    ``inf`` — never flags — exactly as ``dynamic_threshold`` does via
    ``.fillna(np.inf)``.
    """
    combined = shifted_mean + z * shifted_std
    result: np.ndarray[Any, Any] = np.where(np.isnan(combined), np.inf, combined)
    return result


def sweep_channel(
    smoothed: np.ndarray[Any, Any],
    labels: np.ndarray[Any, Any],
    *,
    threshold_window: int,
    min_run_length: int,
    z_values: list[float],
    floor_values: list[float],
    eval_slice: slice | None = None,
) -> dict[GridPoint, float]:
    """Return {(z, floor): seg_f0_5} for one channel over the grid.

    ``eval_slice`` selects the reported portion of the window axis — pass the
    same slice ``model.scoring.score_channel`` would use for the eval split
    being compared, or None for the full array. ``smoothed`` and ``labels``
    must be the same length (the caller is comparing a saved errors array
    against ``load_window_labels`` output; a mismatch means they came from
    different settings profiles).
    """
    if smoothed.shape != labels.shape:
        raise ValueError(
            f"smoothed{smoothed.shape} and labels{labels.shape} must have the same "
            "shape — a mismatch means they were produced by different "
            "window_size/prediction_horizon settings."
        )
    sl = eval_slice if eval_slice is not None else slice(None)
    shifted_mean, shifted_std = precompute_threshold_terms(smoothed, threshold_window)

    out: dict[GridPoint, float] = {}
    for z in z_values:
        threshold = threshold_from_terms(shifted_mean, shifted_std, z)
        for floor in floor_values:
            flags = flag_anomalies(smoothed, threshold, min_run_length, floor)
            out[(z, floor)] = evaluate_overlap(labels[sl], flags[sl])["seg_f0_5"]
    return out


def sweep_group(
    per_channel: dict[str, tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]],
    *,
    threshold_window: int,
    min_run_length: int,
    z_values: list[float],
    floor_values: list[float],
    eval_slice: slice | None = None,
) -> dict[GridPoint, float]:
    """Mean seg_f0_5 across a group of channels, per grid point.

    The mean over channels is the same aggregation ``run_hpo_sweep`` optimises,
    so a ceiling from here is directly comparable to a sweep's reported best.
    """
    if not per_channel:
        raise ValueError("per_channel is empty — nothing to sweep.")
    totals: dict[GridPoint, list[float]] = {}
    for smoothed, labels in per_channel.values():
        for point, score in sweep_channel(
            smoothed, labels,
            threshold_window=threshold_window,
            min_run_length=min_run_length,
            z_values=z_values,
            floor_values=floor_values,
            eval_slice=eval_slice,
        ).items():
            totals.setdefault(point, []).append(score)
    return {point: float(np.mean(scores)) for point, scores in totals.items()}


def best_point(grid: dict[GridPoint, float]) -> tuple[GridPoint, float]:
    """Return ((z, floor), score) for the grid's maximum.

    Ties break toward the LOWER z and LOWER floor — the more conservative
    detector of two equal-scoring configs, and deterministic regardless of dict
    ordering.
    """
    if not grid:
        raise ValueError("grid is empty.")
    best = max(grid.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))
    return best[0], best[1]


def bounds_report(
    grid: dict[GridPoint, float], z_values: list[float], floor_values: list[float]
) -> dict[str, Any]:
    """Flag whether the grid's optimum sits on an edge of the swept region.

    An optimum on the edge means the ceiling is a LOWER BOUND — the true
    optimum may lie outside what was swept, and the grid should be widened
    before the number is quoted. This is the same class of check the plan-021
    Validation section requires for search-space bounds, applied to the grid
    itself so the diagnostic cannot make the mistake it exists to catch.
    """
    (z, floor), score = best_point(grid)
    at_z_edge = z in (min(z_values), max(z_values))
    at_floor_edge = floor in (min(floor_values), max(floor_values))
    return {
        "best_z": z,
        "best_floor": floor,
        "best_score": score,
        "at_z_edge": at_z_edge,
        "at_floor_edge": at_floor_edge,
        "is_lower_bound": at_z_edge or at_floor_edge,
    }
