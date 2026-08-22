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

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from spacecraft_telemetry.model.scoring import evaluate_overlap, flag_anomalies

# A (z, floor) pair and the mean objective it achieved.
GridPoint = tuple[float, float]

# (lower, upper) bound each axis physically cannot cross, exclusive. `None`
# means unbounded in that direction. threshold_z must stay > 0 (a
# non-positive z-score threshold is meaningless); min_error_value is an
# absolute error floor and cannot go negative. Both are conventionally
# written as (0.0, None) here — see ray_fanout.threshold_search._axis_edges'
# natural-bound check for how the 0.0 boundary is treated identically
# regardless of whether the underlying parameter is strictly-positive
# (threshold_z) or zero-inclusive (min_error_value): either way, there is
# nothing to widen into below it. Lives here, not in threshold_search.py,
# because that driver is axis-name-generic and must not carry an implicit
# opinion about these two specific axes (docs/reviews/022, item A1).
NATURAL_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    "threshold_z": (0.0, None),
    "min_error_value": (0.0, None),
}


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
    prepared: tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]] | None = None,
) -> dict[GridPoint, float]:
    """Return {(z, floor): seg_f0_5} for one channel over the grid.

    ``eval_slice`` selects the reported portion of the window axis — pass the
    same slice ``model.scoring.score_channel`` would use for the eval split
    being compared, or None for the full array. ``smoothed`` and ``labels``
    must be the same length (the caller is comparing a saved errors array
    against ``load_window_labels`` output; a mismatch means they came from
    different settings profiles).

    ``prepared``: this channel's precomputed ``(shifted_mean, shifted_std)``
    from :func:`precompute_threshold_terms`, when a caller already has it
    (e.g. a widening driver sweeping the same channel across several rounds
    — see ``ray_fanout.threshold_search``). Recomputed here when ``None``,
    exactly as before this parameter existed — additive, not a behaviour
    change on the default path (docs/reviews/022, item P2).
    """
    if smoothed.shape != labels.shape:
        raise ValueError(
            f"smoothed{smoothed.shape} and labels{labels.shape} must have the same "
            "shape — a mismatch means they were produced by different "
            "window_size/prediction_horizon settings."
        )
    sl = eval_slice if eval_slice is not None else slice(None)
    shifted_mean, shifted_std = (
        prepared
        if prepared is not None
        else precompute_threshold_terms(smoothed, threshold_window)
    )

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
    prepared: dict[str, tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]] | None = None,
) -> dict[GridPoint, float]:
    """Mean seg_f0_5 across a group of channels, per grid point.

    The mean over channels is the same aggregation ``run_hpo_sweep`` optimises,
    so a ceiling from here is directly comparable to a sweep's reported best.

    ``prepared``: per-channel precomputed rolling terms, forwarded to
    :func:`sweep_channel` — see its docstring (docs/reviews/022, item P2).
    """
    if not per_channel:
        raise ValueError("per_channel is empty — nothing to sweep.")
    totals: dict[GridPoint, list[float]] = {}
    for channel, (smoothed, labels) in per_channel.items():
        for point, score in sweep_channel(
            smoothed, labels,
            threshold_window=threshold_window,
            min_run_length=min_run_length,
            z_values=z_values,
            floor_values=floor_values,
            eval_slice=eval_slice,
            prepared=prepared.get(channel) if prepared is not None else None,
        ).items():
            totals.setdefault(point, []).append(score)
    return {point: float(np.mean(scores)) for point, scores in totals.items()}


def sweep_group_mission_level(
    per_channel: dict[str, tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]],
    channel_timestamps: dict[str, np.ndarray[Any, Any]],
    mission_events: Any,
    mission_timeline: Any,
    *,
    threshold_window: int,
    min_run_length: int,
    z_values: list[float],
    floor_values: list[float],
    eval_slice: slice | None = None,
    prepared: dict[str, tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]] | None = None,
) -> dict[GridPoint, float]:
    """Grid sweep scored on MISSION-LEVEL corrected event-wise F0.5.

    :func:`sweep_group` optimises mean per-channel seg_f0_5, which is what Ray
    Tune optimises — and which **diverges sharply** from the mission-level
    metric the ESA-ADB report actually publishes. Measured on plan 021's arms:
    a config that maximised per-channel seg_f0_5 drove mission-level detections
    from 33 to 112 and F0.5 from 0.376 to 0.123, because per-channel-optimal
    thresholds fire on correlated events that compound under OR-aggregation.
    Selecting on the reported metric requires scoring on the reported metric.

    Reproduces the same aggregation ray_fanout.tune's trial function performs
    for its observational mission_f0_5 (stage 021.4b): per-channel flags ->
    time intervals -> OR-aggregated mission detections -> corrected_event_wise
    over the "all_events" scope (excludes only Communication Gap, the paper's
    primary table).

    Costs interval math per grid point rather than array ops, so it is
    materially slower than :func:`sweep_group` — seed it from a coarse
    neighbourhood around that sweep's optimum rather than sweeping wide.

    ``channel_timestamps`` must be index-aligned with each channel's arrays in
    ``per_channel`` and sliced identically by ``eval_slice``.

    ``prepared``: per-channel precomputed ``(shifted_mean, shifted_std)``,
    when a caller already has it — e.g. a widening driver, which otherwise
    re-pays this rolling pass over a multi-million-element array on every
    round (docs/reviews/022, item P2). Recomputed internally when ``None``,
    exactly as before this parameter existed.
    """
    from spacecraft_telemetry.esa_adb.detections import (
        _flags_to_intervals,
        mission_intervals_from_per_channel,
    )
    from spacecraft_telemetry.esa_adb.metrics import corrected_event_wise

    if not per_channel:
        raise ValueError("per_channel is empty — nothing to sweep.")
    missing = set(per_channel) - set(channel_timestamps)
    if missing:
        raise ValueError(
            f"channel_timestamps missing for {sorted(missing)} — every channel "
            "must supply timestamps, or its detections silently vanish from the "
            "OR-aggregation and the mission metric is computed on a subset."
        )

    sl = eval_slice if eval_slice is not None else slice(None)
    terms: dict[str, tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]] = (
        prepared
        if prepared is not None
        else {
            channel: precompute_threshold_terms(smoothed, threshold_window)
            for channel, (smoothed, _labels) in per_channel.items()
        }
    )

    out: dict[GridPoint, float] = {}
    for z in z_values:
        thresholds = {
            ch: threshold_from_terms(mean, std, z) for ch, (mean, std) in terms.items()
        }
        for floor in floor_values:
            per_channel_intervals = {}
            for channel, (smoothed, _labels) in per_channel.items():
                flags = flag_anomalies(smoothed, thresholds[channel], min_run_length, floor)
                per_channel_intervals[channel] = _flags_to_intervals(
                    flags[sl], channel_timestamps[channel][sl]
                )
            detections = mission_intervals_from_per_channel(per_channel_intervals)
            cew = corrected_event_wise(
                mission_events, detections, mission_timeline,
                excluded_categories=frozenset({"Communication Gap"}),
            )
            out[(z, floor)] = float(cew["f_beta"])
    return out


def threshold_grid_sweep_fn(
    sweep: Callable[..., dict[GridPoint, float]], **fixed_kwargs: object
) -> Callable[[dict[str, list[float]]], dict[GridPoint, float]]:
    """Adapt a sweep function in this module to widen_to_convergence's interface.

    Binds ``axes["threshold_z"]``/``axes["min_error_value"]`` onto
    :func:`sweep_group`'s (or :func:`sweep_group_mission_level`'s)
    ``z_values=``/``floor_values=`` keywords. Both those functions already
    return a grid keyed by ``(z, floor)`` tuples, which is exactly
    ``ray_fanout.threshold_search.widen_to_convergence``'s expected shape
    PROVIDED axes are passed in ``{"threshold_z": ..., "min_error_value":
    ...}`` order — enforced below rather than trusted, since a silent
    transposition would pair z with floor's grid position and vice versa.

    Lives here, not in ``threshold_search.py``, since it is a domain
    adapter that hardcodes THIS module's keyword names and tuple order —
    the generic driver it feeds carries no knowledge of either
    (docs/reviews/022, item A3).

    Args:
        sweep: :func:`sweep_group` or :func:`sweep_group_mission_level`.
        **fixed_kwargs: Everything else those functions need
            (``per_channel``, ``threshold_window``, ``min_run_length``,
            ``eval_slice``, and for the mission-level variant
            ``channel_timestamps``/``mission_events``/``mission_timeline``).
    """

    def _sweep(axes: dict[str, list[float]]) -> dict[GridPoint, float]:
        if list(axes) != ["threshold_z", "min_error_value"]:
            raise ValueError(
                "threshold_grid_sweep_fn requires axes in "
                "{'threshold_z': ..., 'min_error_value': ...} order to match "
                f"this module's (z, floor) tuple keys; got {list(axes)}."
            )
        return sweep(
            z_values=axes["threshold_z"],
            floor_values=axes["min_error_value"],
            **fixed_kwargs,
        )

    return _sweep
