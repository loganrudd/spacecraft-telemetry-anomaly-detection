"""Mechanical grid-widening driver for threshold selection (docs/plans/022, stage 022.1).

Replaces the hand-driven loop in ``scripts/threshold_ceiling.py``: sweep a grid,
read the "optimum sits on a GRID EDGE" warning, widen the range by eye, re-run.
That loop ran four times during plan 021.7 alone, each time with ranges chosen
by a human at a keyboard — the same kind of ad-hoc tuning that made 021.5's
first result wrong by 3x. This module makes the widening mechanical.

**Generalizes at the driver boundary, not in the swept primitives.**
:mod:`spacecraft_telemetry.ray_fanout.threshold_grid` is pinned bit-for-bit
against ``outputs/mission_grid_h10_first.json`` and is never imported here —
this module has zero domain coupling. Its ``threshold_z``/``min_error_value``
bounds (:data:`threshold_grid.NATURAL_BOUNDS`) and the ``threshold_grid_sweep_fn``
adapter that binds this driver's axis-name-generic ``dict[str, list[float]]``
interface onto threshold_grid's fixed ``z_values=``/``floor_values=`` keywords
both live in :mod:`threshold_grid` instead (docs/reviews/022, items A1/A3) —
callers pass them in explicitly rather than this module reaching for them by
name.

The driver itself (:func:`widen_to_convergence`) knows nothing about
``threshold_z`` or ``min_error_value`` — it operates on whatever axes,
``natural_bounds``, and ``sweep_fn`` it is given, in the axis order the caller
supplies. That is what lets DC-VAE (plan 023) reuse it with a different
``sweep_fn``, bounds, and axis set, paying only for a second thin adapter
rather than a rewrite.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from spacecraft_telemetry.core.logging import get_logger

log = get_logger(__name__)

# A point in axis-order-many dimensions and the mean objective it achieved.
# Matches ray_fanout.threshold_grid.GridPoint's shape (a plain tuple of
# floats) generalized past exactly two axes.
GridPoint = tuple[float, ...]
SweepFn = Callable[[dict[str, list[float]]], dict[GridPoint, float]]


class NonConvergenceError(RuntimeError):
    """Raised when the optimum still sits on a grid edge after max_expansions.

    Never caught internally — a caller must not silently accept a ceiling
    that is actually a lower bound. Widen ``n_expand``/``max_expansions`` or
    inspect why the objective keeps improving toward an axis's edge.
    """


@dataclass(frozen=True)
class WideningResult:
    """Outcome of :func:`widen_to_convergence`.

    Always describes an interior (or natural-bound) optimum: a non-convergent
    result raises :class:`NonConvergenceError` instead of returning one whose
    optimum is a grid-edge lower bound — there is no partially-successful
    return to represent, which is why this carries no ``interior`` flag
    (docs/reviews/022, item C2 — the field could only ever be ``True``).

    Attributes:
        grid:        Every swept point, original and widened, keyed by
            axis-order tuples. Original points are present at their
            original values — widening only adds keys, never recomputes one.
        axes:        The (possibly widened) axis value lists, keyed by name,
            in the same order used to build ``grid``'s tuple keys.
        axis_order:  ``list(axes)`` at call time — the tuple-position ->
            axis-name mapping for ``grid`` and ``best_point``.
        best_point:  The winning axis-order tuple.
        best_score:  Its objective value.
        expansions:  Number of widening rounds performed (0 if the initial
            sweep's optimum was already interior or sat at a natural bound).
    """

    grid: dict[GridPoint, float]
    axes: dict[str, list[float]]
    axis_order: list[str]
    best_point: GridPoint
    best_score: float
    expansions: int


def best_point(grid: dict[GridPoint, float]) -> tuple[GridPoint, float]:
    """Return (point, score) for the grid's maximum.

    Ties break toward the LOWER value on every axis, in axis order — the
    more conservative point among equal-scoring configs, generalizing
    threshold_grid's now-deleted (lower z, lower floor) tie-break to however
    many axes are in play, deterministic regardless of dict ordering.
    """
    if not grid:
        raise ValueError("grid is empty.")
    best = max(grid.items(), key=lambda kv: (kv[1], *(-v for v in kv[0])))
    return best[0], best[1]


def _axis_edges(
    best_point: GridPoint,
    axes: dict[str, list[float]],
    axis_order: list[str],
    natural_bounds: dict[str, tuple[float | None, float | None]],
) -> dict[str, str]:
    """Return {axis_name: "low"|"high"} for axes needing expansion.

    An axis whose best value sits at the low/high edge of its current swept
    range is a candidate — UNLESS that edge already sits at (or past) the
    axis's natural bound in that direction, in which case there is nothing
    to widen into and the axis converges silently (not included in the
    returned dict).
    """
    edges: dict[str, str] = {}
    for i, name in enumerate(axis_order):
        values = axes[name]
        lo, hi = min(values), max(values)
        value = best_point[i]
        lower_bound, upper_bound = natural_bounds.get(name, (None, None))
        if value == lo and (lower_bound is None or lo > lower_bound):
            edges[name] = "low"
        elif value == hi and (upper_bound is None or hi < upper_bound):
            edges[name] = "high"
    return edges


def _expansion_points(
    direction: str,
    values: list[float],
    n_expand: int,
    bound: float | None,
) -> list[float]:
    """New points extending ``values`` by up to ``n_expand``, spaced by the
    interval between the two points nearest the edge being widened.

    Stops early (fewer than ``n_expand`` points) once a candidate would
    cross ``bound`` — the caller never receives a point beyond the
    parameter's natural bound. The bound itself is emitted once, as the
    last point, if not already present in ``values``: ``bound`` is a
    legitimate, reachable setting (e.g. ``min_error_value=0.0``, "no
    absolute error floor") and excluding it from every grid that doesn't
    happen to start there would make it structurally unreachable (docs/
    reviews/022, item C1). ``bound`` is exclusive for the purpose of
    treating an edge value AT the bound as converged (see ``_axis_edges``),
    but inclusive here, for the purpose of sweeping it.
    """
    ordered = sorted(values)
    if len(ordered) < 2:
        interval = 1.0
    elif direction == "low":
        interval = ordered[1] - ordered[0]
    else:
        interval = ordered[-1] - ordered[-2]

    edge = ordered[0] if direction == "low" else ordered[-1]
    sign = -1 if direction == "low" else 1

    points: list[float] = []
    for i in range(1, n_expand + 1):
        candidate = edge + sign * i * interval
        if bound is not None:
            crossed = (
                candidate <= bound if direction == "low" else candidate >= bound
            )
            if crossed:
                if bound not in ordered and bound not in points:
                    points.append(bound)
                break
        points.append(candidate)
    return points


def widen_to_convergence(
    sweep_fn: SweepFn,
    axes: dict[str, list[float]],
    *,
    natural_bounds: dict[str, tuple[float | None, float | None]],
    n_expand: int = 3,
    max_expansions: int = 3,
) -> WideningResult:
    """Sweep ``axes``, widening any axis whose optimum sits on a grid edge.

    1. Sweep the current axes via ``sweep_fn``.
    2. For each axis whose optimum sits at an edge (and not at that axis's
       natural bound — see :func:`_axis_edges`), extend THAT axis only, in
       the direction of the edge, by up to ``n_expand`` new points spaced by
       the interval nearest the edge.
    3. Re-sweep only the new points (crossed with every other axis's current
       full range) and merge into the existing grid — grid values are
       deterministic per point, so merging is exact and re-sweeping an
       already-computed point would be pure waste.
    4. Repeat until every remaining edge is a natural bound, or until
       ``max_expansions`` rounds have run without convergence, in which case
       this raises :class:`NonConvergenceError` rather than returning a
       config whose optimum is a lower bound.

    Args:
        sweep_fn: Given ``{axis_name: [values, ...], ...}``, returns
            ``{point_tuple: score, ...}`` for every combination, where
            ``point_tuple[i]`` corresponds to ``list(axes)[i]`` at the time
            of the call — i.e. the SAME axis order this function was called
            with. Adapters (e.g. ``threshold_grid.threshold_grid_sweep_fn``)
            must preserve that order.
        axes: Initial axis value lists, keyed by name. Order fixes the
            tuple-position mapping used throughout.
        natural_bounds: Per-axis (lower, upper) bound, exclusive, ``None``
            meaning unbounded in that direction. Required — not defaulted —
            so this module carries no implicit opinion about ``threshold_z``
            or ``min_error_value`` by name (docs/reviews/022, item A1); pass
            :data:`threshold_grid.NATURAL_BOUNDS` for those two axes, or an
            explicit ``{}`` for an unbounded axis set (e.g. a future DC-VAE
            axis with no known natural floor).
        n_expand: New points added per widened axis per round.
        max_expansions: Widening rounds attempted before raising.

    Returns:
        WideningResult — see its docstring.

    Raises:
        NonConvergenceError: still on a (non-natural-bound) edge after
            ``max_expansions`` rounds.
    """
    axis_order = list(axes)
    axes = {name: list(values) for name, values in axes.items()}

    grid: dict[GridPoint, float] = dict(sweep_fn({name: axes[name] for name in axis_order}))

    for expansion_round in range(max_expansions + 1):
        point, score = best_point(grid)
        edges = _axis_edges(point, axes, axis_order, natural_bounds)
        if not edges:
            return WideningResult(
                grid=grid,
                axes=axes,
                axis_order=axis_order,
                best_point=point,
                best_score=score,
                expansions=expansion_round,
            )

        if expansion_round == max_expansions:
            raise NonConvergenceError(
                f"Threshold search did not converge after {max_expansions} "
                f"expansion(s): axes {sorted(edges)} still sit on a grid edge "
                f"(best_point={point}, best_score={score}). Widen "
                "n_expand/max_expansions, or investigate why the objective "
                "keeps improving toward the edge."
            )

        any_widened = False
        for name, direction in edges.items():
            lower_bound, upper_bound = natural_bounds.get(name, (None, None))
            bound = lower_bound if direction == "low" else upper_bound
            new_points = _expansion_points(direction, axes[name], n_expand, bound)
            if not new_points:
                continue
            any_widened = True
            sweep_axes = {
                other: (new_points if other == name else axes[other]) for other in axis_order
            }
            new_grid = sweep_fn(sweep_axes)
            grid.update(new_grid)
            axes[name] = sorted(set(axes[name]) | set(new_points))
            # docs/reviews/022, item P3: one line per widened axis per round —
            # the only signal separating "round 2 of 3" from "hung" on a run
            # that can exceed 30 minutes.
            log.info(
                "threshold_search.widen",
                round=expansion_round,
                axis=name,
                direction=direction,
                new_points=new_points,
                best_score=score,
                grid_size=len(grid),
            )

        if not any_widened:
            # Every flagged axis was blocked from producing a single new
            # point (bound reached to within one interval) — nothing changed
            # this round, so another round would repeat identically. Fail
            # now rather than loop uselessly to max_expansions.
            raise NonConvergenceError(
                f"Threshold search stuck on a grid edge with no room to widen: "
                f"axes {sorted(edges)} (best_point={point}, "
                f"best_score={score})."
            )

    # Unreachable: the loop always returns or raises within max_expansions+1
    # iterations, but mypy cannot see that from the range() bound alone.
    raise AssertionError("unreachable")
