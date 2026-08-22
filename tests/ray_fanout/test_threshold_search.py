"""Tests for ray_fanout.threshold_search (docs/plans/022, stage 022.1).

All tests use a synthetic sweep_fn — a closure over a known 2-D function, not
real error arrays. The driver's widening logic is independent of what it
sweeps; that independence is exactly what the sweep_fn seam is for.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from spacecraft_telemetry.ray_fanout.threshold_search import (
    GridPoint,
    NonConvergenceError,
    threshold_grid_sweep_fn,
    widen_to_convergence,
)


def _counting_sweep_fn(
    objective: Callable[[float, float], float],
) -> tuple[Callable[[dict[str, list[float]]], dict[GridPoint, float]], dict[GridPoint, int]]:
    """Wrap objective(a, b) -> float as a sweep_fn over named axes "a"/"b".

    ``calls`` counts how many times each point was actually computed — the
    no-drift gate needs to see that widening never recomputes an
    already-swept point.
    """
    calls: dict[GridPoint, int] = {}

    def _sweep(axes: dict[str, list[float]]) -> dict[GridPoint, float]:
        out: dict[GridPoint, float] = {}
        for a in axes["a"]:
            for b in axes["b"]:
                point = (a, b)
                calls[point] = calls.get(point, 0) + 1
                out[point] = objective(a, b)
        return out

    return _sweep, calls


class TestInteriorOptimum:
    def test_zero_expansions_when_optimum_is_interior(self) -> None:
        def objective(a: float, b: float) -> float:
            return -((a - 5.0) ** 2) - ((b - 0.2) ** 2)

        sweep_fn, calls = _counting_sweep_fn(objective)
        axes = {"a": [1.0, 3.0, 5.0, 7.0, 9.0], "b": [0.0, 0.1, 0.2, 0.3, 0.4]}

        result = widen_to_convergence(sweep_fn, axes, natural_bounds={})

        assert result.expansions == 0
        assert result.best_point == (5.0, 0.2)
        assert result.best_score == pytest.approx(0.0)
        assert result.axes == axes
        assert all(n == 1 for n in calls.values())

    def test_ties_break_toward_lower_value_on_every_axis(self) -> None:
        # Tied region sits strictly inside both axes so the tie-break itself
        # is under test, not an edge-expansion side effect.
        def objective(a: float, b: float) -> float:
            return 0.0 if a in (2.0, 3.0) and b in (1.0, 2.0) else -1.0

        sweep_fn, _calls = _counting_sweep_fn(objective)
        axes = {"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [0.0, 1.0, 2.0, 3.0, 4.0]}

        result = widen_to_convergence(sweep_fn, axes, natural_bounds={})

        assert result.best_point == (2.0, 1.0)
        assert result.expansions == 0


class TestEdgeExpansion:
    def test_low_edge_expands_downward_to_reach_the_true_optimum(self) -> None:
        # Unit-spaced axis so the widened points land exactly on the target
        # (a=-4) regardless of interval arithmetic — the point of this test
        # is edge detection + widening direction, not grid/target alignment.
        def objective(a: float, b: float) -> float:
            return -((a + 4.0) ** 2) - ((b - 0.2) ** 2)

        sweep_fn, calls = _counting_sweep_fn(objective)
        axes = {"a": [1.0, 2.0, 3.0, 4.0], "b": [0.0, 0.1, 0.2, 0.3, 0.4]}

        result = widen_to_convergence(sweep_fn, axes, natural_bounds={})

        assert result.expansions > 0
        assert result.best_point == (-4.0, 0.2)
        assert min(result.axes["a"]) < -4.0  # -4 is interior, not the final edge
        # Original points survive at their original values, computed once.
        for a in [1.0, 2.0, 3.0, 4.0]:
            point = (a, 0.2)
            assert calls[point] == 1
            assert result.grid[point] == pytest.approx(objective(*point))

    def test_high_edge_expands_upward_to_reach_the_true_optimum(self) -> None:
        def objective(a: float, b: float) -> float:
            return -((a - 10.0) ** 2) - ((b - 0.2) ** 2)

        sweep_fn, calls = _counting_sweep_fn(objective)
        axes = {"a": [1.0, 2.0, 3.0, 4.0], "b": [0.0, 0.1, 0.2, 0.3, 0.4]}

        result = widen_to_convergence(sweep_fn, axes, natural_bounds={})

        assert result.expansions > 0
        assert result.best_point == (10.0, 0.2)
        assert max(result.axes["a"]) > 10.0  # 10 is interior, not the final edge
        for a in [1.0, 2.0, 3.0, 4.0]:
            point = (a, 0.2)
            assert calls[point] == 1
            assert result.grid[point] == pytest.approx(objective(*point))

    def test_merged_grid_has_no_re_sweep_drift(self) -> None:
        """Every original point keeps its original (once-computed) value."""

        # b's optimum (0.5) sits strictly inside [0.0, 1.0] regardless of how
        # far a's axis widens, so only a's edge drives expansion here.
        def objective(a: float, b: float) -> float:
            return -((a - 8.0) ** 2) - ((b - 0.5) ** 2)

        sweep_fn, calls = _counting_sweep_fn(objective)
        axes = {"a": [1.0, 2.0, 3.0], "b": [0.0, 0.5, 1.0]}

        result = widen_to_convergence(sweep_fn, axes, natural_bounds={})

        assert result.expansions > 0
        assert result.best_point == (8.0, 0.5)
        for a in [1.0, 2.0, 3.0]:
            for b in [0.0, 0.5, 1.0]:
                point = (a, b)
                assert calls[point] == 1
                assert result.grid[point] == pytest.approx(objective(*point))

    def test_both_axes_widen_in_the_same_round_fills_the_corner(self) -> None:
        """Widening a and b in one round must still cover their full cross product."""

        def objective(a: float, b: float) -> float:
            return -((a - 20.0) ** 2) - ((b - 5.0) ** 2)

        sweep_fn, _calls = _counting_sweep_fn(objective)
        axes = {"a": [1.0, 3.0, 5.0], "b": [1.0, 2.0, 3.0]}

        result = widen_to_convergence(
            sweep_fn, axes, natural_bounds={}, n_expand=5, max_expansions=5
        )

        # The full cross product of the final axes must be present — no gaps
        # left by widening one axis before the other's range was updated.
        for a in result.axes["a"]:
            for b in result.axes["b"]:
                assert (a, b) in result.grid


class TestNaturalBound:
    def test_optimum_at_natural_bound_converges_without_expanding(self) -> None:
        # Score strictly decreases in b, so the true optimum sits at b's
        # lowest value — which is also the axis's natural bound. There is
        # nothing to widen into; must converge immediately, never try
        # negative b.
        def objective(a: float, b: float) -> float:
            return -((a - 5.0) ** 2) - b

        sweep_fn, _calls = _counting_sweep_fn(objective)
        axes = {"a": [1.0, 3.0, 5.0, 7.0, 9.0], "b": [0.0, 0.1, 0.2]}

        result = widen_to_convergence(sweep_fn, axes, natural_bounds={"b": (0.0, None)})

        assert result.expansions == 0
        assert result.best_point == (5.0, 0.0)
        assert result.axes["b"] == [0.0, 0.1, 0.2]
        assert all(b >= 0.0 for _a, b in result.grid)


class TestInclusiveBound:
    """docs/reviews/022, item C1: candidate <= bound used to be an exclusive
    check, rejecting the bound itself. A floor axis stopping one interval
    short of 0.0 (e.g. [0.1, 0.2, 0.3]) would then hard-fail rather than
    converging at the legitimate, reachable min_error_value=0.0 setting."""

    def test_optimum_reaches_the_natural_bound_when_grid_starts_short_of_it(self) -> None:
        def objective(a: float, b: float) -> float:
            return -((a - 5.0) ** 2) - b  # wants max a, MIN b -> b's optimum is the bound

        sweep_fn, calls = _counting_sweep_fn(objective)
        axes = {"a": [1.0, 3.0, 5.0, 7.0, 9.0], "b": [0.1, 0.2, 0.3]}

        result = widen_to_convergence(sweep_fn, axes, natural_bounds={"b": (0.0, None)})

        assert result.best_point == (5.0, 0.0)
        assert 0.0 in result.axes["b"]
        assert result.expansions > 0
        # The bound is swept exactly once, not repeatedly across rounds.
        assert calls[(5.0, 0.0)] == 1

    def test_bound_already_present_is_not_re_swept(self) -> None:
        """If the bound happens to already be a swept value, _expansion_points
        must not re-add (and therefore re-sweep) it."""
        def objective(a: float, b: float) -> float:
            return -((a - 5.0) ** 2) - b

        sweep_fn, calls = _counting_sweep_fn(objective)
        # b's low edge (0.05) is already interior to the bound, and 0.0 is
        # already present elsewhere in the axis — nothing to widen for b.
        axes = {"a": [1.0, 3.0, 5.0, 7.0, 9.0], "b": [0.0, 0.05, 0.1]}

        result = widen_to_convergence(sweep_fn, axes, natural_bounds={"b": (0.0, None)})

        assert result.best_point == (5.0, 0.0)
        assert result.axes["b"] == [0.0, 0.05, 0.1]
        assert calls[(5.0, 0.0)] == 1


class TestNonConvergence:
    def test_raises_when_the_optimum_keeps_moving_toward_an_unbounded_edge(self) -> None:
        def objective(a: float, b: float) -> float:
            return a - b  # wants max a, min b — both edges, neither bounded

        sweep_fn, _calls = _counting_sweep_fn(objective)
        axes = {"a": [1.0, 2.0, 3.0], "b": [0.0, 0.1]}

        with pytest.raises(NonConvergenceError):
            widen_to_convergence(sweep_fn, axes, natural_bounds={}, max_expansions=2)

    def test_stuck_with_no_room_to_widen_is_a_distinct_branch(self) -> None:
        """T2: a DISTINCT failure branch from max_expansions exhaustion above
        — every flagged axis's widen attempt yields zero new points
        (n_expand=0), so nothing changes and the driver must fail immediately
        with its own message rather than looping uselessly to
        max_expansions. Previously uncovered — exactly why C1 shipped."""

        def objective(a: float, b: float) -> float:
            return a  # wants max a -> unbounded high edge; b is a flat tie

        sweep_fn, _calls = _counting_sweep_fn(objective)
        axes = {"a": [1.0, 2.0, 3.0], "b": [0.0]}

        with pytest.raises(NonConvergenceError, match="no room to widen"):
            widen_to_convergence(sweep_fn, axes, natural_bounds={}, n_expand=0)


class TestThresholdGridSweepFn:
    def test_binds_named_axes_onto_z_values_and_floor_values(self) -> None:
        captured: dict[str, object] = {}

        def fake_sweep(
            *, z_values: list[float], floor_values: list[float], extra: str | None = None
        ) -> dict[GridPoint, float]:
            captured["z_values"] = z_values
            captured["floor_values"] = floor_values
            captured["extra"] = extra
            return {(z, f): 0.0 for z in z_values for f in floor_values}

        sweep_fn = threshold_grid_sweep_fn(fake_sweep, extra="pinned")
        grid = sweep_fn({"threshold_z": [1.0, 2.0], "min_error_value": [0.0, 0.5]})

        assert captured["z_values"] == [1.0, 2.0]
        assert captured["floor_values"] == [0.0, 0.5]
        assert captured["extra"] == "pinned"
        assert set(grid) == {(1.0, 0.0), (1.0, 0.5), (2.0, 0.0), (2.0, 0.5)}

    def test_wrong_axis_order_raises_rather_than_silently_transposing(self) -> None:
        sweep_fn = threshold_grid_sweep_fn(lambda **_kwargs: {})
        with pytest.raises(ValueError, match="threshold_z"):
            sweep_fn({"min_error_value": [0.0], "threshold_z": [1.0]})
