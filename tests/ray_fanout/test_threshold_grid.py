"""Tests for ray_fanout.threshold_grid — post-hoc (z, floor) ceiling sweeps.

The load-bearing property is the precompute equivalence: the whole module is a
speed optimisation over calling dynamic_threshold once per grid point, so if
the recombination drifts from dynamic_threshold the ceilings are quietly wrong
and the tuning-parity check (docs/plans/021 stage 021.5b) silently lies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacecraft_telemetry.model.scoring import dynamic_threshold
from spacecraft_telemetry.ray_fanout.threshold_grid import (
    GridPoint,
    precompute_threshold_terms,
    sweep_channel,
    sweep_group,
    threshold_from_terms,
    threshold_grid_sweep_fn,
)


def _series(n: int = 400, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.abs(rng.standard_normal(n)) * 0.05
    x[120:130] += 1.2  # a clear excursion so flags are non-degenerate
    x[300:308] += 0.8
    return x.astype(np.float64)


def _labels_for(n: int = 400) -> np.ndarray:
    lab = np.zeros(n, dtype=bool)
    lab[120:130] = True
    lab[300:308] = True
    return lab


class TestPrecomputeEquivalence:
    """The optimisation must be exact, not approximate."""

    @pytest.mark.parametrize("z", [0.0, 1.0, 2.5, 3.0, 7.978, 20.0])
    @pytest.mark.parametrize("window", [2, 10, 50, 250])
    def test_matches_dynamic_threshold_exactly(self, z: float, window: int) -> None:
        smoothed = _series()
        expected = dynamic_threshold(smoothed, window, z)
        mean, std = precompute_threshold_terms(smoothed, window)
        actual = threshold_from_terms(mean, std, z)
        np.testing.assert_array_equal(actual, expected)

    def test_position_zero_is_inf(self) -> None:
        """No history at t=0 → threshold inf → never flags, same as
        dynamic_threshold's fillna(inf)."""
        mean, std = precompute_threshold_terms(_series(), 10)
        assert threshold_from_terms(mean, std, 3.0)[0] == np.inf

    def test_terms_are_z_independent(self) -> None:
        """The point of the split: the rolling passes don't depend on z."""
        smoothed = _series()
        a = precompute_threshold_terms(smoothed, 25)
        b = precompute_threshold_terms(smoothed, 25)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])


class TestSweepChannel:
    def test_covers_full_grid(self) -> None:
        grid = sweep_channel(
            _series(), _labels_for(),
            threshold_window=50, min_run_length=2,
            z_values=[2.0, 3.0], floor_values=[0.0, 0.1, 0.2],
        )
        assert set(grid) == {(z, f) for z in (2.0, 3.0) for f in (0.0, 0.1, 0.2)}
        assert all(0.0 <= v <= 1.0 for v in grid.values())

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same shape"):
            sweep_channel(
                _series(400), _labels_for(300),
                threshold_window=10, min_run_length=1,
                z_values=[3.0], floor_values=[0.0],
            )

    def test_eval_slice_restricts_scoring(self) -> None:
        """A slice covering only nominal windows must score differently from
        one covering the excursions — otherwise the slice is being ignored."""
        smoothed, labels = _series(), _labels_for()
        kw = dict(
            threshold_window=50, min_run_length=2,
            z_values=[2.0], floor_values=[0.0],
        )
        anomalous = sweep_channel(smoothed, labels, eval_slice=slice(100, 200), **kw)
        nominal = sweep_channel(smoothed, labels, eval_slice=slice(200, 290), **kw)
        assert anomalous[(2.0, 0.0)] != nominal[(2.0, 0.0)]

    def test_matches_naive_per_point_computation(self) -> None:
        """End-to-end equivalence against the obvious slow implementation."""
        from spacecraft_telemetry.model.scoring import evaluate_overlap, flag_anomalies

        smoothed, labels = _series(), _labels_for()
        grid = sweep_channel(
            smoothed, labels,
            threshold_window=40, min_run_length=3,
            z_values=[2.0, 4.0], floor_values=[0.0, 0.15],
        )
        for (z, floor), got in grid.items():
            th = dynamic_threshold(smoothed, 40, z)
            fl = flag_anomalies(smoothed, th, 3, floor)
            assert got == pytest.approx(evaluate_overlap(labels, fl)["seg_f0_5"])


class TestSweepGroup:
    def test_averages_across_channels(self) -> None:
        a = (_series(seed=1), _labels_for())
        b = (_series(seed=2), _labels_for())
        kw = dict(
            threshold_window=40, min_run_length=2,
            z_values=[2.0, 3.0], floor_values=[0.0],
        )
        per_a = sweep_channel(*a, **kw)
        per_b = sweep_channel(*b, **kw)
        grouped = sweep_group({"a": a, "b": b}, **kw)
        for point in grouped:
            assert grouped[point] == pytest.approx((per_a[point] + per_b[point]) / 2)

    def test_empty_group_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            sweep_group({}, threshold_window=10, min_run_length=1,
                        z_values=[3.0], floor_values=[0.0])


class TestSweepGroupMissionLevel:
    """The correction to 021.5b attempt 1: selecting on per-channel seg_f0_5
    and reporting mission-level corrected event-wise F0.5 are different
    objectives, and a config optimal on the first was catastrophic on the
    second (33 -> 112 detections, 0.376 -> 0.123)."""

    def _fixture(self) -> tuple[dict, dict]:
        n = 400
        ts = pd.date_range("2000-01-01", periods=n, freq="90s").to_numpy()
        per_channel = {
            "channel_41": (_series(seed=1), _labels_for()),
            "channel_42": (_series(seed=2), _labels_for()),
        }
        stamps = {"channel_41": ts, "channel_42": ts}
        return per_channel, stamps

    def test_requires_timestamps_for_every_channel(self) -> None:
        """A channel without timestamps would silently drop out of the
        OR-aggregation, computing the mission metric on a subset."""
        from spacecraft_telemetry.ray_fanout.threshold_grid import (
            sweep_group_mission_level,
        )

        per_channel, stamps = self._fixture()
        del stamps["channel_42"]
        with pytest.raises(ValueError, match="channel_timestamps missing"):
            sweep_group_mission_level(
                per_channel, stamps, [], None,
                threshold_window=40, min_run_length=2,
                z_values=[3.0], floor_values=[0.0],
            )

    def test_empty_group_raises(self) -> None:
        from spacecraft_telemetry.ray_fanout.threshold_grid import (
            sweep_group_mission_level,
        )

        with pytest.raises(ValueError, match="empty"):
            sweep_group_mission_level(
                {}, {}, [], None,
                threshold_window=40, min_run_length=2,
                z_values=[3.0], floor_values=[0.0],
            )

    def test_covers_full_grid_and_returns_scores(self) -> None:
        """With no ground-truth events the metric is well-defined (0.0) — the
        point here is that every grid point is evaluated and the aggregation
        pipeline runs end to end."""
        from spacecraft_telemetry.ray_fanout.threshold_grid import (
            sweep_group_mission_level,
        )

        per_channel, stamps = self._fixture()
        timeline = [
            (pd.Timestamp("2000-01-01", tz="UTC"),
             pd.Timestamp("2000-01-01", tz="UTC") + pd.Timedelta(hours=10))
        ]
        grid = sweep_group_mission_level(
            per_channel, stamps, [], timeline,
            threshold_window=40, min_run_length=2,
            z_values=[2.0, 3.0], floor_values=[0.0, 0.1],
        )
        assert set(grid) == {(z, f) for z in (2.0, 3.0) for f in (0.0, 0.1)}
        assert all(isinstance(v, float) for v in grid.values())


class TestThresholdGridSweepFn:
    """docs/reviews/022, item A3: moved from ray_fanout.threshold_search —
    this is a domain adapter (hardcodes THIS module's z/floor keyword names
    and tuple order), not part of the generic widening driver."""

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
