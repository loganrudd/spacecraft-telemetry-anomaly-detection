"""Unit tests for injection/faults.py.

All tests are pure numpy — no I/O, no torch, fast.
"""

from __future__ import annotations

import numpy as np
import pytest

from spacecraft_telemetry.injection.faults import (
    ChannelProfile,
    inject_drift,
    inject_faults,
    inject_flatline,
    inject_spike,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nominal(n: int = 500, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n).astype(np.float32)


def _single_segment(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Segment IDs (all same) + no-LOS mask."""
    return np.zeros(n, dtype=np.int32), np.zeros(n, dtype=bool)


# ---------------------------------------------------------------------------
# ChannelProfile.from_dict validation
# ---------------------------------------------------------------------------

class TestChannelProfileValidation:
    def test_empty_dict_uses_defaults(self) -> None:
        profile = ChannelProfile.from_dict({})
        assert profile.signal_class == "slow_lownoise"
        assert profile.magnitude_sigma_range == [0.4, 1.5]

    def test_valid_dict_round_trips(self) -> None:
        d = {
            "signal_class": "smooth_ramp",
            "fault_type_weights": {"spike": 0.3, "drift": 0.4, "flatline": 0.3},
            "magnitude_sigma_range": [0.5, 2.0],
            "spike_duration_range": [2, 10],
            "drift_duration_range": [20, 200],
            "flatline_duration_range": [20, 200],
        }
        profile = ChannelProfile.from_dict(d)
        assert profile.signal_class == "smooth_ramp"
        assert profile.fault_type_weights == {"spike": 0.3, "drift": 0.4, "flatline": 0.3}

    def test_unknown_fault_type_key_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown fault_type_weights keys"):
            ChannelProfile.from_dict({"fault_type_weights": {"spike": 1.0, "explode": 0.5}})

    def test_all_zero_weights_raises(self) -> None:
        with pytest.raises(ValueError, match="All fault_type_weights are <= 0"):
            ChannelProfile.from_dict(
                {"fault_type_weights": {"spike": 0.0, "drift": 0.0, "flatline": 0.0}}
            )

    def test_reversed_magnitude_range_raises(self) -> None:
        with pytest.raises(ValueError, match="magnitude_sigma_range"):
            ChannelProfile.from_dict({"magnitude_sigma_range": [2.0, 0.5]})

    def test_equal_magnitude_range_raises(self) -> None:
        with pytest.raises(ValueError, match="magnitude_sigma_range"):
            ChannelProfile.from_dict({"magnitude_sigma_range": [1.0, 1.0]})

    def test_reversed_spike_duration_raises(self) -> None:
        with pytest.raises(ValueError, match="spike_duration_range"):
            ChannelProfile.from_dict({"spike_duration_range": [10, 2]})


# ---------------------------------------------------------------------------
# inject_spike
# ---------------------------------------------------------------------------

class TestInjectSpike:
    def test_output_shape_preserved(self) -> None:
        v = _nominal(200)
        out, mask = inject_spike(v, start=50, magnitude_sigma=2.0, duration=3)
        assert out.shape == v.shape
        assert mask.shape == v.shape

    def test_mask_true_only_over_fault_span(self) -> None:
        v = _nominal(200)
        _, mask = inject_spike(v, start=50, magnitude_sigma=2.0, duration=5)
        assert mask[50:55].all()
        assert not mask[:50].any()
        assert not mask[55:].any()

    def test_values_unchanged_outside_fault(self) -> None:
        v = _nominal(200)
        out, _ = inject_spike(v, start=80, magnitude_sigma=3.0, duration=4)
        np.testing.assert_array_equal(out[:80], v[:80])
        np.testing.assert_array_equal(out[84:], v[84:])

    def test_fault_region_differs_from_original(self) -> None:
        v = _nominal(200)
        out, _ = inject_spike(v, start=50, magnitude_sigma=2.0, duration=3)
        assert not np.allclose(out[50:53], v[50:53])

    def test_duration_clipped_at_series_end(self) -> None:
        v = _nominal(10)
        _, mask = inject_spike(v, start=8, magnitude_sigma=1.0, duration=100)
        # Should only affect positions 8 and 9
        assert mask[8:10].all()
        assert not mask[:8].any()

    def test_single_point_spike(self) -> None:
        v = _nominal(100)
        _, mask = inject_spike(v, start=50, magnitude_sigma=1.0, duration=1)
        assert mask.sum() == 1
        assert mask[50]


# ---------------------------------------------------------------------------
# inject_drift
# ---------------------------------------------------------------------------

class TestInjectDrift:
    def test_output_shape_preserved(self) -> None:
        v = _nominal(500)
        out, mask = inject_drift(v, start=100, duration=60, total_shift_sigma=1.5)
        assert out.shape == v.shape
        assert mask.shape == v.shape

    def test_mask_true_only_over_fault_span(self) -> None:
        v = _nominal(500)
        _, mask = inject_drift(v, start=100, duration=60, total_shift_sigma=1.5)
        assert mask[100:160].all()
        assert not mask[:100].any()
        assert not mask[160:].any()

    def test_values_unchanged_outside_fault(self) -> None:
        v = _nominal(500)
        out, _ = inject_drift(v, start=100, duration=60, total_shift_sigma=1.5)
        np.testing.assert_array_equal(out[:100], v[:100])
        np.testing.assert_array_equal(out[160:], v[160:])

    def test_ramp_then_hold_structure(self) -> None:
        v = np.zeros(200, dtype=np.float32)
        out, _ = inject_drift(v, start=10, duration=10, total_shift_sigma=2.0)
        segment = out[10:20].astype(float)
        # Should be monotonically increasing in ramp half
        ramp = segment[:5]
        assert ramp[-1] > ramp[0], "ramp should increase"
        # Hold half should be approximately constant
        hold = segment[5:]
        assert float(hold.std()) < 1e-5, "hold should be constant"
        # Final offset matches target
        assert abs(float(hold[-1]) - 2.0) < 1e-4

    def test_duration_one(self) -> None:
        v = _nominal(50)
        _, mask = inject_drift(v, start=20, duration=1, total_shift_sigma=1.0)
        assert mask.sum() == 1

    def test_clipped_drift_duration_two_hold_at_shift(self) -> None:
        """actual_duration=2: ramp=[0.0], hold=[shift] → out[start+1] ≈ shift."""
        v = np.zeros(50, dtype=np.float32)
        out, mask = inject_drift(v, start=10, duration=2, total_shift_sigma=1.5)
        assert mask[10:12].all()
        assert mask.sum() == 2
        assert abs(float(out[11]) - 1.5) < 1e-4, "hold should be at total_shift_sigma"


# ---------------------------------------------------------------------------
# inject_flatline
# ---------------------------------------------------------------------------

class TestInjectFlatline:
    def test_output_shape_preserved(self) -> None:
        v = _nominal(300)
        out, mask = inject_flatline(v, start=100, duration=50)
        assert out.shape == v.shape
        assert mask.shape == v.shape

    def test_mask_true_only_over_fault_span(self) -> None:
        v = _nominal(300)
        _, mask = inject_flatline(v, start=100, duration=50)
        assert mask[100:150].all()
        assert not mask[:100].any()
        assert not mask[150:].any()

    def test_values_unchanged_outside_fault(self) -> None:
        v = _nominal(300)
        out, _ = inject_flatline(v, start=100, duration=50)
        np.testing.assert_array_equal(out[:100], v[:100])
        np.testing.assert_array_equal(out[150:], v[150:])

    def test_flatline_region_is_constant(self) -> None:
        v = _nominal(300)
        out, _ = inject_flatline(v, start=100, duration=50)
        assert float(out[100:150].std()) == 0.0

    def test_flatline_value_is_pre_fault(self) -> None:
        v = _nominal(300)
        expected = float(v[99])
        out, _ = inject_flatline(v, start=100, duration=50)
        assert float(out[100]) == pytest.approx(expected, abs=1e-5)


# ---------------------------------------------------------------------------
# inject_faults (orchestrator)
# ---------------------------------------------------------------------------

class TestInjectFaults:
    def _rng(self, seed: int = 42) -> np.random.Generator:
        return np.random.default_rng(seed)

    def _default_profile(self) -> ChannelProfile:
        return ChannelProfile()  # uses all defaults

    def test_output_shapes_match_input(self) -> None:
        v = _nominal(1000)
        seg, los = _single_segment(1000)
        out, mask, _records = inject_faults(v, seg, los, self._rng(), 6, self._default_profile())
        assert out.shape == v.shape
        assert mask.shape == v.shape

    def test_mask_aligns_with_fault_records(self) -> None:
        v = _nominal(1000)
        seg, los = _single_segment(1000)
        _, mask, records = inject_faults(v, seg, los, self._rng(), 6, self._default_profile())
        # Reconstruct expected mask from records
        expected = np.zeros(1000, dtype=bool)
        for rec in records:
            expected[rec["start"] : rec["end"]] = True
        np.testing.assert_array_equal(mask, expected)

    def test_values_match_mask(self) -> None:
        v = _nominal(1000)
        seg, los = _single_segment(1000)
        out, mask, _ = inject_faults(v, seg, los, self._rng(), 6, self._default_profile())
        # Outside fault regions, values must be unchanged
        np.testing.assert_array_equal(out[~mask], v[~mask])

    def test_no_fault_in_los_region(self) -> None:
        v = _nominal(1000)
        seg = np.zeros(1000, dtype=np.int32)
        is_los = np.zeros(1000, dtype=bool)
        is_los[200:400] = True  # large LOS block
        _, mask, _ = inject_faults(v, seg, is_los, self._rng(), 10, self._default_profile())
        assert not mask[200:400].any(), "no fault should land inside LOS region"

    def test_no_fault_crosses_segment_boundary(self) -> None:
        n = 1000
        v = _nominal(n)
        # Two segments: 0-499 and 500-999
        seg = np.zeros(n, dtype=np.int32)
        seg[500:] = 1
        los = np.zeros(n, dtype=bool)
        _, _, records = inject_faults(v, seg, los, self._rng(), 8, self._default_profile())
        for rec in records:
            s, e = rec["start"], rec["end"]
            # All fault positions must lie within a single segment
            assert len(np.unique(seg[s:e])) == 1, (
                f"Fault [{s}:{e}] crosses a segment boundary"
            )

    def test_both_hpo_and_held_out_get_faults(self) -> None:
        n = 2000
        v = _nominal(n)
        seg, los = _single_segment(n)
        _, mask, _records = inject_faults(
            v, seg, los, self._rng(), 6, self._default_profile(), hpo_fraction=0.6
        )
        n_hpo = int(n * 0.6)
        assert mask[:n_hpo].any(), "HPO portion must have at least one fault"
        assert mask[n_hpo:].any(), "Held-out portion must have at least one fault"

    def test_determinism_under_same_seed(self) -> None:
        v = _nominal(1000)
        seg, los = _single_segment(1000)
        profile = self._default_profile()
        _, mask1, rec1 = inject_faults(v, seg, los, np.random.default_rng(99), 5, profile)
        _, mask2, rec2 = inject_faults(v, seg, los, np.random.default_rng(99), 5, profile)
        np.testing.assert_array_equal(mask1, mask2)
        assert rec1 == rec2

    def test_different_seeds_produce_different_faults(self) -> None:
        v = _nominal(1000)
        seg, los = _single_segment(1000)
        profile = self._default_profile()
        _, mask1, _ = inject_faults(v, seg, los, np.random.default_rng(0), 5, profile)
        _, mask2, _ = inject_faults(v, seg, los, np.random.default_rng(1), 5, profile)
        assert not np.array_equal(mask1, mask2), (
            "different seeds should produce different placements"
        )

    def test_fault_records_contain_required_fields(self) -> None:
        v = _nominal(1000)
        seg, los = _single_segment(1000)
        _, _, records = inject_faults(v, seg, los, self._rng(), 4, self._default_profile())
        for rec in records:
            assert "type" in rec
            assert "start" in rec
            assert "end" in rec
            assert "duration" in rec
            assert "magnitude_sigma" in rec
            assert "signal_class" in rec

    def test_flatline_magnitude_sigma_is_none(self) -> None:
        n = 1000
        v = _nominal(n)
        seg, los = _single_segment(n)
        flatline_profile = ChannelProfile(
            fault_type_weights={"spike": 0.0, "drift": 0.0, "flatline": 1.0},
            magnitude_sigma_range=[0.5, 1.5],
            flatline_duration_range=[5, 30],
        )
        _, _, records = inject_faults(v, seg, los, self._rng(), 4, flatline_profile)
        for rec in records:
            assert rec["magnitude_sigma"] is None, (
                "flatline records must have magnitude_sigma=None"
            )

    def test_spike_drift_magnitude_sigma_is_float(self) -> None:
        n = 1000
        v = _nominal(n)
        seg, los = _single_segment(n)
        spike_drift_profile = ChannelProfile(
            fault_type_weights={"spike": 0.5, "drift": 0.5, "flatline": 0.0},
        )
        _, _, records = inject_faults(v, seg, los, self._rng(), 6, spike_drift_profile)
        for rec in records:
            assert isinstance(rec["magnitude_sigma"], float), (
                f"spike/drift magnitude_sigma must be float, got {type(rec['magnitude_sigma'])}"
            )

    def test_flatline_guard_skips_flat_region(self) -> None:
        # Flat series — flatline injection should be skipped (all-flat guard)
        n = 1000
        v = np.zeros(n, dtype=np.float32)
        seg, los = _single_segment(n)
        flatline_profile = ChannelProfile(
            fault_type_weights={"spike": 0.0, "drift": 0.0, "flatline": 1.0},
            flatline_duration_range=[5, 10],
        )
        _, _, records = inject_faults(
            v, seg, los, self._rng(), 5, flatline_profile,
            flat_variance_threshold=0.01,
        )
        # flatlines should be skipped because pre-window variance is ~0
        assert all(r["type"] != "flatline" for r in records), (
            "flatline should be skipped on already-flat series"
        )

    def test_fault_region_actually_changed(self) -> None:
        v = _nominal(1000)
        v_orig = v.copy()
        seg, los = _single_segment(1000)
        out, mask, _ = inject_faults(v, seg, los, self._rng(), 6, self._default_profile())
        assert mask.any(), "at least one fault must be placed"
        assert not np.array_equal(out[mask], v_orig[mask]), (
            "fault region must differ from original signal"
        )

    def test_empty_series_returns_unchanged(self) -> None:
        v = np.zeros(0, dtype=np.float32)
        seg = np.zeros(0, dtype=np.int32)
        los = np.zeros(0, dtype=bool)
        out, mask, records = inject_faults(v, seg, los, self._rng(), 3, self._default_profile())
        assert len(out) == 0
        assert len(mask) == 0
        assert records == []
