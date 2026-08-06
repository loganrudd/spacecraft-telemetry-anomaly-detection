"""Tests for esa_adb.intervals — half-open interval algebra on pd.Timestamp."""

from __future__ import annotations

import numpy as np
import pandas as pd

from spacecraft_telemetry.esa_adb.intervals import (
    intersect,
    normalize,
    overlaps,
    subtract,
    total_duration,
    union,
)


def _ts(*minutes: int) -> list[pd.Timestamp]:
    """Build timestamps at t0 + N minutes for compact interval literals."""
    base = pd.Timestamp("2000-01-01T00:00:00Z")
    return [base + pd.Timedelta(minutes=m) for m in minutes]


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------


class TestNormalize:
    def test_empty_input(self) -> None:
        assert normalize([]) == []

    def test_single_interval_unchanged(self) -> None:
        s, e = _ts(0, 10)
        assert normalize([(s, e)]) == [(s, e)]

    def test_sorts_out_of_order_intervals(self) -> None:
        a0, a1, b0, b1 = _ts(20, 30, 0, 10)
        result = normalize([(a0, a1), (b0, b1)])
        assert result == [(b0, b1), (a0, a1)]

    def test_merges_overlapping_intervals(self) -> None:
        s0, s1, s2, s3 = _ts(0, 10, 5, 15)
        result = normalize([(s0, s1), (s2, s3)])
        assert result == [(s0, s3)]

    def test_merges_touching_intervals(self) -> None:
        """s2 == e1 exactly — touching, not gapped, must merge (half-open)."""
        s0, s1, s3 = _ts(0, 10, 20)
        result = normalize([(s0, s1), (s1, s3)])
        assert result == [(s0, s3)]

    def test_keeps_gapped_intervals_separate(self) -> None:
        s0, e0, s1, e1 = _ts(0, 10, 11, 20)
        result = normalize([(s0, e0), (s1, e1)])
        assert result == [(s0, e0), (s1, e1)]

    def test_nested_interval_absorbed(self) -> None:
        s0, s1, s2, s3 = _ts(0, 5, 8, 20)
        result = normalize([(s0, s3), (s1, s2)])
        assert result == [(s0, s3)]

    def test_three_way_chain_merge(self) -> None:
        s0, s1, s2, s3, s4, s5 = _ts(0, 10, 8, 20, 19, 30)
        result = normalize([(s0, s1), (s2, s3), (s4, s5)])
        assert result == [(s0, s5)]


# ---------------------------------------------------------------------------
# union
# ---------------------------------------------------------------------------


class TestUnion:
    def test_union_of_empty_lists(self) -> None:
        assert union([], []) == []

    def test_union_merges_across_lists(self) -> None:
        s0, s1, s2, s3 = _ts(0, 10, 5, 15)
        result = union([(s0, s1)], [(s2, s3)])
        assert result == [(s0, s3)]

    def test_union_keeps_disjoint_from_both_lists(self) -> None:
        s0, e0, s1, e1 = _ts(0, 5, 10, 15)
        result = union([(s0, e0)], [(s1, e1)])
        assert result == [(s0, e0), (s1, e1)]


# ---------------------------------------------------------------------------
# intersect
# ---------------------------------------------------------------------------


class TestIntersect:
    def test_intersect_empty(self) -> None:
        s0, e0 = _ts(0, 10)
        assert intersect([(s0, e0)], []) == []
        assert intersect([], [(s0, e0)]) == []

    def test_intersect_no_overlap(self) -> None:
        s0, e0, s1, e1 = _ts(0, 5, 10, 15)
        assert intersect([(s0, e0)], [(s1, e1)]) == []

    def test_intersect_partial_overlap(self) -> None:
        s0, s1, s2, s3 = _ts(0, 10, 5, 15)
        result = intersect([(s0, s1)], [(s2, s3)])
        assert result == [(s2, s1)]

    def test_intersect_touching_is_empty(self) -> None:
        """Half-open: [0,10) and [10,20) touch but do not intersect."""
        s0, s1, s2 = _ts(0, 10, 20)
        assert intersect([(s0, s1)], [(s1, s2)]) == []

    def test_intersect_nested(self) -> None:
        s0, s1, s2, s3 = _ts(0, 5, 8, 20)
        result = intersect([(s0, s3)], [(s1, s2)])
        assert result == [(s1, s2)]

    def test_intersect_multiple_segments(self) -> None:
        # a: [0,5) [10,15)   b: [3,12) [14,20)
        a0, a1, a2, a3 = _ts(0, 5, 10, 15)
        b0, b1, b2, b3 = _ts(3, 12, 14, 20)
        result = intersect([(a0, a1), (a2, a3)], [(b0, b1), (b2, b3)])
        # expect [3,5), [10,12), [14,15)
        assert result == [(b0, a1), (a2, b1), (b2, a3)]


# ---------------------------------------------------------------------------
# subtract
# ---------------------------------------------------------------------------


class TestSubtract:
    def test_subtract_no_overlap_unchanged(self) -> None:
        s0, e0, s1, e1 = _ts(0, 5, 10, 15)
        assert subtract([(s0, e0)], [(s1, e1)]) == [(s0, e0)]

    def test_subtract_full_removal(self) -> None:
        s0, e0 = _ts(0, 10)
        assert subtract([(s0, e0)], [(s0, e0)]) == []

    def test_subtract_removes_middle_chunk(self) -> None:
        s0, s1, s2, s3 = _ts(0, 5, 8, 20)
        result = subtract([(s0, s3)], [(s1, s2)])
        assert result == [(s0, s1), (s2, s3)]

    def test_subtract_removes_prefix(self) -> None:
        s0, s1, s2 = _ts(0, 5, 10)
        result = subtract([(s0, s2)], [(s0, s1)])
        assert result == [(s1, s2)]

    def test_subtract_removes_suffix(self) -> None:
        s0, s1, s2 = _ts(0, 5, 10)
        result = subtract([(s0, s2)], [(s1, s2)])
        assert result == [(s0, s1)]

    def test_subtract_empty_b_is_noop(self) -> None:
        s0, e0 = _ts(0, 10)
        assert subtract([(s0, e0)], []) == [(s0, e0)]


# ---------------------------------------------------------------------------
# total_duration
# ---------------------------------------------------------------------------


class TestTotalDuration:
    def test_empty_is_zero(self) -> None:
        assert total_duration([]) == np.timedelta64(0, "ns")

    def test_single_interval(self) -> None:
        s0, e0 = _ts(0, 10)
        assert total_duration([(s0, e0)]) == np.timedelta64(10, "m")

    def test_sums_disjoint_intervals(self) -> None:
        s0, e0, s1, e1 = _ts(0, 5, 10, 25)
        assert total_duration([(s0, e0), (s1, e1)]) == np.timedelta64(20, "m")

    def test_single_point_interval_is_zero_duration(self) -> None:
        s0 = _ts(0)[0]
        assert total_duration([(s0, s0)]) == np.timedelta64(0, "ns")

    def test_double_counts_unnormalized_overlap(self) -> None:
        """total_duration does not normalize — overlapping input double-counts.

        Callers that need the no-double-counting guarantee must call
        normalize() first (documented in the docstring).
        """
        s0, s1, s2 = _ts(0, 10, 5)
        raw = total_duration([(s0, s1), (s2, s1)])
        assert raw == np.timedelta64(15, "m")
        merged = total_duration(normalize([(s0, s1), (s2, s1)]))
        assert merged == np.timedelta64(10, "m")


# ---------------------------------------------------------------------------
# overlaps
# ---------------------------------------------------------------------------


class TestOverlaps:
    def test_no_candidates(self) -> None:
        s0, e0 = _ts(0, 10)
        assert overlaps((s0, e0), []) is False

    def test_overlapping_candidate(self) -> None:
        s0, e0, s1, e1 = _ts(0, 10, 5, 15)
        assert overlaps((s0, e0), [(s1, e1)]) is True

    def test_touching_candidate_does_not_overlap(self) -> None:
        s0, s1, s2 = _ts(0, 10, 20)
        assert overlaps((s0, s1), [(s1, s2)]) is False

    def test_disjoint_candidate(self) -> None:
        s0, e0, s1, e1 = _ts(0, 5, 10, 15)
        assert overlaps((s0, e0), [(s1, e1)]) is False

    def test_matches_any_of_several_candidates(self) -> None:
        s0, e0 = _ts(100, 110)
        far0, far1 = _ts(0, 5)
        near0, near1 = _ts(105, 120)
        assert overlaps((s0, e0), [(far0, far1), (near0, near1)]) is True
