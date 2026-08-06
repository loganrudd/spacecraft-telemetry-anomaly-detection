"""Half-open interval algebra on ``pd.Timestamp``, used by the ESA-ADB metrics.

Every interval is ``[start, end)`` — start inclusive, end exclusive — matching
the convention already used by ``preprocess.transforms.label_timesteps``.
A "canonical" interval list is sorted by start and has no two intervals that
overlap or touch (adjacent, zero-gap intervals are merged into one).

This module has no MLflow / settings dependencies — it is pure interval
arithmetic so it can be exhaustively unit tested in isolation from the rest
of the ESA-ADB report.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

    Interval = tuple[pd.Timestamp, pd.Timestamp]


def normalize(intervals: list[Interval]) -> list[Interval]:
    """Sort by start and merge overlapping or touching intervals.

    Two intervals ``(s1, e1)`` and ``(s2, e2)`` (with ``s1 <= s2``) merge iff
    ``s2 <= e1`` — i.e. they overlap or share a boundary. A strict gap
    (``s2 > e1``) keeps them separate.
    """
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda iv: iv[0])
    merged: list[list[pd.Timestamp]] = [list(ordered[0])]
    for start, end in ordered[1:]:
        last = merged[-1]
        if start <= last[1]:
            if end > last[1]:
                last[1] = end
        else:
            merged.append([start, end])
    return [(m[0], m[1]) for m in merged]


def union(a: list[Interval], b: list[Interval]) -> list[Interval]:
    """Union of two interval lists, returned canonical (sorted, merged)."""
    return normalize(list(a) + list(b))


def intersect(a: list[Interval], b: list[Interval]) -> list[Interval]:
    """Intersection of two interval lists, returned canonical.

    Both inputs are normalized first so the sweep below can assume each list
    is individually sorted and non-overlapping.
    """
    norm_a = normalize(a)
    norm_b = normalize(b)
    result: list[Interval] = []
    i, j = 0, 0
    while i < len(norm_a) and j < len(norm_b):
        a_start, a_end = norm_a[i]
        b_start, b_end = norm_b[j]
        lo = max(a_start, b_start)
        hi = min(a_end, b_end)
        if lo < hi:
            result.append((lo, hi))
        if a_end < b_end:
            i += 1
        else:
            j += 1
    return result


def subtract(a: list[Interval], b: list[Interval]) -> list[Interval]:
    """Return ``a`` with every sub-interval overlapping ``b`` removed.

    Both inputs are normalized first. Implemented by walking ``a`` and
    clipping out every ``b`` interval that overlaps the current remainder.
    """
    norm_a = normalize(a)
    norm_b = normalize(b)
    result: list[Interval] = []
    for a_start, a_end in norm_a:
        cur = a_start
        for b_start, b_end in norm_b:
            if b_end <= cur or b_start >= a_end:
                continue
            if b_start > cur:
                result.append((cur, min(b_start, a_end)))
            cur = max(cur, b_end)
            if cur >= a_end:
                break
        if cur < a_end:
            result.append((cur, a_end))
    return result


def total_duration(intervals: list[Interval]) -> np.timedelta64:
    """Sum of ``(end - start)`` across all intervals (input need not be normalized).

    Callers that need the "no double counting" guarantee for overlapping
    input must call :func:`normalize` first.
    """
    total = np.timedelta64(0, "ns")
    for start, end in intervals:
        total = total + (end - start).to_timedelta64()
    return total


def overlaps(interval: Interval, candidates: list[Interval]) -> bool:
    """True iff ``interval`` overlaps any interval in ``candidates`` (half-open)."""
    s, e = interval
    return any(cs < e and ce > s for cs, ce in candidates)
