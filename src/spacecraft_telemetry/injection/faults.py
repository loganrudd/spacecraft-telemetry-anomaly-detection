"""Pure-numpy anomaly injection primitives (Phase 15).

All functions operate on a 1-D ``value_normalized`` array already in z-score
units (sigma).  Amplitudes are therefore in sigma, no normalization_params lookup needed.

Mirror discipline: pure numpy, no torch, no I/O — same as model/scoring.py.
This module is also the future building block for Phase 16/17 serving-side
tick-by-tick injection (same math, different call site).

Public API
----------
inject_spike(values, start, magnitude_sigma, duration)
    → (values_out, mask)

inject_drift(values, start, duration, total_shift_sigma)
    → (values_out, mask)

inject_flatline(values, start, duration)
    → (values_out, mask)

inject_faults(values, segment_ids, is_los, rng, cfg, channel_profile)
    → (values_out, is_anomaly_mask, fault_records)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Primitive injectors — each returns (modified copy, boolean mask)
# ---------------------------------------------------------------------------

def inject_spike(
    values: np.ndarray[Any, Any],
    start: int,
    magnitude_sigma: float,
    duration: int = 1,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, np.dtype[np.bool_]]]:
    """Inject an additive spike burst.

    Adds ±magnitude_sigma to values[start : start+duration].  The sign is
    chosen so the spike moves *away* from the local pre-window mean (amplifies
    the deviation rather than cancelling it — maximises forecast residual).

    Args:
        values:          1-D float32 normalized series.
        start:           First index of the fault (inclusive).
        magnitude_sigma: Burst amplitude in sigma units.
        duration:        Number of steps the spike lasts (default 1 = point spike).

    Returns:
        (values_out, mask) — modified copy and boolean mask (True over [start, start+duration)).
    """
    n = len(values)
    end = min(start + duration, n)
    out = values.copy()

    # Pre-window mean to choose sign
    window = values[max(0, start - 20) : start]
    pre_mean = float(window.mean()) if len(window) > 0 else 0.0
    sign = 1.0 if values[start] >= pre_mean else -1.0

    out[start:end] = out[start:end] + sign * magnitude_sigma

    mask = np.zeros(n, dtype=bool)
    mask[start:end] = True
    return out, mask


def inject_drift(
    values: np.ndarray[Any, Any],
    start: int,
    duration: int,
    total_shift_sigma: float,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, np.dtype[np.bool_]]]:
    """Inject a linear ramp then hold.

    Adds a linearly increasing offset from 0 to total_shift_sigma over
    [start, start+duration), then holds the constant offset for the remainder
    of the duration.  The hold ensures the anomaly segment has sustained
    deviation rather than returning to nominal.

    Args:
        values:            1-D float32 normalized series.
        start:             First index of the fault.
        duration:          Total length of the drift segment (ramp + hold).
        total_shift_sigma: Total offset reached at end of ramp, in sigma units.

    Returns:
        (values_out, mask).
    """
    n = len(values)
    end = min(start + duration, n)
    actual_dur = end - start
    out = values.copy()

    ramp_len = max(1, actual_dur // 2)
    hold_len = actual_dur - ramp_len

    ramp = np.linspace(0.0, total_shift_sigma, ramp_len, dtype=np.float64)
    hold = np.full(hold_len, total_shift_sigma, dtype=np.float64)
    offset = np.concatenate([ramp, hold])

    out[start:end] = (out[start:end].astype(np.float64) + offset).astype(np.float32)

    mask = np.zeros(n, dtype=bool)
    mask[start:end] = True
    return out, mask


def inject_flatline(
    values: np.ndarray[Any, Any],
    start: int,
    duration: int,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, np.dtype[np.bool_]]]:
    """Replace a segment with the last pre-fault value (sensor death).

    Holds the value immediately before ``start`` constant over
    [start, start+duration).

    Args:
        values:   1-D float32 normalized series.
        start:    First index of the fault (must be > 0 for meaningful pre-value).
        duration: Length of the flatline segment.

    Returns:
        (values_out, mask).
    """
    n = len(values)
    end = min(start + duration, n)
    out = values.copy()

    pre_value = float(values[start - 1]) if start > 0 else float(values[start])
    out[start:end] = np.float32(pre_value)

    mask = np.zeros(n, dtype=bool)
    mask[start:end] = True
    return out, mask


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

_VALID_FAULT_TYPES: frozenset[str] = frozenset({"spike", "drift", "flatline"})
_RANGE_KEYS = (
    "magnitude_sigma_range",
    "spike_duration_range",
    "drift_duration_range",
    "flatline_duration_range",
)


@dataclass
class ChannelProfile:
    """Per-channel injection profile (from configs/injection_profiles.json)."""

    signal_class: str = "slow_lownoise"
    fault_type_weights: dict[str, float] = field(
        default_factory=lambda: {"drift": 0.45, "flatline": 0.45, "spike": 0.10}
    )
    magnitude_sigma_range: list[float] = field(default_factory=lambda: [0.4, 1.5])
    spike_duration_range: list[int] = field(default_factory=lambda: [1, 5])
    drift_duration_range: list[int] = field(default_factory=lambda: [30, 300])
    flatline_duration_range: list[int] = field(default_factory=lambda: [30, 300])

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ChannelProfile:
        """Build a ChannelProfile from a dict, validating keys and ranges.

        Raises ValueError on unknown fault_type_weights keys, all-zero weights,
        or any *_range where lo >= hi.
        """
        if "fault_type_weights" in d:
            unknown = set(d["fault_type_weights"]) - _VALID_FAULT_TYPES
            if unknown:
                raise ValueError(
                    f"Unknown fault_type_weights keys: {sorted(unknown)!r}. "
                    f"Valid keys: {sorted(_VALID_FAULT_TYPES)!r}"
                )
            if all(v <= 0 for v in d["fault_type_weights"].values()):
                raise ValueError("All fault_type_weights are <= 0; at least one must be positive")

        for key in _RANGE_KEYS:
            if key in d:
                lo, hi = d[key]
                if lo >= hi:
                    raise ValueError(f"{key} must have lo < hi, got [{lo}, {hi}]")

        return cls(
            signal_class=d.get("signal_class", "slow_lownoise"),
            fault_type_weights=d.get(
                "fault_type_weights",
                {"drift": 0.45, "flatline": 0.45, "spike": 0.10},
            ),
            magnitude_sigma_range=d.get("magnitude_sigma_range", [0.4, 1.5]),
            spike_duration_range=d.get("spike_duration_range", [1, 5]),
            drift_duration_range=d.get("drift_duration_range", [30, 300]),
            flatline_duration_range=d.get("flatline_duration_range", [30, 300]),
        )


def inject_faults(
    values: np.ndarray[Any, Any],
    segment_ids: np.ndarray[Any, Any],
    is_los: np.ndarray[Any, Any],
    rng: np.random.Generator,
    faults_per_channel: int,
    profile: ChannelProfile,
    *,
    hpo_fraction: float = 0.6,
    min_gap: int = 50,
    window_size: int = 250,
    flat_variance_threshold: float = 0.01,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], list[dict[str, Any]]]:
    """Place a reproducible set of faults across a series.

    Placement constraints
    ---------------------
    - Never inject inside ``is_los`` buckets.
    - Never inject across ``segment_id`` boundaries — windows can't span them,
      so a fault straddling a boundary is undetectable.
    - Distribute faults so BOTH the HPO portion (first ``hpo_fraction``) and the
      held-out portion (last 1-hpo_fraction) receive at least one fault.  This
      prevents the "no labeled anomaly windows" warnings from _prepare_channel_data.
    - For flatline faults: skip segments where local pre-window variance is already
      ~0 (eclipse-flat regions on power channels → label poison).
    - Maintain a minimum gap of ``min_gap`` steps between fault endpoints.

    Args:
        values:               1-D float32 normalized series.
        segment_ids:          (N,) int32 segment IDs (gap boundaries).
        is_los:               (N,) bool LOS flag.
        rng:                  Seeded numpy RNG for determinism.
        faults_per_channel:   Total number of faults to inject.
        profile:              Per-channel injection parameters.
        hpo_fraction:         Fraction of windows assigned to HPO (must be covered).
        min_gap:              Minimum steps between consecutive faults.
        window_size:          Model window size (used to ensure fault ends before series end).
        flat_variance_threshold: Pre-window variance below which flatline is skipped.

    Returns:
        (values_out, is_anomaly_mask, fault_records)
        fault_records: list of dicts with keys
            {type, start, end, duration, magnitude_sigma, signal_class}
    """
    n = len(values)
    n_hpo = int(n * hpo_fraction)

    # Build candidate start positions: not in LOS, not in last window (needs tail)
    candidates = np.flatnonzero(~is_los)
    candidates = candidates[candidates < n - window_size]
    if len(candidates) == 0:
        return values.copy(), np.zeros(n, dtype=bool), []

    # Build segment lookup in one O(N) pass using run boundaries (np.diff trick).
    # Segments are contiguous blocks (monotonically assigned by detect_gaps),
    # so a single scan suffices — no per-segment boolean mask needed.
    seg_extents: dict[int, tuple[int, int]] = {}
    if n > 0:
        boundaries = np.flatnonzero(np.diff(segment_ids) != 0) + 1
        starts = np.r_[0, boundaries]
        ends = np.r_[boundaries, n]
        for s, e in zip(starts.tolist(), ends.tolist(), strict=True):
            seg_extents[int(segment_ids[s])] = (s, e)

    # Fault type sampling
    types = list(profile.fault_type_weights.keys())
    weights = np.array([profile.fault_type_weights[t] for t in types], dtype=float)
    weights = weights / weights.sum()

    out = values.copy()
    anomaly_mask = np.zeros(n, dtype=bool)
    fault_records: list[dict[str, Any]] = []
    occupied_end = -min_gap  # last occupied position

    # We want at least 1 fault in HPO portion and 1 in held-out portion.
    # Split budget: allocate proportionally, minimum 1 each if budget allows.
    n_hpo_faults = max(1, round(faults_per_channel * hpo_fraction))
    n_held_faults = max(1, faults_per_channel - n_hpo_faults)

    def _sample_fault_params(fault_type: str) -> tuple[int, float | None]:
        """Return (duration, magnitude_sigma) for a fault type.

        Flatline does not use a magnitude; returns None to avoid a meaningless draw
        that would also shift the RNG stream for subsequent spike/drift placements.
        """
        if fault_type == "spike":
            lo, hi = profile.spike_duration_range
            mag_low, mag_high = profile.magnitude_sigma_range
            magnitude: float | None = float(rng.uniform(mag_low, mag_high))
        elif fault_type == "drift":
            lo, hi = profile.drift_duration_range
            mag_low, mag_high = profile.magnitude_sigma_range
            magnitude = float(rng.uniform(mag_low, mag_high))
        else:
            lo, hi = profile.flatline_duration_range
            magnitude = None

        duration = int(rng.integers(lo, max(lo + 1, hi + 1)))
        return duration, magnitude

    def _try_place(pool: np.ndarray[Any, Any], fault_type: str) -> bool:
        """Try to inject one fault into a random candidate from ``pool``."""
        nonlocal occupied_end

        if len(pool) == 0:
            return False

        # Shuffle and try up to 30 positions
        idxs = rng.permutation(len(pool))[:30]
        for idx in idxs:
            start = int(pool[idx])
            if start < occupied_end + min_gap:
                continue

            # Ensure the fault stays within its segment
            seg_id = int(segment_ids[start])
            seg_start, seg_end = seg_extents[seg_id]
            if start <= seg_start:
                continue  # need at least one pre-fault value for flatline sign

            duration, magnitude = _sample_fault_params(fault_type)
            end = min(start + duration, seg_end, n - 1)
            actual_duration = end - start
            if actual_duration < 1:
                continue
            # linspace(0, shift, 1) == [0.0]: zero offset applied with mask=True → poisoned label.
            if fault_type == "drift" and actual_duration < 2:
                continue

            # Flatline guard: skip if pre-window variance is already near zero
            if fault_type == "flatline":
                pre_win = values[max(seg_start, start - window_size) : start]
                if len(pre_win) > 1 and float(pre_win.var()) < flat_variance_threshold:
                    continue

            # Apply the fault — each primitive returns a full-length array;
            # splice only the fault region back into out so previously-injected
            # faults outside [start:end] are not overwritten.
            if fault_type == "spike":
                assert magnitude is not None  # spike always draws a magnitude
                full_out, _ = inject_spike(out, start, magnitude, actual_duration)
            elif fault_type == "drift":
                assert magnitude is not None  # drift always draws a magnitude
                full_out, _ = inject_drift(out, start, actual_duration, magnitude)
            else:
                full_out, _ = inject_flatline(out, start, actual_duration)

            out[start:end] = full_out[start:end]
            anomaly_mask[start:end] = True

            occupied_end = end
            fault_records.append({
                "type": fault_type,
                "start": start,
                "end": end,
                "duration": actual_duration,
                "magnitude_sigma": magnitude,
                "signal_class": profile.signal_class,
            })
            return True
        return False

    # Place faults in HPO portion
    hpo_pool = candidates[candidates < n_hpo]
    for _ in range(n_hpo_faults):
        ft = str(rng.choice(types, p=weights))
        _try_place(hpo_pool, ft)

    # Place faults in held-out portion
    held_pool = candidates[candidates >= n_hpo]
    occupied_end = -min_gap  # reset gap tracking for held-out region
    for _ in range(n_held_faults):
        ft = str(rng.choice(types, p=weights))
        _try_place(held_pool, ft)

    return out, anomaly_mask, fault_records
