"""Framework-agnostic feature definitions for spacecraft telemetry.

This module is the single source of truth for what each feature means and how
it is computed. Every phase that produces or consumes features imports from here:

    Phase 2 (Spark)   — translates each FeatureDefinition to a Spark window
                        function; tests assert Spark output matches compute_numpy
    Phase 3 (Feast)   — generates FeatureView schema from FEATURE_DEFINITIONS
    Phase 9 (FastAPI) — calls compute_features_numpy() on a sliding value buffer
                        and pushes results to the Feast online store
    Phase 8 (Evidently) — references feature names for drift detection columns

Adding a new feature here automatically propagates it to all phases that iterate
FEATURE_DEFINITIONS — no hardcoded column lists elsewhere.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FeatureDefinition:
    """A single engineered feature: name, type, description, and reference impl.

    Attributes:
        name: Column name used everywhere (Parquet, Feast, Evidently).
        dtype: Feast/Spark type string — "float32" for all current features.
        description: Plain-English definition. Phase 9 must match this exactly.
        window_size: Number of values the computation looks back over.
                     None means the feature uses a fixed look-back (e.g. 2 for ROC).
        compute_numpy: Pure-NumPy reference implementation.
                       Signature: (values: np.ndarray, timestamps_s: np.ndarray) -> float
                       - values: 1-D array of the most recent raw (normalized) readings,
                         length >= window_size, most recent last
                       - timestamps_s: matching Unix timestamps in seconds, same length
                       Returns NaN when there are insufficient values.
    """

    name: str
    dtype: str
    description: str
    window_size: int
    compute_numpy: Callable[[np.ndarray, np.ndarray], float]


def _rolling_mean(n: int) -> Callable[[np.ndarray, np.ndarray], float]:
    def _fn(values: np.ndarray, _ts: np.ndarray) -> float:
        if len(values) < n:
            return float("nan")
        return float(np.mean(values[-n:]))

    return _fn


def _rolling_std(n: int) -> Callable[[np.ndarray, np.ndarray], float]:
    def _fn(values: np.ndarray, _ts: np.ndarray) -> float:
        if len(values) < n:
            return float("nan")
        return float(np.std(values[-n:], ddof=1))

    return _fn


def _rolling_min(n: int) -> Callable[[np.ndarray, np.ndarray], float]:
    def _fn(values: np.ndarray, _ts: np.ndarray) -> float:
        if len(values) < n:
            return float("nan")
        return float(np.min(values[-n:]))

    return _fn


def _rolling_max(n: int) -> Callable[[np.ndarray, np.ndarray], float]:
    def _fn(values: np.ndarray, _ts: np.ndarray) -> float:
        if len(values) < n:
            return float("nan")
        return float(np.max(values[-n:]))

    return _fn


def _rate_of_change(values: np.ndarray, timestamps_s: np.ndarray) -> float:
    """(value[t] - value[t-1]) / (timestamp[t] - timestamp[t-1]) in seconds."""
    if len(values) < 2 or len(timestamps_s) < 2:
        return float("nan")
    dt = float(timestamps_s[-1] - timestamps_s[-2])
    if dt == 0.0:
        return float("nan")
    return float((values[-1] - values[-2]) / dt)


def _build_registry(windows: list[int]) -> list[FeatureDefinition]:
    defs: list[FeatureDefinition] = []
    for n in windows:
        defs += [
            FeatureDefinition(
                name=f"rolling_mean_{n}",
                dtype="float32",
                description=f"Mean of previous {n} values (inclusive, trailing window)",
                window_size=n,
                compute_numpy=_rolling_mean(n),
            ),
            FeatureDefinition(
                name=f"rolling_std_{n}",
                dtype="float32",
                description=f"Sample std dev of previous {n} values (ddof=1)",
                window_size=n,
                compute_numpy=_rolling_std(n),
            ),
            FeatureDefinition(
                name=f"rolling_min_{n}",
                dtype="float32",
                description=f"Minimum of previous {n} values",
                window_size=n,
                compute_numpy=_rolling_min(n),
            ),
            FeatureDefinition(
                name=f"rolling_max_{n}",
                dtype="float32",
                description=f"Maximum of previous {n} values",
                window_size=n,
                compute_numpy=_rolling_max(n),
            ),
        ]
    defs.append(
        FeatureDefinition(
            name="rate_of_change",
            dtype="float32",
            description="(value[t] - value[t-1]) / interval_seconds — first derivative",
            window_size=2,
            compute_numpy=_rate_of_change,
        )
    )
    return defs


# Default windows match SparkConfig.feature_windows default — [10, 50, 100].
# If SparkConfig is changed, update this list too.
_DEFAULT_WINDOWS: list[int] = [10, 50, 100]

FEATURE_DEFINITIONS: list[FeatureDefinition] = _build_registry(_DEFAULT_WINDOWS)

# Fast lookup by name — built once at import time.
_BY_NAME: dict[str, FeatureDefinition] = {f.name: f for f in FEATURE_DEFINITIONS}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_feature_names() -> list[str]:
    """Return all feature names in registry order."""
    return [f.name for f in FEATURE_DEFINITIONS]


def get_feature_by_name(name: str) -> FeatureDefinition:
    """Return a FeatureDefinition by name.

    Raises:
        KeyError: if name is not in the registry.
    """
    try:
        return _BY_NAME[name]
    except KeyError as err:
        raise KeyError(f"Unknown feature {name!r}. Known: {list(_BY_NAME)}") from err


def compute_features_numpy(
    values: np.ndarray,
    timestamps_s: np.ndarray,
) -> dict[str, float]:
    """Compute all registered features for the most recent point in a buffer.

    This is the reference implementation consumed directly by:
    - Phase 9 FastAPI streaming loop (one call per incoming telemetry tick)
    - Phase 2 Spark equivalence tests (validate Spark output matches this)

    Args:
        values: 1-D NumPy array of recent raw (z-score normalized) values,
                most recent last. Should contain at least max(window_size)
                values for all features to be non-NaN.
        timestamps_s: Matching Unix timestamps in seconds, same shape as values.

    Returns:
        Dict mapping feature name → float value. NaN where buffer is too short.
    """
    values = np.asarray(values, dtype=np.float64)
    timestamps_s = np.asarray(timestamps_s, dtype=np.float64)
    return {fd.name: fd.compute_numpy(values, timestamps_s) for fd in FEATURE_DEFINITIONS}
