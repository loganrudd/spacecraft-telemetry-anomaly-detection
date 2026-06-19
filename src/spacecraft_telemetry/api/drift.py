"""Real-time rolling drift monitor for the FastAPI serving layer (Plan 0095).

``RollingDriftMonitor`` maintains a fixed-size rolling window of telemetry ticks
for a single channel and periodically computes Wasserstein drift scores against a
reference profile loaded at startup.

Evidently is NOT used on the hot serving path — only scipy is. At 100-1000x replay
speeds, the ~58ms Evidently Report overhead per run would saturate the thread pool
and cap effective replay speed. Direct scipy gives the same statistic in <1ms.
Evidently is retained for batch reports (evidently_monitoring/reports.py) where
the HTML report is the actual deliverable.

Public API
----------
FeatureDrift
    Per-feature drift score and flag for one drift run.
DriftSnapshot
    Aggregate result from one drift run for a single channel.
RollingDriftMonitor
    Stateful monitor: push ticks, check cadence, run drift asynchronously.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from scipy import stats

from spacecraft_telemetry.evidently_monitoring.reference import REALTIME_FEATURE_COLS


@dataclass
class FeatureDrift:
    """Drift result for a single monitored feature column."""

    feature: str
    score: float    # Wasserstein distance (normed): W1(ref, cur) / std(ref)
    drifted: bool   # True if score >= feature_drift_threshold


@dataclass
class DriftSnapshot:
    """Aggregate drift result for one channel from a single drift run."""

    timestamp: datetime
    channel: str
    features: list[FeatureDrift]
    percent_drifted: float  # fraction of features flagged as drifted, in [0, 1]
    drifted: bool           # True if percent_drifted >= channel_drift_threshold


class RollingDriftMonitor:
    """Maintains a rolling window per channel and computes Wasserstein drift periodically.

    Each ``push`` appends one telemetry tick to the internal deque
    (evicting the oldest tick once ``window_size`` is reached).
    ``should_run`` returns True every ``tick_interval`` ticks once the
    window is full.  ``run`` offloads the scipy computation to a thread
    via ``asyncio.to_thread`` so the event loop is never blocked.

    Drift is computed as normed Wasserstein distance per feature column,
    matching the formula used by Evidently's batch reports:
        score = scipy.stats.wasserstein_distance(ref, cur) / max(std(ref), 0.001)
    This keeps batch and real-time scores directly comparable.

    Args:
        channel:                 Channel identifier string.
        reference:               Reference-profile DataFrame (MONITORING_FEATURE_COLS columns).
        window_size:             Rolling window capacity in ticks.
        tick_interval:           Number of ticks between drift runs.
        feature_drift_threshold: Normed Wasserstein distance threshold for per-feature drift.
        channel_drift_threshold: Fraction of features that must drift to flag the channel.
    """

    def __init__(
        self,
        *,
        channel: str,
        reference: pd.DataFrame,
        window_size: int,
        tick_interval: int,
        feature_drift_threshold: float,
        channel_drift_threshold: float,
    ) -> None:
        self._channel = channel
        self._reference = reference[REALTIME_FEATURE_COLS].copy()
        self._window: deque[dict[str, float]] = deque(maxlen=window_size)
        self._window_size = window_size
        self._tick_interval = tick_interval
        self._feature_drift_threshold = feature_drift_threshold
        self._channel_drift_threshold = channel_drift_threshold
        self._tick_count: int = 0

    def push(self, row: dict[str, float]) -> None:
        """Append one tick's values to the rolling window.

        Args:
            row: At minimum ``{"value_normalized": float}``.  ``rate_of_change``
                 is recomputed from value_normalized inside ``_compute_drift``
                 so it does not need to be supplied here.
        """
        self._window.append({col: row.get(col, float("nan")) for col in REALTIME_FEATURE_COLS})
        self._tick_count += 1

    def should_run(self) -> bool:
        """Return True when the window is full and a tick-interval boundary is crossed."""
        return (
            len(self._window) >= self._window_size
            and self._tick_count % self._tick_interval == 0
        )

    async def run(self) -> DriftSnapshot | None:
        """Compute drift against the reference profile.

        Off-loads scipy computation (<1ms p50) to a thread pool executor via
        ``asyncio.to_thread`` so the event loop is never blocked.

        Returns:
            A ``DriftSnapshot`` if the window is full, else ``None``.
        """
        if len(self._window) < self._window_size:
            return None
        current = pd.DataFrame(list(self._window))
        return await asyncio.to_thread(self._compute_drift, current)

    @staticmethod
    def _add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
        """Fill rate_of_change from value_normalized diffs.

        The tick bus only pushes value_normalized; rate_of_change is derived here.
        NaN is intentionally not filled: the first-row NaN from diff() and any
        all-NaN windows are handled by dropna() in _compute_drift, which returns
        score=0 for empty arrays rather than computing against spurious zeros.
        Rolling stats (rolling_mean_*, etc.) are deliberately excluded from
        real-time comparison — their distributions are only reliable when the
        buffer is much longer than the rolling window period.  See REALTIME_FEATURE_COLS.
        """
        df["rate_of_change"] = df["value_normalized"].diff()
        return df

    def _compute_drift(self, current: pd.DataFrame) -> DriftSnapshot:
        """Compute normed Wasserstein distance per feature via scipy.

        Mirrors the formula in Evidently's wasserstein_distance_norm.py exactly:
            score = wasserstein_distance(ref, cur) / max(std(ref), 0.001)
        so scores are directly comparable to batch drift reports.
        Empty columns (all-NaN flatlined channel) return score=0, drifted=False.
        """
        current = self._add_rolling_features(current)
        features: list[FeatureDrift] = []

        for col in REALTIME_FEATURE_COLS:
            ref_vals = self._reference[col].dropna().to_numpy(dtype=float)
            cur_vals = current[col].dropna().to_numpy(dtype=float)

            if len(ref_vals) == 0 or len(cur_vals) == 0:
                features.append(FeatureDrift(feature=col, score=0.0, drifted=False))
                continue

            norm = max(float(np.std(ref_vals)), 0.001)
            score = float(stats.wasserstein_distance(ref_vals, cur_vals)) / norm
            features.append(FeatureDrift(
                feature=col,
                score=score,
                drifted=score >= self._feature_drift_threshold,
            ))

        n_drifted = sum(f.drifted for f in features)
        percent_drifted = n_drifted / len(features) if features else 0.0

        return DriftSnapshot(
            timestamp=datetime.now(UTC),
            channel=self._channel,
            features=features,
            percent_drifted=percent_drifted,
            drifted=percent_drifted >= self._channel_drift_threshold,
        )
