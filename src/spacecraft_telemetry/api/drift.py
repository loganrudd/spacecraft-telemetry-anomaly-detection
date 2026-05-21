"""Real-time rolling drift monitor for the FastAPI serving layer (Plan 0095).

``RollingDriftMonitor`` maintains a fixed-size rolling window of telemetry ticks
for a single channel and periodically computes Evidently drift scores against a
reference profile loaded at startup.

Public API
----------
FeatureDrift
    Per-feature drift score and flag for one Evidently run.
DriftSnapshot
    Aggregate result from one Evidently run for a single channel.
RollingDriftMonitor
    Stateful monitor: push ticks, check cadence, run drift asynchronously.
"""

from __future__ import annotations

import asyncio
import warnings
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd
from evidently.legacy.metric_preset import DataDriftPreset
from evidently.legacy.pipeline.column_mapping import ColumnMapping
from evidently.legacy.report import Report

from spacecraft_telemetry.evidently_monitoring.reference import REALTIME_FEATURE_COLS


@dataclass
class FeatureDrift:
    """Drift result for a single monitored feature column."""

    feature: str
    score: float    # Wasserstein distance (Evidently default for numerical columns)
    drifted: bool   # True if KS p-value < threshold


@dataclass
class DriftSnapshot:
    """Aggregate drift result for one channel from a single Evidently run."""

    timestamp: datetime
    channel: str
    features: list[FeatureDrift]
    percent_drifted: float  # fraction of features flagged as drifted, in [0, 1]
    drifted: bool           # True if percent_drifted >= channel_drift_threshold


class RollingDriftMonitor:
    """Maintains a rolling window per channel and runs Evidently periodically.

    Each ``push`` appends one telemetry tick to the internal deque
    (evicting the oldest tick once ``window_size`` is reached).
    ``should_run`` returns True every ``tick_interval`` ticks once the
    window is full.  ``run`` offloads the Evidently ``Report.run()`` call to a
    thread via ``asyncio.to_thread`` so the event loop is never blocked.

    Args:
        channel:                 Channel identifier string.
        reference:               Reference-profile DataFrame (MONITORING_FEATURE_COLS columns).
        window_size:             Rolling window capacity in ticks.
        tick_interval:           Number of ticks between Evidently runs.
        feature_drift_threshold: Fraction of features that must drift to flag the channel.
        channel_drift_threshold: Per-feature drift threshold (Evidently threshold parameter).
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
        self._col_mapping = ColumnMapping(numerical_features=REALTIME_FEATURE_COLS)

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

        Off-loads ``Report.run()`` (≈58 ms p50) to a thread pool executor via
        ``asyncio.to_thread`` so the event loop is never blocked for more than a
        few microseconds.

        Returns:
            A ``DriftSnapshot`` if the window is full, else ``None``.
        """
        if len(self._window) < self._window_size:
            return None
        current = pd.DataFrame(list(self._window))
        return await asyncio.to_thread(self._compute_drift, current)

    @staticmethod
    def _add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
        """Fill rate_of_change from value_normalized.

        The tick bus only pushes value_normalized; rate_of_change is derived here
        so Evidently receives a non-NaN column.  Rolling stats (rolling_mean_*, etc.)
        are deliberately excluded from real-time comparison — their distributions are
        only reliable when the buffer is much longer than the rolling window period,
        which cannot be guaranteed at serve time.  See REALTIME_FEATURE_COLS.
        """
        df["rate_of_change"] = df["value_normalized"].diff().fillna(0.0)
        return df

    def _compute_drift(self, current: pd.DataFrame) -> DriftSnapshot:
        """Run Evidently synchronously — called from a thread pool worker."""
        current = self._add_rolling_features(current)
        report = Report(metrics=[DataDriftPreset()])
        with warnings.catch_warnings():
            # Evidently triggers numpy divide-by-zero warnings on constant-value
            # columns (zero variance).  These are benign and would spam logs at
            # every drift evaluation for flat-signal channels.
            warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
            report.run(
                reference_data=self._reference,
                current_data=current[REALTIME_FEATURE_COLS],
                column_mapping=self._col_mapping,
            )

        by_name = {m["metric"]: m["result"] for m in report.as_dict()["metrics"]}
        drift_table = by_name["DataDriftTable"]["drift_by_columns"]

        features: list[FeatureDrift] = []
        for col in REALTIME_FEATURE_COLS:
            col_info = drift_table.get(col, {})
            score = float(col_info.get("drift_score", 0.0))
            drifted = bool(col_info.get("drift_detected", False))
            features.append(FeatureDrift(feature=col, score=score, drifted=drifted))

        n_drifted = sum(f.drifted for f in features)
        percent_drifted = n_drifted / len(features) if features else 0.0

        return DriftSnapshot(
            timestamp=datetime.now(UTC),
            channel=self._channel,
            features=features,
            percent_drifted=percent_drifted,
            drifted=percent_drifted >= self._channel_drift_threshold,
        )
