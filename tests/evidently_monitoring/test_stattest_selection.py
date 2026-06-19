"""Verify that the explicit stattest pin overrides Evidently's size-based auto-selection.

Evidently auto-selects the numerical drift test by reference sample size:
  n <= 1000  → K-S   (threshold = p-value)
  n > 1000   → Wasserstein distance (normed)  (threshold = distance)

We pin num_stattest="wasserstein" explicitly in both batch (reports.py) and
real-time (drift.py) so the test is deterministic regardless of n — and so the
meaning of feature_drift_threshold (a Wasserstein distance) is consistent everywhere.

These tests confirm the pin actually overrides auto-selection, including when the
reference size would normally trip the KS branch.

Run with:
    pytest tests/evidently_monitoring/test_stattest_selection.py -v -s -m slow
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spacecraft_telemetry.core.config import Settings
from spacecraft_telemetry.evidently_monitoring.reference import compute_feature_dataframe
from spacecraft_telemetry.evidently_monitoring.reports import run_drift_report

_WASSERSTEIN_DISPLAY_NAME = "Wasserstein distance (normed)"


def _make_feature_df(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2020-01-01", periods=n, freq="1s", tz="UTC")
    raw = pd.DataFrame(
        {
            "telemetry_timestamp": timestamps,
            "value_normalized": rng.standard_normal(n).astype(np.float32),
        }
    )
    return compute_feature_dataframe(raw, Settings())


def _stattest_names(report) -> dict[str, str]:
    by_name = {m["metric"]: m["result"] for m in report.as_dict()["metrics"]}
    return {
        col: info.get("stattest_name", "unknown")
        for col, info in by_name["DataDriftTable"]["drift_by_columns"].items()
    }


@pytest.mark.slow
def test_wasserstein_used_at_small_n(capsys) -> None:
    """Explicit pin forces Wasserstein even when n=256 would auto-select K-S."""
    n = 256
    ref = _make_feature_df(n=n, seed=0)
    cur = _make_feature_df(n=n, seed=1)
    report, _ = run_drift_report(ref, cur, Settings())

    names = _stattest_names(report)
    unique = set(names.values())

    with capsys.disabled():
        print(f"\n--- Small n (n={n}, would auto-select K-S without pin) ---")
        print(f"  Unique tests used: {unique}")

    assert unique == {_WASSERSTEIN_DISPLAY_NAME}, (
        f"Expected Wasserstein at n={n} (explicit pin), got {unique}"
    )


@pytest.mark.slow
def test_wasserstein_used_at_prod_asymmetric_shape(capsys) -> None:
    """Production real-time shape: reference=5000, current window=256 → Wasserstein."""
    ref = _make_feature_df(n=5000, seed=0)
    cur = _make_feature_df(n=256, seed=1)
    report, _ = run_drift_report(ref, cur, Settings())

    names = _stattest_names(report)
    unique = set(names.values())

    with capsys.disabled():
        print("\n--- Production asymmetric shape (ref=5000, cur=256) ---")
        print(f"  Unique tests used: {unique}")

    assert unique == {_WASSERSTEIN_DISPLAY_NAME}


@pytest.mark.slow
def test_wasserstein_used_at_batch_scale(capsys) -> None:
    """Batch reference: n=5000 rows — Wasserstein, as before the pin."""
    n = 5000
    ref = _make_feature_df(n=n, seed=0)
    cur = _make_feature_df(n=n, seed=1)
    report, _ = run_drift_report(ref, cur, Settings())

    names = _stattest_names(report)
    unique = set(names.values())

    with capsys.disabled():
        print(f"\n--- Batch scale (n={n}) ---")
        print(f"  Unique tests used: {unique}")

    assert unique == {_WASSERSTEIN_DISPLAY_NAME}
