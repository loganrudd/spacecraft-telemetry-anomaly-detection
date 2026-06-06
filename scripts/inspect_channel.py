"""Inspect one channel's prediction error vs dynamic threshold at its labeled anomalies.

When a channel forecasts well (low val_loss) but still scores seg_f0_5 = 0, there
are two very different causes — this tells them apart:

  Case 1 — forecaster tracks the anomalies. The error barely rises during labeled
           anomalies (spike_ratio ≈ 1), so there's nothing for the threshold to
           catch. A Telemanom blind spot: it only flags anomalies that *surprise*
           the forecaster. Accept or drop the channel.

  Case 2 — threshold too high. The error clearly spikes at anomalies (spike_ratio
           ≫ 1) but stays under the dynamic threshold, so nothing is flagged. The
           subsystem's tuned threshold_z is too conservative for this channel.

  (A third readout: if the error *does* cross the threshold at anomalies yet
   seg_f0_5 is still 0, the flagged runs were dropped by threshold_min_anomaly_len
   pruning.)

Pulls the channel's scoring run from MLflow (errors.npy = smoothed errors,
threshold.npy = dynamic threshold) and the per-window labels, all aligned.

Usage:
    SPACECRAFT_MLFLOW__TRACKING_URI=http://localhost:8090 \\
    SPACECRAFT_PREPROCESS__PROCESSED_DATA_DIR=gs://<PROJECT_ID>-processed-data \\
    SSL_CERT_FILE=$(python -m certifi) \\
    python scripts/inspect_channel.py --env cloud --mission ESA-Mission1 --channel channel_42

Requires: .[tracking,gcp]. configure_mlflow handles Cloud Run GCP auth (or use a
`gcloud run services proxy` localhost URI, which needs no token).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Allow running as a script without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spacecraft_telemetry.core.config import load_settings
from spacecraft_telemetry.core.logging import get_logger
from spacecraft_telemetry.mlflow_tracking import configure_mlflow, experiment_name
from spacecraft_telemetry.model.dataset import load_window_labels
from spacecraft_telemetry.model.io import (
    bytes_to_errors,
    download_artifact_bytes,
    find_latest_run_for_channel,
)

log = get_logger(__name__)


def _diagnose(
    smoothed: np.ndarray[Any, Any],
    threshold: np.ndarray[Any, Any],
    labels: np.ndarray[Any, Any],
) -> dict[str, float]:
    """Compare smoothed error vs threshold at anomaly windows vs nominal windows.

    All three arrays are per-window and aligned (test-loader order). threshold[0]
    and warmup positions can be inf — excluded so they don't skew the medians.
    """
    labels = labels.astype(bool)
    finite = np.isfinite(threshold)
    nominal = smoothed[(~labels) & finite]
    anom = smoothed[labels & finite]
    anom_thr = threshold[labels & finite]

    nominal_med = float(np.median(nominal)) if nominal.size else float("nan")
    anom_med = float(np.median(anom)) if anom.size else float("nan")
    return {
        "n_anomaly_windows": int(labels.sum()),
        "nominal_err_median": nominal_med,
        "anomaly_err_median": anom_med,
        # How much bigger the error is at anomalies than at nominal points.
        "spike_ratio": (anom_med / nominal_med) if nominal_med else float("nan"),
        "threshold_median_at_anomalies": (
            float(np.median(anom_thr)) if anom_thr.size else float("nan")
        ),
        # Fraction of anomaly windows whose error actually clears the threshold.
        "frac_above_threshold": float(np.mean(anom > anom_thr)) if anom.size else 0.0,
    }


def _verdict(d: dict[str, float]) -> str:
    """Map the diagnosis numbers to a one-line, actionable conclusion."""
    if d["n_anomaly_windows"] == 0:
        return "No anomaly windows in the test split — nothing to inspect here."
    if d["frac_above_threshold"] > 0.05:
        return (
            f"Error CROSSES the threshold at {d['frac_above_threshold']:.0%} of anomaly "
            "windows, yet seg_f0_5=0 → the flagged runs were dropped by "
            "threshold_min_anomaly_len. Lower it for this subsystem."
        )
    if d["spike_ratio"] >= 2.0:
        return (
            f"Error SPIKES at anomalies (~{d['spike_ratio']:.1f}x nominal) but stays "
            "UNDER the threshold → Case 2: threshold_z too high for this channel. "
            "Lower threshold_z for its subsystem."
        )
    return (
        f"Error barely rises at anomalies (~{d['spike_ratio']:.1f}x nominal) → Case 1: "
        "the forecaster tracks the anomalies, so there's no spike to detect "
        "(Telemanom blind spot). Accept or drop the channel."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect a channel's error-vs-threshold at anomalies."
    )
    parser.add_argument("--env", default="local", help="Config env (local, cloud, test).")
    parser.add_argument("--mission", required=True, help="Mission name, e.g. ESA-Mission1.")
    parser.add_argument("--channel", required=True, help="Channel ID, e.g. channel_42.")
    args = parser.parse_args()

    settings = load_settings(args.env)
    configure_mlflow(settings)
    uri = settings.mlflow.tracking_uri

    exp = experiment_name("telemanom", "scoring", args.mission)
    run = find_latest_run_for_channel(exp, args.channel, uri)
    if run is None:
        raise SystemExit(f"No scoring run found for {args.channel} in experiment {exp!r}.")

    smoothed = bytes_to_errors(download_artifact_bytes(run.info.run_id, "errors.npy", uri))
    threshold = bytes_to_errors(download_artifact_bytes(run.info.run_id, "threshold.npy", uri))
    labels = load_window_labels(settings, args.mission, args.channel)

    n = min(len(smoothed), len(threshold), len(labels))
    if not (len(smoothed) == len(threshold) == len(labels)):
        log.warning(
            "inspect.length_mismatch",
            errors=len(smoothed), threshold=len(threshold), labels=len(labels),
            note=f"truncating all to {n}",
        )
    d = _diagnose(smoothed[:n], threshold[:n], labels[:n])

    print(f"\n== {args.channel} ({args.mission}) — run {run.info.run_id[:8]} ==")
    print(f"  anomaly windows           : {d['n_anomaly_windows']}")
    print(f"  median error  (nominal)   : {d['nominal_err_median']:.4g}")
    print(f"  median error  (anomaly)   : {d['anomaly_err_median']:.4g}")
    print(f"  spike ratio (anom/nominal): {d['spike_ratio']:.2f}x")
    print(f"  median threshold @ anomaly: {d['threshold_median_at_anomalies']:.4g}")
    print(f"  anomaly windows > threshold: {d['frac_above_threshold']:.0%}")
    print(f"\n  → {_verdict(d)}\n")


if __name__ == "__main__":
    main()
