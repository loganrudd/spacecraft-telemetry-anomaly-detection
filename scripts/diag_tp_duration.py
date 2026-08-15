"""Diagnostic: do true-positive detections run longer than false positives?

Answers the open question left by the plan-019 detection-level diagnostic, which
found that all 17 of arm A's true false positives are 2.3-3.0 min long -- tightly
clustered at the minimum qualifying run length (``threshold_min_anomaly_len=5``,
picked from an interior point of [1, 10]). TP durations were never computed, so
it was unknown whether a duration filter would be a free precision win or would
cut true detections too.

If TP detections run materially longer than FP detections, raising
``threshold_min_anomaly_len`` buys precision with no retraining and no
architecture change. If both distributions sit at the same floor, the lever is
dead. Either answer is worth having before committing to modelling work.

Mirrors the tuned eval window from esa_adb/report.py exactly (``_hpo_cutoff`` ->
intersect with the full mission timeline), so the detections classified here are
the same 33 the report scores -- this is a read-only analysis of existing
scoring runs, it never re-scores.

Detections are classified three ways, matching the plan's diagnostic table:
  * ``anomaly``    -- overlaps an event of category "Anomaly"
  * ``rare_event`` -- overlaps only "Rare Event" (a real annotated phenomenon
                      that the anomalies_only scope excludes by definition)
  * ``false_pos``  -- overlaps no annotated event at all

Usage:
    SSL_CERT_FILE=$(python -c "import certifi;print(certifi.where())") \\
    REQUESTS_CA_BUNDLE=$SSL_CERT_FILE \\
    SPACECRAFT_DATA__SAMPLE_DATA_DIR=gs://spacecraft-telemetry-ads-sample-data \\
    SPACECRAFT_PREPROCESS__PROCESSED_DATA_DIR=gs://spacecraft-telemetry-ads-processed-data \\
    SPACECRAFT_MLFLOW__TRACKING_URI=https://mlflow-pb5fb25noa-uc.a.run.app \\
    python scripts/diag_tp_duration.py --env cloud --mission ESA-Mission1-ADB
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running as a script without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spacecraft_telemetry.core.config import load_settings
from spacecraft_telemetry.esa_adb.detections import mission_detection_intervals
from spacecraft_telemetry.esa_adb.events import group_events, load_events
from spacecraft_telemetry.esa_adb.intervals import intersect, normalize, overlaps
from spacecraft_telemetry.esa_adb.report import LIGHTWEIGHT_CHANNELS, _hpo_cutoff
from spacecraft_telemetry.esa_adb.timeline import mission_timeline
from spacecraft_telemetry.mlflow_tracking import configure_mlflow

# Matches _SCOPE_EXCLUSIONS["all_events"] in esa_adb/report.py: Communication
# Gap spans have no real data behind them, so they are neither TP nor FP.
_EXCLUDED = frozenset({"Communication Gap"})


def _minutes(td: pd.Timedelta) -> float:
    return float(td / pd.Timedelta(minutes=1))


def _describe(name: str, durations: list[float]) -> str:
    if not durations:
        return f"  {name:<12} n=0"
    arr = np.array(durations)
    return (
        f"  {name:<12} n={len(arr):<3} "
        f"min={arr.min():7.2f}  p50={np.median(arr):7.2f}  "
        f"max={arr.max():8.2f}  mean={arr.mean():7.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="TP vs FP detection-duration check.")
    parser.add_argument("--env", default="cloud")
    parser.add_argument("--mission", default="ESA-Mission1-ADB")
    parser.add_argument("--channels", default=None, help="Comma-separated. Default: 41-46.")
    args = parser.parse_args()

    settings = load_settings(args.env)
    configure_mlflow(settings)
    channels = args.channels.split(",") if args.channels else LIGHTWEIGHT_CHANNELS
    mission = args.mission

    # Reproduce report.py's tuned evaluation window exactly.
    events_df = load_events(settings, mission)
    timeline_full = mission_timeline(settings, mission, channels)
    hpo_cutoff = _hpo_cutoff(settings, mission, channels)
    far_future = pd.Timestamp.max.tz_localize("UTC")
    timeline_tuned = intersect(timeline_full, [(hpo_cutoff, far_future)])

    detections = intersect(
        normalize(mission_detection_intervals(settings, mission, channels, tuned=True)),
        timeline_tuned,
    )
    events = group_events(events_df, channels, timeline_tuned)
    included = [e for e in events if e.category not in _EXCLUDED]
    anomalies = [e for e in included if e.category == "Anomaly"]

    anomaly_intervals = [iv for e in anomalies for iv in e.intervals]
    included_intervals = [iv for e in included for iv in e.intervals]

    # Classify each detection and record its duration.
    buckets: dict[str, list[float]] = {"anomaly": [], "rare_event": [], "false_pos": []}
    for det in detections:
        dur = _minutes(det[1] - det[0])
        if overlaps(det, anomaly_intervals):
            buckets["anomaly"].append(dur)
        elif overlaps(det, included_intervals):
            buckets["rare_event"].append(dur)
        else:
            buckets["false_pos"].append(dur)

    print(f"\nmission={mission}  channels={len(channels)}  tuned eval window")
    print(f"HPO cutoff: {hpo_cutoff}")
    print(f"{len(detections)} mission-level detections, {len(included)} included events "
          f"({len(anomalies)} of category 'Anomaly')\n")

    print("DETECTION durations (minutes), by what the detection hit:")
    for name in ("anomaly", "rare_event", "false_pos"):
        print(_describe(name, buckets[name]))

    # The decisive comparison: can a duration floor separate the two classes?
    tp = sorted(buckets["anomaly"])
    fp = sorted(buckets["false_pos"])
    print("\nVERDICT:")
    if not tp or not fp:
        print("  Cannot compare — one class is empty.")
    else:
        print(f"  shortest anomaly-hitting detection : {tp[0]:.2f} min")
        print(f"  longest  false-positive detection  : {max(fp):.2f} min")
        if tp[0] > max(fp):
            kept = len(tp)
            print(
                f"  SEPARABLE — a floor between {max(fp):.2f} and {tp[0]:.2f} min "
                f"removes all {len(fp)} FPs and keeps all {kept} TPs."
            )
        else:
            survivors = [d for d in fp if d >= tp[0]]
            print(
                f"  NOT SEPARABLE — {len(survivors)} of {len(fp)} FPs are at least as long "
                f"as the shortest TP ({tp[0]:.2f} min). A duration floor set to keep every "
                f"TP would remove only {len(fp) - len(survivors)} FP(s)."
            )

    # Ground-truth event durations, for context on what the detector is chasing.
    print("\nGROUND-TRUTH event durations (minutes, clipped to eval window):")
    for label, evs in (("Anomaly", anomalies), ("all included", included)):
        durs = [_minutes(sum((iv[1] - iv[0] for iv in e.intervals), pd.Timedelta(0))) for e in evs]
        print(_describe(label, durs))


if __name__ == "__main__":
    main()
