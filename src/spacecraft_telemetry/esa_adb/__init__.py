"""ESA-ADB-comparable evaluation (Plan 019).

Reproduces the metrics from the ESA Anomaly Detection Benchmark paper
(Kotowski et al. 2024, arXiv:2406.17826) against channels 41-46 of
ESA-Mission1, so our results can sit next to the paper's own Telemanom-ESA /
Telemanom-ESA-Pruned numbers under the same metric definitions.

This is a deliberately separate metric convention from
``model.scoring.evaluate_overlap`` (our serving-parity segment-overlap
metric) — see ``docs/architecture/esa-adb-metrics.md`` for when to use which.

Public API
----------
intervals   normalize, union, intersect, subtract, total_duration, overlaps
events      Event, load_events, group_events
timeline    channel_timeline, mission_timeline
detections  find_scoring_run, channel_detection_intervals,
            channel_detection_intervals_from_spec,
            per_channel_detection_intervals, mission_detection_intervals
metrics     corrected_event_wise, alarming_precision, channel_aware
offline     RunSpec, OfflineRunMap, load_run_map — run the report with no
            MLflow tracking server (see scripts/stage_adb_offline.sh)
report      build_report — the top-level entry point (see scripts/esa_adb_report.py)
"""

from spacecraft_telemetry.esa_adb.detections import (
    channel_detection_intervals,
    channel_detection_intervals_from_spec,
    find_scoring_run,
    mission_detection_intervals,
    per_channel_detection_intervals,
)
from spacecraft_telemetry.esa_adb.events import Event, group_events, load_events
from spacecraft_telemetry.esa_adb.intervals import (
    intersect,
    normalize,
    overlaps,
    subtract,
    total_duration,
    union,
)
from spacecraft_telemetry.esa_adb.metrics import (
    alarming_precision,
    channel_aware,
    corrected_event_wise,
)
from spacecraft_telemetry.esa_adb.offline import OfflineRunMap, RunSpec, load_run_map
from spacecraft_telemetry.esa_adb.report import build_report
from spacecraft_telemetry.esa_adb.timeline import channel_timeline, mission_timeline

__all__ = [
    "Event",
    "OfflineRunMap",
    "RunSpec",
    "alarming_precision",
    "build_report",
    "channel_aware",
    "channel_detection_intervals",
    "channel_detection_intervals_from_spec",
    "channel_timeline",
    "corrected_event_wise",
    "find_scoring_run",
    "group_events",
    "intersect",
    "load_events",
    "load_run_map",
    "mission_detection_intervals",
    "mission_timeline",
    "normalize",
    "overlaps",
    "per_channel_detection_intervals",
    "subtract",
    "total_duration",
    "union",
]
