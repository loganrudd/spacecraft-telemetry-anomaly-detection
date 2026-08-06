"""ESA-ADB-comparable evaluation (Plan 019).

Reproduces the metrics from the ESA Anomaly Detection Benchmark paper
(Kotowski et al. 2024, arXiv:2406.17826) against channels 41-46 of
ESA-Mission1, so our results can sit next to the paper's own Telemanom-ESA /
Telemanom-ESA-Pruned numbers under the same metric definitions.

This is a deliberately separate metric convention from
``model.scoring.evaluate_overlap`` (our serving-parity segment-overlap
metric) — see ``docs/architecture/esa-adb-metrics.md`` for when to use which.

Public API (expanded as each module lands — see docs/plans/019):
----------
intervals   normalize, union, intersect, subtract, total_duration, overlaps
"""

from spacecraft_telemetry.esa_adb.intervals import (
    intersect,
    normalize,
    overlaps,
    subtract,
    total_duration,
    union,
)

__all__ = [
    "intersect",
    "normalize",
    "overlaps",
    "subtract",
    "total_duration",
    "union",
]
