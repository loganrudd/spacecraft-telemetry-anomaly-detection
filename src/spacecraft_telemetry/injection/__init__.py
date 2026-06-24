"""Anomaly injection for ISS telemetry (Phase 15).

Public API
----------
inject_spike     Additive burst at a fixed start position.
inject_drift     Linear ramp then hold.
inject_flatline  Constant-value sensor-death segment.
inject_faults    Orchestrator: place a reproducible set of faults across a series.
"""

from spacecraft_telemetry.injection.faults import (
    inject_drift,
    inject_faults,
    inject_flatline,
    inject_spike,
)

__all__ = ["inject_drift", "inject_faults", "inject_flatline", "inject_spike"]
