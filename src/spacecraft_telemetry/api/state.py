"""Runtime application state container for the FastAPI serving layer.

AppState is constructed once during the lifespan startup event and attached
to the FastAPI app instance (``app.state``).  Request handlers access it via
``request.app.state``.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import torch

from spacecraft_telemetry.core.config import Settings


@dataclass(frozen=True)
class AppState:
    """Immutable runtime state populated by the lifespan handler.

    All fields are set once at startup and are structurally immutable:
    ``frozen=True`` prevents attribute reassignment; ``MappingProxyType``
    wrappers on dict fields prevent content mutation. Request handlers
    must treat this as read-only.

    Drift-related fields (``drift_monitors``, ``tick_buses``) hold mutable
    containers — the field references are frozen but the container interiors
    are intentionally mutable (drift monitors accumulate ticks; tick buses
    deque-append new rows).  This is a deliberate exception to the
    immutability convention: drift state is ephemeral per-session data, not
    configuration.
    """

    settings: Settings
    mission: str
    subsystem: str | None  # None when serving the whole mission
    device: torch.device
    engines: MappingProxyType[str, Any]  # values: ChannelInferenceEngine
    channel_subsystem_map: MappingProxyType[str, str]
    replay_data: MappingProxyType[str, Any]  # values: ReplayData
    startup_monotonic_ns: int
    mlflow_tracking_uri: str
    # Drift monitoring — populated only when drift.enabled is True in settings.
    # drift_monitors: channel_id → RollingDriftMonitor (typed as Any to avoid
    # importing drift.py from state.py and creating a circular dependency).
    drift_monitors: dict[str, Any] = field(default_factory=dict)
    # tick_buses: channel_id → deque of telemetry tick dicts for drift consumers.
    # deque maxlen == drift.window_size; non-blocking append (drops oldest on overflow).
    tick_buses: dict[str, deque[dict[str, float]]] = field(default_factory=dict)

    @property
    def channels_loaded(self) -> list[str]:
        """Sorted list of channel IDs with loaded inference engines."""
        return sorted(self.engines.keys())

    def uptime_seconds(self) -> float:
        """Wall-clock seconds elapsed since ``startup_monotonic_ns`` was recorded."""
        return (time.monotonic_ns() - self.startup_monotonic_ns) / 1e9
