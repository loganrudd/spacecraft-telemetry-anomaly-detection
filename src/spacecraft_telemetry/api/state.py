"""Runtime application state container for the FastAPI serving layer.

AppState is constructed once during the lifespan startup event and attached
to the FastAPI app instance (``app.state``).  Request handlers access it via
``request.app.state``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
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
    """

    settings: Settings
    mission: str
    subsystem: str
    device: torch.device
    engines: MappingProxyType[str, Any]  # values: ChannelInferenceEngine
    channel_subsystem_map: MappingProxyType[str, str]
    replay_data: MappingProxyType[str, Any]  # values: ReplayData
    startup_monotonic_ns: int
    mlflow_tracking_uri: str

    @property
    def channels_loaded(self) -> list[str]:
        """Sorted list of channel IDs with loaded inference engines."""
        return sorted(self.engines.keys())

    def uptime_seconds(self) -> float:
        """Wall-clock seconds elapsed since ``startup_monotonic_ns`` was recorded."""
        return (time.monotonic_ns() - self.startup_monotonic_ns) / 1e9
