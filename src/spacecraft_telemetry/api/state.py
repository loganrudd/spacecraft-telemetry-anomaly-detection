"""Runtime application state container for the FastAPI serving layer.

AppState is constructed once during the lifespan startup event and attached
to the FastAPI app instance (``app.state``).  Request handlers access it via
``request.app.state``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from spacecraft_telemetry.api.inference import ChannelInferenceEngine
from spacecraft_telemetry.core.config import Settings


@dataclass
class AppState:
    """Immutable runtime state populated by the lifespan handler.

    All fields are set once at startup; request handlers treat this as
    read-only after that point.
    """

    settings: Settings
    mission: str
    subsystem: str
    device: torch.device
    engines: dict[str, ChannelInferenceEngine]
    channel_subsystem_map: dict[str, str]
    startup_monotonic_ns: int
    mlflow_tracking_uri: str

    @property
    def channels_loaded(self) -> list[str]:
        """Sorted list of channel IDs with loaded inference engines."""
        return sorted(self.engines.keys())

    def uptime_seconds(self) -> float:
        """Wall-clock seconds elapsed since ``startup_monotonic_ns`` was recorded."""
        return (time.monotonic_ns() - self.startup_monotonic_ns) / 1e9
