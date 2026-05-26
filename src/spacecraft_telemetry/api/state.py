"""Runtime application state container for the FastAPI serving layer.

Two state objects live on ``app.state``:

* ``loading_state`` (LoadingState) — set at the start of lifespan, before any
  models are loaded.  Mutable counters updated by the background load task.
  Always present once lifespan has started.

* ``app_state`` (AppState) — set by the background load task when all engines
  are ready.  May be absent while loading is in progress.

Request handlers should use ``_get_ready_state(request)`` in endpoints.py to
obtain AppState, which raises 503 while loading is in progress.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import torch

from spacecraft_telemetry.core.config import Settings


@dataclass
class LoadingState:
    """Mutable progress tracker written by the background load task.

    Incremented from async coroutines in the event loop only — no lock needed
    (asyncio is single-threaded; counter updates never straddle an await).
    """

    channels_total: int
    channels_done: int = 0    # completed (success + failure)
    channels_ready: int = 0   # loaded successfully
    is_complete: bool = False
    error: str | None = None
    _start_ns: int = field(default_factory=time.monotonic_ns, repr=False)

    def uptime_seconds(self) -> float:
        return (time.monotonic_ns() - self._start_ns) / 1e9


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
    subsystem: str | None  # None when serving the whole mission
    device: torch.device
    engines: MappingProxyType[str, Any]  # values: ChannelInferenceEngine
    channel_subsystem_map: MappingProxyType[str, str]
    replay_data: MappingProxyType[str, Any]  # values: ReplayData
    startup_monotonic_ns: int
    mlflow_tracking_uri: str
    # Reference profiles for drift monitoring — populated only when
    # drift.enabled is True.  Values are pd.DataFrame (typed Any to avoid
    # importing pandas from state.py).  Each drift stream request creates its
    # own per-request RollingDriftMonitor from these immutable profiles.
    drift_references: MappingProxyType[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )

    @property
    def channels_loaded(self) -> list[str]:
        """Sorted list of channel IDs with loaded inference engines."""
        return sorted(self.engines.keys())

    def uptime_seconds(self) -> float:
        """Wall-clock seconds elapsed since ``startup_monotonic_ns`` was recorded."""
        return (time.monotonic_ns() - self.startup_monotonic_ns) / 1e9
