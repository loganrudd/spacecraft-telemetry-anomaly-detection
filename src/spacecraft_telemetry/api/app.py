"""FastAPI application factory and lifespan handler.

Entry point:
    app = create_app(load_settings())

The lifespan handler runs at startup:
  1. configure_mlflow  — FIRST, before any other library touches the URI
  2. torch.set_num_threads(1)  — one process per channel; avoid BLAS oversubscription
  3. resolve_device  — CUDA > MPS > CPU
  4. load_channel_subsystem_map  — channel → subsystem lookup
  5. For each channel in the configured subsystem: load LSTM + scoring params,
     build ChannelInferenceEngine, put in eval() mode.
  6. Attach AppState to app.state.app_state.
  7. Raise RuntimeError if no engines loaded (hard fail — nothing to serve).

The router (endpoints.py) is wired in during create_app, so Step 7 can add
it without modifying this file.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import MappingProxyType

import torch
from fastapi import FastAPI

from spacecraft_telemetry.api import endpoints
from spacecraft_telemetry.api.inference import ChannelInferenceEngine
from spacecraft_telemetry.api.logging_middleware import CorrelationIdMiddleware
from spacecraft_telemetry.api.replay import ReplayData
from spacecraft_telemetry.api.state import AppState
from spacecraft_telemetry.core.config import Settings
from spacecraft_telemetry.core.logging import get_logger, setup_logging
from spacecraft_telemetry.core.metadata import load_channel_subsystem_map
from spacecraft_telemetry.mlflow_tracking.conventions import registered_model_name
from spacecraft_telemetry.mlflow_tracking.runs import configure_mlflow
from spacecraft_telemetry.model.dataset import load_series_parquet
from spacecraft_telemetry.model.device import resolve_device
from spacecraft_telemetry.model.io import load_model_for_scoring, load_scoring_params


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """FastAPI lifespan handler — loads all channel inference engines at startup."""
    settings: Settings = app.state.settings
    log = get_logger("api.lifespan")

    # MLflow URI must be set first — before any third-party library runs.
    configure_mlflow(settings)

    torch.set_num_threads(1)
    device = resolve_device(settings.model.device)

    channel_subsystem_map = load_channel_subsystem_map(settings, settings.api.mission)

    # TODO: add mission-wide fallback (discover all channels) when Mission2/Mission3
    # support is needed — currently an absent channels.csv produces resolved=[] which
    # hits the RuntimeError below with a clear message.
    if settings.api.channels:
        resolved = list(settings.api.channels)
        log.info("api.lifespan.channels.explicit", count=len(resolved))
    else:
        resolved = [
            ch for ch, sub in channel_subsystem_map.items()
            if sub == settings.api.subsystem
        ]
        log.info(
            "api.lifespan.channels.from_subsystem",
            subsystem=settings.api.subsystem,
            count=len(resolved),
        )

    engines: dict[str, ChannelInferenceEngine] = {}
    for ch in resolved:
        try:
            name = registered_model_name("telemanom", settings.api.mission, ch)
            model, window_size = load_model_for_scoring(
                name, device, settings.mlflow.tracking_uri
            )
            model.eval()
            params = load_scoring_params(
                channel=ch,
                mission=settings.api.mission,
                tracking_uri=settings.mlflow.tracking_uri,
            )
            engines[ch] = ChannelInferenceEngine(
                mission=settings.api.mission,
                channel=ch,
                model=model,
                window_size=window_size,
                params=params,
                device=device,
            )
            log.info("api.lifespan.channel.loaded", channel=ch, window_size=window_size)
        except Exception as exc:
            log.warning("api.lifespan.channel.skipped", channel=ch, error=str(exc))

    replay_data: dict[str, ReplayData] = {}
    for ch in engines:
        try:
            values, _seg, anom, timestamps = await asyncio.to_thread(
                load_series_parquet,
                settings.spark.processed_data_dir,
                settings.api.mission,
                ch,
                "test",
            )
            replay_data[ch] = (values, anom, timestamps)
        except Exception as exc:
            log.warning("api.lifespan.replay_data.failed", channel=ch, error=str(exc))

    if not engines:
        raise RuntimeError(
            f"No channels loaded for mission={settings.api.mission!r} "
            f"subsystem={settings.api.subsystem!r}. "
            "Train and score at least one channel first."
        )

    app.state.app_state = AppState(
        settings=settings,
        mission=settings.api.mission,
        subsystem=settings.api.subsystem,
        device=device,
        engines=MappingProxyType(engines),
        channel_subsystem_map=MappingProxyType(channel_subsystem_map),
        replay_data=MappingProxyType(replay_data),
        startup_monotonic_ns=time.monotonic_ns(),
        mlflow_tracking_uri=settings.mlflow.tracking_uri,
    )
    log.info(
        "api.lifespan.startup.complete",
        channels_loaded=sorted(engines.keys()),
    )

    try:
        yield
    finally:
        log.info("api.lifespan.shutdown")


def create_app(settings: Settings) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        settings: Application settings (loaded via ``load_settings()``).

    Returns:
        Configured FastAPI instance with lifespan, middleware, and routes.
        Routes are included in Step 7 (endpoints.py).
    """
    setup_logging(settings.logging)
    app = FastAPI(
        title="Spacecraft Telemetry Serving",
        description="Real-time telemetry replay with LSTM anomaly detection.",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.add_middleware(CorrelationIdMiddleware)
    if settings.api.cors_allowed_origins:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.api.cors_allowed_origins,
            allow_credentials=False,
            allow_methods=["GET"],
            allow_headers=["*"],
            expose_headers=["X-Correlation-Id"],
        )
    app.include_router(endpoints.router)
    return app
