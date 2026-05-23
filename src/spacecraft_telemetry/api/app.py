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
from pathlib import Path
from types import MappingProxyType

import torch
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from spacecraft_telemetry.api import endpoints
from spacecraft_telemetry.api.inference import ChannelInferenceEngine
from spacecraft_telemetry.api.logging_middleware import CorrelationIdMiddleware
from spacecraft_telemetry.api.replay import ReplayData
from spacecraft_telemetry.api.state import AppState
from spacecraft_telemetry.core.config import Settings
from spacecraft_telemetry.core.logging import get_logger, setup_logging
from spacecraft_telemetry.core.metadata import load_channel_subsystem_map
from spacecraft_telemetry.evidently_monitoring.reference import (
    load_reference_profile,
    reference_profile_path,
)
from spacecraft_telemetry.mlflow_tracking.conventions import registered_model_name
from spacecraft_telemetry.mlflow_tracking.runs import configure_mlflow
from spacecraft_telemetry.model.dataset import load_series_parquet
from spacecraft_telemetry.model.device import resolve_device
from spacecraft_telemetry.model.io import load_model_for_scoring, load_scoring_params

# Maximum concurrent MLflow model loads during lifespan startup.  Unbounded
# gather causes OOM or MLflow throttling at whole-mission scale (300+ channels).
_LIFESPAN_LOAD_CONCURRENCY = 8


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

    if settings.api.channels:
        resolved = list(settings.api.channels)
        log.info("api.lifespan.channels.explicit", count=len(resolved))
    elif settings.api.subsystem is not None:
        resolved = [
            ch for ch, sub in channel_subsystem_map.items()
            if sub == settings.api.subsystem
        ]
        log.info(
            "api.lifespan.channels.from_subsystem",
            subsystem=settings.api.subsystem,
            count=len(resolved),
        )
    else:
        resolved = sorted(channel_subsystem_map.keys())
        log.info("api.lifespan.channels.whole_mission", count=len(resolved))

    _load_sem = asyncio.Semaphore(_LIFESPAN_LOAD_CONCURRENCY)

    async def _load_engine(ch: str) -> tuple[str, ChannelInferenceEngine | None]:
        async with _load_sem:
            try:
                name = registered_model_name("telemanom", settings.api.mission, ch)
                model, window_size = await asyncio.to_thread(
                    load_model_for_scoring, name, device, settings.mlflow.tracking_uri
                )
                model.eval()
                params = await asyncio.to_thread(
                    load_scoring_params,
                    channel=ch,
                    mission=settings.api.mission,
                    tracking_uri=settings.mlflow.tracking_uri,
                )
                engine = ChannelInferenceEngine(
                    mission=settings.api.mission,
                    channel=ch,
                    model=model,
                    window_size=window_size,
                    params=params,
                    device=device,
                )
                log.info("api.lifespan.channel.loaded", channel=ch, window_size=window_size)
                return ch, engine
            except Exception as exc:
                log.warning("api.lifespan.channel.skipped", channel=ch, error=str(exc))
                return ch, None

    engine_results = await asyncio.gather(*[_load_engine(ch) for ch in resolved])
    engines: dict[str, ChannelInferenceEngine] = {
        ch: eng for ch, eng in engine_results if eng is not None
    }

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
        scope = (
            f"subsystem={settings.api.subsystem!r}"
            if settings.api.subsystem
            else "whole mission"
        )
        raise RuntimeError(
            f"No channels loaded for mission={settings.api.mission!r} ({scope}). "
            "Train and score at least one channel first."
        )

    # Drift reference profiles — best-effort: missing profile logs a warning but
    # does not prevent startup.  Drift stream returns 503 if no profiles are loaded.
    # Per-request RollingDriftMonitor instances are created by drift_stream at
    # connection time so there is no shared mutable state across clients.
    drift_references: dict[str, object] = {}
    if settings.drift.enabled:
        async def _load_drift_reference(
            ch: str,
        ) -> tuple[str, object]:
            async with _load_sem:
                prof_path = reference_profile_path(settings, settings.api.mission, ch)
                try:
                    ref = await asyncio.to_thread(load_reference_profile, prof_path)
                    log.info("api.lifespan.drift_reference.loaded", channel=ch)
                    return ch, ref
                except FileNotFoundError:
                    log.warning("api.lifespan.drift_reference.missing", channel=ch)
                    return ch, None

        ref_results = await asyncio.gather(
            *[_load_drift_reference(ch) for ch in engines]
        )
        for ch, ref in ref_results:
            if ref is not None:
                drift_references[ch] = ref
    else:
        log.info("api.lifespan.drift.disabled")

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
        drift_references=drift_references,
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

    # Mount the built dashboard at / when configured (Phase 10: same-origin SPA on
    # Cloud Run). Routes registered above match first; html=True serves index.html
    # for unknown paths so client-side routes work on refresh.
    if settings.api.static_dir:
        static_path = Path(settings.api.static_dir)
        if static_path.is_dir():
            app.mount(
                "/",
                StaticFiles(directory=static_path, html=True),
                name="dashboard",
            )
        else:
            get_logger("api.startup").warning(
                "api.static_dir.missing",
                static_dir=str(static_path),
            )
    return app
