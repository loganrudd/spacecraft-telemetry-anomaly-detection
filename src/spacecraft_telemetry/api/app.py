"""FastAPI application factory and lifespan handler.

Entry point:
    app = create_app(load_settings())

Startup sequence (Phase 10 deferred-load design):
  Sync (completes before the first request is accepted):
    1. configure_mlflow  — FIRST, before any third-party library touches the URI
    2. torch.set_num_threads(1)  — avoid BLAS oversubscription under Ray
    3. resolve_device  — CUDA > MPS > CPU
    4. load_channel_subsystem_map  — determines which channels to load
    5. Resolve channel list; create LoadingState; attach to app.state.loading_state

  Background task (runs concurrently while the app serves requests):
    6. Load LSTM + scoring params per channel (semaphore-limited concurrency)
    7. Load replay Parquet for each engine
    8. Load Evidently drift reference profiles
    9. Attach AppState to app.state.app_state
   10. Mark LoadingState.is_complete = True

During steps 6-9, GET /health returns status="loading" with progress counters
so the React dashboard can show a progress bar instead of a blank screen.
SSE endpoints return 503 until app_state is set.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from types import MappingProxyType

import torch
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from spacecraft_telemetry.api import endpoints
from spacecraft_telemetry.api.inference import ChannelInferenceEngine
from spacecraft_telemetry.api.logging_middleware import CorrelationIdMiddleware
from spacecraft_telemetry.api.replay import ReplayData
from spacecraft_telemetry.api.state import AppState, LoadingState
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
    """FastAPI lifespan — fast sync setup, then deferred background model load."""
    settings: Settings = app.state.settings
    log = get_logger("api.lifespan")

    # Sync setup — must complete before the app accepts connections.
    configure_mlflow(settings)  # FIRST — before any third-party library touches the URI
    torch.set_num_threads(1)
    device = resolve_device(settings.model.device)
    channel_subsystem_map = load_channel_subsystem_map(settings, settings.api.mission)

    if settings.api.channels:
        resolved = list(settings.api.channels)
        log.info("api.lifespan.channels.explicit", count=len(resolved))
    elif settings.api.subsystem is not None:
        resolved = [ch for ch, sub in channel_subsystem_map.items()
                    if sub == settings.api.subsystem]
        log.info("api.lifespan.channels.from_subsystem",
                 subsystem=settings.api.subsystem, count=len(resolved))
    else:
        resolved = sorted(channel_subsystem_map.keys())
        log.info("api.lifespan.channels.whole_mission", count=len(resolved))

    loading = LoadingState(channels_total=len(resolved))
    app.state.loading_state = loading

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
                loading.channels_ready += 1
                return ch, engine
            except Exception as exc:
                log.warning("api.lifespan.channel.skipped", channel=ch, error=str(exc))
                return ch, None
            finally:
                loading.channels_done += 1

    async def _background_load() -> None:
        try:
            engine_results = await asyncio.gather(*[_load_engine(ch) for ch in resolved])
            engines: dict[str, ChannelInferenceEngine] = {
                ch: eng for ch, eng in engine_results if eng is not None
            }

            replay_data: dict[str, ReplayData] = {}
            for ch in engines:
                try:
                    values, _seg, anom, timestamps = await asyncio.to_thread(
                        load_series_parquet,
                        settings.preprocess.processed_data_dir,
                        settings.api.mission,
                        ch,
                        "test",
                    )
                    replay_data[ch] = (values, anom, timestamps)
                except Exception as exc:
                    log.warning("api.lifespan.replay_data.failed", channel=ch, error=str(exc))

            if not engines:
                scope = (f"subsystem={settings.api.subsystem!r}"
                         if settings.api.subsystem else "whole mission")
                loading.error = (
                    f"No channels loaded for mission={settings.api.mission!r} ({scope}). "
                    "Train, score, and promote at least one channel first "
                    "(make model-train → make model-score → make mlflow-promote)."
                )
                log.error("api.lifespan.startup.no_engines", error=loading.error)
                loading.is_complete = True
                return

            drift_references: dict[str, object] = {}
            if settings.drift.enabled:
                async def _load_drift_ref(ch: str) -> tuple[str, object]:
                    async with _load_sem:
                        prof_path = reference_profile_path(settings, settings.api.mission, ch)
                        try:
                            ref = await asyncio.to_thread(load_reference_profile, prof_path)
                            log.info("api.lifespan.drift_reference.loaded", channel=ch)
                            return ch, ref
                        except FileNotFoundError:
                            log.warning("api.lifespan.drift_reference.missing", channel=ch)
                            return ch, None

                for ch, ref in await asyncio.gather(*[_load_drift_ref(ch) for ch in engines]):
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
                drift_references=MappingProxyType(drift_references),
                resolved_channels=resolved,
            )
            log.info("api.lifespan.startup.complete", channels_loaded=sorted(engines.keys()))

        except asyncio.CancelledError:
            log.info("api.lifespan.startup.cancelled")
            raise
        except Exception as exc:
            loading.error = str(exc)
            log.exception("api.lifespan.startup.failed", error=str(exc))
        finally:
            loading.is_complete = True

    task = asyncio.create_task(_background_load())

    try:
        yield
    finally:
        if not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
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
