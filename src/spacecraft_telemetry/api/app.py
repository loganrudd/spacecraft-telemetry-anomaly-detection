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
    7. Load Evidently drift reference profiles + anomaly-dense replay slices
       (concurrently — both are I/O-bound GCS reads under the same semaphore)
    8. Attach AppState to app.state.app_state
    9. Mark LoadingState.is_complete = True

  Replay slices (api.replay_max_rows rows per channel, centred on the first
  labeled anomaly) are pre-cached at step 7.  Caching is safe because slicing
  reduces each channel from 9M+ rows to ~3k (~1 MB total vs a SIGKILL when
  the full test series was eagerly loaded).  Pre-caching eliminates per-stream
  GCS reads and makes SSE streams start instantly.  Pass replay_max_rows=0 in
  config to disable slicing and replay the full test set.

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
from typing import Any, cast

import mlflow.exceptions
import torch
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from mlflow.tracking.client import MlflowClient

from spacecraft_telemetry.api import endpoints
from spacecraft_telemetry.api.broadcast import EventBroadcaster, run_shared_loop
from spacecraft_telemetry.api.inference import ChannelInferenceEngine
from spacecraft_telemetry.api.live.normalization import load_normalization_params
from spacecraft_telemetry.api.logging_middleware import CorrelationIdMiddleware
from spacecraft_telemetry.api.replay import ReplayData, _anomaly_slice
from spacecraft_telemetry.api.state import AppState, LoadingState
from spacecraft_telemetry.core.config import Settings
from spacecraft_telemetry.core.logging import get_logger, setup_logging
from spacecraft_telemetry.core.metadata import load_channel_subsystem_map
from spacecraft_telemetry.evidently_monitoring.reference import (
    load_reference_profile,
    reference_profile_path,
)
from spacecraft_telemetry.mlflow_tracking.conventions import registered_model_name
from spacecraft_telemetry.mlflow_tracking.registry import CHAMPION_ALIAS
from spacecraft_telemetry.mlflow_tracking.runs import configure_mlflow
from spacecraft_telemetry.model.dataset import load_series_parquet
from spacecraft_telemetry.model.device import resolve_device
from spacecraft_telemetry.model.io import (
    ModelNotFoundError,
    load_model_for_scoring,
    load_scoring_params,
)

# Maximum concurrent MLflow model loads during lifespan startup.  Unbounded
# gather causes OOM or MLflow throttling at whole-mission scale (100+ channels).
_LIFESPAN_LOAD_CONCURRENCY = 4

# Exceptions that indicate a transient or expected channel-load failure (missing
# model, registry unavailable, network blip).  Programming errors (TypeError,
# AttributeError, KeyError) are intentionally excluded so they propagate loudly.
_CHANNEL_LOAD_ERRORS = (
    mlflow.exceptions.MlflowException,
    OSError,
    FileNotFoundError,
    ConnectionError,
    TimeoutError,
    RuntimeError,
    # No registered version for this channel — expected when serving a mission
    # whose channels weren't all trained. Not a RuntimeError subclass, so it must
    # be listed explicitly or startup would crash instead of skipping the channel.
    ModelNotFoundError,
)


async def _load_engine(
    ch: str,
    settings: Settings,
    device: torch.device,
    sem: asyncio.Semaphore,
    loading: LoadingState,
    log: Any,
) -> tuple[str, ChannelInferenceEngine | None]:
    """Load LSTM model + scoring params for one channel; returns None on expected failure."""
    async with sem:
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
        except _CHANNEL_LOAD_ERRORS as exc:
            log.warning("api.lifespan.channel.skipped", channel=ch, error=str(exc))
            return ch, None
        finally:
            loading.channels_done += 1


async def _load_drift_ref(
    ch: str,
    settings: Settings,
    sem: asyncio.Semaphore,
    log: Any,
) -> tuple[str, object]:
    """Load Evidently reference profile for one channel; returns None when missing."""
    async with sem:
        prof_path = reference_profile_path(settings, settings.api.mission, ch)
        try:
            ref = await asyncio.to_thread(load_reference_profile, prof_path)
            log.info("api.lifespan.drift_reference.loaded", channel=ch)
            return ch, ref
        except FileNotFoundError:
            log.warning("api.lifespan.drift_reference.missing", channel=ch)
            return ch, None


async def _load_replay_slice(
    ch: str,
    settings: Settings,
    sem: asyncio.Semaphore,
    log: Any,
) -> tuple[str, ReplayData | None]:
    """Pre-load and slice the replay series for one channel at startup.

    With replay_max_rows limiting each channel to ~3k rows (vs 9M+ for the
    full test series), the whole-mission pre-cache is ~1 MB — safe in the
    2 GiB Cloud Run instance.  Pre-caching here eliminates per-stream GCS
    reads and makes streams start instantly instead of blocking on a parquet
    read from GCS.
    """
    async with sem:
        try:
            replay_dir = settings.replay_dir
            values, _seg, anom, timestamps = await asyncio.to_thread(
                load_series_parquet,
                replay_dir,
                settings.api.mission,
                ch,
                "test",
            )
            sl = _anomaly_slice(
                anom,
                warmup_rows=settings.api.replay_warmup_rows,
                max_rows=settings.api.replay_max_rows,
            )
            data: ReplayData = (values[sl], anom[sl], timestamps[sl])
            log.info("api.lifespan.replay_slice.loaded", channel=ch, rows=len(data[0]))
            return ch, data
        except Exception as exc:  # GCS unavailable, missing partition, etc.
            log.warning("api.lifespan.replay_slice.failed", channel=ch, error=str(exc))
            return ch, None


def _has_champion_alias(model: Any) -> bool:
    """True when a RegisteredModel carries the @champion alias.

    MLflow's ``RegisteredModel.aliases`` is a ``{alias: version}`` dict in 3.x;
    tolerate a list of objects with ``.alias`` for version resilience.
    """
    aliases = getattr(model, "aliases", None) or {}
    if isinstance(aliases, dict):
        return CHAMPION_ALIAS in aliases
    return any(getattr(a, "alias", None) == CHAMPION_ALIAS for a in aliases)


def _resolve_champion_channels(
    settings: Settings, mission: str, log: Any
) -> set[str] | None:
    """Return the channel IDs that have a ``@champion`` model in the registry.

    The serving loader requires the ``@champion`` alias (``load_model_for_scoring``,
    ``require_champion=True``), so the promoted set is the authoritative list of
    servable channels — it excludes degenerate, untrained, and not-yet-promoted
    channels alike, with no dependency on any side file.

    Returns ``None`` when the registry query itself fails (MLflow unreachable) so
    the caller can fall back to the full channel list rather than booting empty.
    Returns an empty set when the query succeeds but nothing is promoted yet.
    """
    prefix = registered_model_name(settings.model.model_type, mission, "")
    try:
        client = MlflowClient(tracking_uri=settings.mlflow.tracking_uri)
        models = client.search_registered_models(filter_string=f"name LIKE '{prefix}%'")
    except Exception as exc:
        log.warning("api.lifespan.champion_query.failed", error=str(exc))
        return None
    champions = {m.name.removeprefix(prefix) for m in models if _has_champion_alias(m)}
    log.info("api.lifespan.champions.resolved", mission=mission, count=len(champions))
    return champions


def _gate_by_champion(
    candidates: list[str], champions: set[str] | None
) -> tuple[list[str], list[str]]:
    """Split ``candidates`` into (servable, no_champion) using the champion set.

    When ``champions`` is None (registry unavailable) all candidates pass through —
    the loader's ``require_champion`` gate is the backstop.
    """
    if champions is None:
        return sorted(candidates), []
    servable = sorted(c for c in candidates if c in champions)
    no_champion = sorted(c for c in candidates if c not in champions)
    return servable, no_champion


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

    # The serving loader accepts only @champion versions, so the promoted-champion
    # set is the source of truth for which channels are servable. Resolving the
    # load list from it (instead of channels.csv) keeps channels_total equal to
    # what can actually load — otherwise degenerate / untrained / unpromoted
    # channels inflate the total and the frontend's ready==total gate never
    # completes (the "62 / 76" stuck-loading bug).
    champions = _resolve_champion_channels(settings, settings.api.mission, log)

    if settings.api.channels:
        # Explicit override — intersect with champions so the progress gate stays
        # honest; a requested channel with no promoted model could never load.
        resolved, _no_champion = _gate_by_champion(list(settings.api.channels), champions)
        if _no_champion:
            log.warning("api.lifespan.channels.no_champion",
                        scope="explicit", channels=_no_champion)
        log.info("api.lifespan.channels.explicit", count=len(resolved))
    elif settings.api.subsystems is not None:
        in_subsystem = [ch for ch, sub in channel_subsystem_map.items()
                        if sub in settings.api.subsystems]
        resolved, _ = _gate_by_champion(in_subsystem, champions)
        log.info("api.lifespan.channels.from_subsystem",
                 subsystems=settings.api.subsystems, count=len(resolved))
    elif champions is not None:
        # Whole mission = exactly the promoted models; the registry is authoritative.
        resolved = sorted(champions)
        log.info("api.lifespan.channels.whole_mission", count=len(resolved))
    else:
        # Registry query failed — fall back to the full channel list. The loader's
        # require_champion gate still prevents serving anything unpromoted; this
        # only means channels_total may overcount until MLflow is reachable again.
        resolved = sorted(channel_subsystem_map.keys())
        log.warning("api.lifespan.channels.whole_mission_fallback",
                    count=len(resolved), reason="champion registry query unavailable")

    loading = LoadingState(channels_total=len(resolved))
    app.state.loading_state = loading
    sem = asyncio.Semaphore(_LIFESPAN_LOAD_CONCURRENCY)

    async def _background_load() -> None:
        try:
            engine_results = await asyncio.gather(
                *[_load_engine(ch, settings, device, sem, loading, log) for ch in resolved]
            )
            engines: dict[str, ChannelInferenceEngine] = {
                ch: eng for ch, eng in engine_results if eng is not None
            }

            if not engines:
                scope = (f"subsystems={settings.api.subsystems!r}"
                         if settings.api.subsystems else "whole mission")
                loading.error = (
                    f"No channels loaded for mission={settings.api.mission!r} ({scope}). "
                    "Train, score, and promote at least one channel first "
                    "(make model-train → make model-score → make mlflow-promote)."
                )
                log.error("api.lifespan.startup.no_engines", error=loading.error)
                loading.is_complete = True
                return

            drift_references: dict[str, object] = {}
            replay_slices: dict[str, ReplayData] = {}

            # Load drift references and replay slices concurrently — both are
            # I/O-bound GCS reads and fit under the same semaphore. Replay
            # slices are safe to pre-cache now that replay_max_rows limits each
            # channel to ~3k rows (~1 MB total vs 9M+ rows / SIGKILL before).
            load_results = await asyncio.gather(
                *[_load_drift_ref(ch, settings, sem, log) for ch in engines],
                *[_load_replay_slice(ch, settings, sem, log) for ch in engines],
                return_exceptions=False,
            )
            n = len(engines)
            for ch, ref in load_results[:n]:
                if ref is not None:
                    drift_references[ch] = ref
            if not settings.drift.enabled:
                log.info("api.lifespan.drift.disabled")
                drift_references = {}
            for ch, data in cast(list[tuple[str, ReplayData | None]], load_results[n:]):
                if data is not None:
                    replay_slices[ch] = data

            # Load normalization params for the live pump (no-op in replay mode).
            norm_params: dict[str, Any] = {}
            if settings.api.live:
                try:
                    norm_params = await asyncio.to_thread(
                        load_normalization_params,
                        settings.replay_dir,
                        settings.api.mission,
                    )
                    log.info(
                        "api.lifespan.norm_params.loaded", channels=len(norm_params)
                    )
                except (FileNotFoundError, OSError) as exc:
                    log.warning("api.lifespan.norm_params.missing", error=str(exc))

            # In live mode, prime each engine from the tail of its replay slice
            # so the first live tick produces a valid prediction immediately
            # rather than waiting window_size ticks for the buffer to fill.
            if settings.api.live:
                for ch, engine in engines.items():
                    if ch in replay_slices:
                        values, _, _ = replay_slices[ch]
                        seed = values[-engine.window_size :].tolist()
                        engine.prime(seed)
                        log.info(
                            "api.lifespan.engine.primed",
                            channel=ch,
                            seed_len=len(seed),
                        )

            broadcaster = EventBroadcaster()
            app.state.app_state = AppState(
                settings=settings,
                mission=settings.api.mission,
                subsystems=settings.api.subsystems,
                device=device,
                engines=MappingProxyType(engines),
                channel_subsystem_map=MappingProxyType(channel_subsystem_map),
                replay_data=MappingProxyType(replay_slices),
                startup_monotonic_ns=time.monotonic_ns(),
                mlflow_tracking_uri=settings.mlflow.tracking_uri,
                drift_references=MappingProxyType(drift_references),
                resolved_channels=resolved,
                normalization_params=MappingProxyType(norm_params),
                broadcaster=broadcaster,
            )

            if settings.api.live:
                # Live pump: subscribes Lightstreamer, archives to GCS, and
                # feeds the broadcaster directly — no replay loop needed.
                from spacecraft_telemetry.api.live.pump import LivePump

                pump = LivePump(
                    loop=asyncio.get_running_loop(),
                    broadcaster=broadcaster,
                    engines=dict(engines),
                    norm_params=norm_params,
                    collect_config=settings.collect,
                    state=app.state.app_state,
                    archive_to_gcs=settings.api.archive_to_gcs,
                    raw_ticks_dir=(
                        settings.collect.raw_ticks_dir
                        if settings.api.archive_to_gcs
                        else None
                    ),
                    los_stats_median_s=None,  # Step 6 computes this from archive
                )
                app.state.live_pump = pump
                await pump.start()
                log.info("api.lifespan.live_pump.started")
            else:
                # Replay loop: walks pre-collected Parquet and broadcasts events.
                # All SSE subscribers attach to the in-progress stream — no
                # per-connection warmup.  The task runs until lifespan shuts down.
                app.state.loop_task = asyncio.create_task(
                    run_shared_loop(app.state.app_state),
                    name="shared-replay-loop",
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
        loop_task = getattr(app.state, "loop_task", None)
        if loop_task is not None and not loop_task.done():
            loop_task.cancel()
            with suppress(asyncio.CancelledError):
                await loop_task
        live_pump = getattr(app.state, "live_pump", None)
        if live_pump is not None:
            await live_pump.stop()
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
            allow_methods=["GET", "POST"],
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
            index_html = static_path / "index.html"

            # Starlette StaticFiles mounted at "/" strips the leading "/" before
            # path lookup, leaving scope["path"]="". The html=True redirect logic
            # then issues a self-redirect ("" → "/") that Cloud Run's proxy
            # collapses into a 404 loop. An explicit route bypasses this.
            if index_html.is_file():
                _index = str(index_html)

                @app.get("/", include_in_schema=False)
                async def _serve_spa_root() -> FileResponse:
                    return FileResponse(_index)

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
