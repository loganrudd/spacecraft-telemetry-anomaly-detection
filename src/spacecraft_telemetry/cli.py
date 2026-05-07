"""CLI entry point for the spacecraft telemetry anomaly detection system.

Each subcommand follows the same pattern:
    1. Load settings for the given --env
    2. Set up structured logging (--verbose overrides the config level to DEBUG)
    3. Delegate to the relevant ingest module
"""

from __future__ import annotations

import contextlib as _contextlib
from pathlib import Path
from typing import Any

import click

from spacecraft_telemetry.core.config import LoggingConfig, Settings, load_settings
from spacecraft_telemetry.core.logging import get_logger, setup_logging


@click.group()
@click.option(
    "--env",
    default="local",
    show_default=True,
    help="Config environment to load (local | cloud | test).",
)
@click.option("--verbose", is_flag=True, help="Force DEBUG logging regardless of config.")
@click.pass_context
def main(ctx: click.Context, env: str, verbose: bool) -> None:
    """Spacecraft Telemetry Anomaly Detection System."""
    ctx.ensure_object(dict)
    settings = load_settings(env)
    log_config = (
        LoggingConfig(level="DEBUG", format=settings.logging.format)
        if verbose
        else settings.logging
    )
    setup_logging(log_config)
    ctx.obj["settings"] = settings


# ---------------------------------------------------------------------------
# version
# ---------------------------------------------------------------------------


@main.command()
def version() -> None:
    """Print the package version and exit."""
    from spacecraft_telemetry import __version__

    click.echo(__version__)


# ---------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------


@main.command()
@click.option("--mission", required=True, help="Mission name to download (e.g. ESA-Mission1).")
@click.option(
    "--sample",
    "create_sample",
    is_flag=True,
    help="Create a Parquet sample after downloading.",
)
@click.option(
    "--sample-fraction",
    type=float,
    default=None,
    show_default=True,
    help="Fraction of rows to keep in the sample (overrides config).",
)
@click.pass_context
def download(
    ctx: click.Context,
    mission: str,
    create_sample: bool,
    sample_fraction: float | None,
) -> None:
    """Download an ESA mission from Zenodo and optionally create a local sample.

    The Zenodo record ID is read from config (data.zenodo_record_id).
    Files are written to data.raw_data_dir/{mission}/.

    Examples:

        # Download only
        spacecraft-telemetry download --mission ESA-Mission1

        # Download and create a 1% sample with 5 channels
        spacecraft-telemetry download --mission ESA-Mission1 --sample

        # Download and create a 5% sample
        spacecraft-telemetry download --mission ESA-Mission1 --sample --sample-fraction 0.05
    """
    from spacecraft_telemetry.ingest.download import ZenodoDownloader
    from spacecraft_telemetry.ingest.sample import SampleCreator

    settings = ctx.obj["settings"]
    log = get_logger(__name__)

    raw_dir = Path(str(settings.data.raw_data_dir))

    log.info("starting download", mission=mission, dest=str(raw_dir))
    downloader = ZenodoDownloader(
        record_id=settings.data.zenodo_record_id,
        dest_dir=raw_dir,
    )
    downloader.download_mission(mission)
    click.echo(f"Downloaded {mission} to {raw_dir / mission}")

    if create_sample:
        fraction = (
            sample_fraction if sample_fraction is not None else settings.data.sample_fraction
        )
        log.info("creating sample", mission=mission, fraction=fraction)
        creator = SampleCreator(
            raw_dir=raw_dir,
            sample_dir=Path(str(settings.data.sample_data_dir)),
            sample_fraction=fraction,
            sample_channels=settings.data.sample_channels,
        )
        manifest = creator.create_sample(mission)
        click.echo(f"Sample written to {manifest.sample_dir}")
        click.echo(f"  Channels : {', '.join(manifest.channels)}")
        click.echo(f"  Rows     : { {ch: n for ch, n in manifest.row_counts.items()} }")


# ---------------------------------------------------------------------------
# explore
# ---------------------------------------------------------------------------


@main.command()
@click.option("--mission", required=True, help="Mission name to explore.")
@click.option(
    "--channel",
    default=None,
    help="Show detailed stats for one channel only (default: all channels).",
)
@click.option(
    "--data-dir",
    default=None,
    type=click.Path(exists=False),
    help="Override the data directory (default: settings.data.sample_data_dir).",
)
@click.pass_context
def explore(
    ctx: click.Context,
    mission: str,
    channel: str | None,
    data_dir: str | None,
) -> None:
    """Print a structured exploration report of the sample dataset.

    Examples:

        # Full mission report
        spacecraft-telemetry explore --mission ESA-Mission1

        # Single-channel detail
        spacecraft-telemetry explore --mission ESA-Mission1 --channel A-1

        # Point at a custom data dir
        spacecraft-telemetry explore --mission ESA-Mission1 --data-dir data/raw
    """
    from rich.console import Console
    from rich.table import Table

    from spacecraft_telemetry.ingest.explore import DataExplorer

    settings = ctx.obj["settings"]
    resolved = Path(data_dir) if data_dir else Path(str(settings.data.sample_data_dir))
    explorer = DataExplorer(resolved)

    if channel:
        try:
            summary = explorer.channel_summary(mission, channel)
        except FileNotFoundError as exc:
            raise click.ClickException(str(exc)) from exc

        console = Console()
        tbl = Table(title=f"{mission} / {channel}", show_header=True)
        tbl.add_column("Property", style="bold cyan")
        tbl.add_column("Value")
        tbl.add_row("Rows", f"{summary.n_rows:,}")
        tbl.add_row("Columns", str(summary.n_columns))
        if summary.time_range:
            tbl.add_row("Time range", f"{summary.time_range[0]}  →  {summary.time_range[1]}")
        for col, stats in summary.value_stats.items():
            tbl.add_row(
                col,
                f"min={stats['min']:.4g}  max={stats['max']:.4g}  "
                f"mean={stats['mean']:.4g}  std={stats['std']:.4g}",
            )
        console.print(tbl)
    else:
        explorer.print_report(mission)


# ---------------------------------------------------------------------------
# spark group
# ---------------------------------------------------------------------------


@main.group()
def spark() -> None:
    """PySpark preprocessing pipeline commands."""


@spark.command("preprocess")
@click.option("--mission", required=True, help="Mission name to preprocess (e.g. ESA-Mission1).")
@click.option(
    "--sample-fraction",
    type=float,
    default=None,
    show_default=True,
    help="Ignored — reads from sample_data_dir (already sampled in ingest phase).",
)
@click.option(
    "--train-fraction",
    type=float,
    default=None,
    show_default=True,
    help="Temporal train split fraction (overrides config, default 0.8).",
)
@click.pass_context
def spark_preprocess(
    ctx: click.Context,
    mission: str,
    sample_fraction: float | None,
    train_fraction: float | None,
) -> None:
    """Run the Spark preprocessing pipeline for one mission.

    Reads channel Parquet files from sample_data_dir/{mission}/channels/,
    processes them (null-fill → gap-detect → normalize → label → features + series),
    and writes partitioned Parquet to spark.processed_data_dir/{mission}/.

    Examples:

        # Default settings (train_fraction=0.8)
        spacecraft-telemetry spark preprocess --mission ESA-Mission1
    """
    from spacecraft_telemetry.spark.pipeline import run_preprocessing
    from spacecraft_telemetry.spark.session import create_spark_session, stop_spark_session

    settings = ctx.obj["settings"]
    log = get_logger(__name__)

    # Apply CLI overrides to SparkConfig.
    spark_overrides: dict[str, object] = {}
    if train_fraction is not None:
        spark_overrides["train_fraction"] = train_fraction
    if spark_overrides:
        settings = settings.model_copy(
            update={"spark": settings.spark.model_copy(update=spark_overrides)}
        )

    log.info(
        "spark.preprocess.start",
        mission=mission,
        train_fraction=settings.spark.train_fraction,
    )

    session = create_spark_session(settings.spark)
    try:
        summary = run_preprocessing(session, settings, mission)
    finally:
        stop_spark_session(session)

    click.echo(f"Mission           : {mission}")
    click.echo(f"Channels processed: {summary['channels_processed']}")
    click.echo(f"Rows in           : {summary['rows_in']:,}")
    click.echo(f"Train rows        : {summary['train_rows']:,}")
    click.echo(f"Test rows         : {summary['test_rows']:,}")


# ---------------------------------------------------------------------------
# model group
# ---------------------------------------------------------------------------


@main.group()
def model() -> None:
    """Telemanom LSTM training + scoring commands."""


@model.command("train")
@click.option("--mission", required=True, help="Mission name (e.g. ESA-Mission1).")
@click.option("--channel", required=True, help="Channel ID (e.g. channel_1).")
@click.option(
    "--epochs",
    type=int,
    default=None,
    help="Override model.epochs from config.",
)
@click.option(
    "--batch-size",
    type=int,
    default=None,
    help="Override model.batch_size from config.",
)
@click.option(
    "--device",
    type=click.Choice(["auto", "cpu", "mps", "cuda"]),
    default=None,
    help="Override model.device from config.",
)
@click.option(
    "--window-size",
    type=int,
    default=None,
    help="Override model.window_size from config (LSTM sliding-window length).",
)
@click.option(
    "--prediction-horizon",
    type=int,
    default=None,
    help="Override model.prediction_horizon from config.",
)
@click.pass_context
def model_train(
    ctx: click.Context,
    mission: str,
    channel: str,
    epochs: int | None,
    batch_size: int | None,
    device: str | None,
    window_size: int | None,
    prediction_horizon: int | None,
) -> None:
    """Train a TelemanomLSTM on a single telemetry channel.

    Reads per-timestep series Parquet from spark.processed_data_dir/{mission}/train/
    and writes model artifacts to model.artifacts_dir/{mission}/{channel}/.
    Windows are constructed on-the-fly by the DataLoader (Plan 002.5).

    Examples:

        # Default config
        spacecraft-telemetry model train --mission ESA-Mission1 --channel channel_1

        # Quick dev run
        spacecraft-telemetry model train --mission ESA-Mission1 --channel channel_1 \\
            --epochs 5 --device cpu
    """
    from spacecraft_telemetry.model.training import train_channel

    settings = ctx.obj["settings"]
    model_overrides: dict[str, object] = {}
    if epochs is not None:
        model_overrides["epochs"] = epochs
    if batch_size is not None:
        model_overrides["batch_size"] = batch_size
    if device is not None:
        model_overrides["device"] = device
    if window_size is not None:
        model_overrides["window_size"] = window_size
    if prediction_horizon is not None:
        model_overrides["prediction_horizon"] = prediction_horizon
    if model_overrides:
        settings = settings.model_copy(
            update={"model": settings.model.model_copy(update=model_overrides)}
        )

    result = train_channel(settings, mission, channel)

    click.echo(f"Mission      : {mission}")
    click.echo(f"Channel      : {channel}")
    click.echo(f"Epochs run   : {result.epochs_run}")
    click.echo(f"Best epoch   : {result.best_epoch}")
    click.echo(f"Best val loss: {result.best_val_loss:.6f}")
    click.echo(f"MLflow URI   : {settings.mlflow.tracking_uri}")


@model.command("score")
@click.option("--mission", required=True, help="Mission name (e.g. ESA-Mission1).")
@click.option("--channel", required=True, help="Channel ID (e.g. channel_1).")
@click.pass_context
def model_score(
    ctx: click.Context,
    mission: str,
    channel: str,
) -> None:
    """Score a trained model against its test split and persist metrics.

    Loads model artifacts from model.artifacts_dir/{mission}/{channel}/ and
    reads test Parquet from spark.processed_data_dir/{mission}/test/.
    Writes errors.npy, threshold.json, metrics.json to the artifacts dir.

    Examples:

        spacecraft-telemetry model score --mission ESA-Mission1 --channel channel_1
    """
    from spacecraft_telemetry.model.scoring import score_channel

    settings = ctx.obj["settings"]
    metrics = score_channel(settings, mission, channel)

    click.echo(f"Mission  : {mission}")
    click.echo(f"Channel  : {channel}")
    click.echo(f"Precision: {metrics['precision']:.4f}")
    click.echo(f"Recall   : {metrics['recall']:.4f}")
    click.echo(f"F1       : {metrics['f1']:.4f}")
    click.echo(f"F0.5     : {metrics['f0_5']:.4f}")


# ---------------------------------------------------------------------------
# ray group
# ---------------------------------------------------------------------------


@_contextlib.contextmanager
def _ray_session(settings: Settings) -> Any:
    """Context manager that initialises a Ray cluster and shuts it down on exit.

    Propagates the current virtualenv's sys.path to Ray worker processes so
    they can find installed packages when launched via `uv run`.

    NOTE: The PYTHONPATH injection works correctly for local dev where driver
    and workers share the same filesystem. On Cloud Run / Dataproc (Phase 11),
    workers run in the container image, so PYTHONPATH from the driver node is
    irrelevant. Phase 11 must replace this with runtime_env derived from the
    container image (e.g. runtime_env={"pip": requirements_path}) or rely on
    the image having the package pre-installed.
    """
    import os
    import sys

    import ray

    _pythonpath = os.pathsep.join(p for p in sys.path if p)
    ray.init(
        address=settings.ray.address,
        num_cpus=settings.ray.num_cpus,
        ignore_reinit_error=True,
        runtime_env={"env_vars": {
            "PYTHONPATH": _pythonpath,
            # Propagate the resolved tracking URI so workers start with the
            # correct database and configure_mlflow() never sees a mismatch.
            "MLFLOW_TRACKING_URI": settings.mlflow.tracking_uri,
        }},
    )
    try:
        yield
    finally:
        ray.shutdown()


@click.group()
def ray_group() -> None:
    """Ray Core parallel training, scoring, and tuning commands."""


@ray_group.command("train")
@click.option("--mission", required=True, help="Mission name (e.g. ESA-Mission1).")
@click.option(
    "--channels",
    default=None,
    help="Comma-separated channel IDs to train. Defaults to all discovered channels.",
)
@click.option(
    "--max-channels",
    type=int,
    default=None,
    help="Cap sweep at this many channels (useful for smoke tests).",
)
@click.pass_context
def ray_train(
    ctx: click.Context,
    mission: str,
    channels: str | None,
    max_channels: int | None,
) -> None:
    """Train channels in parallel using Ray Core.

    Trains all discovered channels by default. Discovers channels by scanning
    the Spark processed-data directory unless --channels is supplied explicitly
    (required for gs:// artifact stores).

    Examples:

        # Train all discovered channels
        spacecraft-telemetry --env local ray train --mission ESA-Mission1

        # Smoke test: only first 3 channels
        spacecraft-telemetry ray train --mission ESA-Mission1 --max-channels 3

        # Cloud: explicit channel list (gs:// scan not supported)
        spacecraft-telemetry --env cloud ray train \\
            --mission ESA-Mission1 \\
            --channels channel_1,channel_2,channel_3
    """
    from spacecraft_telemetry.ray_training import discover_channels, train_all_channels

    settings = ctx.obj["settings"]
    log = get_logger(__name__)

    with _ray_session(settings):
        channel_list: list[str]
        if channels is not None:
            channel_list = [c.strip() for c in channels.split(",") if c.strip()]
        else:
            channel_list = discover_channels(settings, mission)
            if not channel_list:
                raise click.ClickException(
                    f"No preprocessed channels found for {mission}. "
                    "Run `spacecraft-telemetry spark preprocess` first, "
                    "or pass --channels explicitly."
                )

        log.info("ray.train.start", mission=mission, n_channels=len(channel_list))
        results = train_all_channels(
            settings, mission, channel_list, max_channels=max_channels
        )

    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_err = len(results) - n_ok
    click.echo(f"Mission  : {mission}")
    click.echo(f"Channels : {len(results)}")
    click.echo(f"OK       : {n_ok}")
    click.echo(f"Errors   : {n_err}")
    for r in results:
        if r["status"] == "ok":
            click.echo(
                f"  {r['channel']:20s}  epoch={r['best_epoch']:3d}"
                f"  val_loss={r['best_val_loss']:.6f}"
            )
        else:
            click.echo(f"  {r['channel']:20s}  ERROR — see logs")

    if n_err:
        raise SystemExit(1)


@ray_group.command("score")
@click.option("--mission", required=True, help="Mission name (e.g. ESA-Mission1).")
@click.option(
    "--channels",
    default=None,
    help="Comma-separated channel IDs to score. Defaults to all discovered channels.",
)
@click.option(
    "--max-channels",
    type=int,
    default=None,
    help="Cap sweep at this many channels.",
)
@click.option(
    "--tuned-configs",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to JSON file with per-subsystem scoring param overrides (Phase 6 output).",
)
@click.pass_context
def ray_score(
    ctx: click.Context,
    mission: str,
    channels: str | None,
    max_channels: int | None,
    tuned_configs: Path | None,
) -> None:
    """Score channels in parallel using Ray Core.

    Loads trained model artifacts and runs anomaly scoring on test data.
    Scores all channels by default. Optionally applies per-subsystem scoring
    param overrides from Phase 6 HPO output (--tuned-configs path/to/tuned_configs.json).

    Examples:

        # Score all channels with Hundman defaults
        spacecraft-telemetry ray score --mission ESA-Mission1

        # With Phase 6 HPO-tuned params
        spacecraft-telemetry ray score --mission ESA-Mission1 \\
            --tuned-configs outputs/tuned_configs.json
    """
    import json

    from spacecraft_telemetry.ray_training import discover_channels, score_all_channels

    settings = ctx.obj["settings"]
    log = get_logger(__name__)

    tuned: dict[str, Any] | None = None
    if tuned_configs is not None:
        with tuned_configs.open() as f:
            tuned = json.load(f)
        log.info("ray.score.tuned_configs_loaded", path=str(tuned_configs))

    with _ray_session(settings):
        channel_list: list[str]
        if channels is not None:
            channel_list = [c.strip() for c in channels.split(",") if c.strip()]
        else:
            channel_list = discover_channels(settings, mission)
            if not channel_list:
                raise click.ClickException(
                    f"No preprocessed channels found for {mission}. "
                    "Run `spacecraft-telemetry spark preprocess` first, "
                    "or pass --channels explicitly."
                )

        log.info("ray.score.start", mission=mission, n_channels=len(channel_list))
        results = score_all_channels(
            settings,
            mission,
            channel_list,
            max_channels=max_channels,
            tuned_configs=tuned,
        )

    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_err = len(results) - n_ok
    click.echo(f"Mission  : {mission}")
    click.echo(f"Channels : {len(results)}")
    click.echo(f"OK       : {n_ok}")
    click.echo(f"Errors   : {n_err}")
    for r in results:
        if r["status"] == "ok":
            click.echo(
                f"  {r['channel']:20s}  "
                f"P={r['precision']:.3f}  R={r['recall']:.3f}  "
                f"F1={r['f1']:.3f}  F0.5={r['f0_5']:.3f}"
            )
        else:
            click.echo(f"  {r['channel']:20s}  ERROR — see logs")

    if n_err:
        raise SystemExit(1)


@ray_group.command("tune")
@click.option("--mission", required=True, help="Mission name (e.g. ESA-Mission1).")
@click.option(
    "--channels",
    default=None,
    help="Comma-separated channel IDs to tune. Defaults to all discovered channels.",
)
@click.option(
    "--subsystem",
    default=None,
    help="Optional subsystem name to tune a single group (e.g. subsystem_1).",
)
@click.option(
    "--num-samples",
    type=click.IntRange(1),
    default=None,
    help="Override settings.tune.num_samples for this run.",
)
@click.option(
    "--overwrite-existing",
    is_flag=True,
    default=False,
    help="Overwrite existing tuned_configs.json when it contains invalid JSON.",
)
@click.pass_context
def ray_tune(
    ctx: click.Context,
    mission: str,
    channels: str | None,
    subsystem: str | None,
    num_samples: int | None,
    overwrite_existing: bool,
) -> None:
    """Run Ray Tune HPO for scoring parameters (Phase 6).

    Runs one Tune sweep per subsystem by default, or a single named subsystem
    when --subsystem is provided. Uses channel discovery unless --channels is
    passed explicitly.

    Examples:

        # Tune all discovered subsystems
        spacecraft-telemetry ray tune --mission ESA-Mission1

        # Tune one subsystem only
        spacecraft-telemetry ray tune --mission ESA-Mission1 --subsystem subsystem_1

        # Override sample count for a quick local check
        spacecraft-telemetry ray tune --mission ESA-Mission1 --num-samples 5
    """
    import json

    from spacecraft_telemetry.ray_training import (
        discover_channels,
        load_channel_subsystem_map,
        run_all_sweeps,
        run_hpo_sweep,
        write_tuned_configs,
    )

    settings = ctx.obj["settings"]
    log = get_logger(__name__)

    tune_settings = settings
    if num_samples is not None:
        tune_settings = settings.model_copy(
            update={
                "tune": settings.tune.model_copy(update={"num_samples": num_samples}),
            }
        )

    with _ray_session(tune_settings):
        channel_list: list[str]
        if channels is not None:
            channel_list = [c.strip() for c in channels.split(",") if c.strip()]
        else:
            channel_list = discover_channels(tune_settings, mission)
            if not channel_list:
                raise click.ClickException(
                    f"No preprocessed channels found for {mission}. "
                    "Run `spacecraft-telemetry spark preprocess` first, "
                    "or pass --channels explicitly."
                )

        log.info(
            "ray.tune.start",
            mission=mission,
            n_channels=len(channel_list),
            subsystem=subsystem,
            num_samples=tune_settings.tune.num_samples,
        )

        if subsystem is None:
            output_path = run_all_sweeps(tune_settings, mission, channel_list)
            click.echo(f"Mission       : {mission}")
            click.echo(f"Channels      : {len(channel_list)}")
            click.echo("Subsystems    : all")
            click.echo(f"Num samples   : {tune_settings.tune.num_samples}")
            click.echo(f"Output        : {output_path}")
            return

        subsystem_map = load_channel_subsystem_map(tune_settings, mission)
        if not subsystem_map:
            raise click.ClickException(
                "channels.csv not found or empty; cannot resolve --subsystem. "
                "Pass --channels explicitly or ensure data/raw/{mission}/channels.csv exists."
            )

        subsystem_channels = [
            ch for ch in channel_list if subsystem_map.get(ch) == subsystem
        ]
        if not subsystem_channels:
            raise click.ClickException(
                f"No channels found for subsystem {subsystem!r} in mission {mission}."
            )

        best = run_hpo_sweep(subsystem, subsystem_channels, tune_settings, mission)
        output_path = Path(tune_settings.model.artifacts_dir) / mission / "tuned_configs.json"

        existing: dict[str, dict[str, Any]] = {}
        if output_path.exists():
            try:
                loaded = json.loads(output_path.read_text())
                if isinstance(loaded, dict):
                    existing = {
                        str(k): v for k, v in loaded.items() if isinstance(v, dict)
                    }
            except json.JSONDecodeError as err:
                if not overwrite_existing:
                    raise click.ClickException(
                        "Existing tuned config file contains invalid JSON. "
                        "Fix/remove the file, or re-run with --overwrite-existing."
                    ) from err
                log.warning("ray.tune.output.invalid_json.overwriting", path=str(output_path))

        entry: dict[str, Any] = {
            **best.get("config", {}),
            "_meta": {
                "run_id": best.get("run_id"),
                "f0_5": best.get("f0_5", 0.0),
            },
        }
        existing[subsystem] = entry
        write_tuned_configs(existing, output_path)

    click.echo(f"Mission       : {mission}")
    click.echo(f"Channels      : {len(subsystem_channels)}")
    click.echo(f"Subsystem     : {subsystem}")
    click.echo(f"Num samples   : {tune_settings.tune.num_samples}")
    click.echo(f"Best config   : {best.get('config', best)}")
    click.echo(f"Output        : {output_path}")


main.add_command(ray_group, name="ray")


# ---------------------------------------------------------------------------
# mlflow group
# ---------------------------------------------------------------------------


@click.group()
def mlflow_group() -> None:
    """MLflow model registry commands."""


@mlflow_group.command("promote")
@click.option(
    "--name",
    required=True,
    help="Registered model name (e.g. telemanom-ESA-Mission1-channel_1).",
)
@click.option(
    "--version",
    "model_version",
    type=int,
    default=None,
    help="Model version number. Defaults to latest non-archived version.",
)
@click.option(
    "--stage",
    required=True,
    type=click.Choice(["Staging", "Production", "Archived"]),
    help="Target stage for the model version.",
)
@click.pass_context
def mlflow_promote(ctx: click.Context, name: str, model_version: int | None, stage: str) -> None:
    """Promote a registered model version to Staging, Production, or Archived."""
    import mlflow
    from spacecraft_telemetry.mlflow_tracking.registry import promote

    settings: Settings = ctx.obj["settings"]
    tracking_uri = settings.mlflow.tracking_uri
    mlflow.set_tracking_uri(tracking_uri)

    try:
        promote(name=name, version=model_version, stage=stage)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    resolved_version = model_version if model_version is not None else "(latest non-archived)"
    click.echo(f"Model         : {name}")
    click.echo(f"Version       : {resolved_version}")
    click.echo(f"Stage         : {stage}")
    click.echo(f"Tracking URI  : {tracking_uri}")


@mlflow_group.command("ui")
@click.option(
    "--port",
    type=int,
    default=5001,
    show_default=True,
    help="Port to serve MLflow UI on.",
)
@click.pass_context
def mlflow_ui(ctx: click.Context, port: int) -> None:
    """Start the MLflow tracking server against the configured backend store.

    Starts a full MLflow server (not the read-only UI) so that concurrent Ray
    workers can write runs through the HTTP REST API, avoiding SQLite write-lock
    contention.  The UI is also served at the same port.

    When tracking_uri is an HTTP endpoint, the server is started against
    backend_store_uri (the underlying SQLite file).  When tracking_uri is a
    SQLite URI directly, it is used as both the backend and the client endpoint.
    """
    import os

    settings: Settings = ctx.obj["settings"]
    tracking_uri = settings.mlflow.tracking_uri

    if tracking_uri.startswith(("http://", "https://")):
        backend = settings.mlflow.backend_store_uri
        if not backend:
            raise click.ClickException(
                "tracking_uri is an HTTP endpoint but mlflow.backend_store_uri is not "
                "set in config. Add backend_store_uri: 'sqlite:///mlflow.db' under the "
                "mlflow section of your config file."
            )
        _db_path = backend.removeprefix("sqlite:///")
    else:
        backend = tracking_uri
        _db_path = tracking_uri.removeprefix("sqlite:///")

    from pathlib import Path as _Path
    _artifact_root = str(_Path(_db_path).parent / "mlartifacts")
    click.echo(f"Backend store : {backend}")
    click.echo(f"Artifact root : {_artifact_root}")
    click.echo(f"Port          : {port}")
    click.echo(f"UI URL        : http://127.0.0.1:{port}")
    os.execvp(
        "mlflow",
        [
            "mlflow", "server",
            "--backend-store-uri", backend,
            "--default-artifact-root", _artifact_root,
            "--host", "127.0.0.1",
            "--port", str(port),
        ],
    )


main.add_command(mlflow_group, name="mlflow")
