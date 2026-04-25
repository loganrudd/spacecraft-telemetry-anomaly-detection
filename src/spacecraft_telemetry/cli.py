"""CLI entry point for the spacecraft telemetry anomaly detection system.

Each subcommand follows the same pattern:
    1. Load settings for the given --env
    2. Set up structured logging (--verbose overrides the config level to DEBUG)
    3. Delegate to the relevant ingest module
"""

from __future__ import annotations

from pathlib import Path

import click

from spacecraft_telemetry.core.config import LoggingConfig, load_settings
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
    "--window-size",
    type=int,
    default=None,
    show_default=True,
    help="LSTM sliding-window length (overrides config, default 250).",
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
    window_size: int | None,
    train_fraction: float | None,
) -> None:
    """Run the Spark preprocessing pipeline for one mission.

    Reads channel Parquet files from sample_data_dir/{mission}/channels/,
    processes them (null-fill → gap-detect → normalize → features + windows),
    and writes partitioned Parquet to spark.processed_data_dir/{mission}/.

    Examples:

        # Default settings (window_size=250, train_fraction=0.8)
        spacecraft-telemetry spark preprocess --mission ESA-Mission1

        # Quick dev run with smaller windows
        spacecraft-telemetry spark preprocess --mission ESA-Mission1 --window-size 50
    """
    from spacecraft_telemetry.spark.pipeline import run_preprocessing
    from spacecraft_telemetry.spark.session import create_spark_session, stop_spark_session

    settings = ctx.obj["settings"]
    log = get_logger(__name__)

    # Apply CLI overrides to SparkConfig.
    spark_overrides: dict = {}
    if window_size is not None:
        spark_overrides["window_size"] = window_size
    if train_fraction is not None:
        spark_overrides["train_fraction"] = train_fraction
    if spark_overrides:
        settings = settings.model_copy(
            update={"spark": settings.spark.model_copy(update=spark_overrides)}
        )

    log.info(
        "spark.preprocess.start",
        mission=mission,
        window_size=settings.spark.window_size,
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
    click.echo(f"Feature rows out  : {summary['feature_rows_out']:,}")
    click.echo(f"Train windows     : {summary['train_windows']:,}")
    click.echo(f"Test windows      : {summary['test_windows']:,}")
    click.echo(f"Total windows     : {summary['windows_out']:,}")
