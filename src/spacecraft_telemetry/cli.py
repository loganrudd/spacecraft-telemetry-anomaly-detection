"""CLI entry point for the spacecraft telemetry anomaly detection system.

Each subcommand follows the same pattern:
    1. Load settings for the given --env
    2. Set up structured logging (--verbose overrides the config level to DEBUG)
    3. Delegate to the relevant ingest module
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

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
    spark_overrides: dict[str, object] = {}
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


# ---------------------------------------------------------------------------
# feast group
# ---------------------------------------------------------------------------


@main.group()
def feast() -> None:
    """Feast feature store commands."""


def _resolve_feast_settings(
    ctx: click.Context,
    mission: str | None,
) -> Settings:
    """Return settings, optionally overriding source_path for the given mission."""
    settings = ctx.obj["settings"]
    if mission is not None:
        new_source_path = (
            Path(str(settings.spark.processed_data_dir)) / mission / "features"
        )
        settings = settings.model_copy(
            update={"feast": settings.feast.model_copy(update={"source_path": new_source_path})}
        )
    return settings


@feast.command("apply")
@click.option(
    "--mission",
    default=None,
    help="Override source_path to spark.processed_data_dir/{mission}/features. "
         "Default: use settings.feast.source_path from config.",
)
@click.pass_context
def feast_apply(ctx: click.Context, mission: str | None) -> None:
    """Register feature view definitions to the local Feast registry.

    Examples:

        # Use config default source_path
        spacecraft-telemetry feast apply

        # Override source path for a specific mission
        spacecraft-telemetry feast apply --mission ESA-Mission1
    """
    from spacecraft_telemetry.feast_client.store import apply_definitions, create_feature_store

    settings = _resolve_feast_settings(ctx, mission)
    log = get_logger(__name__)

    log.info("feast.apply.start", repo_path=str(settings.feast.repo_path))
    store = create_feature_store(settings)
    counts = apply_definitions(store)
    click.echo(f"Registry      : {settings.feast.repo_path}/data/registry.db")
    click.echo(f"Entities      : {counts['entities']}")
    click.echo(f"Feature views : {counts['feature_views']}")


@feast.command("materialize")
@click.option(
    "--mission",
    default=None,
    help="Override source_path to spark.processed_data_dir/{mission}/features.",
)
@click.option(
    "--end-date",
    type=click.DateTime(),
    default=None,
    help="Materialize up to this date (default: now).",
)
@click.option(
    "--start-date",
    type=click.DateTime(),
    default=None,
    help="Explicit backfill start. Omit for incremental (since last materialization).",
)
@click.pass_context
def feast_materialize(
    ctx: click.Context,
    mission: str | None,
    end_date: datetime | None,
    start_date: datetime | None,
) -> None:
    """Materialize offline features into the online (SQLite) store.

    Incremental by default — only processes rows added since the last run.
    Use --start-date for a full backfill window.

    Examples:

        # Incremental (most common)
        spacecraft-telemetry feast materialize --mission ESA-Mission1

        # Explicit backfill window
        spacecraft-telemetry feast materialize --mission ESA-Mission1 \\
            --start-date 2000-01-01 --end-date 2001-01-01
    """
    from datetime import datetime, timezone

    from spacecraft_telemetry.feast_client.store import (
        apply_definitions,
        create_feature_store,
        materialize,
    )

    settings = _resolve_feast_settings(ctx, mission)
    log = get_logger(__name__)

    end = end_date or datetime.now(tz=timezone.utc)
    store = create_feature_store(settings)
    # Ensure definitions are registered before materializing.
    apply_definitions(store)

    log.info(
        "feast.materialize.cli.start",
        end_date=str(end),
        start_date=str(start_date) if start_date else "incremental",
    )
    materialize(store, end_date=end, start_date=start_date)
    click.echo(f"Materialized up to {end.isoformat()}")
    click.echo(f"Online store   : {settings.feast.repo_path}/data/online_store.db")


@feast.command("retrieve")
@click.option("--channel", required=True, help="Channel entity key (e.g. channel_1).")
@click.option("--mission", required=True, help="Mission entity key (e.g. ESA-Mission1).")
@click.option(
    "--mode",
    type=click.Choice(["online", "historical"]),
    default="online",
    show_default=True,
    help="online: latest materialized values; historical: point-in-time window.",
)
@click.option(
    "--start",
    type=click.DateTime(),
    default=None,
    help="Historical window start (required when --mode=historical).",
)
@click.option(
    "--end",
    type=click.DateTime(),
    default=None,
    help="Historical window end (default: now, used when --mode=historical).",
)
@click.pass_context
def feast_retrieve(
    ctx: click.Context,
    channel: str,
    mission: str,
    mode: str,
    start: datetime | None,
    end: datetime | None,
) -> None:
    """Retrieve features for a channel/mission pair.

    Examples:

        # Latest values from the online store
        spacecraft-telemetry feast retrieve --channel channel_1 --mission ESA-Mission1

        # Point-in-time historical window
        spacecraft-telemetry feast retrieve --channel channel_1 --mission ESA-Mission1 \\
            --mode historical --start 2000-01-01 --end 2000-02-01
    """
    import json
    from datetime import datetime, timezone

    import pandas as pd

    from spacecraft_telemetry.feast_client.client import (
        get_historical_features,
        get_online_features_for_channel,
    )
    from spacecraft_telemetry.feast_client.store import apply_definitions, create_feature_store

    settings = _resolve_feast_settings(ctx, mission)
    store = create_feature_store(settings)
    apply_definitions(store)

    if mode == "online":
        result = get_online_features_for_channel(store, channel_id=channel, mission_id=mission)
        click.echo(json.dumps(result, indent=2, default=str))
    else:
        if start is None:
            raise click.UsageError("--start is required when --mode=historical")
        end_ts = end or datetime.now(tz=timezone.utc)
        # Build a small entity_df spanning the requested window using real timestamps
        # from the offline Parquet. For the CLI we synthesize a grid at 90s intervals;
        # Phase 4 (training) will supply real telemetry timestamps instead.
        start_utc = start.replace(tzinfo=timezone.utc) if start.tzinfo is None else start
        end_utc = end_ts.replace(tzinfo=timezone.utc) if end_ts.tzinfo is None else end_ts
        timestamps = pd.date_range(start=start_utc, end=end_utc, freq="90s", tz="UTC")
        entity_df = pd.DataFrame(
            {
                "channel_id": channel,
                "mission_id": mission,
                "event_timestamp": timestamps,
            }
        )
        df = get_historical_features(store, entity_df)
        click.echo(df.to_string())
