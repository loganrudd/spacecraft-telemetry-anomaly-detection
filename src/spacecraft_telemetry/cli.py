"""CLI entry point — implemented in Step 8."""

import click


@click.group()
@click.option("--env", default="local", show_default=True, help="Config environment to load.")
@click.option("--verbose", is_flag=True, help="Enable debug logging.")
@click.pass_context
def main(ctx: click.Context, env: str, verbose: bool) -> None:
    """Spacecraft Telemetry Anomaly Detection System."""
    ctx.ensure_object(dict)
    ctx.obj["env"] = env
    ctx.obj["verbose"] = verbose


@main.command()
def version() -> None:
    """Print the package version."""
    from spacecraft_telemetry import __version__

    click.echo(__version__)
