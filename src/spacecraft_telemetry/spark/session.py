"""SparkSession factory for the spacecraft telemetry preprocessing pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

from spacecraft_telemetry.core.config import SparkConfig
from spacecraft_telemetry.core.logging import get_logger

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

log = get_logger(__name__)


def create_spark_session(
    config: SparkConfig, app_name: str = "spacecraft-telemetry"
) -> SparkSession:
    """Build and return a SparkSession configured for the given environment.

    Imports PySpark lazily so the rest of the package works without the spark
    optional dependency installed.

    Args:
        config: SparkConfig controlling memory, parallelism, and Spark settings.
        app_name: Spark application name shown in the UI.

    Returns:
        A configured SparkSession (local mode when num_cores <= 8).
    """
    try:
        from pyspark.sql import SparkSession
    except ImportError as exc:
        raise ImportError(
            "PySpark is required for the preprocessing pipeline. "
            "Install it with: uv sync --extra spark"
        ) from exc

    master = f"local[{config.num_cores}]"

    log.info(
        "creating_spark_session",
        master=master,
        driver_memory=config.driver_memory,
        app_name=app_name,
    )

    session = (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.driver.memory", config.driver_memory)
        # Silence noisy Hadoop ViewFS warning on macOS local mode
        .config(
            "spark.hadoop.fs.viewfs.overload.scheme.target.file.impl",
            "org.apache.hadoop.fs.LocalFileSystem",
        )
        # Keep UI disabled by default for CLI runs; enable via env var if needed
        .config("spark.ui.enabled", "false")
        # Reduce default 200 shuffle partitions — local mode with 2 cores doesn't need them
        # and 200 * n_channels produces thousands of tiny output files locally.
        .config("spark.sql.shuffle.partitions", "8")
        # ANSI mode stays on (PySpark 4.x default) — strict types are correct
        .getOrCreate()
    )

    # Reduce Spark's verbose logging to WARNING for cleaner CLI output
    session.sparkContext.setLogLevel("WARN")

    return session


def stop_spark_session(session) -> None:  # type: ignore[no-untyped-def]
    """Stop a SparkSession. Safe to call if already stopped."""
    try:
        session.stop()
        log.info("spark_session_stopped")
    except Exception as exc:
        log.warning("spark_session_stop_failed", exc=str(exc))
