"""Structured logging setup using structlog.

Usage:
    from spacecraft_telemetry.core.logging import get_logger, setup_logging
    from spacecraft_telemetry.core.config import LoggingConfig

    setup_logging(LoggingConfig(level="DEBUG", format="console"))
    log = get_logger(__name__)
    log.info("pipeline started", mission="ESA-Mission1", channels=5)
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
import structlog.contextvars
import structlog.dev
import structlog.processors
import structlog.types

from spacecraft_telemetry.core.config import LoggingConfig


def setup_logging(config: LoggingConfig) -> None:
    """Configure structlog and the stdlib root logger.

    Two output modes:
    - "console": human-readable with colours (local dev)
    - "json":    one JSON object per line (cloud / log aggregators)

    Call once at process startup before any logging occurs. Calling again
    reconfigures structlog in place (useful in tests).

    Args:
        config: LoggingConfig carrying level and format fields.
    """
    numeric_level: int = getattr(logging, config.level.upper(), logging.INFO)
    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)

    processors: list[Any] = [
        # Merge any context vars set via structlog.contextvars.bind_contextvars()
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        timestamper,
    ]

    if config.format == "json":
        # dict_tracebacks MUST precede JSONRenderer: it expands exc_info into a
        # structured "exception" field. Without it, JSONRenderer serializes
        # exc_info=True as the literal boolean `true` and the traceback is lost
        # entirely — exactly what blinded the Phase 12 collector flush errors.
        # ConsoleRenderer (below) formats exc_info on its own, so this is only
        # needed on the JSON path.
        processors += [
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    # Configure the stdlib root logger so third-party library logs are visible.
    logging.basicConfig(
        format="%(levelname)-8s %(name)s: %(message)s",
        stream=sys.stdout,
        level=numeric_level,
        force=True,
    )


def get_logger(name: str) -> structlog.types.FilteringBoundLogger:
    """Return a structlog bound logger for the given name.

    Args:
        name: Logger name, typically the calling module's __name__.

    Returns:
        A structlog BoundLogger pre-bound with {"logger": name}.
    """
    return structlog.get_logger().bind(logger=name)  # type: ignore[no-any-return]
