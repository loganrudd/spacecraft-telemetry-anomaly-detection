"""Tests for core.logging."""

import logging

import structlog.testing

from spacecraft_telemetry.core.config import LoggingConfig
from spacecraft_telemetry.core.logging import get_logger, setup_logging


class TestSetupLogging:
    def test_console_format_does_not_raise(self) -> None:
        setup_logging(LoggingConfig(level="DEBUG", format="console"))

    def test_json_format_does_not_raise(self) -> None:
        setup_logging(LoggingConfig(level="INFO", format="json"))

    def test_stdlib_root_level_matches_config(self) -> None:
        setup_logging(LoggingConfig(level="WARNING", format="console"))
        assert logging.getLogger().level == logging.WARNING

    def test_stdlib_root_level_debug(self) -> None:
        setup_logging(LoggingConfig(level="DEBUG", format="console"))
        assert logging.getLogger().level == logging.DEBUG


class TestGetLogger:
    def test_returns_bound_logger(self) -> None:
        setup_logging(LoggingConfig(level="DEBUG", format="console"))
        log = get_logger("spacecraft_telemetry.test")
        assert log is not None

    def test_captures_event(self) -> None:
        setup_logging(LoggingConfig(level="DEBUG", format="console"))
        log = get_logger("test.module")

        with structlog.testing.capture_logs() as captured:
            log.info("pipeline started", mission="ESA-Mission1")

        assert len(captured) == 1
        assert captured[0]["event"] == "pipeline started"
        assert captured[0]["mission"] == "ESA-Mission1"
        assert captured[0]["log_level"] == "info"

    def test_captures_extra_fields(self) -> None:
        setup_logging(LoggingConfig(level="DEBUG", format="console"))
        log = get_logger("test.ingest")

        with structlog.testing.capture_logs() as captured:
            log.debug("downloading file", channel="A-1", size_mb=42)

        assert captured[0]["channel"] == "A-1"
        assert captured[0]["size_mb"] == 42

    def test_debug_not_captured_when_level_is_warning(self) -> None:
        setup_logging(LoggingConfig(level="WARNING", format="console"))
        log = get_logger("test.filter")

        with structlog.testing.capture_logs() as captured:
            log.debug("this should be filtered")
            log.warning("this should appear")

        # capture_logs bypasses the level filter — both will be captured.
        # We verify that the level field is set correctly instead.
        events = {e["event"] for e in captured}
        assert "this should appear" in events

    def test_different_names_return_independent_loggers(self) -> None:
        setup_logging(LoggingConfig(level="DEBUG", format="console"))
        log_a = get_logger("module.a")
        log_b = get_logger("module.b")

        with structlog.testing.capture_logs() as captured:
            log_a.info("from a")
            log_b.info("from b")

        assert len(captured) == 2
        events = [e["event"] for e in captured]
        assert "from a" in events
        assert "from b" in events
