"""Tests for the CLI `api serve` command."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from spacecraft_telemetry.cli import main


class TestApiServeHelp:
    def test_help_exits_zero(self) -> None:
        result = CliRunner().invoke(main, ["--env=test", "api", "serve", "--help"])
        assert result.exit_code == 0

    def test_help_lists_host_option(self) -> None:
        result = CliRunner().invoke(main, ["--env=test", "api", "serve", "--help"])
        assert "--host" in result.output

    def test_help_lists_port_option(self) -> None:
        result = CliRunner().invoke(main, ["--env=test", "api", "serve", "--help"])
        assert "--port" in result.output

    def test_help_lists_mission_option(self) -> None:
        result = CliRunner().invoke(main, ["--env=test", "api", "serve", "--help"])
        assert "--mission" in result.output

    def test_help_lists_subsystem_option(self) -> None:
        result = CliRunner().invoke(main, ["--env=test", "api", "serve", "--help"])
        assert "--subsystem" in result.output

    def test_help_lists_channels_option(self) -> None:
        result = CliRunner().invoke(main, ["--env=test", "api", "serve", "--help"])
        assert "--channels" in result.output

    def test_api_group_help_exits_zero(self) -> None:
        result = CliRunner().invoke(main, ["--env=test", "api", "--help"])
        assert result.exit_code == 0

    def test_api_group_lists_serve_subcommand(self) -> None:
        result = CliRunner().invoke(main, ["--env=test", "api", "--help"])
        assert "serve" in result.output


class TestApiServeRun:
    """Test that uvicorn.run is called with the right host/port settings."""

    def _run_serve(self, extra_args: list[str]) -> tuple[MagicMock, MagicMock]:
        """Invoke `api serve` with both uvicorn.run and create_app patched.

        Returns (mock_uvicorn_run, mock_create_app).
        """
        mock_app = MagicMock()
        with (
            patch("uvicorn.run") as mock_run,
            patch(
                "spacecraft_telemetry.api.app.create_app", return_value=mock_app
            ) as mock_create,
        ):
            result = CliRunner().invoke(
                main, ["--env=test", "api", "serve", *extra_args]
            )
            assert result.exit_code == 0, result.output
        return mock_run, mock_create

    def test_uvicorn_run_called(self) -> None:
        mock_run, _ = self._run_serve([])
        mock_run.assert_called_once()

    def test_default_host_from_config(self) -> None:
        """Without --host, host comes from ApiConfig defaults."""
        mock_run, _ = self._run_serve([])
        _app, kwargs = mock_run.call_args[0][0], mock_run.call_args[1]
        # host is passed as a keyword argument
        assert "host" in kwargs

    def test_override_host_is_forwarded(self) -> None:
        mock_run, _ = self._run_serve(["--host=0.0.0.0"])
        kwargs = mock_run.call_args[1]
        assert kwargs["host"] == "0.0.0.0"

    def test_override_port_is_forwarded(self) -> None:
        mock_run, _ = self._run_serve(["--port=9999"])
        kwargs = mock_run.call_args[1]
        assert kwargs["port"] == 9999

    def test_reload_flag_forwarded(self) -> None:
        mock_run, _ = self._run_serve(["--reload"])
        kwargs = mock_run.call_args[1]
        assert kwargs["reload"] is True

    def test_reload_off_by_default(self) -> None:
        mock_run, _ = self._run_serve([])
        kwargs = mock_run.call_args[1]
        assert kwargs["reload"] is False

    def test_channels_csv_parsed(self) -> None:
        """--channels a,b should reach create_app with channels=['a','b']."""
        mock_app = MagicMock()
        captured: list[object] = []

        def _capture_create_app(settings):  # type: ignore[no-untyped-def]
            captured.append(settings)
            return mock_app

        with patch("uvicorn.run"), patch(
            "spacecraft_telemetry.api.app.create_app", side_effect=_capture_create_app
        ):
            result = CliRunner().invoke(
                main, ["--env=test", "api", "serve", "--channels=ch-a,ch-b"]
            )
        assert result.exit_code == 0, result.output
        assert len(captured) == 1
        settings = captured[0]
        assert hasattr(settings, "api")
        assert settings.api.channels == ["ch-a", "ch-b"]

    def test_mission_override_applied(self) -> None:
        mock_app = MagicMock()
        captured: list[object] = []

        def _capture(settings):  # type: ignore[no-untyped-def]
            captured.append(settings)
            return mock_app

        with patch("uvicorn.run"), patch(
            "spacecraft_telemetry.api.app.create_app", side_effect=_capture
        ):
            CliRunner().invoke(
                main, ["--env=test", "api", "serve", "--mission=my-mission"]
            )
        assert captured[0].api.mission == "my-mission"

    def test_subsystem_override_applied(self) -> None:
        mock_app = MagicMock()
        captured: list[object] = []

        def _capture(settings):  # type: ignore[no-untyped-def]
            captured.append(settings)
            return mock_app

        with patch("uvicorn.run"), patch(
            "spacecraft_telemetry.api.app.create_app", side_effect=_capture
        ):
            CliRunner().invoke(
                main, ["--env=test", "api", "serve", "--subsystem=my-sub"]
            )
        assert captured[0].api.subsystem == "my-sub"
