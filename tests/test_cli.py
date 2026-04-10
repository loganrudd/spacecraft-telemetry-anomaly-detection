"""Tests for the CLI entry point."""

from __future__ import annotations

import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner

from spacecraft_telemetry import __version__
from spacecraft_telemetry.cli import main

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _base_args(tmp_path: Path, env: str = "test") -> list[str]:
    """Return CLI args that point config at a temp dir so no real YAML is needed."""
    return [f"--env={env}"]


def _write_sample_mission(sample_dir: Path, mission: str, n_rows: int = 20) -> None:
    """Write a tiny Parquet + labels setup for the explore command."""
    rng = np.random.default_rng(0)
    ch_dir = sample_dir / mission / "channels"
    ch_dir.mkdir(parents=True)
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=n_rows, freq="1s"),
            "value": rng.random(n_rows),
        }
    )
    df.to_parquet(ch_dir / "A-1.parquet", index=False)
    pd.DataFrame([{"channel": "A-1", "start": 0, "end": 5}]).to_csv(
        sample_dir / mission / "labels.csv", index=False
    )


def _write_raw_mission(raw_dir: Path, mission: str, n_rows: int = 100) -> None:
    """Write a tiny pickle channel for the download → sample path."""
    rng = np.random.default_rng(0)
    ch_dir = raw_dir / mission / "channels"
    ch_dir.mkdir(parents=True)
    df = pd.DataFrame({"value": rng.random(n_rows)})
    with (ch_dir / "A-1.pkl").open("wb") as fh:
        pickle.dump(df, fh)


# ---------------------------------------------------------------------------
# Top-level group
# ---------------------------------------------------------------------------


class TestMainGroup:
    def test_help_lists_all_subcommands(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "download" in result.output
        assert "explore" in result.output
        assert "version" in result.output

    def test_unknown_option_exits_nonzero(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["--not-a-flag"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# version
# ---------------------------------------------------------------------------


class TestVersionCommand:
    def test_prints_package_version(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["version"])
        assert result.exit_code == 0
        assert __version__ in result.output

    def test_version_help(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["version", "--help"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------


class TestDownloadCommand:
    def test_help_shows_options(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["download", "--help"])
        assert result.exit_code == 0
        assert "--mission" in result.output
        assert "--sample" in result.output
        assert "--sample-fraction" in result.output

    def test_requires_mission_option(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["download"])
        assert result.exit_code != 0
        assert "mission" in result.output.lower()

    def test_download_calls_download_mission(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SPACECRAFT_CONFIG_DIR", str(tmp_path))
        monkeypatch.setenv("SPACECRAFT_DATA__RAW_DATA_DIR", str(tmp_path / "raw"))

        mock_downloader = MagicMock()
        mock_downloader.download_mission.return_value = tmp_path / "raw" / "M1"

        with patch(
            "spacecraft_telemetry.ingest.download.ZenodoDownloader",
            return_value=mock_downloader,
        ):
            result = runner.invoke(main, ["--env=local", "download", "--mission=M1"])

        assert result.exit_code == 0, result.output
        mock_downloader.download_mission.assert_called_once_with("M1")

    def test_download_with_sample_calls_create_sample(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        raw_dir = tmp_path / "raw"
        sample_dir = tmp_path / "sample"
        monkeypatch.setenv("SPACECRAFT_CONFIG_DIR", str(tmp_path))
        monkeypatch.setenv("SPACECRAFT_DATA__RAW_DATA_DIR", str(raw_dir))
        monkeypatch.setenv("SPACECRAFT_DATA__SAMPLE_DATA_DIR", str(sample_dir))

        _write_raw_mission(raw_dir, "M1")

        mock_downloader = MagicMock()
        mock_downloader.download_mission.return_value = raw_dir / "M1"

        with patch(
            "spacecraft_telemetry.ingest.download.ZenodoDownloader",
            return_value=mock_downloader,
        ):
            result = runner.invoke(main, ["--env=local", "download", "--mission=M1", "--sample"])

        assert result.exit_code == 0, result.output
        assert "Sample written to" in result.output

    def test_sample_fraction_override(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        raw_dir = tmp_path / "raw"
        sample_dir = tmp_path / "sample"
        monkeypatch.setenv("SPACECRAFT_DATA__RAW_DATA_DIR", str(raw_dir))
        monkeypatch.setenv("SPACECRAFT_DATA__SAMPLE_DATA_DIR", str(sample_dir))

        _write_raw_mission(raw_dir, "M1", n_rows=200)

        mock_downloader = MagicMock()
        mock_downloader.download_mission.return_value = raw_dir / "M1"

        with patch(
            "spacecraft_telemetry.ingest.download.ZenodoDownloader",
            return_value=mock_downloader,
        ):
            result = runner.invoke(
                main,
                ["--env=local", "download", "--mission=M1", "--sample", "--sample-fraction=0.5"],
            )

        assert result.exit_code == 0, result.output
        # 200 rows * 0.5 = 100 rows written
        parquet = sample_dir / "M1" / "channels" / "A-1.parquet"
        assert parquet.exists()
        assert len(pd.read_parquet(parquet)) == 100


# ---------------------------------------------------------------------------
# explore
# ---------------------------------------------------------------------------


class TestExploreCommand:
    def test_help_shows_options(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["explore", "--help"])
        assert result.exit_code == 0
        assert "--mission" in result.output
        assert "--channel" in result.output
        assert "--data-dir" in result.output

    def test_requires_mission_option(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["explore"])
        assert result.exit_code != 0
        assert "mission" in result.output.lower()

    def test_full_mission_report(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sample_dir = tmp_path / "sample"
        _write_sample_mission(sample_dir, "M1")
        monkeypatch.setenv("SPACECRAFT_DATA__SAMPLE_DATA_DIR", str(sample_dir))

        result = runner.invoke(main, ["--env=local", "explore", "--mission=M1"])

        assert result.exit_code == 0, result.output
        assert "M1" in result.output

    def test_single_channel_report(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sample_dir = tmp_path / "sample"
        _write_sample_mission(sample_dir, "M1")
        monkeypatch.setenv("SPACECRAFT_DATA__SAMPLE_DATA_DIR", str(sample_dir))

        result = runner.invoke(main, ["--env=local", "explore", "--mission=M1", "--channel=A-1"])

        assert result.exit_code == 0, result.output
        assert "A-1" in result.output

    def test_data_dir_override(self, runner: CliRunner, tmp_path: Path) -> None:
        custom_dir = tmp_path / "custom"
        _write_sample_mission(custom_dir, "M1")

        result = runner.invoke(
            main,
            ["--env=local", "explore", "--mission=M1", f"--data-dir={custom_dir}"],
        )

        assert result.exit_code == 0, result.output
        assert "M1" in result.output

    def test_missing_channel_exits_with_error(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sample_dir = tmp_path / "sample"
        _write_sample_mission(sample_dir, "M1")
        monkeypatch.setenv("SPACECRAFT_DATA__SAMPLE_DATA_DIR", str(sample_dir))

        result = runner.invoke(main, ["--env=local", "explore", "--mission=M1", "--channel=Z-99"])

        assert result.exit_code != 0

    def test_verbose_flag_accepted(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sample_dir = tmp_path / "sample"
        _write_sample_mission(sample_dir, "M1")
        monkeypatch.setenv("SPACECRAFT_DATA__SAMPLE_DATA_DIR", str(sample_dir))

        result = runner.invoke(main, ["--env=local", "--verbose", "explore", "--mission=M1"])

        assert result.exit_code == 0, result.output
