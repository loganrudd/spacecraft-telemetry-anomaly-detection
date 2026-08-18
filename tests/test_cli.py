"""Tests for the CLI entry point."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pyarrow as _pa
import pyarrow.parquet as _pq
import pytest
from click.testing import CliRunner

from spacecraft_telemetry import __version__
from spacecraft_telemetry.cli import main
from spacecraft_telemetry.core.config import load_settings

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
    df.to_parquet(ch_dir / "channel_1.parquet", index=False)
    pd.DataFrame([{"channel": "channel_1", "start": 0, "end": 5}]).to_csv(
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

        result = runner.invoke(main, ["--env=local", "explore", "--mission=M1", "--channel=1"])

        assert result.exit_code == 0, result.output
        assert "1" in result.output

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

        result = runner.invoke(main, ["--env=local", "explore", "--mission=M1", "--channel=99"])

        assert result.exit_code != 0

    def test_verbose_flag_accepted(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sample_dir = tmp_path / "sample"
        _write_sample_mission(sample_dir, "M1")
        monkeypatch.setenv("SPACECRAFT_DATA__SAMPLE_DATA_DIR", str(sample_dir))

        result = runner.invoke(main, ["--env=local", "--verbose", "explore", "--mission=M1"])

        assert result.exit_code == 0, result.output


# ---------------------------------------------------------------------------
# ray group
# ---------------------------------------------------------------------------


class TestRayTuneCommand:
    def test_help_shows_options(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["ray", "tune", "--help"])
        assert result.exit_code == 0
        assert "--mission" in result.output
        assert "--subsystem" in result.output
        assert "--num-samples" in result.output

    def test_tune_all_calls_run_all_sweeps(
        self, runner: CliRunner
    ) -> None:
        settings = load_settings("test")
        mock_cm = MagicMock()
        mock_cm.__enter__.return_value = None
        mock_cm.__exit__.return_value = None

        with (
            patch("spacecraft_telemetry.cli.load_settings", return_value=settings),
            patch("spacecraft_telemetry.cli._ray_session", return_value=mock_cm),
            patch(
                "spacecraft_telemetry.ray_fanout.discover_channels",
                return_value=["channel_1", "channel_2"],
            ),
            patch(
                "spacecraft_telemetry.ray_fanout.run_all_sweeps",
                return_value=Path("models/ESA-Mission1/tuned_configs.json"),
            ) as mock_run_all,
        ):
            result = runner.invoke(
                main,
                ["--env=test", "ray", "tune", "--mission=ESA-Mission1", "--num-samples=5"],
            )

        assert result.exit_code == 0, result.output
        call_args = mock_run_all.call_args
        assert call_args is not None
        passed_settings = call_args.args[0]
        assert passed_settings.tune.num_samples == 5
        assert "Output" in result.output

    def test_tune_single_subsystem_calls_run_hpo_sweep(
        self, runner: CliRunner
    ) -> None:
        settings = load_settings("test")
        mock_cm = MagicMock()
        mock_cm.__enter__.return_value = None
        mock_cm.__exit__.return_value = None

        with (
            patch("spacecraft_telemetry.cli.load_settings", return_value=settings),
            patch("spacecraft_telemetry.cli._ray_session", return_value=mock_cm),
            patch(
                "spacecraft_telemetry.ray_fanout.discover_channels",
                return_value=["channel_1", "channel_2"],
            ),
            patch(
                "spacecraft_telemetry.ray_fanout.load_channel_subsystem_map",
                return_value={"channel_1": "subsystem_1", "channel_2": "subsystem_6"},
            ),
            patch(
                "spacecraft_telemetry.ray_fanout.run_hpo_sweep",
                return_value={
                    "config": {
                        "error_smoothing_window": 10,
                        "threshold_window": 100,
                        "threshold_z": 2.5,
                        "threshold_min_anomaly_len": 2,
                    },
                    "f0_5": 0.75,
                    "run_id": "fake-run-id",
                },
            ) as mock_run_one,
            patch("spacecraft_telemetry.ray_fanout.write_tuned_configs") as mock_write,
        ):
            result = runner.invoke(
                main,
                [
                    "--env=test",
                    "ray",
                    "tune",
                    "--mission=ESA-Mission1",
                    "--subsystem=subsystem_1",
                ],
            )

        assert result.exit_code == 0, result.output
        mock_run_one.assert_called_once()
        called_channels = mock_run_one.call_args.args[1]
        assert called_channels == ["channel_1"]
        mock_write.assert_called_once()
        assert "Subsystem" in result.output

    def test_tune_errors_when_no_channels_discovered(self, runner: CliRunner) -> None:
        settings = load_settings("test")
        mock_cm = MagicMock()
        mock_cm.__enter__.return_value = None
        mock_cm.__exit__.return_value = None

        with (
            patch("spacecraft_telemetry.cli.load_settings", return_value=settings),
            patch("spacecraft_telemetry.cli._ray_session", return_value=mock_cm),
            patch("spacecraft_telemetry.ray_fanout.discover_channels", return_value=[]),
        ):
            result = runner.invoke(
                main,
                ["--env=test", "ray", "tune", "--mission=ESA-Mission1"],
            )

        assert result.exit_code != 0
        assert "No preprocessed channels found" in result.output

    def test_tune_errors_when_no_channels_discovered_names_the_variant(
        self, runner: CliRunner
    ) -> None:
        """A variant whose processed tree doesn't exist (e.g. a typo) must name

        the variant and the expected path in the error, not just the mission --
        the base mission's channels are likely present and preprocess run would
        be the wrong remedy to suggest.
        """
        settings = load_settings("test").model_copy(update={"variant": "adb-42m"})
        mock_cm = MagicMock()
        mock_cm.__enter__.return_value = None
        mock_cm.__exit__.return_value = None

        with (
            patch("spacecraft_telemetry.cli.load_settings", return_value=settings),
            patch("spacecraft_telemetry.cli._ray_session", return_value=mock_cm),
            patch("spacecraft_telemetry.ray_fanout.discover_channels", return_value=[]),
        ):
            result = runner.invoke(
                main,
                ["--env=test", "ray", "tune", "--mission=ESA-Mission1"],
            )

        assert result.exit_code != 0
        assert "variant='adb-42m'" in result.output
        assert "ESA-Mission1/adb-42m/train" in result.output

    def test_tune_errors_when_subsystem_map_missing(self, runner: CliRunner) -> None:
        settings = load_settings("test")
        mock_cm = MagicMock()
        mock_cm.__enter__.return_value = None
        mock_cm.__exit__.return_value = None

        with (
            patch("spacecraft_telemetry.cli.load_settings", return_value=settings),
            patch("spacecraft_telemetry.cli._ray_session", return_value=mock_cm),
            patch(
                "spacecraft_telemetry.ray_fanout.discover_channels",
                return_value=["channel_1", "channel_2"],
            ),
            patch("spacecraft_telemetry.ray_fanout.load_channel_subsystem_map", return_value={}),
        ):
            result = runner.invoke(
                main,
                [
                    "--env=test",
                    "ray",
                    "tune",
                    "--mission=ESA-Mission1",
                    "--subsystem=subsystem_1",
                ],
            )

        assert result.exit_code != 0
        assert "cannot resolve --subsystem" in result.output

    def test_tune_errors_when_subsystem_has_no_channels(self, runner: CliRunner) -> None:
        settings = load_settings("test")
        mock_cm = MagicMock()
        mock_cm.__enter__.return_value = None
        mock_cm.__exit__.return_value = None

        with (
            patch("spacecraft_telemetry.cli.load_settings", return_value=settings),
            patch("spacecraft_telemetry.cli._ray_session", return_value=mock_cm),
            patch(
                "spacecraft_telemetry.ray_fanout.discover_channels",
                return_value=["channel_1", "channel_2"],
            ),
            patch(
                "spacecraft_telemetry.ray_fanout.load_channel_subsystem_map",
                return_value={"channel_1": "subsystem_6", "channel_2": "subsystem_6"},
            ),
        ):
            result = runner.invoke(
                main,
                [
                    "--env=test",
                    "ray",
                    "tune",
                    "--mission=ESA-Mission1",
                    "--subsystem=subsystem_1",
                ],
            )

        assert result.exit_code != 0
        assert "No channels found for subsystem" in result.output

    def test_tune_single_subsystem_invalid_existing_json_requires_overwrite(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        base = load_settings("test")
        settings = load_settings("test").model_copy(
            update={
                "model": base.model.model_copy(
                    update={"artifacts_dir": tmp_path / "models"}
                )
            }
        )
        output = Path(settings.model.artifacts_dir) / "ESA-Mission1" / "tuned_configs.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("{not-json")

        mock_cm = MagicMock()
        mock_cm.__enter__.return_value = None
        mock_cm.__exit__.return_value = None

        with (
            patch("spacecraft_telemetry.cli.load_settings", return_value=settings),
            patch("spacecraft_telemetry.cli._ray_session", return_value=mock_cm),
            patch(
                "spacecraft_telemetry.ray_fanout.discover_channels",
                return_value=["channel_1"],
            ),
            patch(
                "spacecraft_telemetry.ray_fanout.load_channel_subsystem_map",
                return_value={"channel_1": "subsystem_1"},
            ),
            patch(
                "spacecraft_telemetry.ray_fanout.run_hpo_sweep",
                return_value={
                    "config": {
                        "error_smoothing_window": 10,
                        "threshold_window": 100,
                        "threshold_z": 2.5,
                        "threshold_min_anomaly_len": 2,
                    },
                    "f0_5": 0.75,
                    "run_id": "fake-run-id",
                },
            ),
        ):
            result = runner.invoke(
                main,
                [
                    "--env=test",
                    "ray",
                    "tune",
                    "--mission=ESA-Mission1",
                    "--subsystem=subsystem_1",
                ],
            )

        assert result.exit_code != 0
        assert "invalid JSON" in result.output

    def test_tune_single_subsystem_invalid_existing_json_overwrite(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        base = load_settings("test")
        settings = load_settings("test").model_copy(
            update={
                "model": base.model.model_copy(
                    update={"artifacts_dir": tmp_path / "models"}
                )
            }
        )
        output = Path(settings.model.artifacts_dir) / "ESA-Mission1" / "tuned_configs.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("{not-json")

        mock_cm = MagicMock()
        mock_cm.__enter__.return_value = None
        mock_cm.__exit__.return_value = None

        with (
            patch("spacecraft_telemetry.cli.load_settings", return_value=settings),
            patch("spacecraft_telemetry.cli._ray_session", return_value=mock_cm),
            patch(
                "spacecraft_telemetry.ray_fanout.discover_channels",
                return_value=["channel_1"],
            ),
            patch(
                "spacecraft_telemetry.ray_fanout.load_channel_subsystem_map",
                return_value={"channel_1": "subsystem_1"},
            ),
            patch(
                "spacecraft_telemetry.ray_fanout.run_hpo_sweep",
                return_value={
                    "config": {
                        "error_smoothing_window": 10,
                        "threshold_window": 100,
                        "threshold_z": 2.5,
                        "threshold_min_anomaly_len": 2,
                    },
                    "f0_5": 0.75,
                    "run_id": "fake-run-id",
                },
            ),
        ):
            result = runner.invoke(
                main,
                [
                    "--env=test",
                    "ray",
                    "tune",
                    "--mission=ESA-Mission1",
                    "--subsystem=subsystem_1",
                    "--overwrite-existing",
                ],
            )

        assert result.exit_code == 0, result.output


# ---------------------------------------------------------------------------
# ray train command
# ---------------------------------------------------------------------------


class TestRayTrainCommand:
    def _mock_cm(self) -> MagicMock:
        cm = MagicMock()
        cm.__enter__.return_value = None
        cm.__exit__.return_value = None
        return cm

    def test_subsystem_filters_channels(self, runner: CliRunner) -> None:
        settings = load_settings("test")
        with (
            patch("spacecraft_telemetry.cli.load_settings", return_value=settings),
            patch("spacecraft_telemetry.cli._ray_session", return_value=self._mock_cm()),
            patch(
                "spacecraft_telemetry.ray_fanout.discover_channels",
                return_value=["ch_a", "ch_b", "ch_c"],
            ),
            patch(
                "spacecraft_telemetry.ray_fanout.load_channel_subsystem_map",
                return_value={"ch_a": "subsystem_1", "ch_b": "subsystem_6", "ch_c": "subsystem_1"},
            ),
            patch(
                "spacecraft_telemetry.ray_fanout.train_all_channels",
                return_value=[
                    {"status": "ok", "channel": "ch_a", "best_epoch": 5, "best_val_loss": 0.01},
                    {"status": "ok", "channel": "ch_c", "best_epoch": 5, "best_val_loss": 0.01},
                ],
            ) as mock_train,
        ):
            result = runner.invoke(
                main,
                [
                    "--env=test", "ray", "train",
                    "--mission=ESA-Mission1", "--subsystem=subsystem_1",
                ],
            )

        assert result.exit_code == 0, result.output
        called_channels = mock_train.call_args.args[2]
        assert set(called_channels) == {"ch_a", "ch_c"}

    def test_subsystem_nonexistent_raises_error(self, runner: CliRunner) -> None:
        settings = load_settings("test")
        with (
            patch("spacecraft_telemetry.cli.load_settings", return_value=settings),
            patch("spacecraft_telemetry.cli._ray_session", return_value=self._mock_cm()),
            patch(
                "spacecraft_telemetry.ray_fanout.discover_channels",
                return_value=["ch_a", "ch_b"],
            ),
            patch(
                "spacecraft_telemetry.ray_fanout.load_channel_subsystem_map",
                return_value={"ch_a": "subsystem_1", "ch_b": "subsystem_1"},
            ),
        ):
            result = runner.invoke(
                main,
                [
                    "--env=test", "ray", "train",
                    "--mission=ESA-Mission1", "--subsystem=nonexistent",
                ],
            )

        assert result.exit_code != 0
        assert "No channels found for subsystem" in result.output

    def test_subsystem_map_empty_raises_error(self, runner: CliRunner) -> None:
        settings = load_settings("test")
        with (
            patch("spacecraft_telemetry.cli.load_settings", return_value=settings),
            patch("spacecraft_telemetry.cli._ray_session", return_value=self._mock_cm()),
            patch(
                "spacecraft_telemetry.ray_fanout.discover_channels",
                return_value=["ch_a", "ch_b"],
            ),
            patch(
                "spacecraft_telemetry.ray_fanout.load_channel_subsystem_map",
                return_value={},
            ),
        ):
            result = runner.invoke(
                main,
                [
                    "--env=test", "ray", "train",
                    "--mission=ESA-Mission1", "--subsystem=subsystem_1",
                ],
            )

        assert result.exit_code != 0
        assert "cannot resolve --subsystem" in result.output

    def test_explicit_channels_ignores_subsystem(self, runner: CliRunner) -> None:
        settings = load_settings("test")
        with (
            patch("spacecraft_telemetry.cli.load_settings", return_value=settings),
            patch("spacecraft_telemetry.cli._ray_session", return_value=self._mock_cm()),
            patch(
                "spacecraft_telemetry.ray_fanout.train_all_channels",
                return_value=[
                    {"status": "ok", "channel": "ch_a", "best_epoch": 5, "best_val_loss": 0.01},
                    {"status": "ok", "channel": "ch_b", "best_epoch": 5, "best_val_loss": 0.01},
                ],
            ) as mock_train,
        ):
            result = runner.invoke(
                main,
                [
                    "--env=test",
                    "ray",
                    "train",
                    "--mission=ESA-Mission1",
                    "--channels=ch_a,ch_b",
                    "--subsystem=subsystem_1",
                ],
            )

        assert result.exit_code == 0, result.output
        called_channels = mock_train.call_args.args[2]
        # subsystem is ignored when --channels is given; both channels passed through
        assert set(called_channels) == {"ch_a", "ch_b"}

    def test_help_shows_subsystem_option(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["ray", "train", "--help"])
        assert result.exit_code == 0
        assert "--subsystem" in result.output


# ---------------------------------------------------------------------------
# ray score command
# ---------------------------------------------------------------------------


class TestRayScoreCommand:
    def _mock_cm(self) -> MagicMock:
        cm = MagicMock()
        cm.__enter__.return_value = None
        cm.__exit__.return_value = None
        return cm

    def test_subsystem_filters_channels(self, runner: CliRunner) -> None:
        settings = load_settings("test")
        with (
            patch("spacecraft_telemetry.cli.load_settings", return_value=settings),
            patch("spacecraft_telemetry.cli._ray_session", return_value=self._mock_cm()),
            patch(
                "spacecraft_telemetry.ray_fanout.discover_channels",
                return_value=["ch_a", "ch_b", "ch_c"],
            ),
            patch(
                "spacecraft_telemetry.ray_fanout.load_channel_subsystem_map",
                return_value={"ch_a": "subsystem_1", "ch_b": "subsystem_6", "ch_c": "subsystem_1"},
            ),
            patch(
                "spacecraft_telemetry.ray_fanout.score_all_channels",
                return_value=[
                    {
                        "status": "ok",
                        "channel": "ch_a",
                        "precision": 0.9,
                        "recall": 0.8,
                        "f1": 0.85,
                        "f0_5": 0.87,
                        "seg_f0_5": 0.80,
                        "n_true_seqs": 4,
                        "n_pred_seqs": 5,
                        "pruned_seg_f0_5": 0.83,
                        "pruned_n_pred_seqs": 4,
                    },
                    {
                        "status": "ok",
                        "channel": "ch_c",
                        "precision": 0.9,
                        "recall": 0.8,
                        "f1": 0.85,
                        "f0_5": 0.87,
                        "seg_f0_5": 0.80,
                        "n_true_seqs": 4,
                        "n_pred_seqs": 5,
                        "pruned_seg_f0_5": 0.83,
                        "pruned_n_pred_seqs": 4,
                    },
                ],
            ) as mock_score,
        ):
            result = runner.invoke(
                main,
                [
                    "--env=test", "ray", "score",
                    "--mission=ESA-Mission1", "--subsystem=subsystem_1",
                ],
            )

        assert result.exit_code == 0, result.output
        called_channels = mock_score.call_args.args[2]
        assert set(called_channels) == {"ch_a", "ch_c"}

    def test_subsystem_nonexistent_raises_error(self, runner: CliRunner) -> None:
        settings = load_settings("test")
        with (
            patch("spacecraft_telemetry.cli.load_settings", return_value=settings),
            patch("spacecraft_telemetry.cli._ray_session", return_value=self._mock_cm()),
            patch(
                "spacecraft_telemetry.ray_fanout.discover_channels",
                return_value=["ch_a", "ch_b"],
            ),
            patch(
                "spacecraft_telemetry.ray_fanout.load_channel_subsystem_map",
                return_value={"ch_a": "subsystem_1", "ch_b": "subsystem_1"},
            ),
        ):
            result = runner.invoke(
                main,
                [
                    "--env=test", "ray", "score",
                    "--mission=ESA-Mission1", "--subsystem=nonexistent",
                ],
            )

        assert result.exit_code != 0
        assert "No channels found for subsystem" in result.output

    def test_subsystem_map_empty_raises_error(self, runner: CliRunner) -> None:
        settings = load_settings("test")
        with (
            patch("spacecraft_telemetry.cli.load_settings", return_value=settings),
            patch("spacecraft_telemetry.cli._ray_session", return_value=self._mock_cm()),
            patch(
                "spacecraft_telemetry.ray_fanout.discover_channels",
                return_value=["ch_a", "ch_b"],
            ),
            patch(
                "spacecraft_telemetry.ray_fanout.load_channel_subsystem_map",
                return_value={},
            ),
        ):
            result = runner.invoke(
                main,
                [
                    "--env=test", "ray", "score",
                    "--mission=ESA-Mission1", "--subsystem=subsystem_1",
                ],
            )

        assert result.exit_code != 0
        assert "cannot resolve --subsystem" in result.output

    def test_explicit_channels_ignores_subsystem(self, runner: CliRunner) -> None:
        settings = load_settings("test")
        with (
            patch("spacecraft_telemetry.cli.load_settings", return_value=settings),
            patch("spacecraft_telemetry.cli._ray_session", return_value=self._mock_cm()),
            patch(
                "spacecraft_telemetry.ray_fanout.score_all_channels",
                return_value=[
                    {
                        "status": "ok",
                        "channel": "ch_a",
                        "precision": 0.9,
                        "recall": 0.8,
                        "f1": 0.85,
                        "f0_5": 0.87,
                        "seg_f0_5": 0.80,
                        "n_true_seqs": 4,
                        "n_pred_seqs": 5,
                        "pruned_seg_f0_5": 0.83,
                        "pruned_n_pred_seqs": 4,
                    },
                    {
                        "status": "ok",
                        "channel": "ch_b",
                        "precision": 0.9,
                        "recall": 0.8,
                        "f1": 0.85,
                        "f0_5": 0.87,
                        "seg_f0_5": 0.80,
                        "n_true_seqs": 4,
                        "n_pred_seqs": 5,
                        "pruned_seg_f0_5": 0.83,
                        "pruned_n_pred_seqs": 4,
                    },
                ],
            ) as mock_score,
        ):
            result = runner.invoke(
                main,
                [
                    "--env=test",
                    "ray",
                    "score",
                    "--mission=ESA-Mission1",
                    "--channels=ch_a,ch_b",
                    "--subsystem=subsystem_1",
                ],
            )

        assert result.exit_code == 0, result.output
        called_channels = mock_score.call_args.args[2]
        # subsystem is ignored when --channels is given; both channels passed through
        assert set(called_channels) == {"ch_a", "ch_b"}

    def test_help_shows_subsystem_option(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["ray", "score", "--help"])
        assert result.exit_code == 0
        assert "--subsystem" in result.output


# ---------------------------------------------------------------------------
# _discover_registered_channels (unit-level, no CLI invocation)
# ---------------------------------------------------------------------------


class TestDiscoverRegisteredChannels:
    """docs/reviews/020-experiment-variant-axis.md item C4: the previous
    fallback (`v.name.removeprefix(prefix)`) is dead in the common case
    (matching is already filtered to mission_id == mission, and
    model/training.py sets mission_id and channel_id together) and WRONG in
    the one case it could fire under a variant, where removeprefix leaves
    the variant slug glued to the front of the channel id. Replaced with a
    log-and-skip; these tests pin that a version missing the channel_id tag
    is dropped, not fabricated into a bad channel id.
    """

    @staticmethod
    def _make_version(name: str, *, mission_id: str, variant: str | None = None) -> Any:
        mv = MagicMock()
        mv.name = name
        mv.tags = {"mission_id": mission_id}
        if variant:
            mv.tags["variant"] = variant
        return mv

    def test_missing_channel_id_tag_is_skipped_not_fabricated(self) -> None:
        from spacecraft_telemetry.cli import _discover_registered_channels

        mission, variant = "ESA-Mission1", "adb-24m"
        # No channel_id tag. Under the old fallback,
        # "telemanom-ESA-Mission1-adb-24m-channel_41".removeprefix(
        # "telemanom-ESA-Mission1-") would wrongly yield "adb-24m-channel_41"
        # (the variant slug glued to the channel id) instead of being skipped.
        untagged = self._make_version(
            f"telemanom-{mission}-{variant}-channel_41", mission_id=mission, variant=variant
        )
        tagged = self._make_version(
            f"telemanom-{mission}-{variant}-channel_1", mission_id=mission, variant=variant
        )
        tagged.tags["channel_id"] = "channel_1"

        with patch("mlflow.tracking.client.MlflowClient") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.search_model_versions.return_value = [untagged, tagged]

            result = _discover_registered_channels(mission, variant)

        assert result == ["channel_1"]
        assert "adb-24m-channel_41" not in result


class TestMlflowCli:
    def test_mlflow_promote_help(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["--env=test", "mlflow", "promote", "--help"])
        assert result.exit_code == 0
        assert "--name" in result.output
        assert "--stage" not in result.output

    def test_mlflow_ui_help(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["--env=test", "mlflow", "ui", "--help"])
        assert result.exit_code == 0
        assert "--port" in result.output

    def test_mlflow_promote_resolves_latest_version(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """promote without --version resolves the latest version and sets @champion."""
        from unittest.mock import MagicMock

        settings = load_settings("test").model_copy(
            update={
                "mlflow": load_settings("test").mlflow.model_copy(
                    update={"tracking_uri": f"sqlite:///{tmp_path}/mlflow.db"}
                )
            }
        )

        mock_version = MagicMock()
        mock_version.version = "3"

        with (
            patch("spacecraft_telemetry.cli.load_settings", return_value=settings),
            patch("spacecraft_telemetry.mlflow_tracking.registry.MlflowClient") as mock_client_cls,
        ):
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.search_model_versions.return_value = [mock_version]
            mock_client.set_registered_model_alias.return_value = None

            result = runner.invoke(
                main,
                [
                    "--env=test",
                    "mlflow",
                    "promote",
                    "--name=telemanom-ESA-Mission1-channel_1",
                ],
            )

        assert result.exit_code == 0, result.output
        mock_client.set_registered_model_alias.assert_called_once_with(
            "telemanom-ESA-Mission1-channel_1", "champion", "3"
        )

    def test_mlflow_promote_no_versions_errors(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """promote should fail with a clear message when no versions exist."""
        from unittest.mock import MagicMock

        settings = load_settings("test").model_copy(
            update={
                "mlflow": load_settings("test").mlflow.model_copy(
                    update={"tracking_uri": f"sqlite:///{tmp_path}/mlflow.db"}
                )
            }
        )

        with (
            patch("spacecraft_telemetry.cli.load_settings", return_value=settings),
            patch("spacecraft_telemetry.mlflow_tracking.registry.MlflowClient") as mock_client_cls,
        ):
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.search_model_versions.return_value = []

            result = runner.invoke(
                main,
                [
                    "--env=test",
                    "mlflow",
                    "promote",
                    "--name=telemanom-ESA-Mission1-channel_1",
                ],
            )

        assert result.exit_code != 0
        assert "No versions found" in result.output

    def test_mlflow_promote_all_discovers_from_registry(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """--mission alone discovers all registered models from the registry."""
        from unittest.mock import MagicMock

        settings = load_settings("test").model_copy(
            update={
                "mlflow": load_settings("test").mlflow.model_copy(
                    update={"tracking_uri": f"sqlite:///{tmp_path}/mlflow.db"}
                )
            }
        )

        def make_version(
            name: str, ver: str, *, mission_id: str = "", channel_id: str = "",
            variant: str | None = None,
        ) -> MagicMock:
            mv = MagicMock()
            mv.name = name
            mv.version = ver
            mv.aliases = []
            mv.tags = {"mission_id": mission_id, "channel_id": channel_id}
            if variant:
                mv.tags["variant"] = variant
            return mv

        channel_ids = ["channel_1", "channel_10", "channel_11"]
        mission = "ESA-Mission1"
        prefix = f"telemanom-{mission}-"
        discovery_versions = [
            make_version(f"{prefix}{ch}", "1", mission_id=mission, channel_id=ch)
            for ch in channel_ids
        ]

        with (
            patch("spacecraft_telemetry.cli.load_settings", return_value=settings),
            patch("mlflow.tracking.client.MlflowClient") as mock_discovery_client_cls,
            patch(
                "spacecraft_telemetry.mlflow_tracking.registry.MlflowClient"
            ) as mock_registry_client_cls,
        ):
            mock_discovery_client = MagicMock()
            mock_registry_client = MagicMock()
            mock_discovery_client_cls.return_value = mock_discovery_client
            mock_registry_client_cls.return_value = mock_registry_client

            # First call: registry discovery. Subsequent calls: per-model version lookup.
            per_model_version = make_version("", "1")
            mock_discovery_client.search_model_versions.return_value = discovery_versions
            mock_registry_client.search_model_versions.return_value = [per_model_version]
            mock_registry_client.set_registered_model_alias.return_value = None

            result = runner.invoke(
                main,
                ["--env=test", "mlflow", "promote", "--mission", mission],
            )

        assert result.exit_code == 0, result.output
        assert f"Promoted      : {len(channel_ids)}/{len(channel_ids)}" in result.output
        assert mock_registry_client.set_registered_model_alias.call_count == len(channel_ids)
        # The mock can't apply the filter string (no real backend), but the
        # string itself is the contract with a real MLflow registry — pin it
        # so a malformed query doesn't pass this suite and fail on first use
        # against a real server.
        mock_discovery_client.search_model_versions.assert_called_once_with(
            f"name LIKE '{prefix}%' and tags.mission_id = '{mission}'"
        )

    def test_mlflow_promote_all_excludes_pseudo_mission_arms(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """--mission ESA-Mission1 (no --variant) must not sweep in ADB pseudo-mission models.

        Regression test for the plan 020 registry hazard: `telemanom-ESA-Mission1-`
        is a string prefix of `telemanom-ESA-Mission1-ADB-24m-channel_41`, so a
        name-prefix-only discovery query incorrectly matches both. This must fail
        against the pre-020 cli.py, which filtered on name prefix alone.
        """
        from unittest.mock import MagicMock

        settings = load_settings("test").model_copy(
            update={
                "mlflow": load_settings("test").mlflow.model_copy(
                    update={"tracking_uri": f"sqlite:///{tmp_path}/mlflow.db"}
                )
            }
        )

        def make_version(name: str, ver: str, *, mission_id: str, channel_id: str) -> MagicMock:
            mv = MagicMock()
            mv.name = name
            mv.version = ver
            mv.aliases = []
            mv.tags = {"mission_id": mission_id, "channel_id": channel_id}
            return mv

        mission = "ESA-Mission1"
        real_versions = [
            make_version(
                f"telemanom-{mission}-channel_1", "1", mission_id=mission, channel_id="channel_1"
            ),
        ]
        # A legacy pseudo-mission arm: registered under mission="ESA-Mission1-ADB-24m",
        # so its name shares the "telemanom-ESA-Mission1-" prefix but its mission_id
        # tag reflects the pseudo-mission, not the real one.
        pseudo_mission_versions = [
            make_version(
                "telemanom-ESA-Mission1-ADB-24m-channel_41", "1",
                mission_id="ESA-Mission1-ADB-24m", channel_id="channel_41",
            ),
        ]
        all_versions = real_versions + pseudo_mission_versions

        with (
            patch("spacecraft_telemetry.cli.load_settings", return_value=settings),
            patch("mlflow.tracking.client.MlflowClient") as mock_discovery_client_cls,
            patch(
                "spacecraft_telemetry.mlflow_tracking.registry.MlflowClient"
            ) as mock_registry_client_cls,
        ):
            mock_discovery_client = MagicMock()
            mock_registry_client = MagicMock()
            mock_discovery_client_cls.return_value = mock_discovery_client
            mock_registry_client_cls.return_value = mock_registry_client

            # search_model_versions is mocked at the client level, so it can't
            # actually apply the `tags.mission_id = ...` filter string cli.py
            # sends — return everything and let _discover_registered_channels'
            # Python-side tag filtering do the real work being tested here.
            mock_discovery_client.search_model_versions.return_value = all_versions
            mock_registry_client.search_model_versions.return_value = [make_version(
                "", "1", mission_id="", channel_id="",
            )]
            mock_registry_client.set_registered_model_alias.return_value = None

            result = runner.invoke(
                main,
                ["--env=test", "mlflow", "promote", "--mission", mission],
            )

        assert result.exit_code == 0, result.output
        assert "Promoted      : 1/1" in result.output
        promoted_names = [
            call.args[0] for call in mock_registry_client.set_registered_model_alias.call_args_list
        ]
        assert "telemanom-ESA-Mission1-channel_1" in promoted_names
        # If the mission_id tag filter were missing, channel_41 (from the
        # pseudo-mission arm) would leak in and be promoted under this name.
        assert "telemanom-ESA-Mission1-channel_41" not in promoted_names

    def test_mlflow_promote_all_excludes_same_mission_variants(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """--mission ESA-Mission1 (no --variant) must not sweep in a REAL
        variant's models — the one discrimination case the mission_id tag
        filter alone cannot solve, since a variant model shares mission_id
        with the base model by construction (docs/reviews/020-experiment-
        variant-axis.md item T1). This is exactly the case that becomes
        load-bearing once plan 020.5 migrates the ADB arms to
        mission=ESA-Mission1 with a variant slug instead of a pseudo-mission:
        deleting the `variant` tag comparison from _discover_registered_channels
        would leave this test failing while
        test_mlflow_promote_all_excludes_pseudo_mission_arms stays green.
        """
        from unittest.mock import MagicMock

        settings = load_settings("test").model_copy(
            update={
                "mlflow": load_settings("test").mlflow.model_copy(
                    update={"tracking_uri": f"sqlite:///{tmp_path}/mlflow.db"}
                )
            }
        )

        def make_version(
            name: str, ver: str, *, mission_id: str, channel_id: str,
            variant: str | None = None,
        ) -> MagicMock:
            mv = MagicMock()
            mv.name = name
            mv.version = ver
            mv.aliases = []
            mv.tags = {"mission_id": mission_id, "channel_id": channel_id}
            if variant:
                mv.tags["variant"] = variant
            return mv

        mission = "ESA-Mission1"
        base_version = make_version(
            f"telemanom-{mission}-channel_1", "1", mission_id=mission, channel_id="channel_1"
        )
        # Same REAL mission_id as base_version — only the variant tag
        # distinguishes it. A filter that checked mission_id alone would
        # wrongly sweep this in under a base (--variant unset) query.
        variant_version = make_version(
            f"telemanom-{mission}-adb-24m-channel_41", "1",
            mission_id=mission, channel_id="channel_41", variant="adb-24m",
        )

        with (
            patch("spacecraft_telemetry.cli.load_settings", return_value=settings),
            patch("mlflow.tracking.client.MlflowClient") as mock_discovery_client_cls,
            patch(
                "spacecraft_telemetry.mlflow_tracking.registry.MlflowClient"
            ) as mock_registry_client_cls,
        ):
            mock_discovery_client = MagicMock()
            mock_registry_client = MagicMock()
            mock_discovery_client_cls.return_value = mock_discovery_client
            mock_registry_client_cls.return_value = mock_registry_client

            mock_discovery_client.search_model_versions.return_value = [
                base_version, variant_version,
            ]
            mock_registry_client.search_model_versions.return_value = [make_version(
                "", "1", mission_id="", channel_id="",
            )]
            mock_registry_client.set_registered_model_alias.return_value = None

            result = runner.invoke(
                main,
                ["--env=test", "mlflow", "promote", "--mission", mission],
            )

        assert result.exit_code == 0, result.output
        assert "Promoted      : 1/1" in result.output
        promoted_names = [
            call.args[0] for call in mock_registry_client.set_registered_model_alias.call_args_list
        ]
        assert promoted_names == ["telemanom-ESA-Mission1-channel_1"]
        # If the variant tag comparison were missing, channel_41 (a REAL
        # variant of THIS mission) would leak into the base promotion set.
        assert "telemanom-ESA-Mission1-adb-24m-channel_41" not in promoted_names

    def test_mlflow_promote_variant_discovers_only_that_variant(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """--mission ESA-Mission1 --variant adb-24m discovers only that variant's models."""
        from unittest.mock import MagicMock

        settings = load_settings("test").model_copy(
            update={
                "mlflow": load_settings("test").mlflow.model_copy(
                    update={"tracking_uri": f"sqlite:///{tmp_path}/mlflow.db"}
                )
            }
        )

        def make_version(
            name: str, ver: str, *, mission_id: str, channel_id: str,
            variant: str | None = None,
        ) -> MagicMock:
            mv = MagicMock()
            mv.name = name
            mv.version = ver
            mv.aliases = []
            mv.tags = {"mission_id": mission_id, "channel_id": channel_id}
            if variant:
                mv.tags["variant"] = variant
            return mv

        mission = "ESA-Mission1"
        base_version = make_version(
            f"telemanom-{mission}-channel_1", "1", mission_id=mission, channel_id="channel_1"
        )
        variant_version = make_version(
            f"telemanom-{mission}-adb-24m-channel_41", "1",
            mission_id=mission, channel_id="channel_41", variant="adb-24m",
        )

        with (
            patch("spacecraft_telemetry.cli.load_settings", return_value=settings),
            patch("mlflow.tracking.client.MlflowClient") as mock_discovery_client_cls,
            patch(
                "spacecraft_telemetry.mlflow_tracking.registry.MlflowClient"
            ) as mock_registry_client_cls,
        ):
            mock_discovery_client = MagicMock()
            mock_registry_client = MagicMock()
            mock_discovery_client_cls.return_value = mock_discovery_client
            mock_registry_client_cls.return_value = mock_registry_client

            mock_discovery_client.search_model_versions.return_value = [
                base_version, variant_version,
            ]
            mock_registry_client.search_model_versions.return_value = [make_version(
                "", "1", mission_id="", channel_id="",
            )]
            mock_registry_client.set_registered_model_alias.return_value = None

            result = runner.invoke(
                main,
                ["--env=test", "mlflow", "promote", "--mission", mission, "--variant", "adb-24m"],
            )

        assert result.exit_code == 0, result.output
        assert "Promoted      : 1/1" in result.output
        promoted_names = [
            call.args[0] for call in mock_registry_client.set_registered_model_alias.call_args_list
        ]
        assert promoted_names == ["telemanom-ESA-Mission1-adb-24m-channel_41"]

    def test_mlflow_promote_variant_requires_mission(self, runner: CliRunner) -> None:
        result = runner.invoke(
            main, ["--env=test", "mlflow", "promote", "--variant", "adb-24m"]
        )
        assert result.exit_code != 0
        assert "--variant requires --mission" in result.output

    def test_mlflow_promote_resolves_variant_from_settings_when_flag_absent(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """A --mission-only invocation must scope to settings.variant (e.g.
        from an exported SPACECRAFT_VARIANT) rather than silently falling
        through to the base production models — the hazard docs/reviews/
        020-experiment-variant-axis.md item A4 exists to close: a train ->
        score -> tune session run under SPACECRAFT_VARIANT would otherwise
        promote base models at the promote step.
        """
        from unittest.mock import MagicMock

        settings = load_settings("test").model_copy(
            update={
                "variant": "adb-24m",
                "mlflow": load_settings("test").mlflow.model_copy(
                    update={"tracking_uri": f"sqlite:///{tmp_path}/mlflow.db"}
                ),
            }
        )

        def make_version(
            name: str, ver: str, *, mission_id: str, channel_id: str,
            variant: str | None = None,
        ) -> MagicMock:
            mv = MagicMock()
            mv.name = name
            mv.version = ver
            mv.aliases = []
            mv.tags = {"mission_id": mission_id, "channel_id": channel_id}
            if variant:
                mv.tags["variant"] = variant
            return mv

        mission = "ESA-Mission1"
        base_version = make_version(
            f"telemanom-{mission}-channel_1", "1", mission_id=mission, channel_id="channel_1"
        )
        variant_version = make_version(
            f"telemanom-{mission}-adb-24m-channel_41", "1",
            mission_id=mission, channel_id="channel_41", variant="adb-24m",
        )

        with (
            patch("spacecraft_telemetry.cli.load_settings", return_value=settings),
            patch("mlflow.tracking.client.MlflowClient") as mock_discovery_client_cls,
            patch(
                "spacecraft_telemetry.mlflow_tracking.registry.MlflowClient"
            ) as mock_registry_client_cls,
        ):
            mock_discovery_client = MagicMock()
            mock_registry_client = MagicMock()
            mock_discovery_client_cls.return_value = mock_discovery_client
            mock_registry_client_cls.return_value = mock_registry_client

            mock_discovery_client.search_model_versions.return_value = [
                base_version, variant_version,
            ]
            mock_registry_client.search_model_versions.return_value = [make_version(
                "", "1", mission_id="", channel_id="",
            )]
            mock_registry_client.set_registered_model_alias.return_value = None

            # No --variant flag — must resolve from settings.variant.
            result = runner.invoke(
                main, ["--env=test", "mlflow", "promote", "--mission", mission],
            )

        assert result.exit_code == 0, result.output
        assert "Variant       : adb-24m" in result.output
        promoted_names = [
            call.args[0] for call in mock_registry_client.set_registered_model_alias.call_args_list
        ]
        assert promoted_names == ["telemanom-ESA-Mission1-adb-24m-channel_41"]

    def test_mlflow_promote_explicit_variant_overrides_settings(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """An explicit --variant flag wins over settings.variant.

        Uses two distinct non-empty variants (rather than --variant "" to
        mean "override back to base") because "" and settings.variant=None
        are not the same thing to _discover_registered_channels: an
        untagged model's variant tag resolves to None (see _version_tag's
        falsy-to-None normalization), which "" would not equal. That
        asymmetry is a separate, narrower question from what this test — the
        explicit-flag-wins precedence — is checking.
        """
        from unittest.mock import MagicMock

        settings = load_settings("test").model_copy(
            update={
                "variant": "adb-24m",
                "mlflow": load_settings("test").mlflow.model_copy(
                    update={"tracking_uri": f"sqlite:///{tmp_path}/mlflow.db"}
                ),
            }
        )

        def make_version(
            name: str, ver: str, *, mission_id: str, channel_id: str,
            variant: str | None = None,
        ) -> MagicMock:
            mv = MagicMock()
            mv.name = name
            mv.version = ver
            mv.aliases = []
            mv.tags = {"mission_id": mission_id, "channel_id": channel_id}
            if variant:
                mv.tags["variant"] = variant
            return mv

        mission = "ESA-Mission1"
        env_variant_version = make_version(
            f"telemanom-{mission}-adb-24m-channel_41", "1",
            mission_id=mission, channel_id="channel_41", variant="adb-24m",
        )
        flag_variant_version = make_version(
            f"telemanom-{mission}-other-arm-channel_42", "1",
            mission_id=mission, channel_id="channel_42", variant="other-arm",
        )

        with (
            patch("spacecraft_telemetry.cli.load_settings", return_value=settings),
            patch("mlflow.tracking.client.MlflowClient") as mock_discovery_client_cls,
            patch(
                "spacecraft_telemetry.mlflow_tracking.registry.MlflowClient"
            ) as mock_registry_client_cls,
        ):
            mock_discovery_client = MagicMock()
            mock_registry_client = MagicMock()
            mock_discovery_client_cls.return_value = mock_discovery_client
            mock_registry_client_cls.return_value = mock_registry_client

            mock_discovery_client.search_model_versions.return_value = [
                env_variant_version, flag_variant_version,
            ]
            mock_registry_client.search_model_versions.return_value = [make_version(
                "", "1", mission_id="", channel_id="",
            )]
            mock_registry_client.set_registered_model_alias.return_value = None

            # settings.variant is "adb-24m"; --variant explicitly says
            # "other-arm" and must win.
            result = runner.invoke(
                main,
                ["--env=test", "mlflow", "promote", "--mission", mission,
                 "--variant", "other-arm"],
            )

        assert result.exit_code == 0, result.output
        assert "Variant       : other-arm" in result.output
        promoted_names = [
            call.args[0] for call in mock_registry_client.set_registered_model_alias.call_args_list
        ]
        assert promoted_names == ["telemanom-ESA-Mission1-other-arm-channel_42"]

    def test_mlflow_promote_all_no_registered_models_errors(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """--mission with no registered models raises a clear error."""
        from unittest.mock import MagicMock

        settings = load_settings("test").model_copy(
            update={
                "mlflow": load_settings("test").mlflow.model_copy(
                    update={"tracking_uri": f"sqlite:///{tmp_path}/mlflow.db"}
                )
            }
        )

        with (
            patch("spacecraft_telemetry.cli.load_settings", return_value=settings),
            patch("mlflow.tracking.client.MlflowClient") as mock_client_cls,
        ):
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.search_model_versions.return_value = []

            result = runner.invoke(
                main,
                ["--env=test", "mlflow", "promote", "--mission", "ESA-Mission1"],
            )

        assert result.exit_code != 0
        assert "No registered models found" in result.output
        # No untagged legacy versions exist in this mock, so no hint is appended.
        assert "predate the mission_id tag" not in result.output

    def test_mlflow_promote_all_untagged_legacy_models_hint(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """A name-prefix match with no mission_id tag (pre-2026-06-03) gets a diagnostic hint.

        _discover_registered_channels's tag-filtered query finds nothing, but a
        model matching the name prefix does exist — just registered before the
        mission_id tag was added. The blanket "no registered models found"
        message would misreport this as "nothing is trained"; the hint must
        name the real cause and remedy instead.
        """
        from unittest.mock import MagicMock

        settings = load_settings("test").model_copy(
            update={
                "mlflow": load_settings("test").mlflow.model_copy(
                    update={"tracking_uri": f"sqlite:///{tmp_path}/mlflow.db"}
                )
            }
        )

        legacy_version = MagicMock()
        legacy_version.name = "telemanom-ESA-Mission1-channel_22"
        legacy_version.version = "1"
        legacy_version.tags = {"window_size": "250"}  # no mission_id tag at all

        def _search(filter_string: str) -> list[MagicMock]:
            # The tag-filtered discovery query (the primary lookup) finds
            # nothing tagged; only the unfiltered name-prefix query run by
            # _untagged_registry_hint sees the legacy version — a real server
            # applying `tags.mission_id = ...` would behave identically.
            if "tags.mission_id" in filter_string:
                return []
            return [legacy_version]

        with (
            patch("spacecraft_telemetry.cli.load_settings", return_value=settings),
            patch("mlflow.tracking.client.MlflowClient") as mock_client_cls,
        ):
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.search_model_versions.side_effect = _search

            result = runner.invoke(
                main,
                ["--env=test", "mlflow", "promote", "--mission", "ESA-Mission1"],
            )

        assert result.exit_code != 0
        assert "1 version(s) across 1 model(s)" in result.output
        assert "predate the mission_id tag" in result.output
        assert "registered before 2026-06-03" in result.output

    def test_mlflow_demote_all_excludes_pseudo_mission_arms(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """--mission ESA-Mission1 (no --variant) must not demote ADB pseudo-mission models."""
        from unittest.mock import MagicMock

        settings = load_settings("test").model_copy(
            update={
                "mlflow": load_settings("test").mlflow.model_copy(
                    update={"tracking_uri": f"sqlite:///{tmp_path}/mlflow.db"}
                )
            }
        )

        def make_version(name: str, ver: str, *, mission_id: str, channel_id: str) -> MagicMock:
            mv = MagicMock()
            mv.name = name
            mv.version = ver
            mv.aliases = []
            mv.tags = {"mission_id": mission_id, "channel_id": channel_id}
            return mv

        mission = "ESA-Mission1"
        all_versions = [
            make_version(
                f"telemanom-{mission}-channel_1", "1", mission_id=mission, channel_id="channel_1"
            ),
            make_version(
                "telemanom-ESA-Mission1-ADB-24m-channel_41", "1",
                mission_id="ESA-Mission1-ADB-24m", channel_id="channel_41",
            ),
        ]

        with (
            patch("spacecraft_telemetry.cli.load_settings", return_value=settings),
            patch("mlflow.tracking.client.MlflowClient") as mock_discovery_client_cls,
            patch(
                "spacecraft_telemetry.mlflow_tracking.registry.MlflowClient"
            ) as mock_registry_client_cls,
        ):
            mock_discovery_client = MagicMock()
            mock_registry_client = MagicMock()
            mock_discovery_client_cls.return_value = mock_discovery_client
            mock_registry_client_cls.return_value = mock_registry_client

            mock_discovery_client.search_model_versions.return_value = all_versions
            mock_registered_model = MagicMock()
            mock_registered_model.aliases = {"champion": "1"}
            mock_registry_client.get_registered_model.return_value = mock_registered_model
            mock_registry_client.delete_registered_model_alias.return_value = None

            result = runner.invoke(
                main,
                ["--env=test", "mlflow", "demote", "--mission", mission],
            )

        assert result.exit_code == 0, result.output
        assert "Demoted       : 1/1" in result.output
        demoted_names = [
            call.args[0] for call in mock_registry_client.get_registered_model.call_args_list
        ]
        assert demoted_names == ["telemanom-ESA-Mission1-channel_1"]

    def test_mlflow_demote_all_untagged_legacy_models_hint(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Mirrors the promote-side hint test: demote's "no models found" error
        must also distinguish a tagging gap from nothing being registered."""
        from unittest.mock import MagicMock

        settings = load_settings("test").model_copy(
            update={
                "mlflow": load_settings("test").mlflow.model_copy(
                    update={"tracking_uri": f"sqlite:///{tmp_path}/mlflow.db"}
                )
            }
        )

        legacy_version = MagicMock()
        legacy_version.name = "telemanom-ESA-Mission1-channel_22"
        legacy_version.version = "1"
        legacy_version.tags = {"window_size": "250"}

        def _search(filter_string: str) -> list[MagicMock]:
            if "tags.mission_id" in filter_string:
                return []
            return [legacy_version]

        with (
            patch("spacecraft_telemetry.cli.load_settings", return_value=settings),
            patch("mlflow.tracking.client.MlflowClient") as mock_client_cls,
        ):
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.search_model_versions.side_effect = _search

            result = runner.invoke(
                main,
                ["--env=test", "mlflow", "demote", "--mission", "ESA-Mission1"],
            )

        assert result.exit_code != 0
        assert "1 version(s) across 1 model(s)" in result.output
        assert "predate the mission_id tag" in result.output

    def test_mlflow_demote_variant_discovers_only_that_variant(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """--mission ESA-Mission1 --variant adb-24m demotes only that variant's champions.

        Mirrors test_mlflow_promote_variant_discovers_only_that_variant. Demote
        is the more destructive command -- it clears @champion aliases, and the
        bulk "--mission with no filter" form is its documented full-reset
        purpose -- so registered_model_name(..., variant) at the demote call
        site deserves the same variant-discrimination coverage as promote, not
        less.
        """
        from unittest.mock import MagicMock

        settings = load_settings("test").model_copy(
            update={
                "mlflow": load_settings("test").mlflow.model_copy(
                    update={"tracking_uri": f"sqlite:///{tmp_path}/mlflow.db"}
                )
            }
        )

        def make_version(
            name: str, ver: str, *, mission_id: str, channel_id: str,
            variant: str | None = None,
        ) -> MagicMock:
            mv = MagicMock()
            mv.name = name
            mv.version = ver
            mv.aliases = []
            mv.tags = {"mission_id": mission_id, "channel_id": channel_id}
            if variant:
                mv.tags["variant"] = variant
            return mv

        mission = "ESA-Mission1"
        base_version = make_version(
            f"telemanom-{mission}-channel_1", "1", mission_id=mission, channel_id="channel_1"
        )
        variant_version = make_version(
            f"telemanom-{mission}-adb-24m-channel_41", "1",
            mission_id=mission, channel_id="channel_41", variant="adb-24m",
        )

        with (
            patch("spacecraft_telemetry.cli.load_settings", return_value=settings),
            patch("mlflow.tracking.client.MlflowClient") as mock_discovery_client_cls,
            patch(
                "spacecraft_telemetry.mlflow_tracking.registry.MlflowClient"
            ) as mock_registry_client_cls,
        ):
            mock_discovery_client = MagicMock()
            mock_registry_client = MagicMock()
            mock_discovery_client_cls.return_value = mock_discovery_client
            mock_registry_client_cls.return_value = mock_registry_client

            mock_discovery_client.search_model_versions.return_value = [
                base_version, variant_version,
            ]
            mock_registered_model = MagicMock()
            mock_registered_model.aliases = {"champion": "1"}
            mock_registry_client.get_registered_model.return_value = mock_registered_model
            mock_registry_client.delete_registered_model_alias.return_value = None

            result = runner.invoke(
                main,
                ["--env=test", "mlflow", "demote", "--mission", mission, "--variant", "adb-24m"],
            )

        assert result.exit_code == 0, result.output
        assert "Variant       : adb-24m" in result.output
        assert "Demoted       : 1/1" in result.output
        demoted_names = [
            call.args[0] for call in mock_registry_client.get_registered_model.call_args_list
        ]
        assert demoted_names == ["telemanom-ESA-Mission1-adb-24m-channel_41"]

    def test_mlflow_demote_variant_requires_mission(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["--env=test", "mlflow", "demote", "--variant", "adb-24m"])
        assert result.exit_code != 0
        assert "--variant requires --mission" in result.output


# ---------------------------------------------------------------------------
# drift group
# ---------------------------------------------------------------------------

_DRIFT_SERIES_SCHEMA = _pa.schema(
    [
        _pa.field("telemetry_timestamp", _pa.timestamp("us", tz="UTC")),
        _pa.field("value_normalized", _pa.float32()),
        _pa.field("segment_id", _pa.int32()),
        _pa.field("is_anomaly", _pa.bool_()),
    ]
)


def _write_split_parquet(
    base: Path,
    mission: str,
    channel: str,
    split: str,
    n: int = 300,
    seed: int = 0,
) -> None:
    """Write a tiny Hive-partitioned series Parquet for one mission/channel/split."""
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2020-01-01", periods=n, freq="1s", tz="UTC")
    table = _pa.table(
        {
            "telemetry_timestamp": _pa.array(timestamps.astype("datetime64[us, UTC]")),
            "value_normalized": _pa.array(rng.standard_normal(n).astype("float32")),
            "segment_id": _pa.array(np.zeros(n, dtype=np.int32)),
            "is_anomaly": _pa.array(np.zeros(n, dtype=bool)),
        },
        schema=_DRIFT_SERIES_SCHEMA,
    )
    partition_dir = (
        base / mission / split / f"mission_id={mission}" / f"channel_id={channel}"
    )
    partition_dir.mkdir(parents=True, exist_ok=True)
    _pq.write_table(table, partition_dir / "part.parquet")


class TestDriftCommands:
    """CLI smoke tests for `drift batch` and `drift batch-mission`."""

    def test_drift_batch_help(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["--env=test", "drift", "batch", "--help"])
        assert result.exit_code == 0
        assert "--mission" in result.output
        assert "--channel" in result.output

    def test_drift_batch_mission_help(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["--env=test", "drift", "batch-mission", "--help"])
        assert result.exit_code == 0
        assert "--mission" in result.output
        assert "--max-channels" in result.output

    def test_drift_batch_runs_and_prints_channel(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Happy-path smoke test: writes train+test Parquet, runs drift batch."""
        from spacecraft_telemetry.core.config import (
            DriftConfig,
            PreprocessingConfig,
            Settings,
        )

        mission = "TEST-Mission"
        channel = "ch_1"
        _write_split_parquet(tmp_path, mission, channel, "train")
        _write_split_parquet(tmp_path, mission, channel, "test", seed=1)

        mlflow_uri = f"sqlite:///{tmp_path}/mlflow.db"
        # Use Settings() defaults — test.yaml has feature_windows=[3, 5] which
        # doesn't match MONITORING_FEATURE_COLS (built from [10, 50, 100]).
        settings = Settings(
            preprocess=PreprocessingConfig(processed_data_dir=str(tmp_path)),
            mlflow=Settings().mlflow.model_copy(update={"tracking_uri": mlflow_uri}),
            drift=DriftConfig(reference_profiles_dir=str(tmp_path / "profiles")),
        )

        with patch("spacecraft_telemetry.cli.load_settings", return_value=settings):
            result = runner.invoke(
                main,
                ["--env=test", "drift", "batch", f"--mission={mission}", f"--channel={channel}"],
            )

        assert result.exit_code == 0, result.output
        assert "Channel" in result.output
        assert channel in result.output

    def test_drift_batch_missing_channel_errors(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """drift batch fails with FileNotFoundError when channel data is absent."""
        from spacecraft_telemetry.core.config import PreprocessingConfig, Settings

        mission = "TEST-Mission"
        settings = Settings(
            preprocess=PreprocessingConfig(processed_data_dir=str(tmp_path)),
            mlflow=Settings().mlflow.model_copy(
                update={"tracking_uri": f"sqlite:///{tmp_path}/mlflow.db"}
            ),
        )

        with patch("spacecraft_telemetry.cli.load_settings", return_value=settings):
            result = runner.invoke(
                main,
                ["--env=test", "drift", "batch", f"--mission={mission}", "--channel=nonexistent"],
            )

        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# collect
# ---------------------------------------------------------------------------


class TestCollectCommand:
    def test_help_shows_options(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["collect", "--help"])
        assert result.exit_code == 0
        assert "--channel-set" in result.output
        assert "--duration" in result.output
        # --mission was removed (ISSLive is a single-mission feed)
        assert "--mission" not in result.output

    def test_collect_calls_collector_run(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        mock_collector = MagicMock()
        with patch(
            "spacecraft_telemetry.ingest.collector.LightstreamerCollector",
            return_value=mock_collector,
        ):
            result = runner.invoke(main, ["--env=local", "collect", "--duration=0"])

        assert result.exit_code == 0, result.output
        mock_collector.run.assert_called_once_with(seconds=0.0)

    def test_channel_set_override_reaches_config(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """--channel-set all propagates into the CollectorConfig passed to the ctor."""
        import spacecraft_telemetry.ingest.collector as _mod

        captured_configs: list[object] = []
        orig_init = _mod.LightstreamerCollector.__init__

        def _capture_init(self: object, config: object, dest_dir: object) -> None:
            captured_configs.append(config)
            orig_init(self, config, dest_dir)  # type: ignore[arg-type]

        mock_run = MagicMock()
        with (
            patch.object(_mod.LightstreamerCollector, "__init__", _capture_init),
            patch.object(_mod.LightstreamerCollector, "run", mock_run),
        ):
            result = runner.invoke(
                main, ["--env=local", "collect", "--channel-set=all", "--duration=0"]
            )

        assert result.exit_code == 0, result.output
        assert captured_configs, "LightstreamerCollector was not constructed"
        assert captured_configs[0].channel_set == "all"  # type: ignore[union-attr]

    def test_dest_dir_preserves_gcs_uri(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """dest_dir passed to LightstreamerCollector must not mangle gs:// URIs.

        pathlib.Path('gs://bucket/x') collapses the double-slash, causing UPath
        to raise ValueError on the first flush. The CLI must pass the raw string.
        """
        monkeypatch.setenv(
            "SPACECRAFT_COLLECT__RAW_TICKS_DIR", "gs://my-project-raw-data"
        )

        import spacecraft_telemetry.ingest.collector as _mod

        captured_dest: list[str] = []
        orig_init = _mod.LightstreamerCollector.__init__

        def _capture_init(self: object, config: object, dest_dir: object) -> None:
            captured_dest.append(str(dest_dir))
            orig_init(self, config, dest_dir)  # type: ignore[arg-type]

        mock_run = MagicMock()
        with (
            patch.object(_mod.LightstreamerCollector, "__init__", _capture_init),
            patch.object(_mod.LightstreamerCollector, "run", mock_run),
        ):
            result = runner.invoke(main, ["--env=local", "collect", "--duration=0"])

        assert result.exit_code == 0, result.output
        assert captured_dest, "LightstreamerCollector was not constructed"
        assert captured_dest[0].startswith("gs://"), (
            f"gs:// URI mangled: {captured_dest[0]!r}"
        )


# ---------------------------------------------------------------------------
# inject group (Phase 15)
# ---------------------------------------------------------------------------


class TestInjectGroup:
    def test_inject_help(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["inject", "--help"])
        assert result.exit_code == 0
        assert "inject" in result.output.lower()

    def test_inject_run_help(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["inject", "run", "--help"])
        assert result.exit_code == 0
        assert "--mission" in result.output
        assert "--processed-dir" in result.output
        assert "--output-dir" in result.output

    def test_inject_run_calls_generate(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from unittest.mock import patch

        manifest = {"S1000003": [{"type": "drift", "start": 10, "end": 50, "duration": 40,
                                  "magnitude_sigma": 1.0, "signal_class": "slow_lownoise"}]}

        with patch(
            "spacecraft_telemetry.injection.generate_injected_dataset",
            return_value=manifest,
        ) as mock_gen:
            result = runner.invoke(
                main,
                ["--env=local", "inject", "run", "--mission=ISS",
                 "--channels=S1000003",
                 f"--processed-dir={tmp_path}/proc",
                 f"--output-dir={tmp_path}/injected"],
            )

        assert result.exit_code == 0, result.output
        mock_gen.assert_called_once()
        call_kwargs = mock_gen.call_args
        assert call_kwargs.args[1] == "ISS"  # mission
        assert call_kwargs.args[2] == ["S1000003"]  # channel_list

    def test_inject_run_processed_dir_override_applied(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from unittest.mock import patch

        captured_settings: list = []

        def _fake_gen(settings, mission, channels):
            captured_settings.append(settings)
            return {}

        with patch("spacecraft_telemetry.injection.generate_injected_dataset", _fake_gen):
            result = runner.invoke(
                main,
                ["--env=local", "inject", "run", "--mission=ISS",
                 f"--processed-dir={tmp_path}/custom_proc",
                 f"--output-dir={tmp_path}/injected"],
            )

        assert result.exit_code == 0, result.output
        assert captured_settings
        assert str(captured_settings[0].preprocess.processed_data_dir) == str(
            tmp_path / "custom_proc"
        )


class TestRayScoreProcessedDirOption:
    def test_processed_dir_flag_in_help(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["ray", "score", "--help"])
        assert result.exit_code == 0
        assert "--processed-dir" in result.output


class TestRayScoreInjectedOption:
    """--injected (Phase 15) tags scoring runs so ray_fanout.tune can locate a
    channel's nominal baseline run independently of its injected-data run for
    the HPO false-positive-rate penalty."""

    def _mock_cm(self) -> MagicMock:
        cm = MagicMock()
        cm.__enter__.return_value = None
        cm.__exit__.return_value = None
        return cm

    def test_injected_flag_in_help(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["ray", "score", "--help"])
        assert result.exit_code == 0
        assert "--injected" in result.output

    def test_injected_flag_passes_injected_data_source(self, runner: CliRunner) -> None:
        settings = load_settings("test")
        with (
            patch("spacecraft_telemetry.cli.load_settings", return_value=settings),
            patch("spacecraft_telemetry.cli._ray_session", return_value=self._mock_cm()),
            patch(
                "spacecraft_telemetry.ray_fanout.discover_channels",
                return_value=["ch_a"],
            ),
            patch(
                "spacecraft_telemetry.ray_fanout.score_all_channels",
                return_value=[],
            ) as mock_score,
        ):
            result = runner.invoke(
                main,
                ["--env=test", "ray", "score", "--mission=ISS", "--injected"],
            )

        assert result.exit_code == 0, result.output
        assert mock_score.call_args.kwargs["data_source"] == "injected"

    def test_no_injected_flag_defaults_to_nominal_data_source(self, runner: CliRunner) -> None:
        settings = load_settings("test")
        with (
            patch("spacecraft_telemetry.cli.load_settings", return_value=settings),
            patch("spacecraft_telemetry.cli._ray_session", return_value=self._mock_cm()),
            patch(
                "spacecraft_telemetry.ray_fanout.discover_channels",
                return_value=["ch_a"],
            ),
            patch(
                "spacecraft_telemetry.ray_fanout.score_all_channels",
                return_value=[],
            ) as mock_score,
        ):
            result = runner.invoke(
                main,
                ["--env=test", "ray", "score", "--mission=ISS"],
            )

        assert result.exit_code == 0, result.output
        assert mock_score.call_args.kwargs["data_source"] == "nominal"


class TestRayTuneProcessedDirOption:
    def test_processed_dir_flag_in_help(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["ray", "tune", "--help"])
        assert result.exit_code == 0
        assert "--processed-dir" in result.output
