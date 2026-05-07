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
                "spacecraft_telemetry.ray_training.discover_channels",
                return_value=["channel_1", "channel_2"],
            ),
            patch(
                "spacecraft_telemetry.ray_training.run_all_sweeps",
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
                "spacecraft_telemetry.ray_training.discover_channels",
                return_value=["channel_1", "channel_2"],
            ),
            patch(
                "spacecraft_telemetry.ray_training.load_channel_subsystem_map",
                return_value={"channel_1": "subsystem_1", "channel_2": "subsystem_6"},
            ),
            patch(
                "spacecraft_telemetry.ray_training.run_hpo_sweep",
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
            patch("spacecraft_telemetry.ray_training.write_tuned_configs") as mock_write,
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
            patch("spacecraft_telemetry.ray_training.discover_channels", return_value=[]),
        ):
            result = runner.invoke(
                main,
                ["--env=test", "ray", "tune", "--mission=ESA-Mission1"],
            )

        assert result.exit_code != 0
        assert "No preprocessed channels found" in result.output

    def test_tune_errors_when_subsystem_map_missing(self, runner: CliRunner) -> None:
        settings = load_settings("test")
        mock_cm = MagicMock()
        mock_cm.__enter__.return_value = None
        mock_cm.__exit__.return_value = None

        with (
            patch("spacecraft_telemetry.cli.load_settings", return_value=settings),
            patch("spacecraft_telemetry.cli._ray_session", return_value=mock_cm),
            patch(
                "spacecraft_telemetry.ray_training.discover_channels",
                return_value=["channel_1", "channel_2"],
            ),
            patch("spacecraft_telemetry.ray_training.load_channel_subsystem_map", return_value={}),
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
                "spacecraft_telemetry.ray_training.discover_channels",
                return_value=["channel_1", "channel_2"],
            ),
            patch(
                "spacecraft_telemetry.ray_training.load_channel_subsystem_map",
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
                "spacecraft_telemetry.ray_training.discover_channels",
                return_value=["channel_1"],
            ),
            patch(
                "spacecraft_telemetry.ray_training.load_channel_subsystem_map",
                return_value={"channel_1": "subsystem_1"},
            ),
            patch(
                "spacecraft_telemetry.ray_training.run_hpo_sweep",
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
                "spacecraft_telemetry.ray_training.discover_channels",
                return_value=["channel_1"],
            ),
            patch(
                "spacecraft_telemetry.ray_training.load_channel_subsystem_map",
                return_value={"channel_1": "subsystem_1"},
            ),
            patch(
                "spacecraft_telemetry.ray_training.run_hpo_sweep",
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
# mlflow group
# ---------------------------------------------------------------------------


class TestMlflowCli:
    def test_mlflow_promote_help(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["--env=test", "mlflow", "promote", "--help"])
        assert result.exit_code == 0
        assert "--name" in result.output
        assert "--stage" in result.output

    def test_mlflow_ui_help(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["--env=test", "mlflow", "ui", "--help"])
        assert result.exit_code == 0
        assert "--port" in result.output

    def test_mlflow_promote_resolves_latest_version(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """promote without --version resolves the latest non-archived version."""
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
        mock_version.current_stage = "None"

        with (
            patch("spacecraft_telemetry.cli.load_settings", return_value=settings),
            patch("mlflow.tracking.MlflowClient") as mock_client_cls,
        ):
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client
            mock_client.search_model_versions.return_value = [mock_version]
            mock_client.transition_model_version_stage.return_value = None

            result = runner.invoke(
                main,
                [
                    "--env=test",
                    "mlflow",
                    "promote",
                    "--name=telemanom-ESA-Mission1-channel_1",
                    "--stage=Production",
                ],
            )

        assert result.exit_code == 0, result.output
        mock_client.transition_model_version_stage.assert_called_once_with(
            name="telemanom-ESA-Mission1-channel_1",
            version="3",
            stage="Production",
        )

    def test_mlflow_promote_no_versions_errors(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """promote should fail with a clear message when no non-archived versions exist."""
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
            patch("mlflow.tracking.MlflowClient") as mock_client_cls,
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
                    "--stage=Production",
                ],
            )

        assert result.exit_code != 0
        assert "No promotable versions" in result.output
