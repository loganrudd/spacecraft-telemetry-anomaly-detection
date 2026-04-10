"""Tests for core.config."""

import pytest
from pathlib import Path

from spacecraft_telemetry.core.config import (
    DataConfig,
    LoggingConfig,
    Settings,
    load_settings,
)

_YAML_LOCAL = """\
env: local
data:
  raw_data_dir: data/raw
  sample_data_dir: data/sample
  zenodo_record_id: "12528696"
  missions:
    - ESA-Mission1
    - ESA-Mission2
    - ESA-Mission3
  sample_fraction: 0.01
  sample_channels: 5
logging:
  level: DEBUG
  format: console
"""

_YAML_MINIMAL = """\
data:
  raw_data_dir: data/raw
  sample_data_dir: data/sample
  missions: [ESA-Mission1]
  sample_fraction: 0.01
  sample_channels: 2
logging:
  level: WARNING
  format: console
"""


# ---------------------------------------------------------------------------
# DataConfig
# ---------------------------------------------------------------------------


class TestDataConfig:
    def test_defaults(self) -> None:
        cfg = DataConfig()
        assert cfg.zenodo_record_id == "12528696"
        assert cfg.sample_fraction == 0.01
        assert cfg.sample_channels == 5
        assert len(cfg.missions) == 3

    def test_fraction_zero_is_invalid(self) -> None:
        with pytest.raises(ValueError, match="sample_fraction"):
            DataConfig(sample_fraction=0.0)

    def test_fraction_above_one_is_invalid(self) -> None:
        with pytest.raises(ValueError, match="sample_fraction"):
            DataConfig(sample_fraction=1.1)

    def test_fraction_exactly_one_is_valid(self) -> None:
        cfg = DataConfig(sample_fraction=1.0)
        assert cfg.sample_fraction == 1.0

    def test_channels_zero_is_invalid(self) -> None:
        with pytest.raises(ValueError, match="sample_channels"):
            DataConfig(sample_channels=0)


# ---------------------------------------------------------------------------
# LoggingConfig
# ---------------------------------------------------------------------------


class TestLoggingConfig:
    def test_defaults(self) -> None:
        cfg = LoggingConfig()
        assert cfg.level == "INFO"
        assert cfg.format == "console"

    def test_json_format_is_valid(self) -> None:
        cfg = LoggingConfig(format="json")
        assert cfg.format == "json"

    def test_invalid_format_raises(self) -> None:
        with pytest.raises(ValueError, match="format"):
            LoggingConfig(format="xml")


# ---------------------------------------------------------------------------
# load_settings / Settings
# ---------------------------------------------------------------------------


class TestLoadSettings:
    def test_loads_yaml_values(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        (config_dir / "local.yaml").write_text(_YAML_LOCAL)
        monkeypatch.setenv("SPACECRAFT_CONFIG_DIR", str(config_dir))
        monkeypatch.delenv("SPACECRAFT_ENV", raising=False)

        settings = load_settings("local")

        assert settings.env == "local"
        assert settings.logging.level == "DEBUG"
        assert settings.data.sample_fraction == 0.01
        assert len(settings.data.missions) == 3

    def test_env_var_overrides_yaml(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        (config_dir / "local.yaml").write_text(_YAML_MINIMAL)
        monkeypatch.setenv("SPACECRAFT_CONFIG_DIR", str(config_dir))
        monkeypatch.setenv("SPACECRAFT_LOGGING__LEVEL", "ERROR")

        settings = load_settings("local")

        assert settings.logging.level == "ERROR"  # env var wins over YAML

    def test_missing_yaml_falls_back_to_defaults(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        config_dir = tmp_path / "no-configs-here"
        monkeypatch.setenv("SPACECRAFT_CONFIG_DIR", str(config_dir))
        monkeypatch.delenv("SPACECRAFT_ENV", raising=False)

        settings = load_settings("local")

        assert settings.data.sample_fraction == 0.01  # DataConfig default
        assert settings.logging.level == "INFO"  # LoggingConfig default

    def test_load_settings_does_not_leak_env_var(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        (config_dir / "test.yaml").write_text(_YAML_MINIMAL)
        monkeypatch.setenv("SPACECRAFT_CONFIG_DIR", str(config_dir))
        monkeypatch.delenv("SPACECRAFT_ENV", raising=False)

        import os

        load_settings("test")

        # SPACECRAFT_ENV must not linger after load_settings returns
        assert "SPACECRAFT_ENV" not in os.environ

    def test_settings_direct_construction(self) -> None:
        settings = Settings(
            data=DataConfig(sample_fraction=0.5, missions=["M1"], sample_channels=3),
            logging=LoggingConfig(level="WARNING", format="json"),
        )
        assert settings.data.sample_fraction == 0.5
        assert settings.logging.format == "json"
