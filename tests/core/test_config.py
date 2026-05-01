"""Tests for core.config."""

from pathlib import Path

import pytest

from spacecraft_telemetry.core.config import (
    DataConfig,
    FeastConfig,
    LoggingConfig,
    ModelConfig,
    Settings,
    SparkConfig,
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
spark:
  train_fraction: 0.8
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
# SparkConfig
# ---------------------------------------------------------------------------


class TestSparkConfig:
    def test_defaults(self) -> None:
        cfg = SparkConfig()
        # window_size and prediction_horizon live on ModelConfig since Plan 002.5
        assert cfg.train_fraction == 0.8
        assert cfg.normalization == "z-score"
        assert cfg.gap_multiplier == 3.0
        assert cfg.feature_windows == [10, 50, 100]
        assert cfg.driver_memory == "1536m"
        assert cfg.num_cores == 2

    def test_train_fraction_zero_invalid(self) -> None:
        with pytest.raises(ValueError, match="train_fraction"):
            SparkConfig(train_fraction=0.0)

    def test_train_fraction_one_invalid(self) -> None:
        with pytest.raises(ValueError, match="train_fraction"):
            SparkConfig(train_fraction=1.0)

    def test_train_fraction_valid_range(self) -> None:
        cfg = SparkConfig(train_fraction=0.7)
        assert cfg.train_fraction == 0.7

    def test_invalid_normalization_raises(self) -> None:
        with pytest.raises(ValueError, match="normalization"):
            SparkConfig(normalization="l2")  # type: ignore[arg-type]

    def test_min_max_normalization_rejected(self) -> None:
        with pytest.raises((ValueError, Exception)):
            SparkConfig(normalization="min-max")  # type: ignore[arg-type]

    def test_window_size_zero_invalid(self) -> None:
        # window_size moved to ModelConfig in Plan 002.5
        with pytest.raises(ValueError, match="must be >= 1"):
            ModelConfig(window_size=0)

    def test_gap_multiplier_zero_invalid(self) -> None:
        with pytest.raises(ValueError, match="gap_multiplier"):
            SparkConfig(gap_multiplier=0.0)

    def test_feature_windows_empty_invalid(self) -> None:
        with pytest.raises(ValueError, match="feature_windows"):
            SparkConfig(feature_windows=[])

    def test_feature_windows_zero_entry_invalid(self) -> None:
        with pytest.raises(ValueError, match="feature_windows"):
            SparkConfig(feature_windows=[10, 0])

    def test_custom_feature_windows(self) -> None:
        cfg = SparkConfig(feature_windows=[5, 20])
        assert cfg.feature_windows == [5, 20]


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

    def test_missing_yaml_falls_back_to_defaults(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        config_dir = tmp_path / "no-configs-here"
        monkeypatch.setenv("SPACECRAFT_CONFIG_DIR", str(config_dir))
        monkeypatch.delenv("SPACECRAFT_ENV", raising=False)

        settings = load_settings("local")

        assert settings.data.sample_fraction == 0.01  # DataConfig default
        assert settings.logging.level == "INFO"  # LoggingConfig default

    def test_load_settings_does_not_leak_env_var(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        (config_dir / "test.yaml").write_text(_YAML_MINIMAL)
        monkeypatch.setenv("SPACECRAFT_CONFIG_DIR", str(config_dir))
        monkeypatch.delenv("SPACECRAFT_ENV", raising=False)

        import os

        load_settings("test")

        # SPACECRAFT_ENV must not linger after load_settings returns
        assert "SPACECRAFT_ENV" not in os.environ

    def test_spark_config_present_with_defaults(self) -> None:
        settings = Settings()
        # window_size moved to ModelConfig in Plan 002.5
        assert settings.model.window_size == 250
        assert settings.spark.train_fraction == 0.8
        assert settings.spark.normalization == "z-score"
        assert settings.spark.feature_windows == [10, 50, 100]

    def test_settings_direct_construction(self) -> None:
        settings = Settings(
            data=DataConfig(sample_fraction=0.5, missions=["M1"], sample_channels=3),
            logging=LoggingConfig(level="WARNING", format="json"),
        )
        assert settings.data.sample_fraction == 0.5
        assert settings.logging.format == "json"


# ---------------------------------------------------------------------------
# FeastConfig
# ---------------------------------------------------------------------------

_YAML_WITH_FEAST = """\
feast:
  repo_path: feature_repo
  project: spacecraft_telemetry
  feature_view_name: telemetry_features
  source_path: data/processed/ESA-Mission1/features
  source_root: data/processed
  ttl_days: 365
"""


class TestFeastConfig:
    def test_defaults(self) -> None:
        cfg = FeastConfig()
        assert cfg.project == "spacecraft_telemetry"
        assert cfg.feature_view_name == "telemetry_features"
        assert cfg.ttl_days == 365
        assert cfg.repo_path == Path("feature_repo")
        assert cfg.source_path == Path("data/processed/ESA-Mission1/features")
        assert cfg.source_root == Path("data/processed")

    def test_ttl_zero_is_invalid(self) -> None:
        with pytest.raises(ValueError, match="ttl_days"):
            FeastConfig(ttl_days=0)

    def test_ttl_negative_is_invalid(self) -> None:
        with pytest.raises(ValueError, match="ttl_days"):
            FeastConfig(ttl_days=-1)

    def test_ttl_one_is_valid(self) -> None:
        cfg = FeastConfig(ttl_days=1)
        assert cfg.ttl_days == 1

    def test_project_digits_first_is_invalid(self) -> None:
        with pytest.raises(ValueError, match="project"):
            FeastConfig(project="123bad")

    def test_project_hyphen_is_invalid(self) -> None:
        with pytest.raises(ValueError, match="project"):
            FeastConfig(project="my-project")

    def test_project_underscore_start_is_valid(self) -> None:
        cfg = FeastConfig(project="_my_project")
        assert cfg.project == "_my_project"

    def test_project_alphanumeric_is_valid(self) -> None:
        cfg = FeastConfig(project="spacecraft_telemetry")
        assert cfg.project == "spacecraft_telemetry"

    def test_load_settings_includes_feast(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        (config_dir / "local.yaml").write_text(_YAML_WITH_FEAST)
        monkeypatch.setenv("SPACECRAFT_CONFIG_DIR", str(config_dir))
        monkeypatch.delenv("SPACECRAFT_ENV", raising=False)

        settings = load_settings("local")

        assert settings.feast.project == "spacecraft_telemetry"
        assert settings.feast.ttl_days == 365
        assert settings.feast.source_path == Path("data/processed/ESA-Mission1/features")
        assert settings.feast.source_root == Path("data/processed")

    def test_feast_env_var_override(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        (config_dir / "local.yaml").write_text(_YAML_WITH_FEAST)
        monkeypatch.setenv("SPACECRAFT_CONFIG_DIR", str(config_dir))
        monkeypatch.setenv("SPACECRAFT_FEAST__TTL_DAYS", "30")
        monkeypatch.delenv("SPACECRAFT_ENV", raising=False)

        settings = load_settings("local")

        assert settings.feast.ttl_days == 30  # env var overrides YAML
