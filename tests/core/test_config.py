"""Tests for core.config."""

from pathlib import Path

import pytest

from spacecraft_telemetry.core.config import (
    ApiConfig,
    DataConfig,
    LoggingConfig,
    MlflowConfig,
    ModelConfig,
    MonitoringConfig,
    Settings,
    SparkConfig,
    TuneConfig,
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

    def test_mlflow_section_round_trips(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        (config_dir / "local.yaml").write_text(
            "mlflow:\n  tracking_uri: 'sqlite:///test.db'\n  experiment_prefix: 'ci-'\n"
        )
        monkeypatch.setenv("SPACECRAFT_CONFIG_DIR", str(config_dir))
        monkeypatch.delenv("SPACECRAFT_ENV", raising=False)

        settings = load_settings("local")

        assert settings.mlflow.tracking_uri.endswith("/test.db")
        assert Path(settings.mlflow.tracking_uri.removeprefix("sqlite:///")).is_absolute()
        assert settings.mlflow.experiment_prefix == "ci-"
        assert settings.mlflow.registry_uri is None

    def test_hpo_eval_fraction_round_trips(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        (config_dir / "local.yaml").write_text("tune:\n  hpo_eval_fraction: 0.7\n")
        monkeypatch.setenv("SPACECRAFT_CONFIG_DIR", str(config_dir))
        monkeypatch.delenv("SPACECRAFT_ENV", raising=False)

        settings = load_settings("local")

        assert settings.tune.hpo_eval_fraction == 0.7

    def test_model_type_default(self) -> None:
        settings = Settings()
        assert settings.model.model_type == "telemanom"


# ---------------------------------------------------------------------------
# TuneConfig
# ---------------------------------------------------------------------------


class TestTuneConfig:
    def test_defaults(self) -> None:
        cfg = TuneConfig()
        assert cfg.num_samples == 50
        assert cfg.max_concurrent_trials == 2
        assert cfg.hpo_eval_fraction == 0.6
        assert cfg.parallel_subsystems is False

    def test_hpo_eval_fraction_zero_is_invalid(self) -> None:
        with pytest.raises(ValueError, match="hpo_eval_fraction"):
            TuneConfig(hpo_eval_fraction=0.0)

    def test_hpo_eval_fraction_one_is_invalid(self) -> None:
        with pytest.raises(ValueError, match="hpo_eval_fraction"):
            TuneConfig(hpo_eval_fraction=1.0)

    def test_hpo_eval_fraction_valid(self) -> None:
        cfg = TuneConfig(hpo_eval_fraction=0.75)
        assert cfg.hpo_eval_fraction == 0.75

    def test_num_samples_zero_is_invalid(self) -> None:
        with pytest.raises(ValueError, match="must be >= 1"):
            TuneConfig(num_samples=0)


# ---------------------------------------------------------------------------
# MlflowConfig
# ---------------------------------------------------------------------------


class TestMlflowConfig:
    def test_defaults(self) -> None:
        cfg = MlflowConfig()
        # sqlite:///mlflow.db is resolved to an absolute path at construction time (A4).
        assert cfg.tracking_uri.startswith("sqlite:///")
        assert Path(cfg.tracking_uri.removeprefix("sqlite:///")).is_absolute()
        assert cfg.tracking_uri.endswith("mlflow.db")
        assert cfg.registry_uri is None
        assert cfg.experiment_prefix == ""

    def test_custom_tracking_uri(self) -> None:
        cfg = MlflowConfig(tracking_uri="http://localhost:5000")
        assert cfg.tracking_uri == "http://localhost:5000"

    def test_registry_uri_can_be_set(self) -> None:
        cfg = MlflowConfig(registry_uri="sqlite:///registry.db")
        assert cfg.registry_uri == "sqlite:///registry.db"

    def test_experiment_prefix_stored(self) -> None:
        cfg = MlflowConfig(experiment_prefix="dev-")
        assert cfg.experiment_prefix == "dev-"


# ---------------------------------------------------------------------------
# MonitoringConfig
# ---------------------------------------------------------------------------


class TestMonitoringConfig:
    def test_defaults(self) -> None:
        cfg = MonitoringConfig()
        assert cfg.drift_threshold == 0.30
        assert cfg.reference_profiles_dir == Path("monitoring/reference_profiles")
        assert cfg.report_output_dir == Path("monitoring/reports")
        assert cfg.reference_sample_rows == 5000

    def test_drift_threshold_zero_is_invalid(self) -> None:
        with pytest.raises(ValueError, match="drift_threshold"):
            MonitoringConfig(drift_threshold=0.0)

    def test_drift_threshold_one_is_invalid(self) -> None:
        with pytest.raises(ValueError, match="drift_threshold"):
            MonitoringConfig(drift_threshold=1.0)

    def test_drift_threshold_valid(self) -> None:
        cfg = MonitoringConfig(drift_threshold=0.5)
        assert cfg.drift_threshold == 0.5

    def test_reference_sample_rows_zero_is_invalid(self) -> None:
        with pytest.raises(ValueError, match="reference_sample_rows"):
            MonitoringConfig(reference_sample_rows=0)

    def test_reference_sample_rows_negative_is_invalid(self) -> None:
        with pytest.raises(ValueError, match="reference_sample_rows"):
            MonitoringConfig(reference_sample_rows=-1)

    def test_reference_sample_rows_custom(self) -> None:
        cfg = MonitoringConfig(reference_sample_rows=500)
        assert cfg.reference_sample_rows == 500

    def test_custom_dirs(self) -> None:
        cfg = MonitoringConfig(
            reference_profiles_dir=Path("custom/profiles"),
            report_output_dir=Path("custom/reports"),
        )
        assert cfg.reference_profiles_dir == Path("custom/profiles")
        assert cfg.report_output_dir == Path("custom/reports")

    def test_settings_has_monitoring_field(self) -> None:
        settings = Settings()
        assert isinstance(settings.monitoring, MonitoringConfig)
        assert settings.monitoring.drift_threshold == 0.30

    def test_monitoring_round_trips_yaml(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        (config_dir / "local.yaml").write_text(
            "monitoring:\n"
            "  drift_threshold: 0.25\n"
            "  reference_sample_rows: 1000\n"
        )
        monkeypatch.setenv("SPACECRAFT_CONFIG_DIR", str(config_dir))
        monkeypatch.delenv("SPACECRAFT_ENV", raising=False)

        settings = load_settings("local")

        assert settings.monitoring.drift_threshold == 0.25
        assert settings.monitoring.reference_sample_rows == 1000

    def test_monitoring_env_var_override(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        (config_dir / "local.yaml").write_text("monitoring:\n  drift_threshold: 0.40\n")
        monkeypatch.setenv("SPACECRAFT_CONFIG_DIR", str(config_dir))
        monkeypatch.setenv("SPACECRAFT_MONITORING__DRIFT_THRESHOLD", "0.20")
        monkeypatch.delenv("SPACECRAFT_ENV", raising=False)

        settings = load_settings("local")

        assert settings.monitoring.drift_threshold == 0.20  # env var wins


# ---------------------------------------------------------------------------
# ApiConfig
# ---------------------------------------------------------------------------


class TestApiConfig:
    def test_defaults(self) -> None:
        cfg = ApiConfig()
        assert cfg.host == "127.0.0.1"
        assert cfg.port == 8000
        assert cfg.mission == "ESA-Mission1"
        assert cfg.subsystem == "subsystem_6"
        assert cfg.channels == []
        assert cfg.replay_speed_default == 10.0
        assert cfg.replay_tick_interval_seconds == 1.0
        assert cfg.stream_buffer_max_events == 256
        assert cfg.request_timeout_seconds == 30

    def test_port_negative_is_invalid(self) -> None:
        with pytest.raises(ValueError, match="port"):
            ApiConfig(port=-1)

    def test_port_above_65535_is_invalid(self) -> None:
        with pytest.raises(ValueError, match="port"):
            ApiConfig(port=65536)

    def test_port_boundary_values_valid(self) -> None:
        assert ApiConfig(port=0).port == 0  # OS-assigned (useful in tests)
        assert ApiConfig(port=1).port == 1
        assert ApiConfig(port=65535).port == 65535

    def test_replay_speed_zero_is_invalid(self) -> None:
        with pytest.raises(ValueError, match="replay_speed_default"):
            ApiConfig(replay_speed_default=0.0)

    def test_replay_speed_negative_is_invalid(self) -> None:
        with pytest.raises(ValueError, match="replay_speed_default"):
            ApiConfig(replay_speed_default=-1.0)

    def test_tick_interval_zero_is_invalid(self) -> None:
        with pytest.raises(ValueError, match="replay_tick_interval_seconds"):
            ApiConfig(replay_tick_interval_seconds=0.0)

    def test_stream_buffer_zero_is_invalid(self) -> None:
        with pytest.raises(ValueError, match="must be >= 1"):
            ApiConfig(stream_buffer_max_events=0)

    def test_request_timeout_zero_is_invalid(self) -> None:
        with pytest.raises(ValueError, match="must be >= 1"):
            ApiConfig(request_timeout_seconds=0)

    def test_settings_has_api_field(self) -> None:
        settings = Settings()
        assert isinstance(settings.api, ApiConfig)
        assert settings.api.port == 8000

    def test_api_round_trips_yaml(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        (config_dir / "local.yaml").write_text(
            "api:\n"
            "  host: \"0.0.0.0\"\n"
            "  port: 9000\n"
            "  subsystem: \"subsystem_1\"\n"
        )
        monkeypatch.setenv("SPACECRAFT_CONFIG_DIR", str(config_dir))
        monkeypatch.delenv("SPACECRAFT_ENV", raising=False)

        settings = load_settings("local")

        assert settings.api.host == "0.0.0.0"
        assert settings.api.port == 9000
        assert settings.api.subsystem == "subsystem_1"

    def test_api_env_var_override(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        (config_dir / "local.yaml").write_text("api:\n  subsystem: \"subsystem_6\"\n")
        monkeypatch.setenv("SPACECRAFT_CONFIG_DIR", str(config_dir))
        monkeypatch.setenv("SPACECRAFT_API__SUBSYSTEM", "subsystem_1")
        monkeypatch.delenv("SPACECRAFT_ENV", raising=False)

        settings = load_settings("local")

        assert settings.api.subsystem == "subsystem_1"  # env var wins

    def test_test_env_loads_correct_api_defaults(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify test.yaml's api overrides are applied correctly."""
        monkeypatch.delenv("SPACECRAFT_API__SUBSYSTEM", raising=False)
        settings = load_settings("test")
        assert settings.api.port == 0
        assert settings.api.replay_speed_default == 1000.0
        assert settings.api.replay_tick_interval_seconds == 0.001
        assert settings.api.stream_buffer_max_events == 32

