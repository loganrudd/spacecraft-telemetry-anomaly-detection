"""Config management for the spacecraft telemetry system.

Settings are loaded from a YAML file at configs/{env}.yaml, with environment
variables taking higher priority (SPACECRAFT_ prefix, __ nested delimiter).

Priority (highest → lowest):
    1. Constructor kwargs (for tests)
    2. Environment variables  (SPACECRAFT_LOGGING__LEVEL, etc.)
    3. YAML config file       (configs/{env}.yaml)
    4. Field defaults
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, field_validator
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

# Repo root — used to locate configs/ in local dev.
# Override with SPACECRAFT_CONFIG_DIR for non-standard layouts.
_REPO_ROOT = Path(__file__).parents[3]


class DataConfig(BaseModel):
    raw_data_dir: Path = Path("data/raw")
    sample_data_dir: Path = Path("data/sample")
    zenodo_record_id: str = "12528696"
    missions: list[str] = ["ESA-Mission1", "ESA-Mission2", "ESA-Mission3"]
    sample_fraction: float = 0.01  # fraction of rows to keep in local dev sample
    sample_channels: int = 5  # channels per mission in local dev sample

    @field_validator("sample_fraction")
    @classmethod
    def fraction_in_range(cls, v: float) -> float:
        if not 0 < v <= 1.0:
            raise ValueError(f"sample_fraction must be in (0, 1], got {v}")
        return v

    @field_validator("sample_channels")
    @classmethod
    def channels_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"sample_channels must be >= 1, got {v}")
        return v


class SparkConfig(BaseModel):
    processed_data_dir: Path = Path("data/processed")
    driver_memory: str = "1536m"
    num_cores: int = 2
    train_fraction: float = 0.8
    normalization: Literal["z-score"] = "z-score"
    gap_multiplier: float = 3.0
    feature_windows: list[int] = [10, 50, 100]

    @field_validator("train_fraction")
    @classmethod
    def train_fraction_in_range(cls, v: float) -> float:
        if not 0 < v < 1.0:
            raise ValueError(f"train_fraction must be in (0, 1), got {v}")
        return v

    @field_validator("num_cores")
    @classmethod
    def positive_int(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"must be >= 1, got {v}")
        return v

    @field_validator("gap_multiplier")
    @classmethod
    def positive_float(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"gap_multiplier must be > 0, got {v}")
        return v

    @field_validator("feature_windows")
    @classmethod
    def windows_nonempty_positive(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("feature_windows must not be empty")
        if any(w < 1 for w in v):
            raise ValueError(f"all feature_windows must be >= 1, got {v}")
        return v


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "console"  # "console" for local dev, "json" for cloud

    @field_validator("format")
    @classmethod
    def valid_format(cls, v: str) -> str:
        allowed = ("json", "console")
        if v not in allowed:
            raise ValueError(f"format must be one of {allowed}, got {v!r}")
        return v


class FeastConfig(BaseModel):
    repo_path: Path = Path("feature_repo")
    project: str = "spacecraft_telemetry"
    feature_view_name: str = "telemetry_features"
    source_path: Path = Path("data/processed/ESA-Mission1/features")
    source_root: Path = Path("data/processed")
    ttl_days: int = 365

    @field_validator("ttl_days")
    @classmethod
    def ttl_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"ttl_days must be >= 1, got {v}")
        return v

    @field_validator("project")
    @classmethod
    def valid_project_name(cls, v: str) -> str:
        if not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]*", v):
            raise ValueError(
                f"project must match [a-zA-Z_][a-zA-Z0-9_]*, got {v!r}"
            )
        return v


class ModelConfig(BaseModel):
    # Identifies which model family this config applies to.
    # Used by io.load_model() and MLflow experiment schema.
    # Future values: "dc_vae"
    scorer_type: Literal["telemanom"] = "telemanom"
    hidden_dim: int = 80
    num_layers: int = 2
    dropout: float = 0.3
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 35
    window_size: int = 250
    prediction_horizon: int = 1
    early_stopping_patience: int = 5
    val_fraction: float = 0.1
    seed: int = 42
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    artifacts_dir: Path = Path("models")
    # DataLoader workers — 0 on MPS/macOS; 4 on cloud GPU nodes
    num_workers: int = 0
    # Scoring (Hundman §3.1 / §3.2 rolling-window simplification)
    inference_batch_size: int = 256
    error_smoothing_window: int = 30
    threshold_window: int = 250
    threshold_z: float = 3.0
    threshold_min_anomaly_len: int = 3

    @field_validator(
        "hidden_dim", "num_layers", "batch_size", "epochs", "early_stopping_patience",
        "seed", "error_smoothing_window", "threshold_window", "threshold_min_anomaly_len",
        "inference_batch_size", "window_size", "prediction_horizon",
    )
    @classmethod
    def positive_int(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"must be >= 1, got {v}")
        return v

    @field_validator("num_workers")
    @classmethod
    def non_negative_int(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"num_workers must be >= 0, got {v}")
        return v

    @field_validator("val_fraction")
    @classmethod
    def val_fraction_in_range(cls, v: float) -> float:
        if not 0 < v < 1.0:
            raise ValueError(f"val_fraction must be in (0, 1), got {v}")
        return v

    @field_validator("dropout")
    @classmethod
    def dropout_in_range(cls, v: float) -> float:
        if not 0.0 <= v < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {v}")
        return v

    @field_validator("learning_rate", "threshold_z")
    @classmethod
    def positive_float(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"must be > 0, got {v}")
        return v


class RayConfig(BaseModel):
    """Ray cluster and task resource configuration."""

    num_cpus: int = 2                  # passed to ray.init() — constrains local parallelism
    num_gpus_per_task: float = 0.0     # 0.0 = CPU-only; 0.25 on T4 packs 4 models per GPU
    max_retries: int = 3               # @ray.remote max_retries for preemptible-VM resilience
    address: str | None = None         # None = start local cluster; "auto" = attach to existing

    @field_validator("num_cpus")
    @classmethod
    def positive_cpus(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"num_cpus must be >= 1, got {v}")
        return v

    @field_validator("num_gpus_per_task")
    @classmethod
    def non_negative_gpus(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"num_gpus_per_task must be >= 0, got {v}")
        return v

    @field_validator("max_retries")
    @classmethod
    def non_negative_retries(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"max_retries must be >= 0, got {v}")
        return v


class TuneConfig(BaseModel):
    """Ray Tune HPO configuration (Phase 6)."""

    num_samples: int = 50               # trials per subsystem sweep
    max_concurrent_trials: int = 2      # M1 constraint: 2 parallel numpy workers
    parallel_subsystems: bool = False   # run subsystem sweeps concurrently (cloud default)
    max_parallel_subsystems: int = 2    # cap concurrent subsystem sweeps when enabled
    mlflow_experiment_prefix: str = "hpo"   # experiments named hpo-{subsystem}
    mlflow_tracking_uri: str = "mlruns"     # file-based local default

    @field_validator("num_samples", "max_concurrent_trials", "max_parallel_subsystems")
    @classmethod
    def positive_int(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"must be >= 1, got {v}")
        return v


class _YamlConfigSource(PydanticBaseSettingsSource):
    """Reads settings from configs/{env}.yaml.

    Config directory resolution order:
        1. SPACECRAFT_CONFIG_DIR env var
        2. {repo_root}/configs/  (default)
    """

    def __init__(self, settings_cls: type[BaseSettings]) -> None:
        super().__init__(settings_cls)
        env_name = os.environ.get("SPACECRAFT_ENV", "local")
        config_dir_str = os.environ.get("SPACECRAFT_CONFIG_DIR", "")
        config_dir = Path(config_dir_str) if config_dir_str else _REPO_ROOT / "configs"
        self._path = config_dir / f"{env_name}.yaml"

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        # Not used — __call__ returns the full config dict directly.
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        if not self._path.exists():
            return {}
        with self._path.open() as f:
            return yaml.safe_load(f) or {}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SPACECRAFT_",
        env_nested_delimiter="__",
    )

    env: str = "local"
    data: DataConfig = DataConfig()
    logging: LoggingConfig = LoggingConfig()
    spark: SparkConfig = SparkConfig()
    feast: FeastConfig = FeastConfig()
    model: ModelConfig = ModelConfig()
    ray: RayConfig = RayConfig()
    tune: TuneConfig = TuneConfig()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (init_settings, env_settings, _YamlConfigSource(settings_cls))


def load_settings(env: str | None = None) -> Settings:
    """Load settings for the given environment.

    Args:
        env: Environment name ('local', 'cloud', 'test'). Defaults to
             SPACECRAFT_ENV env var, or 'local' if not set.

    Returns:
        Populated Settings instance. Env vars override YAML values.
    """
    env_name = env or os.environ.get("SPACECRAFT_ENV", "local")

    # Set SPACECRAFT_ENV so _YamlConfigSource picks up the right file,
    # then restore the original value to avoid leaking state between calls.
    previous = os.environ.get("SPACECRAFT_ENV")
    os.environ["SPACECRAFT_ENV"] = env_name
    try:
        return Settings()
    finally:
        if previous is None:
            os.environ.pop("SPACECRAFT_ENV", None)
        else:
            os.environ["SPACECRAFT_ENV"] = previous
