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




class ModelConfig(BaseModel):
    # Identifies which model family this config applies to.
    # Used by io.load_model() and MLflow experiment / registry naming.
    # Future values: "dc_vae"
    model_type: Literal["telemanom"] = "telemanom"
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
    """Ray Tune HPO configuration (Phase 5)."""

    num_samples: int = 50               # trials per subsystem sweep
    max_concurrent_trials: int = 2      # M1 constraint: 2 parallel numpy workers
    parallel_subsystems: bool = False   # run subsystem sweeps concurrently (cloud default)
    max_parallel_subsystems: int = 2    # cap concurrent subsystem sweeps when enabled
    # Fraction of each channel's test-set windows used for HPO optimisation;
    # the remaining (1 - fraction) are the held-out final-eval portion.
    # This prevents the HPO target and the reported metric from being the same data.
    hpo_eval_fraction: float = 0.6

    @field_validator("num_samples", "max_concurrent_trials", "max_parallel_subsystems")
    @classmethod
    def positive_int(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"must be >= 1, got {v}")
        return v

    @field_validator("hpo_eval_fraction")
    @classmethod
    def fraction_in_open_unit_interval(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError(f"hpo_eval_fraction must be in (0, 1), got {v}")
        return v


class MlflowConfig(BaseModel):
    """MLflow tracking and registry configuration (Phase 6)."""

    # Default is resolved to an absolute path at import time so Ray workers
    # (which run from a session temp dir) always point to the same database.
    tracking_uri: str = f"sqlite:///{(_REPO_ROOT / 'mlflow.db').resolve()}"
    # When None, the registry falls back to tracking_uri (MLflow default behaviour).
    registry_uri: str | None = None
    # SQLite backend URI used when starting the MLflow server locally.  When
    # tracking_uri is an HTTP endpoint, mlflow-server reads this to know which
    # database file to serve.  Ignored when tracking_uri is already a SQLite URI.
    backend_store_uri: str | None = None
    # Optional prefix applied to every experiment name, e.g. "dev-" to isolate
    # development runs from production ones without a separate tracking server.
    experiment_prefix: str = ""

    @field_validator("backend_store_uri")
    @classmethod
    def _resolve_backend_store_uri(cls, v: str | None) -> str | None:
        """Resolve a relative sqlite:///relpath backend_store_uri to an absolute path."""
        if v is None:
            return v
        if not v.startswith("sqlite:///"):
            return v
        rest = v[len("sqlite:///"):]
        if rest.startswith("/"):
            return v
        resolved = (_REPO_ROOT / rest).resolve()
        return f"sqlite:///{resolved}"

    @field_validator("tracking_uri")
    @classmethod
    def _resolve_sqlite_relative_uri(cls, v: str) -> str:
        """Resolve sqlite:///relpath to sqlite:////abs/path at construction time.

        Ray workers run from a session temp directory.  Without this, a relative
        SQLite URI such as ``sqlite:///mlflow.db`` resolves to a fresh database
        in the worker's temp dir, silently discarding all logged runs.

        Absolute SQLite URIs (``sqlite:////abs/path``), HTTP(S) URIs, and
        non-SQLite schemes are passed through unchanged.
        """
        if not v.startswith("sqlite:///"):
            return v
        rest = v[len("sqlite:///"):]
        if rest.startswith("/"):
            return v  # already absolute (sqlite:////abs/path on Unix)
        # Relative path — resolve against the repo root so the database always
        # lands next to configs/ and pyproject.toml regardless of CWD.
        resolved = (_REPO_ROOT / rest).resolve()
        return f"sqlite:///{resolved}"


class MonitoringConfig(BaseModel):
    """Evidently drift monitoring configuration (Phase 7)."""

    # Fraction of features that must drift to emit a retraining trigger.
    drift_threshold: float = 0.30
    # Where reference profiles (train-split feature DataFrames) are persisted.
    reference_profiles_dir: Path = Path("monitoring/reference_profiles")
    # Where HTML drift reports are written before upload to MLflow.
    report_output_dir: Path = Path("monitoring/reports")
    # Max rows sampled from the train split when building a reference profile.
    # Keeps memory bounded for long-running channels (full train can be 100K+ rows).
    reference_sample_rows: int = 5000

    @field_validator("drift_threshold")
    @classmethod
    def threshold_in_range(cls, v: float) -> float:
        if not 0.0 < v < 1.0:
            raise ValueError(f"drift_threshold must be in (0, 1), got {v}")
        return v

    @field_validator("reference_sample_rows")
    @classmethod
    def positive_int(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"reference_sample_rows must be >= 1, got {v}")
        return v


class ApiConfig(BaseModel):
    """FastAPI serving configuration (Phase 8)."""

    host: str = "127.0.0.1"
    port: int = 8000
    mission: str = "ESA-Mission1"
    subsystem: str = "subsystem_6"
    # Explicit channel list; when non-empty, overrides subsystem discovery.
    channels: list[str] = []
    replay_speed_default: float = 10.0
    replay_tick_interval_seconds: float = 1.0
    stream_buffer_max_events: int = 256
    request_timeout_seconds: int = 30

    @field_validator("port")
    @classmethod
    def port_in_range(cls, v: int) -> int:
        # Port 0 is valid — the OS assigns an available port at bind time (useful in tests).
        if not (0 <= v <= 65535):
            raise ValueError(f"port must be in [0, 65535], got {v}")
        return v

    @field_validator("replay_speed_default")
    @classmethod
    def speed_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"replay_speed_default must be > 0, got {v}")
        return v

    @field_validator("replay_tick_interval_seconds")
    @classmethod
    def tick_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"replay_tick_interval_seconds must be > 0, got {v}")
        return v

    @field_validator("stream_buffer_max_events", "request_timeout_seconds")
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
    model: ModelConfig = ModelConfig()
    ray: RayConfig = RayConfig()
    tune: TuneConfig = TuneConfig()
    mlflow: MlflowConfig = MlflowConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    api: ApiConfig = ApiConfig()

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
