"""Artifact I/O for Telemanom model files.

ALL reads and writes of model files go through _write_bytes / _read_bytes.
NEVER call path.write_bytes(...), torch.save(path), or np.save(path, ...) directly
in training.py or scoring.py.

Phase 4: local filesystem only.
Phase 5 swap: widen _write_bytes / _read_bytes to branch on "gs://" URI prefixes
and delegate to gcsfs/fsspec. That is a one-file change — callers stay untouched.
"""

from __future__ import annotations

import io as _io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from spacecraft_telemetry.core.config import ModelConfig, Settings

if TYPE_CHECKING:
    import torch

    from spacecraft_telemetry.model.architecture import TelemanomLSTM


# ---------------------------------------------------------------------------
# Indirection helpers — the only place that touches the filesystem
# ---------------------------------------------------------------------------


def _write_bytes(path: Path | str, data: bytes) -> None:
    """Write bytes to path, creating parent directories as needed.

    Phase 5 swap point: add a branch here to handle gs:// URIs via gcsfs.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)


def _read_bytes(path: Path | str) -> bytes:
    """Read bytes from path.

    Phase 5 swap point: add a branch here to handle gs:// URIs via gcsfs.
    """
    return Path(path).read_bytes()


# ---------------------------------------------------------------------------
# Artifact path layout
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelArtifactPaths:
    """Paths for all artifacts produced by training and scoring a single channel."""

    root: Path | str
    model: Path | str       # model.pt         — serialized state dict
    config: Path | str      # model_config.json — architecture hyperparams
    norm: Path | str        # normalization_params.json — copied from Spark output
    errors: Path | str      # errors.npy        — per-window prediction errors (scoring)
    threshold: Path | str   # threshold.json    — threshold series + config (scoring)
    metrics: Path | str     # metrics.json      — precision/recall/F0.5 (scoring)
    train_log: Path | str   # train_log.json    — per-epoch train/val losses (training)


def artifact_paths(settings: Settings, mission: str, channel: str) -> ModelArtifactPaths:
    """Return the canonical artifact paths for a mission/channel pair."""
    root = settings.model.artifacts_dir / mission / channel
    return ModelArtifactPaths(
        root=root,
        model=root / "model.pt",
        config=root / "model_config.json",
        norm=root / "normalization_params.json",
        errors=root / "errors.npy",
        threshold=root / "threshold.json",
        metrics=root / "metrics.json",
        train_log=root / "train_log.json",
    )


# ---------------------------------------------------------------------------
# Model save / load
# ---------------------------------------------------------------------------


def save_model(
    model: "TelemanomLSTM",
    paths: ModelArtifactPaths,
    model_config: ModelConfig,
    window_size: int,
) -> None:
    """Serialize model weights and architecture config via BytesIO, then write.

    Weights are serialized to an in-memory BytesIO buffer before being passed
    to _write_bytes so the same function works for gs:// URIs in Phase 5.
    Architecture hyperparams are saved separately so load_model can reconstruct
    the correct TelemanomLSTM regardless of the current Settings.model defaults.
    window_size is saved so scoring can validate that loaded Parquet matches
    the sequence length the model was trained on.
    """
    import torch

    buf = _io.BytesIO()
    torch.save(model.state_dict(), buf)
    _write_bytes(paths.model, buf.getvalue())

    config_bytes = json.dumps(
        {
            "hidden_dim": model_config.hidden_dim,
            "num_layers": model_config.num_layers,
            "dropout": model_config.dropout,
            "window_size": window_size,
        },
        indent=2,
    ).encode()
    _write_bytes(paths.config, config_bytes)


def load_model(
    paths: ModelArtifactPaths,
    device: "torch.device",
) -> "tuple[TelemanomLSTM, ModelConfig, int]":
    """Reconstruct TelemanomLSTM from saved config and weights.

    Uses the saved model_config.json (not current Settings.model) so a model
    trained at hidden_dim=80 loads correctly after config defaults change.

    Returns:
        model:        Reconstructed TelemanomLSTM with weights loaded.
        model_config: Architecture hyperparams from the saved JSON.
        window_size:  Sequence length the model was trained on — callers should
                      validate that loaded Parquet data matches this value.
    """
    import torch

    from spacecraft_telemetry.model.architecture import build_model

    saved = json.loads(_read_bytes(paths.config).decode())
    model_config = ModelConfig(
        hidden_dim=saved["hidden_dim"],
        num_layers=saved["num_layers"],
        dropout=saved["dropout"],
    )
    window_size: int = saved["window_size"]
    model = build_model(model_config)

    buf = _io.BytesIO(_read_bytes(paths.model))
    state_dict = torch.load(buf, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)

    return model, model_config, window_size


# ---------------------------------------------------------------------------
# Normalization params copy (called by training.py after save_model)
# ---------------------------------------------------------------------------


def save_norm_params(
    paths: ModelArtifactPaths,
    processed_data_dir: Path | str,
    mission: str,
    channel: str,
) -> None:
    """Copy the per-channel normalization params into the model artifact directory.

    Reads {processed_data_dir}/{mission}/normalization_params.json, extracts
    the entry for `channel`, and writes it to paths.norm so the artifact
    directory is self-contained for Phase 9 serving (no dependency on the
    upstream Spark output layout at inference time).

    Raises:
        FileNotFoundError: If normalization_params.json doesn't exist.
        KeyError:          If `channel` has no entry in the params file.
    """
    source = Path(processed_data_dir) / mission / "normalization_params.json"
    if not source.exists():
        raise FileNotFoundError(
            f"normalization_params.json not found: {source}. "
            "Run the Spark preprocessing pipeline first."
        )
    all_params: dict[str, Any] = json.loads(source.read_bytes())
    if channel not in all_params:
        raise KeyError(
            f"Channel {channel!r} not found in {source}. "
            f"Available channels: {list(all_params.keys())[:10]}..."
        )
    _write_bytes(paths.norm, json.dumps(all_params[channel], indent=2).encode())


# ---------------------------------------------------------------------------
# Auxiliary artifact saves (called by training.py and scoring.py)
# ---------------------------------------------------------------------------


def save_train_log(
    paths: ModelArtifactPaths,
    entries: list[dict[str, Any]],
) -> None:
    """Write per-epoch train/val losses to train_log.json."""
    _write_bytes(paths.train_log, json.dumps(entries, indent=2).encode())


def save_metrics(paths: ModelArtifactPaths, metrics: dict[str, Any]) -> None:
    """Write scoring metrics dict to metrics.json."""
    _write_bytes(paths.metrics, json.dumps(metrics, indent=2).encode())


def save_threshold(paths: ModelArtifactPaths, data: dict[str, Any]) -> None:
    """Write threshold series and config to threshold.json."""
    _write_bytes(paths.threshold, json.dumps(data, indent=2).encode())


def save_errors(paths: ModelArtifactPaths, errors: "Any") -> None:
    """Serialize errors numpy array via BytesIO and write to errors.npy."""
    import numpy as np

    buf = _io.BytesIO()
    np.save(buf, errors)
    _write_bytes(paths.errors, buf.getvalue())
