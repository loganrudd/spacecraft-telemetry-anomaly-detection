"""Telemanom LSTM training loop for a single telemetry channel.

Entrypoint: train_channel(settings, mission, channel) -> TrainingResult

Follows the Explore → Plan → Execute discipline:
- Reads per-timestep series Parquet written by the Phase 2 preprocessing pipeline (Plan 002.5).
- Trains a TelemanomLSTM with early stopping on the val split.
- Logs model artifacts, normalization params, and per-epoch losses to MLflow.
  MLflow is the single source of truth — no filesystem artifact copies.

All artifact writes go through MLflow logging APIs (mlflow.log_dict,
log_artifact_bytes, mlflow.pytorch.log_model via register_pytorch_model).
Never call path.write_bytes() or torch.save(path) directly here.
"""

from __future__ import annotations

import copy
import json
from contextlib import nullcontext, suppress
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from spacecraft_telemetry.core.config import Settings
from spacecraft_telemetry.core.logging import get_logger
from spacecraft_telemetry.core.metadata import load_channel_subsystem_map
from spacecraft_telemetry.core.paths import to_upath
from spacecraft_telemetry.mlflow_tracking import (
    common_tags,
    configure_mlflow,
    experiment_name,
    log_dict,
    log_input_dataset,
    log_metrics_final,
    log_metrics_step,
    log_params,
    open_run,
    refresh_mlflow_auth,
    register_pytorch_model,
    registered_model_name,
    training_data_hash,
)
from spacecraft_telemetry.model.architecture import build_model
from spacecraft_telemetry.model.dataset import make_dataloaders
from spacecraft_telemetry.model.device import resolve_device

log = get_logger(__name__)


@dataclass(frozen=True)
class TrainingResult:
    """Summary of a completed training run."""

    train_losses: list[float]
    val_losses: list[float]
    best_epoch: int
    best_val_loss: float
    epochs_run: int


def train_channel(
    settings: Settings,
    mission: str,
    channel: str,
) -> TrainingResult:
    """Train a TelemanomLSTM on one channel and persist artifacts.

    Steps:
    1. Seed RNG for reproducibility (per-call, not at module import — Ray-safe).
    2. Load train Parquet → numpy → DataLoaders.
    3. Build model + Adam optimizer + MSE loss, move to resolved device.
    4. Train/val loop with early stopping on val loss.
    5. Restore best-epoch weights, save via model.io.
    6. Log run, per-epoch metrics, final metrics, and model version to MLflow.

    Args:
        settings: Fully resolved Settings (env vars + YAML).
        mission:  Mission name, e.g. "ESA-Mission1".
        channel:  Channel ID, e.g. "channel_1".

    Returns:
        TrainingResult with per-epoch losses and best-epoch metadata.
    """
    torch.manual_seed(settings.model.seed)
    np.random.seed(settings.model.seed)
    # One thread per worker — Ray spawns one process per channel; without this,
    # BLAS thread pools compound to 100+ threads across concurrent tasks.
    torch.set_num_threads(1)

    device = resolve_device(settings.model.device)
    cfg = settings.model

    # configure_mlflow mutates process-global state; guard so a misconfigured
    # tracking URI never aborts the training loop (open_run is also guarded).
    with suppress(Exception):
        configure_mlflow(settings)

    log.info(
        "model.train.start",
        mission=mission,
        channel=channel,
        device=str(device),
        epochs=cfg.epochs,
        hidden_dim=cfg.hidden_dim,
    )

    train_loader, val_loader = make_dataloaders(settings, mission, channel)

    model = build_model(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    loss_fn = nn.MSELoss()
    # BF16 autocast on CUDA only — bfloat16 has float32's exponent range so no
    # overflow risk. MPS stays float32: float16's limited range causes nan in the
    # LSTM cell state after 250 sequential timesteps with untrained weights.
    _amp_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )
    _io_dtype: torch.dtype | None = None

    train_losses: list[float] = []
    val_losses: list[float] = []
    best_val_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    patience_counter = 0

    # Subsystem lookup — best-effort metadata; never breaks training on failure.
    # load_channel_subsystem_map is in core.metadata (no ray_fanout dep).
    _subsystem: str | None = None
    with suppress(Exception):
        _subsystem = load_channel_subsystem_map(settings, mission).get(channel)

    _data_hash: str | None = None
    with suppress(Exception):
        _data_hash = training_data_hash(settings.preprocess.processed_data_dir, mission, channel)

    _exp = experiment_name(cfg.model_type, "training", mission)
    _tags = common_tags(
        model_type=cfg.model_type,
        mission=mission,
        phase="training",
        channel=channel,
        subsystem=_subsystem,
        training_data_hash=_data_hash,
    )

    with open_run(experiment=_exp, run_name=channel, tags=_tags) as _run:
        # Log the train partition as the run's input dataset so the Dataset
        # column in the MLflow UI shows which data produced this model.
        log_input_dataset(
            source=str(
                to_upath(settings.preprocess.processed_data_dir)
                / mission / "train"
                / f"mission_id={mission}"
                / f"channel_id={channel}"
            ),
            name=f"{mission}-{channel}-train",
            digest=_data_hash,
            context="training",
        )
        log_params({
            "model_type": cfg.model_type,
            "hidden_dim": cfg.hidden_dim,
            "num_layers": cfg.num_layers,
            "dropout": cfg.dropout,
            "learning_rate": cfg.learning_rate,
            "batch_size": cfg.batch_size,
            "window_size": cfg.window_size,
            "prediction_horizon": cfg.prediction_horizon,
            "early_stopping_patience": cfg.early_stopping_patience,
            "seed": cfg.seed,
        })

        for epoch in range(cfg.epochs):
            # --- train pass ---
            model.train()
            epoch_train_loss = 0.0
            n_train = 0
            for x, y in train_loader:
                x = x.to(device=device, dtype=_io_dtype)
                y = y.to(device=device, dtype=_io_dtype)
                optimizer.zero_grad(set_to_none=True)
                with _amp_ctx:
                    pred = model(x).squeeze(1)
                    loss = loss_fn(pred, y)
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item() * len(x)
                n_train += len(x)
            epoch_train_loss /= n_train

            # --- val pass ---
            model.eval()
            epoch_val_loss = 0.0
            n_val = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device=device, dtype=_io_dtype)
                    y = y.to(device=device, dtype=_io_dtype)
                    with _amp_ctx:
                        pred = model(x).squeeze(1)
                        epoch_val_loss += loss_fn(pred, y).item() * len(x)
                    n_val += len(x)
            epoch_val_loss /= n_val

            train_losses.append(epoch_train_loss)
            val_losses.append(epoch_val_loss)

            log.info(
                "model.train.epoch",
                mission=mission,
                channel=channel,
                epoch=epoch,
                train_loss=round(epoch_train_loss, 6),
                val_loss=round(epoch_val_loss, 6),
            )
            refresh_mlflow_auth()
            log_metrics_step(
                {"train_loss": epoch_train_loss, "val_loss": epoch_val_loss},
                step=epoch,
            )

            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= cfg.early_stopping_patience:
                    log.info(
                        "model.train.early_stop",
                        mission=mission,
                        channel=channel,
                        triggered_at_epoch=epoch,
                        best_epoch=best_epoch,
                    )
                    break

        # Restore best-epoch weights before saving.
        if best_state is not None:
            model.load_state_dict(best_state)

        log_metrics_final({
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "epochs_run": len(train_losses),
        })

        # Log normalization params for this channel.
        _norm_src = (
            to_upath(settings.preprocess.processed_data_dir)
            / mission / "normalization_params.json"
        )
        _all_norm = json.loads(_norm_src.read_bytes())
        log_dict(_all_norm[channel], "normalization_params.json")

        # Log per-epoch loss history.
        log_dict(
            [
                {"epoch": i, "train_loss": tl, "val_loss": vl}
                for i, (tl, vl) in enumerate(zip(train_losses, val_losses, strict=False))
            ],
            "train_log.json",
        )

        # Register model in MLflow registry (calls mlflow.pytorch.log_model internally).
        # Mirror compact lineage onto the model version so it is readable without
        # following a run link: window_size, mission, channel, and the data hash.
        if _run is not None:
            _vtags: dict[str, str] = {
                "window_size": str(cfg.window_size),
                "mission_id": mission,
                "channel_id": channel,
            }
            if _data_hash is not None:
                _vtags["training_data_hash"] = _data_hash
            register_pytorch_model(
                model=model,
                name=registered_model_name(cfg.model_type, mission, channel),
                run_id=_run.info.run_id,
                source_run_model_name=channel,
                version_tags=_vtags,
            )

    log.info(
        "model.train.end",
        mission=mission,
        channel=channel,
        best_epoch=best_epoch,
        best_val_loss=round(best_val_loss, 6),
        epochs_run=len(train_losses),
    )

    return TrainingResult(
        train_losses=train_losses,
        val_losses=val_losses,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        epochs_run=len(train_losses),
    )
