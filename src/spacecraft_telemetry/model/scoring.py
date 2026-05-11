"""Telemanom anomaly scoring — pure numpy/pandas, no torch dependency.

Importable in Phase 8 (FastAPI serving) without a PyTorch install.
The thresholding and evaluation functions are importable without PyTorch —
only predict() and score_channel() require it.

Pipeline:
    loader, target_timestamps, window_is_anomaly = make_test_dataloader(...)
    preds, targets = predict(model, loader, device)
    errors   = preds - targets                       # per-window residuals
    smoothed = smooth_errors(errors, span)           # EWMA of |errors|
    thresh   = dynamic_threshold(smoothed, window, z)
    flags    = flag_anomalies(smoothed, thresh, min_run_length)
    metrics  = evaluate(window_is_anomaly, flags)

Artifact I/O: all writes go through MLflow logging APIs (log_artifact_bytes,
log_metrics_final). Never call path.write_bytes() or np.save(path, ...) here.
"""

from __future__ import annotations

import json
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from spacecraft_telemetry.core.config import Settings
from spacecraft_telemetry.core.logging import get_logger
from spacecraft_telemetry.core.metadata import load_channel_subsystem_map
from spacecraft_telemetry.mlflow_tracking import (
    common_tags,
    configure_mlflow,
    experiment_name,
    log_artifact_bytes,
    log_metrics_final,
    log_params,
    open_run,
    registered_model_name,
)

if TYPE_CHECKING:
    import torch
    from torch.utils.data import DataLoader

    from spacecraft_telemetry.model.architecture import TelemanomLSTM

log = get_logger(__name__)


def predict(
    model: TelemanomLSTM,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> tuple[np.ndarray[Any, np.dtype[np.float32]], np.ndarray[Any, np.dtype[np.float32]]]:
    """Run the LSTM forward over all batches; return (predictions, targets).

    Both arrays are shape (N,) float32 and in DataLoader iteration order
    (i.e. the same order as the window index, since the test loader does not
    shuffle).
    """
    import torch

    model.eval()
    all_preds: list[np.ndarray[Any, np.dtype[np.float32]]] = []
    all_targets: list[np.ndarray[Any, np.dtype[np.float32]]] = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            pred: torch.Tensor = model(x).squeeze(1)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.numpy())

    predictions = np.concatenate(all_preds).astype(np.float32)
    targets = np.concatenate(all_targets).astype(np.float32)
    return predictions, targets


def smooth_errors(
    errors: np.ndarray[Any, Any],
    span: int,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """EWMA of absolute per-window errors (Hundman §3.1).

    Uses the recursive (adjust=False) formula:
        alpha = 2 / (span + 1)
        s[0]  = |e[0]|
        s[t]  = alpha * |e[t]| + (1 - alpha) * s[t-1]
    """
    abs_errors = np.abs(errors).astype(np.float64)
    smoothed: np.ndarray[Any, np.dtype[np.float64]] = (
        pd.Series(abs_errors).ewm(span=span, adjust=False).mean().to_numpy(dtype=np.float64)
    )
    return smoothed


def dynamic_threshold(
    smoothed: np.ndarray[Any, Any],
    window: int,
    z: float,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Rolling mean + z*std over the previous `window` steps.

    At position t:
        threshold[t] = mean(smoothed[t-window : t]) + z * std(smoothed[t-window : t])

    Warmup behaviour (documented):
    - Position 0 has no history → threshold[0] = inf (never flags an anomaly).
    - Positions 1 .. window-1 use whatever history is available (min_periods=1),
      so the threshold tightens gradually as the window fills.
    """
    s = pd.Series(smoothed.astype(np.float64))
    rolling_mean = s.rolling(window, min_periods=1).mean()
    rolling_std = s.rolling(window, min_periods=1).std(ddof=0).fillna(0.0)
    threshold: np.ndarray[Any, np.dtype[np.float64]] = (
        (rolling_mean + z * rolling_std)
        .shift(1)
        .fillna(np.inf)
        .to_numpy(dtype=np.float64)
    )
    return threshold


def flag_anomalies(
    smoothed: np.ndarray[Any, Any],
    threshold: np.ndarray[Any, Any],
    min_run_length: int,
) -> np.ndarray[Any, np.dtype[np.bool_]]:
    """Boolean anomaly flags; contiguous runs shorter than min_run_length are dropped.

    A single-tick spike or brief noise burst (run length < min_run_length) is
    zeroed out. Only sustained exceedances are returned as True.
    """
    raw = smoothed > threshold
    result = np.zeros(len(raw), dtype=bool)

    # Detect run boundaries via diff on a padded boolean array.
    padded = np.concatenate(([False], raw, [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]

    for s, e in zip(starts, ends, strict=False):
        if e - s >= min_run_length:
            result[s:e] = True

    return result


def evaluate(
    is_anomaly_true: np.ndarray[Any, Any],
    is_anomaly_pred: np.ndarray[Any, Any],
) -> dict[str, float]:
    """Precision, recall, F1, and F0.5 from binary anomaly label arrays.

    F0.5 weights precision twice as much as recall — appropriate for spacecraft
    telemetry where false alarms are more costly than missed detections.
    """
    tp = int(np.sum(is_anomaly_true & is_anomaly_pred))
    fp = int(np.sum(~is_anomaly_true & is_anomaly_pred))
    fn = int(np.sum(is_anomaly_true & ~is_anomaly_pred))

    n_true_positive_labels = int(np.sum(is_anomaly_true))
    n_predicted_positive_labels = int(np.sum(is_anomaly_pred))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    beta_sq = 0.25  # beta=0.5 -> beta^2=0.25 (precision weighted 2x recall)
    f0_5 = (
        (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)
        if (beta_sq * precision + recall) > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f0_5": f0_5,
        "n_true_positive_labels": n_true_positive_labels,
        "n_predicted_positive_labels": n_predicted_positive_labels,
    }


def score_channel(
    settings: Settings,
    mission: str,
    channel: str,
    *,
    eval_split: Literal["full_test", "hpo_portion", "final_portion"] = "full_test",
    parent_hpo_run_id: str | None = None,
) -> dict[str, Any]:
    """Load model + test Parquet → predict → score → persist artifacts.

    Writes errors.npy, threshold.npy, threshold_config.json, and metrics.json
    to the artifacts dir using the FULL test-split pipeline (all windows).
    The reported metrics dict is computed over the portion selected by
    ``eval_split``:

    - ``"full_test"``      — all test windows (default, backward-compatible).
    - ``"hpo_portion"``    — first ``hpo_eval_fraction`` of windows (diagnostic).
    - ``"final_portion"``  — remaining windows; the held-out eval set that was
      NOT used for HPO in ``_prepare_channel_data``.

    ``errors.npy`` is always saved from the full smoothed array so that Phase 5
    HPO can keep consuming it regardless of which split was evaluated last.

    Args:
        settings:          Fully resolved Settings.
        mission:           Mission name, e.g. "ESA-Mission1".
        channel:           Channel ID, e.g. "channel_1".
        eval_split:        Which temporal slice of the test set to evaluate.
        parent_hpo_run_id: MLflow run ID of the HPO trial that produced the
                           scoring params (used to set ``tuned_from_run`` tag).

    Returns:
        Metrics dict {precision, recall, f1, f0_5, n_true_positive_labels,
        n_predicted_positive_labels} computed over the selected eval portion.
    """
    from spacecraft_telemetry.model.dataset import make_test_dataloader
    from spacecraft_telemetry.model.device import resolve_device
    from spacecraft_telemetry.model.io import (
        errors_to_bytes,
        load_model_for_scoring,
        threshold_to_bytes,
    )

    cfg = settings.model
    device = resolve_device(cfg.device)

    # Guard so a misconfigured tracking URI never aborts scoring (open_run is
    # also guarded, but configure_mlflow must not raise before we even get there).
    with suppress(Exception):
        configure_mlflow(settings)

    # Load model from MLflow registry (single source of truth post-A1 pivot).
    name = registered_model_name(cfg.model_type, mission, channel)
    model, saved_window_size = load_model_for_scoring(
        name, device, settings.mlflow.tracking_uri
    )
    if cfg.window_size != saved_window_size:
        raise ValueError(
            f"settings.model.window_size={cfg.window_size} does not match the "
            f"value the model was trained on (window_size={saved_window_size}). "
            "Re-train with consistent settings."
        )

    loader, _target_timestamps, is_anomaly_true = make_test_dataloader(
        settings, mission, channel
    )
    preds, targets = predict(model, loader, device)
    errors = preds - targets
    smoothed = smooth_errors(errors, cfg.error_smoothing_window)
    threshold = dynamic_threshold(smoothed, cfg.threshold_window, cfg.threshold_z)
    is_anomaly_pred = flag_anomalies(smoothed, threshold, cfg.threshold_min_anomaly_len)

    # Slice true/pred labels for the reported metrics.
    # errors.npy is saved from the full smoothed array below, unaffected.
    n = len(is_anomaly_true)
    n_hpo = int(n * settings.tune.hpo_eval_fraction)
    if eval_split == "hpo_portion":
        eval_true = is_anomaly_true[:n_hpo]
        eval_pred = is_anomaly_pred[:n_hpo]
    elif eval_split == "final_portion":
        eval_true = is_anomaly_true[n_hpo:]
        eval_pred = is_anomaly_pred[n_hpo:]
    else:  # "full_test"
        eval_true = is_anomaly_true
        eval_pred = is_anomaly_pred

    metrics = evaluate(eval_true, eval_pred)

    # Serialise numpy artifacts for MLflow logging (all writes inside open_run).
    _errors_bytes = errors_to_bytes(smoothed)
    _threshold_bytes = threshold_to_bytes(threshold)
    _threshold_config_bytes = json.dumps(
        {"window": cfg.threshold_window, "z": cfg.threshold_z}, indent=2
    ).encode()

    # Subsystem lookup — best-effort metadata; never breaks scoring on failure.
    # load_channel_subsystem_map is in core.metadata (no ray_training dep).
    _subsystem: str | None = None
    with suppress(Exception):
        _subsystem = load_channel_subsystem_map(settings, mission).get(channel)

    _extra: dict[str, str] = {"eval_split": eval_split}
    if parent_hpo_run_id is not None:
        _extra["tuned_from_run"] = parent_hpo_run_id

    _exp = experiment_name(cfg.model_type, "scoring", mission)
    _tags = common_tags(
        model_type=cfg.model_type,
        mission=mission,
        phase="scoring",
        channel=channel,
        subsystem=_subsystem,
        extra=_extra,
    )

    with open_run(experiment=_exp, run_name=channel, tags=_tags):
        log_params({
            "error_smoothing_window": cfg.error_smoothing_window,
            "threshold_window": cfg.threshold_window,
            "threshold_z": cfg.threshold_z,
            "threshold_min_anomaly_len": cfg.threshold_min_anomaly_len,
            "eval_split": eval_split,
        })
        log_metrics_final(metrics)
        log_artifact_bytes(_errors_bytes, "errors.npy")
        log_artifact_bytes(_threshold_bytes, "threshold.npy")
        log_artifact_bytes(_threshold_config_bytes, "threshold_config.json")
        log_artifact_bytes(
            json.dumps(metrics, indent=2).encode(),
            "metrics/metrics.json",
        )

    log.info(
        "model.score.end",
        mission=mission,
        channel=channel,
        eval_split=eval_split,
        precision=round(metrics["precision"], 4),
        recall=round(metrics["recall"], 4),
        f0_5=round(metrics["f0_5"], 4),
    )

    return metrics
