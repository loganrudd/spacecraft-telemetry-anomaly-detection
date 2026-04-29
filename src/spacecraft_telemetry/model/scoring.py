"""Telemanom anomaly scoring — pure numpy/pandas, no torch dependency.

Importable in Phase 9 (FastAPI serving) without a PyTorch install.
The thresholding and evaluation functions are importable without PyTorch — 
only predict() and score_channel() require it.

Pipeline:
    values, targets, is_anomaly_true = load_windowed_parquet(..., "test")
    preds    = predict(model, values, device, batch_size)
    errors   = preds - targets                       # per-window residuals
    smoothed = smooth_errors(errors, span)           # EWMA of |errors|
    thresh   = dynamic_threshold(smoothed, window, z)
    flags    = flag_anomalies(smoothed, thresh, min_run_length)
    metrics  = evaluate(is_anomaly_true, flags)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from spacecraft_telemetry.core.config import Settings
from spacecraft_telemetry.core.logging import get_logger

if TYPE_CHECKING:
    import torch

    from spacecraft_telemetry.model.architecture import TelemanomLSTM

log = get_logger(__name__)


def predict(
    model: "TelemanomLSTM",
    values: np.ndarray[Any, np.dtype[np.float32]],
    device: "torch.device",
    batch_size: int,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    """Run the LSTM forward in batches; return shape (N,) predictions."""
    import torch

    model.eval()
    n = len(values)
    predictions = np.zeros(n, dtype=np.float32)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            x = torch.from_numpy(values[start:end]).unsqueeze(-1).to(device)
            pred: torch.Tensor = model(x).squeeze(1)
            predictions[start:end] = pred.cpu().numpy()

    return predictions


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

    for s, e in zip(starts, ends):
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
    beta_sq = 0.25  # beta=0.5 → beta²=0.25 (precision weighted 2× recall)
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


def score_channel(settings: Settings, mission: str, channel: str) -> dict[str, Any]:
    """Load model + test Parquet → predict → score → persist artifacts.

    Writes errors.npy, threshold.json, metrics.json to the artifacts dir.
    Returns the metrics dict {precision, recall, f1, f0_5}.
    """
    from spacecraft_telemetry.model.dataset import load_windowed_parquet
    from spacecraft_telemetry.model.device import resolve_device
    from spacecraft_telemetry.model.io import (
        artifact_paths,
        load_model,
        save_errors,
        save_metrics,
        save_threshold,
    )

    cfg = settings.model
    device = resolve_device(cfg.device)
    paths = artifact_paths(settings, mission, channel)

    model, _, saved_window_size = load_model(paths, device)

    values, targets, is_anomaly_true = load_windowed_parquet(
        settings.spark.processed_data_dir, mission, channel, "test"
    )
    if values.shape[1] != saved_window_size:
        raise ValueError(
            f"Parquet window_size={values.shape[1]} does not match the value the "
            f"model was trained on (window_size={saved_window_size}). "
            "Re-run the Spark pipeline and re-train with consistent settings."
        )

    preds = predict(model, values, device, cfg.inference_batch_size)
    errors = preds - targets
    smoothed = smooth_errors(errors, cfg.error_smoothing_window)
    threshold = dynamic_threshold(smoothed, cfg.threshold_window, cfg.threshold_z)
    is_anomaly_pred = flag_anomalies(smoothed, threshold, cfg.threshold_min_anomaly_len)
    metrics = evaluate(is_anomaly_true, is_anomaly_pred)

    save_errors(paths, smoothed)
    save_threshold(
        paths,
        threshold,
        {"window": cfg.threshold_window, "z": cfg.threshold_z},
    )
    save_metrics(paths, metrics)

    log.info(
        "model.score.end",
        mission=mission,
        channel=channel,
        precision=round(metrics["precision"], 4),
        recall=round(metrics["recall"], 4),
        f0_5=round(metrics["f0_5"], 4),
    )

    return metrics
