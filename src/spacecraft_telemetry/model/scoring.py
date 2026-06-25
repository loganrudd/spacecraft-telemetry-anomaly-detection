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
from spacecraft_telemetry.core.paths import to_upath
from spacecraft_telemetry.mlflow_tracking import (
    common_tags,
    configure_mlflow,
    experiment_name,
    log_artifact_bytes,
    log_input_dataset,
    log_metrics_final,
    log_params,
    open_run,
    partition_hash,
    refresh_mlflow_auth,
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
    *,
    channel: str | None = None,
    log_every: int = 200,
) -> tuple[np.ndarray[Any, np.dtype[np.float32]], np.ndarray[Any, np.dtype[np.float32]]]:
    """Run the LSTM forward over all batches; return (predictions, targets).

    Both arrays are shape (N,) float32 and in DataLoader iteration order
    (i.e. the same order as the window index, since the test loader does not
    shuffle).

    Emits a ``batch X / total`` log line every ``log_every`` batches so a long
    CPU inference pass (≈1M test windows/channel) is observable in the worker
    logs rather than appearing hung. ``channel`` tags the line so concurrent
    tasks on one node stay disambiguated. Pass ``log_every=0`` to silence.
    """
    import torch

    model.eval()
    all_preds: list[np.ndarray[Any, np.dtype[np.float32]]] = []
    all_targets: list[np.ndarray[Any, np.dtype[np.float32]]] = []

    try:
        n_batches = len(loader)
    except TypeError:
        n_batches = -1  # unsized loader; report -1 rather than failing

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(device)
            pred: torch.Tensor = model(x).squeeze(1)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.numpy())
            if log_every and (i + 1) % log_every == 0:
                log.info(
                    "model.score.predict.progress",
                    channel=channel,
                    batch=i + 1,
                    total=n_batches,
                )

    if not all_preds:
        raise ValueError(
            f"predict() got an empty DataLoader — no test windows for channel {channel!r}. "
            "The test split likely has no segments long enough for the model's window_size. "
            "Check for LOS fragmentation or reduce window_size."
        )
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


def _find_sequences(arr: np.ndarray[Any, Any]) -> list[tuple[int, int]]:
    """Return (start, end) half-open intervals of contiguous True runs."""
    padded = np.concatenate(([False], arr.astype(bool), [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    return list(zip(starts, ends, strict=False))


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
    for s, e in _find_sequences(raw):
        if e - s >= min_run_length:
            result[s:e] = True
    return result


def prune_anomalies(
    smoothed: np.ndarray[Any, Any],
    is_anomaly_pred: np.ndarray[Any, Any],
    p: float,
) -> np.ndarray[Any, np.dtype[np.bool_]]:
    """Hundman §3.3 false-positive pruning by relative peak-error decrease.

    For each flagged sequence, take its peak smoothed error. Sort peaks
    descending and append the max *non-flagged* error as a noise baseline.
    Walking down the chain, a sequence is marked for removal whenever the
    step-to-step percent decrease is below ``p`` — but any decrease >= ``p``
    *resets* the removal set. The net effect: everything below the last
    significant drop (the long tail blending into the noise floor) is pruned,
    while sequences separated from the noise by a clear gap are kept.

    ``p == 0`` disables pruning (returns the input unchanged), which keeps the
    pre-pruning behaviour reproducible for ablation.

    Args:
        smoothed:        Full smoothed-error array.
        is_anomaly_pred: Boolean flags from flag_anomalies().
        p:               Minimum relative decrease to treat a step as a real
                         gap (Hundman default 0.13). Must be in [0, 1).

    Returns:
        A pruned copy of is_anomaly_pred.
    """
    is_anomaly_pred = np.asarray(is_anomaly_pred).astype(bool)
    if p <= 0.0:
        return is_anomaly_pred

    seqs = _find_sequences(is_anomaly_pred)
    if not seqs:
        return is_anomaly_pred

    peaks = np.array([smoothed[s:e].max() for s, e in seqs], dtype=np.float64)
    order = np.argsort(peaks)[::-1]  # sequence indices, highest peak first

    non_flagged = smoothed[~is_anomaly_pred]
    baseline = float(non_flagged.max()) if non_flagged.size else 0.0
    chain = np.append(peaks[order], baseline)

    to_remove: list[int] = []
    for i in range(len(chain) - 1):
        if chain[i] <= 0.0:
            continue
        decrease = (chain[i] - chain[i + 1]) / chain[i]
        if decrease < p:
            to_remove.append(int(order[i]))
        else:
            to_remove = []  # a real gap — everything above it stays

    result = is_anomaly_pred.copy()
    for idx in to_remove:
        s, e = seqs[idx]
        result[s:e] = False
    return result


def evaluate(
    is_anomaly: np.ndarray[Any, Any],
    is_anomaly_pred: np.ndarray[Any, Any],
) -> dict[str, float]:
    """Point-level P/R/F1/F0.5 from binary anomaly label arrays.

    F0.5 weights precision twice as much as recall — appropriate for spacecraft
    telemetry where false alarms are more costly than missed detections.
    """
    tp = int(np.sum(is_anomaly & is_anomaly_pred))
    fp = int(np.sum(~is_anomaly & is_anomaly_pred))
    fn = int(np.sum(is_anomaly & ~is_anomaly_pred))

    n_true_positive_labels = int(np.sum(is_anomaly))
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


def evaluate_overlap(
    is_anomaly: np.ndarray[Any, Any],
    is_anomaly_pred: np.ndarray[Any, Any],
) -> dict[str, float]:
    """Segment-overlap P/R/F1/F0.5 per Hundman 2018 §3.4.

    Counts TP/FP/FN at the sequence level, not the timestep level:
      - Recall:    fraction of true anomaly sequences overlapped by any prediction.
      - Precision: fraction of predicted sequences that overlap any true sequence.

    This matches Hundman's reported numbers. The point-level evaluate() is kept
    alongside as a stricter, less optimistic companion metric.
    """
    true_seqs = _find_sequences(is_anomaly)
    pred_seqs = _find_sequences(is_anomaly_pred)

    def _overlaps_any(s: int, e: int, candidates: list[tuple[int, int]]) -> bool:
        return any(cs < e and ce > s for cs, ce in candidates)

    n_true = len(true_seqs)
    n_pred = len(pred_seqs)

    tp_recall = sum(1 for s, e in true_seqs if _overlaps_any(s, e, pred_seqs))
    tp_prec = sum(1 for s, e in pred_seqs if _overlaps_any(s, e, true_seqs))

    recall = tp_recall / n_true if n_true > 0 else 0.0
    precision = tp_prec / n_pred if n_pred > 0 else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    beta_sq = 0.25
    f0_5 = (
        (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)
        if (beta_sq * precision + recall) > 0
        else 0.0
    )

    return {
        "seg_precision": precision,
        "seg_recall": recall,
        "seg_f1": f1,
        "seg_f0_5": f0_5,
        "n_true_seqs": float(n_true),
        "n_pred_seqs": float(n_pred),
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
        Metrics dict over the selected eval portion. Headline keys are computed
        on the UN-pruned pipeline (serving parity): point {precision, recall,
        f1, f0_5, n_true_positive_labels, n_predicted_positive_labels} and
        segment-overlap {seg_precision, seg_recall, seg_f1, seg_f0_5,
        n_true_seqs, n_pred_seqs}. Offline pruned-ceiling keys (Hundman §3.3,
        not produced by serving): {pruned_seg_precision, pruned_seg_recall,
        pruned_seg_f0_5, pruned_n_pred_seqs}.
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
    # require_champion=False: scoring is a training-pipeline step, not serving;
    # we need to score a model before deciding whether to promote it.
    log.info("model.score.start", channel=channel, mission=mission, device=str(device))
    name = registered_model_name(cfg.model_type, mission, channel)
    model, saved_window_size = load_model_for_scoring(
        name, device, settings.mlflow.tracking_uri, require_champion=False
    )
    if cfg.window_size != saved_window_size:
        raise ValueError(
            f"settings.model.window_size={cfg.window_size} does not match the "
            f"value the model was trained on (window_size={saved_window_size}). "
            "Re-train with consistent settings."
        )
    log.info("model.score.model_loaded", channel=channel)

    loader, _target_timestamps, is_anomaly = make_test_dataloader(
        settings, mission, channel
    )
    log.info("model.score.dataloader_ready", channel=channel, n_windows=len(is_anomaly))
    preds, targets = predict(model, loader, device, channel=channel)
    log.info("model.score.predict_done", channel=channel)
    errors = preds - targets
    smoothed = smooth_errors(errors, cfg.error_smoothing_window)
    threshold = dynamic_threshold(smoothed, cfg.threshold_window, cfg.threshold_z)
    # Headline flags are UN-pruned: this is exactly what the online serving
    # engine (api/inference.py) produces tick-by-tick. Hundman §3.3 pruning is
    # a retrospective batch op the streaming path cannot replicate, so we keep
    # train/serve parity by reporting the un-pruned pipeline as the headline and
    # the pruned result only as an offline "ceiling" (see docs/architecture/
    # online-pruning-investigation.md for the path to online pruning).
    flags_raw = flag_anomalies(smoothed, threshold, cfg.threshold_min_anomaly_len)
    flags_pruned = prune_anomalies(smoothed, flags_raw, cfg.prune_min_decrease)

    # Slice true/pred labels for the reported metrics.
    # errors.npy is saved from the full smoothed array below, unaffected.
    n = len(is_anomaly)
    n_hpo = int(n * settings.tune.hpo_eval_fraction)
    if eval_split == "hpo_portion":
        _sl = slice(None, n_hpo)
    elif eval_split == "final_portion":
        _sl = slice(n_hpo, None)
    else:  # "full_test"
        _sl = slice(None, None)
    eval_true = is_anomaly[_sl]
    eval_raw = flags_raw[_sl]
    eval_pruned = flags_pruned[_sl]

    # Headline = serving-parity (un-pruned). Ceiling = offline pruned report.
    metrics = {**evaluate(eval_true, eval_raw), **evaluate_overlap(eval_true, eval_raw)}
    _ceiling = evaluate_overlap(eval_true, eval_pruned)
    metrics["pruned_seg_precision"] = _ceiling["seg_precision"]
    metrics["pruned_seg_recall"] = _ceiling["seg_recall"]
    metrics["pruned_seg_f0_5"] = _ceiling["seg_f0_5"]
    metrics["pruned_n_pred_seqs"] = _ceiling["n_pred_seqs"]

    # Serialise numpy artifacts for MLflow logging (all writes inside open_run).
    _errors_bytes = errors_to_bytes(smoothed)
    _threshold_bytes = threshold_to_bytes(threshold)
    _threshold_config_bytes = json.dumps(
        {"window": cfg.threshold_window, "z": cfg.threshold_z}, indent=2
    ).encode()

    # Subsystem lookup — best-effort metadata; never breaks scoring on failure.
    # load_channel_subsystem_map is in core.metadata (no ray_fanout dep).
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

    # Hash the test partition for the Dataset column — best-effort; failure is
    # expected for GCS paths in local dev where the partition is not cached.
    _eval_hash: str | None = None
    with suppress(Exception):
        _eval_hash = partition_hash(
            settings.preprocess.processed_data_dir, mission, channel, "test"
        )

    # The CPU forward pass above can run tens of minutes for a large channel —
    # long enough to outlive the GCP ID token fetched by configure_mlflow at the
    # top of this task. Refresh before the logging block so the artifact/metric
    # writes below don't 401 at the tail (mirrors the per-epoch refresh in
    # training; no-op for local SQLite backends).
    refresh_mlflow_auth()
    with open_run(experiment=_exp, run_name=channel, tags=_tags):
        # Log the test partition as the evaluation dataset so the Dataset
        # column in the MLflow UI records which data produced these scores.
        log_input_dataset(
            source=str(
                to_upath(settings.preprocess.processed_data_dir)
                / mission / "test"
                / f"mission_id={mission}"
                / f"channel_id={channel}"
            ),
            name=f"{mission}-{channel}-test",
            digest=_eval_hash,
            context="evaluation",
        )
        log_params({
            "error_smoothing_window": cfg.error_smoothing_window,
            "threshold_window": cfg.threshold_window,
            "threshold_z": cfg.threshold_z,
            "threshold_min_anomaly_len": cfg.threshold_min_anomaly_len,
            # Affects only the offline pruned-ceiling metrics, not the headline.
            "prune_min_decrease": cfg.prune_min_decrease,
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
