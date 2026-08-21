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

from spacecraft_telemetry.core.config import ModelConfig, Settings
from spacecraft_telemetry.core.logging import get_logger
from spacecraft_telemetry.core.metadata import load_channel_subsystem_map
from spacecraft_telemetry.core.paths import output_path
from spacecraft_telemetry.mlflow_tracking import (
    common_tags,
    configure_mlflow,
    experiment_name,
    group_partition_hash,
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
    shuffle). For a multivariate model (docs/plans/021-multivariate-telemanom.md)
    both are (N, C) — unchanged code path: ``model(x)`` is already (B, C) and
    ``.squeeze(1)`` is a no-op whenever C > 1, so nothing here needed to move.

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
            pred: torch.Tensor = model(x)
            # Squeeze ONLY the univariate single-step case, where the model
            # emits (B, 1) and every pre-021 caller expects (B,). A blanket
            # .squeeze(1) would also collapse the channel axis of a univariate
            # MULTI-step output (B, 1, H) -> (B, H), silently disagreeing with
            # its (B, 1, F) targets. Explicit beats incidental here.
            if pred.ndim == 2 and pred.shape[1] == 1:
                pred = pred.squeeze(1)
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
    min_error_value: float = 0.0,
) -> np.ndarray[Any, np.dtype[np.bool_]]:
    """Boolean anomaly flags; contiguous runs shorter than min_run_length are dropped.

    A single-tick spike or brief noise burst (run length < min_run_length) is
    zeroed out. Only sustained exceedances are returned as True.

    ``min_error_value`` is an *absolute* floor on the smoothed error, applied
    before the run-length filter: a window below the floor is never anomalous
    no matter how far it exceeds the (relative) dynamic threshold. This is what
    ESA-ADB calls "pruning" (Telemanom-ESA-Pruned; original Telemanom hardcodes
    0.05 in errors.py#L339). It is pointwise and stateless, so the streaming
    serving path applies it identically — unlike Hundman §3.3
    :func:`prune_anomalies`, which is retrospective and batch-only.

    Rationale: a purely relative threshold flags numerically tiny errors on
    quiet channels as multi-sigma events. A floor removes those while leaving
    genuine large-error excursions untouched — which raising ``threshold_z``
    cannot do, since it suppresses real excursions on quiet channels too.

    ``min_error_value=0.0`` (the default) disables the floor and preserves the
    original behaviour exactly.
    """
    raw = smoothed > threshold
    if min_error_value > 0.0:
        raw = raw & (smoothed >= min_error_value)
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


def collapse_forecast_errors(
    errors: np.ndarray[Any, Any],
    reduction: str,
) -> np.ndarray[Any, Any]:
    """Collapse a multi-step error tensor's horizon axis: (..., H) -> (...).

    The whole downstream pipeline — EWMA smoothing, dynamic threshold,
    run-length flagging, esa_adb interval reconstruction — assumes exactly ONE
    error per window per channel. Plan 021's discipline is to change the
    forecaster only, so the H forecast errors collapse here, at the forecaster
    boundary, rather than by generalising the scorer.

    Reductions (all on |error|, matching smooth_errors' own use of magnitude):

    - ``"mean"``  every forecast step contributes, so a fault that only becomes
      visible at longer lead times still raises the score.
    - ``"first"`` the 1-step-ahead error alone. Reproduces the forecast_steps=1
      error series exactly, which makes a longer horizon a pure ablation of the
      TRAINING signal: the model learns from H-step supervision but is scored
      identically to the H=1 model.
    - ``"max"``   worst error over the horizon; most sensitive, and noisiest.

    A 1-D or 2-D input (single-step models) is returned unchanged, so this is a
    no-op on every pre-021.7 path rather than a branch its callers must guard.
    """
    if errors.ndim < 3:
        return errors
    if reduction == "first":
        result: np.ndarray[Any, Any] = np.abs(errors[..., 0])
    elif reduction == "max":
        result = np.abs(errors).max(axis=-1)
    elif reduction == "mean":
        result = np.abs(errors).mean(axis=-1)
    else:
        raise ValueError(
            f"Unknown forecast_error_reduction {reduction!r}; "
            "expected one of 'mean', 'first', 'max'."
        )
    return result


def _score_series(
    errors: np.ndarray[Any, Any],
    is_anomaly: np.ndarray[Any, Any],
    cfg: ModelConfig,
    eval_split: Literal["full_test", "hpo_portion", "final_portion"],
    hpo_eval_fraction: float,
) -> tuple[dict[str, float], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Run the existing scoring pipeline on one channel's 1-D error series.

    Shared by score_channel's univariate and multivariate paths — the Design
    section of docs/plans/021-multivariate-telemanom.md: "only the forecaster
    becomes multivariate. The error, threshold, and evaluation path stay
    exactly as they are." For a multivariate group, score_channel calls this
    once per channel with column i of the joint model's (N, C) error matrix —
    smooth_errors/dynamic_threshold/flag_anomalies/prune_anomalies/evaluate/
    evaluate_overlap are all untouched by this plan.

    Returns:
        metrics:   headline (un-pruned) + offline pruned-ceiling keys, same
                   shape score_channel has always returned for one channel.
        smoothed:  full smoothed-error array (for errors.npy — always the
                   full array regardless of eval_split, pre-021 behaviour).
        threshold: full threshold array (for threshold.npy).
    """
    smoothed = smooth_errors(errors, cfg.error_smoothing_window)
    threshold = dynamic_threshold(smoothed, cfg.threshold_window, cfg.threshold_z)
    # Headline flags are UN-pruned — see score_channel's module-level comment
    # on train/serve parity; pruning is an offline ceiling only.
    flags_raw = flag_anomalies(
        smoothed, threshold, cfg.threshold_min_anomaly_len, cfg.min_error_value
    )
    flags_pruned = prune_anomalies(smoothed, flags_raw, cfg.prune_min_decrease)

    n = len(is_anomaly)
    n_hpo = int(n * hpo_eval_fraction)
    if eval_split == "hpo_portion":
        _sl = slice(None, n_hpo)
    elif eval_split == "final_portion":
        _sl = slice(n_hpo, None)
    else:  # "full_test"
        _sl = slice(None, None)
    eval_true = is_anomaly[_sl]
    eval_raw = flags_raw[_sl]
    eval_pruned = flags_pruned[_sl]

    metrics: dict[str, float] = {
        **evaluate(eval_true, eval_raw),
        **evaluate_overlap(eval_true, eval_raw),
    }
    _ceiling = evaluate_overlap(eval_true, eval_pruned)
    metrics["pruned_seg_precision"] = _ceiling["seg_precision"]
    metrics["pruned_seg_recall"] = _ceiling["seg_recall"]
    metrics["pruned_seg_f0_5"] = _ceiling["seg_f0_5"]
    metrics["pruned_n_pred_seqs"] = _ceiling["n_pred_seqs"]
    return metrics, smoothed, threshold


def _check_training_contract(
    contract: Any,
    cfg: ModelConfig,
    name: str,
) -> None:
    """Refuse to score a model under a configuration it was not trained with.

    ``window_size`` has been guarded at this boundary since Phase 3; plan 021
    introduced two more attributes with exactly the same property — get them
    wrong and inference still *runs*, producing a plausible number that means
    nothing:

    - ``forecast_steps`` decides the dataloader's target geometry and the
      model head's output rank. Scoring an H=10 model under H=1 settings
      compares a (C, 10) head's output against 1-step targets.
    - ``input_channels`` ORDER decides which output column is which channel.
      A reorder mislabels every per-channel metric while every shape still
      matches — plan 021 calls this "a catastrophic, silent failure mode."

    Order-sensitivity is the point: train and score both derive the group from
    channels.csv today, so the orders agree by construction. This guard is what
    makes that a guarantee instead of a coincidence.

    Args:
        contract: model.io.ModelContract for the version just loaded.
        cfg:      The ModelConfig scoring is about to run with.
        name:     Registered model name, for the error message.

    Raises:
        ValueError: On any mismatch, before inference runs.
    """
    if cfg.forecast_steps != contract.forecast_steps:
        raise ValueError(
            f"settings.model.forecast_steps={cfg.forecast_steps} does not match the "
            f"value model {name!r} was trained with "
            f"(forecast_steps={contract.forecast_steps}). The forecast head emits "
            f"{contract.forecast_steps} step(s) per channel, so scoring under a "
            "different horizon compares predictions against the wrong targets. "
            "Set SPACECRAFT_MODEL__FORECAST_STEPS to the trained value, or re-train."
        )

    # A version registered before plan 021 carries no `channels`, which is an
    # accurate statement that it is univariate — not missing metadata. Scoring
    # such a model univariately (input_channels unset) is correct and must keep
    # working; scoring it as a GROUP is the mismatch worth catching.
    trained_group = list(contract.channels) if contract.channels else None
    configured_group = list(cfg.input_channels) if cfg.input_channels else None
    if trained_group != configured_group:
        raise ValueError(
            f"settings.model.input_channels={configured_group} does not match the "
            f"channel group model {name!r} was trained on ({trained_group}). "
            "This comparison is ORDER-SENSITIVE: the model's output columns are "
            "positional, so a reordered group would silently attribute every "
            "channel's errors to the wrong channel. Score with the trained order, "
            "or re-train on the order you want."
        )


def score_channel(
    settings: Settings,
    mission: str,
    channel: str,
    *,
    eval_split: Literal["full_test", "hpo_portion", "final_portion"] = "full_test",
    parent_hpo_run_id: str | None = None,
    tuned_source: str | None = None,
    data_source: Literal["nominal", "injected"] = "nominal",
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
        tuned_source:      Provenance for tuned params that did NOT come from a
                           Ray Tune trial (e.g. an exhaustive grid — see
                           scripts/threshold_ceiling.py). Recorded as the
                           ``tuned_source`` tag. Consumers that ask "was this
                           scored with tuned params or Hundman defaults?"
                           (esa_adb.detections.find_scoring_run) must treat
                           EITHER this or ``tuned_from_run`` as tuned —
                           otherwise a grid-selected run is silently
                           misclassified as the untuned baseline, which is
                           exactly the protocol-matched row the ESA-ADB report
                           compares against the paper.
        parent_hpo_run_id: MLflow run ID of the HPO trial that produced the
                           scoring params (used to set ``tuned_from_run`` tag).
        data_source:       "nominal" (default) or "injected" — recorded as the
                           ``data_source`` tag so ray_fanout.tune can locate a
                           channel's nominal baseline run independently of its
                           injected-data run (used by the false-positive-rate
                           penalty in the HPO objective; see ray_fanout/tune.py).

    Multivariate (docs/plans/021-multivariate-telemanom.md): when
    ``settings.model.input_channels`` is set, ``channel`` is the joint
    model's registry key (e.g. a subsystem name), and the return value is
    ``{channel_id: metrics_dict, ...}`` — one entry per channel in the
    group, each with the exact keys documented below. This is a per-channel
    breakdown, not a mission-level aggregate: the OR-aggregated mission-level
    metric is computed by the caller (ray_fanout/tune.py — see plan stage
    021.4b), not here, so the univariate contract stays exactly what it was.

    Returns:
        Univariate: a flat metrics dict over the selected eval portion.
        Multivariate: ``{channel_id: <the same flat dict>, ...}``.

        Headline keys are computed on the UN-pruned pipeline (serving parity):
        point {precision, recall, f1, f0_5, n_true_positive_labels,
        n_predicted_positive_labels} and segment-overlap {seg_precision,
        seg_recall, seg_f1, seg_f0_5, n_true_seqs, n_pred_seqs}. Offline
        pruned-ceiling keys (Hundman §3.3, not produced by serving):
        {pruned_seg_precision, pruned_seg_recall, pruned_seg_f0_5,
        pruned_n_pred_seqs}.
    """
    from spacecraft_telemetry.model.dataset import (
        check_channel_key_pairing,
        make_test_dataloader,
    )
    from spacecraft_telemetry.model.device import resolve_device
    from spacecraft_telemetry.model.io import (
        errors_artifact,
        errors_to_bytes,
        load_model_contract,
        load_model_for_scoring,
        threshold_artifact,
        threshold_to_bytes,
    )

    cfg = settings.model
    check_channel_key_pairing(cfg, channel)
    device = resolve_device(cfg.device)

    # Guard so a misconfigured tracking URI never aborts scoring (open_run is
    # also guarded, but configure_mlflow must not raise before we even get there).
    with suppress(Exception):
        configure_mlflow(settings)

    # Load model from MLflow registry (single source of truth post-A1 pivot).
    # require_champion=False: scoring is a training-pipeline step, not serving;
    # we need to score a model before deciding whether to promote it.
    log.info("model.score.start", channel=channel, mission=mission, device=str(device))
    name = registered_model_name(cfg.model_type, mission, channel, settings.variant)
    model, saved_window_size = load_model_for_scoring(
        name, device, settings.mlflow.tracking_uri, require_champion=False
    )
    if cfg.window_size != saved_window_size:
        raise ValueError(
            f"settings.model.window_size={cfg.window_size} does not match the "
            f"value the model was trained on (window_size={saved_window_size}). "
            "Re-train with consistent settings."
        )
    _check_training_contract(
        load_model_contract(name, settings.mlflow.tracking_uri), cfg, name
    )
    log.info("model.score.model_loaded", channel=channel)

    loader, _target_timestamps, is_anomaly = make_test_dataloader(
        settings, mission, channel
    )
    log.info("model.score.dataloader_ready", channel=channel, n_windows=len(is_anomaly))
    preds, targets = predict(model, loader, device, channel=channel)
    log.info("model.score.predict_done", channel=channel)
    errors = preds - targets

    # Multi-step horizon (021.7): errors are (N, C, H) — or (N, 1, H) when
    # univariate — and collapse to one value per window per channel BEFORE any
    # of the scoring pipeline runs. No-op for single-step models.
    if errors.ndim >= 3:
        errors = collapse_forecast_errors(errors, cfg.forecast_error_reduction)
        log.info(
            "model.score.forecast_collapsed",
            channel=channel,
            forecast_steps=cfg.forecast_steps,
            reduction=cfg.forecast_error_reduction,
        )
        # A univariate multi-step model leaves a length-1 channel axis behind;
        # drop it so the downstream univariate branch sees the (N,) it expects.
        if not cfg.input_channels and errors.ndim == 2 and errors.shape[1] == 1:
            errors = errors[:, 0]

    # Multivariate (docs/plans/021-multivariate-telemanom.md): `channel` is
    # the joint model's registry key (a subsystem name); `errors`/`targets`/
    # `is_anomaly` are (N, C). Extract column i as channel group[i]'s error
    # series and run the SAME per-channel pipeline as the univariate case —
    # "only the forecaster becomes multivariate" (Design). Univariate is not
    # a degenerate C=1 loop iteration; it stays the exact pre-021 call.
    is_multivariate = cfg.input_channels is not None
    group = list(cfg.input_channels) if cfg.input_channels else [channel]

    # Untyped-Any dict, populated below rather than reassigned per branch —
    # keeps a single settled static type instead of a per-branch union.
    metrics: dict[str, Any] = {}
    _artifact_writes: list[tuple[bytes, str]]
    if is_multivariate:
        _artifact_writes = []
        for i, ch in enumerate(group):
            ch_metrics, ch_smoothed, ch_threshold = _score_series(
                errors[:, i], is_anomaly[:, i], cfg, eval_split, settings.tune.hpo_eval_fraction
            )
            metrics[ch] = ch_metrics
            _artifact_writes.append((errors_to_bytes(ch_smoothed), errors_artifact(ch)))
            _artifact_writes.append(
                (threshold_to_bytes(ch_threshold), threshold_artifact(ch))
            )
    else:
        _flat_metrics, _smoothed, _threshold = _score_series(
            errors, is_anomaly, cfg, eval_split, settings.tune.hpo_eval_fraction
        )
        metrics.update(_flat_metrics)
        _artifact_writes = [
            (errors_to_bytes(_smoothed), errors_artifact()),
            (threshold_to_bytes(_threshold), threshold_artifact()),
        ]
    _threshold_config_bytes = json.dumps(
        {"window": cfg.threshold_window, "z": cfg.threshold_z}, indent=2
    ).encode()

    # Subsystem lookup — best-effort metadata; never breaks scoring on failure.
    # load_channel_subsystem_map is in core.metadata (no ray_fanout dep).
    # Multivariate: `channel` already IS the subsystem key.
    _subsystem: str | None = channel if is_multivariate else None
    if not is_multivariate:
        with suppress(Exception):
            _subsystem = load_channel_subsystem_map(settings, mission).get(channel)

    _extra: dict[str, str] = {"eval_split": eval_split, "data_source": data_source}
    if parent_hpo_run_id is not None:
        _extra["tuned_from_run"] = parent_hpo_run_id
    if tuned_source is not None:
        _extra["tuned_source"] = tuned_source
    if is_multivariate:
        _extra["channels"] = ",".join(group)

    _exp = experiment_name(cfg.model_type, "scoring", mission, settings.variant)
    _tags = common_tags(
        model_type=cfg.model_type,
        mission=mission,
        phase="scoring",
        variant=settings.variant,
        # A multivariate run's `channel` is a subsystem key, not a real
        # channel_id — see model/training.py's identical distinction.
        channel=None if is_multivariate else channel,
        subsystem=_subsystem,
        extra=_extra,
    )

    # Hash the test partition(s) for the Dataset column — best-effort;
    # failure is expected for GCS paths in local dev where the partition is
    # not cached.
    _eval_hash: str | None = None
    with suppress(Exception):
        if is_multivariate:
            _eval_hash = group_partition_hash(
                settings.preprocess.processed_data_dir, mission, group, "test",
                variant=settings.variant,
            )
        else:
            _eval_hash = partition_hash(
                settings.preprocess.processed_data_dir, mission, channel, "test",
                variant=settings.variant,
            )

    # The CPU forward pass above can run tens of minutes for a large channel —
    # long enough to outlive the GCP ID token fetched by configure_mlflow at the
    # top of this task. Refresh before the logging block so the artifact/metric
    # writes below don't 401 at the tail (mirrors the per-epoch refresh in
    # training; no-op for local SQLite backends).
    refresh_mlflow_auth()
    with open_run(experiment=_exp, run_name=channel, tags=_tags):
        # Log the test partition(s) as the evaluation dataset so the Dataset
        # column in the MLflow UI records which data produced these scores.
        if is_multivariate:
            _test_source = "; ".join(
                str(
                    output_path(
                        settings.preprocess.processed_data_dir, mission, settings.variant,
                        "test", f"mission_id={mission}", f"channel_id={ch}",
                    )
                )
                for ch in group
            )
        else:
            _test_source = str(
                output_path(
                    settings.preprocess.processed_data_dir, mission, settings.variant,
                    "test", f"mission_id={mission}", f"channel_id={channel}",
                )
            )
        log_input_dataset(
            source=_test_source,
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
            # Absolute error floor (ESA-ADB's "pruning"); 0.0 = disabled.
            # Logged so esa_adb/detections.py can reconstruct flags exactly.
            "min_error_value": cfg.min_error_value,
            "eval_split": eval_split,
            # 021.7: two scoring runs off the SAME H=10 model differ only by
            # this, so without it the runs are indistinguishable in MLflow and
            # the mean-vs-first comparison cannot be reconstructed later.
            "forecast_steps": cfg.forecast_steps,
            "forecast_error_reduction": cfg.forecast_error_reduction,
        })
        if is_multivariate:
            # log_metrics_final wants a flat {name: float} dict; MLflow has no
            # native nesting for scalar metrics, so per-channel keys are
            # prefixed rather than logged as one channel's worth per call —
            # one open_run per group keeps the run count == model count.
            log_metrics_final({
                f"{ch}.{k}": v for ch, ch_metrics in metrics.items() for k, v in ch_metrics.items()
            })
        else:
            log_metrics_final(metrics)
        for _data, _artifact_file in _artifact_writes:
            log_artifact_bytes(_data, _artifact_file)
        log_artifact_bytes(_threshold_config_bytes, "threshold_config.json")
        log_artifact_bytes(
            json.dumps(metrics, indent=2).encode(),
            "metrics/metrics.json",
        )

    if is_multivariate:
        log.info(
            "model.score.end",
            mission=mission,
            channel=channel,
            eval_split=eval_split,
            channels=group,
            f0_5_by_channel={ch: round(m["f0_5"], 4) for ch, m in metrics.items()},
        )
    else:
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
