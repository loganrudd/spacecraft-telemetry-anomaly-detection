"""Ray Tune HPO sweep for Telemanom scoring parameters (Phase 6).

Tunes all 4 scoring parameters (error_smoothing_window, threshold_window,
threshold_z, threshold_min_anomaly_len) without re-training any LSTM models.
Each trial is a single pure-numpy scoring pass (~50ms), so FIFOScheduler is
used rather than ASHA — ASHA's pruning requires intermediate checkpoints that
don't exist for single-pass trials.

Channels are grouped by spacecraft subsystem before sweeping. Individual
channels have only 2-5 anomaly events (too few for stable F0.5 optimisation);
pooling channels within a subsystem (6-30 channels each) gives a robust signal.

Public API
----------
SEARCH_SPACE         Ray Tune search space over the 4 scoring parameters.
run_hpo_sweep        Run one Tune experiment for a named subsystem.
run_all_sweeps       Group channels by subsystem, run all sweeps, write JSON.
write_tuned_configs  Persist subsystem → best_config mapping as JSON.
"""

from __future__ import annotations

import concurrent.futures
import json
import warnings
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np
import ray
from ray import tune

from spacecraft_telemetry.core.config import Settings
from spacecraft_telemetry.core.logging import get_logger
from spacecraft_telemetry.mlflow_tracking.conventions import (
    experiment_name as _mlflow_experiment_name,
)
from spacecraft_telemetry.core.metadata import load_channel_subsystem_map
from spacecraft_telemetry.model.io import (
    bytes_to_errors,
    download_artifact_bytes,
    find_latest_run_for_channel,
)
from spacecraft_telemetry.ray_training.runner import _with_abs_paths

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Search space — 4 scoring params only; no model architecture changes.
# Upper bounds are exclusive for randint; inclusive for uniform.
# ---------------------------------------------------------------------------

SEARCH_SPACE: dict[str, Any] = {
    "error_smoothing_window":    tune.randint(5, 101),   # EWMA span: [5, 100]
    "threshold_window":          tune.randint(50, 501),  # rolling window: [50, 500]
    "threshold_z":               tune.uniform(1.5, 5.0), # z-score multiplier
    "threshold_min_anomaly_len": tune.randint(1, 11),    # min run length: [1, 10]
}


def _prepare_channel_data(
    settings: Settings,
    mission: str,
    channels: list[str],
) -> dict[str, tuple[np.ndarray[Any, Any], np.ndarray[Any, np.dtype[np.bool_]]]]:
    """Load, validate, and temporally slice per-channel trial inputs.

    Reads smoothed errors for each channel from the most recent MLflow scoring
    run (``errors.npy`` artifact).  Channels with no scoring run in MLflow are
    skipped.  Raises when no usable channel remains, or when labels/errors
    shapes are incompatible.

    Emits ``UserWarning`` when either the HPO or held-out portion contains no
    labeled anomaly windows across all channels — both cases produce misleading
    F0.5 scores (always 0) that would silently corrupt HPO or final eval.
    """
    from spacecraft_telemetry.model.dataset import load_window_labels

    missing_errors: list[str] = []
    shape_mismatches: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    load_failures: list[tuple[str, str]] = []
    prepared: dict[str, tuple[np.ndarray[Any, Any], np.ndarray[Any, np.dtype[np.bool_]]]] = {}

    _scoring_exp = _mlflow_experiment_name(settings.model.model_type, "scoring", mission)

    for channel in channels:
        # Find the most recent scoring run for this channel in MLflow.
        _run = find_latest_run_for_channel(
            _scoring_exp, channel, settings.mlflow.tracking_uri
        )
        if _run is None:
            missing_errors.append(channel)
            continue
        try:
            raw = download_artifact_bytes(
                _run.info.run_id, "errors.npy", settings.mlflow.tracking_uri
            )
        except (OSError, Exception):
            missing_errors.append(channel)
            continue

        errors: np.ndarray[Any, Any] = bytes_to_errors(raw)
        try:
            labels = load_window_labels(settings, mission, channel)
        except (FileNotFoundError, OSError, ValueError, RuntimeError) as exc:
            load_failures.append((channel, str(exc)))
            continue

        if labels.shape != errors.shape:
            shape_mismatches.append((channel, tuple(labels.shape), tuple(errors.shape)))
            continue

        prepared[channel] = (errors, labels)

    if shape_mismatches:
        details = "; ".join(
            f"{ch}: labels{lbl} vs errors{err}"
            for ch, lbl, err in shape_mismatches[:3]
        )
        raise ValueError(
            "run_hpo_sweep input mismatch: load_window_labels() shape does not "
            f"match saved errors.npy for {len(shape_mismatches)} channel(s). "
            "This usually means errors were scored with a different settings profile "
            "(window_size/prediction_horizon) than the one passed to tune. "
            f"Examples: {details}."
        )

    if not prepared:
        msg = (
            "run_hpo_sweep has no usable channels: all channels are missing "
            "a scoring run in MLflow or label loading failed."
        )
        if missing_errors:
            msg += f" Missing errors for {len(missing_errors)} channel(s)."
        if load_failures:
            first = load_failures[0]
            msg += f" First label-load failure: {first[0]} -> {first[1]}"
        raise ValueError(msg)

    # Slice each channel to the HPO portion; held-out tail is reserved for
    # final eval in score_channel (Step 5).
    fraction = settings.tune.hpo_eval_fraction
    hpo_anomaly_count = 0
    held_out_anomaly_count = 0
    for ch in list(prepared.keys()):
        errors, labels = prepared[ch]
        n_hpo = int(len(errors) * fraction)
        hpo_anomaly_count += int(labels[:n_hpo].sum())
        held_out_anomaly_count += int(labels[n_hpo:].sum())
        prepared[ch] = (errors[:n_hpo], labels[:n_hpo])

    if hpo_anomaly_count == 0:
        warnings.warn(
            f"HPO portion (first {fraction:.0%} of test data) has no labeled anomaly "
            "windows across all channels in this subsystem. F0.5 will always be 0 "
            "during HPO — consider increasing hpo_eval_fraction or checking labels.",
            UserWarning,
            stacklevel=3,
        )
    if held_out_anomaly_count == 0:
        warnings.warn(
            f"Held-out portion (last {1 - fraction:.0%} of test data) has no labeled "
            "anomaly windows across all channels in this subsystem. Final evaluation "
            "will always score 0 — consider decreasing hpo_eval_fraction or checking labels.",
            UserWarning,
            stacklevel=3,
        )

    return prepared


# ---------------------------------------------------------------------------
# Trial function — pure numpy, no torch dependency
# ---------------------------------------------------------------------------

def _scoring_trial(
    config: dict[str, Any],
    *,
    channel_data: dict[str, tuple[np.ndarray[Any, Any], np.ndarray[Any, np.dtype[np.bool_]]]],
) -> dict[str, float]:
    """Ray Tune trial: score all channels in a subsystem group with config params.

    Each trial applies the scoring pipeline with the sampled hyperparameters
    over pre-loaded immutable channel data and returns mean F0.5.

    No model re-training occurs — only the numpy scoring pass is repeated.
    Returns a dict rather than calling ray.train.report() so that the function
    body has no ray imports. Ray Tune 2.x records a function trainable's return
    value as the trial's final metrics, so get_best_result() works correctly.

    Args:
        config:   Dict of sampled hyperparameter values from SEARCH_SPACE.
        channel_data: Mapping channel -> (errors, labels), pre-validated and
            pre-sliced to the HPO portion by ``_prepare_channel_data``.  Never
            contains the held-out eval tail.
    """
    from spacecraft_telemetry.model.scoring import (
        dynamic_threshold,
        evaluate,
        flag_anomalies,
        smooth_errors,
    )

    f0_5_scores: list[float] = []
    for errors, labels in channel_data.values():

        # Scoring pipeline — identical to score_channel() in scoring.py.
        smoothed = smooth_errors(errors, int(config["error_smoothing_window"]))
        threshold = dynamic_threshold(
            smoothed,
            int(config["threshold_window"]),
            float(config["threshold_z"]),
        )
        flags = flag_anomalies(smoothed, threshold, int(config["threshold_min_anomaly_len"]))
        metrics = evaluate(labels, flags)
        f0_5_scores.append(metrics["f0_5"])

    mean_f0_5 = float(np.mean(f0_5_scores)) if f0_5_scores else 0.0
    return {"f0_5": mean_f0_5}


# ---------------------------------------------------------------------------
# Single subsystem sweep
# ---------------------------------------------------------------------------

def run_hpo_sweep(
    subsystem: str,
    channels: list[str],
    settings: Settings,
    mission: str,
) -> dict[str, Any]:
    """Run a Tune experiment for one subsystem; return result dict.

    Uses HyperOptSearch (Bayesian optimisation) with FIFOScheduler. Each trial
    is a single scoring pass (~50ms) so ASHA's early-stopping is a no-op here.
    All trials land in a single per-mission MLflow experiment
    (``{model_type}-hpo-{mission}``) with subsystem stored as a tag so that
    training, scoring, and HPO runs for a mission are browseable together.

    Args:
        subsystem: Subsystem name, e.g. "subsystem_1".
        channels:  Channel IDs belonging to this subsystem.
        settings:  Fully resolved Settings.
        mission:   Mission name, e.g. "ESA-Mission1".

    Returns:
        Dict with keys:
        - ``"config"``  — best scoring-param dict (4 keys from SEARCH_SPACE).
        - ``"f0_5"``    — best F0.5 achieved over the HPO portion.
        - ``"run_id"``  — MLflow run ID of the best trial (None if unavailable).
    """
    from ray.air.integrations.mlflow import MLflowLoggerCallback
    from ray.tune.schedulers import FIFOScheduler
    from ray.tune.search.hyperopt import HyperOptSearch

    if not ray.is_initialized():
        raise RuntimeError(
            "Ray is not initialized. Initialize Ray in the caller (CLI _ray_session "
            "or test fixture) before calling run_hpo_sweep()."
        )

    cfg = settings.model
    _exp_name = _mlflow_experiment_name(cfg.model_type, "hpo", mission)

    log.info(
        "tune.sweep.start",
        subsystem=subsystem,
        n_channels=len(channels),
        num_samples=settings.tune.num_samples,
        max_concurrent_trials=settings.tune.max_concurrent_trials,
    )

    # Resolve relative paths before serialising settings — Ray Tune workers run
    # from a temp directory where relative paths would not resolve.
    settings_abs = _with_abs_paths(settings)

    # Load immutable per-channel data once and validate compatibility before
    # launching Tune trials.
    channel_data = _prepare_channel_data(settings_abs, mission, channels)

    trial_fn = tune.with_parameters(
        _scoring_trial,
        channel_data=channel_data,
    )

    tuner = tune.Tuner(
        trial_fn,
        param_space=SEARCH_SPACE,
        tune_config=tune.TuneConfig(
            metric="f0_5",
            mode="max",
            num_samples=settings.tune.num_samples,
            max_concurrent_trials=settings.tune.max_concurrent_trials,
            search_alg=HyperOptSearch(metric="f0_5", mode="max"),
            scheduler=FIFOScheduler(),  # type: ignore[no-untyped-call]
        ),
        run_config=tune.RunConfig(
            name=f"hpo-{subsystem}",
            verbose=0,
            callbacks=[
                MLflowLoggerCallback(
                    experiment_name=_exp_name,
                    tracking_uri=settings.mlflow.tracking_uri,
                    tags={
                        "subsystem": subsystem,
                        "eval_split": "hpo_portion",
                        "model_type": cfg.model_type,
                        "mission_id": mission,
                        "phase": "hpo",
                    },
                    save_artifact=False,
                ),
            ],
        ),
    )

    results = tuner.fit()
    best = results.get_best_result(metric="f0_5", mode="max")

    best_config: dict[str, Any] = best.config or {}
    best_f0_5: float = (best.metrics or {}).get("f0_5", 0.0)

    # Look up the MLflow run ID for the best trial so score_channel can record
    # lineage via the tuned_from_run tag (Step 5 / runner.py).
    best_run_id: str | None = None
    with suppress(Exception):
        import mlflow as _mlflow
        _client = _mlflow.tracking.MlflowClient(tracking_uri=settings.mlflow.tracking_uri)
        _exp = _client.get_experiment_by_name(_exp_name)
        if _exp:
            _runs = _client.search_runs(
                [_exp.experiment_id],
                filter_string=f"tags.subsystem = '{subsystem}'",
                order_by=["metrics.f0_5 DESC"],
                max_results=1,
            )
            if _runs:
                best_run_id = _runs[0].info.run_id
    if best_run_id is None:
        log.warning("tune.sweep.best_run_id_missing", subsystem=subsystem,
                    note="lineage tag tuned_from_run will be absent on scoring runs")

    log.info(
        "tune.sweep.end",
        subsystem=subsystem,
        best_f0_5=round(best_f0_5, 4),
        best_config=best_config,
        best_run_id=best_run_id,
    )
    return {"config": best_config, "f0_5": best_f0_5, "run_id": best_run_id}


# ---------------------------------------------------------------------------
# All sweeps + output
# ---------------------------------------------------------------------------

def run_all_sweeps(
    settings: Settings,
    mission: str,
    channels: list[str],
) -> Path:
    """Group channels by subsystem, run HPO for each, write tuned_configs.json.

    Subsystems where no channel has an errors.npy artifact are skipped. This
    allows the sweep to run cleanly when only a subset of channels have been
    scored by Phase 5 (e.g. local dev with 1-5 channels).

    Args:
        settings: Fully resolved Settings.
        mission:  Mission name, e.g. "ESA-Mission1".
        channels: Channel IDs to consider (typically from discover_channels).

    Returns:
        Path to the written tuned_configs.json file.
    """
    if not ray.is_initialized():
        raise RuntimeError(
            "Ray is not initialized. Call run_all_sweeps from a caller that owns "
            "the Ray session (CLI _ray_session or test fixture)."
        )

    subsystem_map = load_channel_subsystem_map(settings, mission)

    # Group provided channels by subsystem.
    by_subsystem: dict[str, list[str]] = {}
    for ch in channels:
        sub = subsystem_map.get(ch)
        if sub is None:
            log.warning("tune.all_sweeps.no_subsystem", channel=ch)
            continue
        by_subsystem.setdefault(sub, []).append(ch)

    # Retain only subsystems that have ≥1 scored channel (scoring run in MLflow).
    _scoring_exp = _mlflow_experiment_name(settings.model.model_type, "scoring", mission)
    eligible: dict[str, list[str]] = {}
    for sub, sub_channels in by_subsystem.items():
        scored = [
            ch
            for ch in sub_channels
            if find_latest_run_for_channel(
                _scoring_exp, ch, settings.mlflow.tracking_uri
            ) is not None
        ]
        if scored:
            eligible[sub] = scored
        else:
            log.info(
                "tune.all_sweeps.skip_subsystem",
                subsystem=sub,
                reason="no scoring run found in MLflow",
            )

    output = Path(settings.model.artifacts_dir) / mission / "tuned_configs.json"

    if not eligible:
        log.warning("tune.all_sweeps.no_eligible_subsystems", mission=mission)
        write_tuned_configs({}, output)
        return output
    def _to_entry(sweep_result: dict[str, Any]) -> dict[str, Any]:
        """Convert run_hpo_sweep result to the on-disk tuned_configs entry."""
        return {
            **sweep_result.get("config", {}),
            "_meta": {
                "run_id": sweep_result.get("run_id"),
                "f0_5": sweep_result.get("f0_5", 0.0),
            },
        }

    sweep_results: dict[str, dict[str, Any]] = {}
    if settings.tune.parallel_subsystems and len(eligible) > 1:
        max_workers = min(settings.tune.max_parallel_subsystems, len(eligible))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(run_hpo_sweep, sub, sub_channels, settings, mission): sub
                for sub, sub_channels in eligible.items()
            }
            for future in concurrent.futures.as_completed(futures):
                subsystem = futures[future]
                sweep_results[subsystem] = _to_entry(future.result())
    else:
        for sub, sub_channels in eligible.items():
            sweep_results[sub] = _to_entry(run_hpo_sweep(sub, sub_channels, settings, mission))

    write_tuned_configs(sweep_results, output)
    return output


def write_tuned_configs(
    results: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    """Write subsystem → config mapping as JSON.

    Schema (written by run_all_sweeps, read by score_all_channels)::

        {
            "subsystem_1": {
                "threshold_z": 2.8, "threshold_window": 200,
                "error_smoothing_window": 25, "threshold_min_anomaly_len": 3,
                "_meta": {"run_id": "abc123...", "f0_5": 0.72}
            }
        }

    ``_meta`` is filtered out by score_all_channels before applying overrides
    (not in _TUNABLE_SCORING_FIELDS). score_channel reads ``_meta.run_id`` to
    set the ``tuned_from_run`` MLflow tag for HPO → scoring run lineage.

    Args:
        results:     Dict keyed by subsystem name → entry dict (params + _meta).
        output_path: Destination path for tuned_configs.json.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    log.info(
        "tune.configs.written",
        path=str(output_path),
        n_subsystems=len(results),
    )
