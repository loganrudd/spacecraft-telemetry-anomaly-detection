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

import io
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from ray import tune

from spacecraft_telemetry.core.config import Settings
from spacecraft_telemetry.core.logging import get_logger
from spacecraft_telemetry.ray_training.runner import (
    _load_channel_subsystem_map,
    _with_abs_paths,
)

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


def _validate_trial_inputs(settings: Settings, mission: str, channels: list[str]) -> None:
    """Fail fast when tune labels and saved errors.npy are incompatible.

    This catches the common case where errors.npy was generated with one model
    window_size (e.g. local config) and run_hpo_sweep is invoked with another
    settings profile (e.g. test config).
    """
    from spacecraft_telemetry.model.dataset import load_window_labels
    from spacecraft_telemetry.model.io import _read_bytes, artifact_paths

    missing_errors: list[str] = []
    shape_mismatches: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    load_failures: list[tuple[str, str]] = []
    n_usable = 0

    for channel in channels:
        paths = artifact_paths(settings, mission, channel)
        try:
            raw = _read_bytes(paths.errors)
        except (FileNotFoundError, OSError):
            missing_errors.append(channel)
            continue

        errors: np.ndarray[Any, Any] = np.load(io.BytesIO(raw))
        try:
            labels = load_window_labels(settings, mission, channel)
        except Exception as exc:
            load_failures.append((channel, str(exc)))
            continue

        if labels.shape != errors.shape:
            shape_mismatches.append((channel, tuple(labels.shape), tuple(errors.shape)))
            continue

        n_usable += 1

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

    if n_usable == 0:
        msg = (
            "run_hpo_sweep has no usable channels: all channels are missing "
            "errors.npy or label loading failed."
        )
        if missing_errors:
            msg += f" Missing errors for {len(missing_errors)} channel(s)."
        if load_failures:
            first = load_failures[0]
            msg += f" First label-load failure: {first[0]} -> {first[1]}"
        raise ValueError(msg)


# ---------------------------------------------------------------------------
# Trial function — pure numpy, no torch dependency
# ---------------------------------------------------------------------------

def _scoring_trial(
    config: dict[str, Any],
    *,
    settings: Settings,
    mission: str,
    channels: list[str],
) -> dict[str, float]:
    """Ray Tune trial: score all channels in a subsystem group with config params.

    Each trial loads pre-computed errors.npy artifacts (written by Phase 5),
    applies the scoring pipeline with the trial's hyperparameters, and returns
    mean F0.5 across all channels in the group as a metrics dict.

    No model re-training occurs — only the numpy scoring pass is repeated.
    Channels whose errors.npy has not yet been written are skipped with a
    warning so a partial Phase 5 state does not crash the sweep.

    Returns a dict rather than calling ray.train.report() so that the function
    body has no ray imports. Ray Tune 2.x records a function trainable's return
    value as the trial's final metrics, so get_best_result() works correctly.
    This avoids the ModuleNotFoundError that arises when trial workers are
    spawned outside the driver's virtualenv PYTHONPATH.

    Args:
        config:   Dict of sampled hyperparameter values from SEARCH_SPACE.
        settings: Resolved Settings (absolute paths, injected by tune.with_parameters).
        mission:  Mission name, e.g. "ESA-Mission1".
        channels: Channel IDs belonging to this subsystem group.
    """
    from spacecraft_telemetry.model.dataset import load_window_labels
    from spacecraft_telemetry.model.io import _read_bytes, artifact_paths
    from spacecraft_telemetry.model.scoring import (
        dynamic_threshold,
        evaluate,
        flag_anomalies,
        smooth_errors,
    )

    _log = get_logger(__name__)

    f0_5_scores: list[float] = []
    for channel in channels:
        paths = artifact_paths(settings, mission, channel)

        # Load pre-computed errors.npy — skip if Phase 5 hasn't run yet.
        try:
            raw = _read_bytes(paths.errors)
        except (FileNotFoundError, OSError):
            _log.warning("tune.trial.skip", channel=channel, reason="errors.npy missing")
            continue

        errors: np.ndarray[Any, Any] = np.load(io.BytesIO(raw))

        # Load per-window anomaly labels (no torch required).
        try:
            labels = load_window_labels(settings, mission, channel)
        except Exception as exc:
            _log.warning("tune.trial.skip", channel=channel, reason=str(exc))
            continue

        if labels.shape != errors.shape:
            _log.warning(
                "tune.trial.skip",
                channel=channel,
                reason=(
                    "labels/errors shape mismatch "
                    f"labels={labels.shape} errors={errors.shape}"
                ),
            )
            continue

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
    """Run a Tune experiment for one subsystem; return the best config dict.

    Uses HyperOptSearch (Bayesian optimisation) with FIFOScheduler. Each trial
    is a single scoring pass (~50ms) so ASHA's early-stopping is a no-op here.
    Trial results are logged to MLflow via MLflowLoggerCallback.

    Args:
        subsystem: Subsystem name, e.g. "subsystem_1".
        channels:  Channel IDs belonging to this subsystem.
        settings:  Fully resolved Settings.
        mission:   Mission name, e.g. "ESA-Mission1".

    Returns:
        Dict of the 4 scoring params from the best trial configuration.
    """
    import ray
    from ray.air.integrations.mlflow import MLflowLoggerCallback
    from ray.tune.schedulers import FIFOScheduler
    from ray.tune.search.hyperopt import HyperOptSearch

    # Ensure Ray is initialized with the venv on PYTHONPATH so that Tune
    # trial subprocess workers can import installed packages (including ray
    # itself). _ray_session (CLI) and the ray_local pytest fixture already
    # call ray.init() with the correct runtime_env; detect that case by
    # checking PYTHONPATH in the cluster env vars via a private API.
    #
    # If Ray is not yet initialized, initialize it here.
    # If Ray IS initialized but PYTHONPATH is missing (e.g. bare ray.init()
    # in a script), restart it — safe because run_hpo_sweep owns the session
    # for standalone use. _ray_session and ray_local always set PYTHONPATH,
    # so they will not trigger the restart.
    _pythonpath = os.pathsep.join(p for p in sys.path if p)
    _need_init = True
    if ray.is_initialized():
        try:
            import ray._private.worker as _w

            _serialized = (
                _w.global_worker.core_worker  # type: ignore[attr-defined]
                .get_job_config()
                .runtime_env_info.serialized_runtime_env
            )
            _env_vars: dict[str, str] = json.loads(_serialized).get("env_vars", {})
            if "PYTHONPATH" in _env_vars:
                _need_init = False
        except Exception:  # pragma: no cover — private API may change
            pass  # conservative: restart with PYTHONPATH
    if _need_init:
        if ray.is_initialized():
            ray.shutdown()
        ray.init(
            ignore_reinit_error=True,
            runtime_env={"env_vars": {"PYTHONPATH": _pythonpath}},
        )

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

    # Fail fast before launching Tune trials to avoid opaque RayTaskError
    # traces when per-window labels and saved errors.npy are incompatible.
    _validate_trial_inputs(settings_abs, mission, channels)

    trial_fn = tune.with_parameters(
        _scoring_trial,
        settings=settings_abs,
        mission=mission,
        channels=channels,
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
                    experiment_name=(
                        f"{settings.tune.mlflow_experiment_prefix}-{subsystem}"
                    ),
                    tracking_uri=settings.tune.mlflow_tracking_uri,
                    save_artifact=False,
                ),
            ],
        ),
    )

    results = tuner.fit()
    best = results.get_best_result(metric="f0_5", mode="max")

    best_config: dict[str, Any] = best.config or {}
    best_f0_5: float = (best.metrics or {}).get("f0_5", 0.0)

    log.info(
        "tune.sweep.end",
        subsystem=subsystem,
        best_f0_5=round(best_f0_5, 4),
        best_config=best_config,
    )
    return best_config


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
    from spacecraft_telemetry.model.io import artifact_paths

    subsystem_map = _load_channel_subsystem_map(settings, mission)

    # Group provided channels by subsystem.
    by_subsystem: dict[str, list[str]] = {}
    for ch in channels:
        sub = subsystem_map.get(ch)
        if sub is None:
            log.warning("tune.all_sweeps.no_subsystem", channel=ch)
            continue
        by_subsystem.setdefault(sub, []).append(ch)

    # Retain only subsystems that have ≥1 scored channel (errors.npy exists).
    eligible: dict[str, list[str]] = {}
    for sub, sub_channels in by_subsystem.items():
        scored = [
            ch
            for ch in sub_channels
            if Path(str(artifact_paths(settings, mission, ch).errors)).exists()
        ]
        if scored:
            eligible[sub] = scored
        else:
            log.info(
                "tune.all_sweeps.skip_subsystem",
                subsystem=sub,
                reason="no errors.npy found",
            )

    output = Path(settings.model.artifacts_dir) / mission / "tuned_configs.json"

    if not eligible:
        log.warning("tune.all_sweeps.no_eligible_subsystems", mission=mission)
        write_tuned_configs({}, output)
        return output

    sweep_results: dict[str, dict[str, Any]] = {}
    for sub, sub_channels in eligible.items():
        best_config = run_hpo_sweep(sub, sub_channels, settings, mission)
        sweep_results[sub] = best_config

    write_tuned_configs(sweep_results, output)
    return output


def write_tuned_configs(
    results: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    """Write subsystem → best_config mapping as JSON.

    The output schema matches what score_all_channels() reads in Phase 5b::

        {
            "subsystem_1": {"threshold_z": 2.8, "threshold_window": 200, ...},
            "subsystem_6": {"threshold_z": 3.2, "error_smoothing_window": 40, ...}
        }

    Args:
        results:     Dict keyed by subsystem name → best config dict.
        output_path: Destination path for tuned_configs.json.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    log.info(
        "tune.configs.written",
        path=str(output_path),
        n_subsystems=len(results),
    )
