"""Ray Tune HPO sweep for Telemanom scoring parameters (Phase 5).

Tunes 4 scoring parameters (error_smoothing_window, threshold_window,
threshold_z, threshold_min_anomaly_len) without re-training any LSTM models.
The objective is segment-overlap F0.5 on the un-pruned pipeline, minus a
penalty on the false-positive rate measured against each channel's nominal
(un-injected) baseline scoring run; Hundman §3.3 pruning is intentionally
excluded to preserve train/serve parity (see SEARCH_SPACE).
Each trial is a single pure-numpy scoring pass (~50ms), so FIFOScheduler is
used rather than ASHA — ASHA's pruning requires intermediate checkpoints that
don't exist for single-pass trials.

Channels are grouped by spacecraft subsystem before sweeping. Individual
channels have only 2-5 anomaly events (too few for stable F0.5 optimisation);
pooling channels within a subsystem (6-30 channels each) gives a robust signal.

Nominal false-positive penalty (ISS Phase 15)
----------------------------------------------
seg_f0_5 alone is computed only against injected-fault labels, which are
common in the injected eval portion. A config can win on seg_f0_5 purely by
flagging *everything* — Telemanom's one-step forecaster barely reacts to slow
drift faults, so the optimizer can floor threshold_z to "detect" them, at the
cost of also firing on nominal noise. ISS replay has no real anomalies, so
the live demo is the worst case for that failure mode: an uncalibrated config
flags constantly. _load_nominal_errors() fetches each channel's nominal
(un-injected) scoring run — tagged ``data_source=nominal`` by score_channel,
see model/scoring.py and cli.py `ray score --injected` — and the objective
subtracts ``fp_penalty_weight * mean(nominal_fp_rate)`` from seg_f0_5. A
channel with no nominal-tagged run yet contributes no penalty (logged once
as a warning) rather than failing the sweep — run `ray score --mission ISS`
(no --injected) to backfill it.

Baseline guard
--------------
A sweep can still land on a config that beats the objective on the HPO
portion (first hpo_eval_fraction of test data) yet scores worse than the
untuned Settings.model defaults on the held-out final_portion — e.g. ISS
Phase 15 tuned runs landed on threshold_min_anomaly_len (K) of 7-10 to
suppress nominal false positives, which also suppressed recall on injected
faults that only sustain a few consecutive threshold exceedances (drift/
flatline ramp+hold over 30-60 ticks). run_hpo_sweep() evaluates the untuned
defaults (clamped into the sweep's own search-space bounds, e.g. the ISS
threshold_window AOS cap) with the same _scoring_trial() objective and keeps
the defaults whenever they are >= the sweep's best trial. When this happens
the returned "run_id" is None (no MLflow trial backs a kept default) — read
by score_channel() as "not tuned", which is the correct interpretation.

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
import time
import warnings
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np
import ray
from ray import tune

from spacecraft_telemetry.core.config import Settings
from spacecraft_telemetry.core.logging import get_logger
from spacecraft_telemetry.core.metadata import load_channel_subsystem_map
from spacecraft_telemetry.mlflow_tracking.conventions import (
    experiment_name as _mlflow_experiment_name,
)
from spacecraft_telemetry.mlflow_tracking.runs import (
    configure_mlflow as _configure_mlflow,
)
from spacecraft_telemetry.mlflow_tracking.runs import (
    keep_mlflow_auth_fresh as _keep_mlflow_auth_fresh,
)
from spacecraft_telemetry.mlflow_tracking.runs import (
    log_artifact_bytes as _log_artifact_bytes,
)
from spacecraft_telemetry.mlflow_tracking.runs import (
    open_run as _open_run,
)
from spacecraft_telemetry.model.io import (
    bytes_to_errors,
    download_artifact_bytes,
    find_latest_run_for_channel,
)
from spacecraft_telemetry.ray_fanout.runner import _with_abs_paths

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Search space — 4 scoring params only; no model architecture changes.
# Upper bounds are exclusive for randint; inclusive for uniform.
#
# prune_min_decrease is deliberately NOT tuned: pruning is a retrospective
# batch op the online serving path cannot replicate, so tuning it would pick a
# z that only works *with* pruning and break train/serve parity. The objective
# below is segment-overlap F0.5 on the UN-pruned pipeline — exactly what the
# serving engine produces. Pruning stays a fixed offline-report knob.
# ---------------------------------------------------------------------------

# ISS threshold_window cap.
# Originally 70 (half the ~140-bucket TDRS AOS window at the 30s grid) so a
# COLD detector's threshold ring buffer could fill and still leave detection
# time within a single pass. That cold-start rationale is obsolete: serving
# warm-starts the threshold via ChannelInferenceEngine.prime_with_scoring()
# at startup and on every LOS recovery (api/app.py, api/live/pump.py), so the
# threshold never waits for a live fill. The old tight cap actively hurt
# detection of sustained faults: a 20-120-bucket drift/flatline (the HPO
# injection population) is a large fraction of a 70-bucket rolling window, so
# the threshold inflated mid-event and absorbed the fault. Raised to 250
# (== the ModelConfig default threshold_window): a fault inflates a
# 250-bucket window far less, and priming needs window_size + 250 grid
# buckets ≈ 3.2 h of history — well within the replay slice serving primes
# from. Kept as a cap (rather than ESA's 500) to bound that priming
# requirement and the live re-prime deque (_recent_buckets in the pump).
_ISS_MAX_THRESHOLD_WINDOW = 250  # buckets at 30s grid ≈ 2.1 h

SEARCH_SPACE: dict[str, Any] = {
    "error_smoothing_window":    tune.randint(5, 101),   # EWMA span: [5, 100]
    "threshold_window":          tune.randint(50, 501),  # rolling window: [50, 500]
    # z floor raised 1.5 -> 2.5: below ~2.5 the optimizer was flooring z to chase
    # undetectable drift faults (see docs/architecture — attitude/solar_array
    # pegged to z~1.6 during ISS Phase 15 HPO, firing on nominal noise in replay).
    # The nominal-FP penalty below is the principled fix; this floor is a backstop.
    "threshold_z":               tune.uniform(2.5, 5.0), # z-score multiplier
    "threshold_min_anomaly_len": tune.randint(1, 11),    # min run length: [1, 10]
}

# ISS-specific search space: threshold_window capped at _ISS_MAX_THRESHOLD_WINDOW
# (bounds the warm-start priming requirement — see the cap's comment above).
#
# threshold_min_anomaly_len (K) is also capped much lower than ESA's [1, 10]:
# ISS injected faults (drift/flatline ramp+hold over 30-60 ticks; a one-step
# forecaster only produces a large residual during the ramp, not the hold) can
# rarely sustain K>=5 consecutive threshold exceedances. With the full ESA
# range available, the optimizer's only lever to suppress nominal false
# positives was pushing K to 7-10 — which also suppressed recall on the very
# faults it was supposed to catch (see the module docstring's Baseline guard
# section). Capping K at 4 keeps recall achievable.
ISS_SEARCH_SPACE: dict[str, Any] = {
    **SEARCH_SPACE,
    "threshold_window": tune.randint(50, _ISS_MAX_THRESHOLD_WINDOW + 1),
    "threshold_min_anomaly_len": tune.randint(1, 5),  # [1, 4], vs ESA's [1, 10]
}


def _prepare_channel_data(
    settings: Settings,
    mission: str,
    channels: list[str],
) -> tuple[
    dict[str, tuple[np.ndarray[Any, Any], np.ndarray[Any, np.dtype[np.bool_]]]],
    dict[str, str | None],
]:
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
    scoring_run_ids: dict[str, str | None] = {}

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
        # Track the scoring run ID so the caller can build a channel manifest.
        scoring_run_ids[channel] = _run.info.run_id

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

    return prepared, scoring_run_ids


def _load_nominal_errors(
    settings: Settings,
    mission: str,
    channels: list[str],
) -> dict[str, np.ndarray[Any, Any]]:
    """Load smoothed-error arrays from each channel's latest nominal scoring run.

    "Nominal" means scored against the un-injected test split — tagged
    ``data_source=nominal`` by score_channel() (the default; pass
    ``ray score --injected`` to tag the injected-data pass instead). Used as
    the false-positive-rate baseline in _scoring_trial(): a trial config that
    fires frequently on nominal data is penalized even if it scores well on
    the injected fault labels (see module docstring).

    Channels with no nominal-tagged scoring run yet are simply omitted from
    the returned dict — the caller treats a missing channel as "no penalty
    data available for this channel" rather than failing the sweep. This is
    expected the first time a mission's HPO runs after this penalty shipped;
    a baseline ``ray score --mission <mission>`` (no --injected) backfills it.
    """
    _scoring_exp = _mlflow_experiment_name(settings.model.model_type, "scoring", mission)
    out: dict[str, np.ndarray[Any, Any]] = {}
    missing: list[str] = []

    for channel in channels:
        run = find_latest_run_for_channel(
            _scoring_exp,
            channel,
            settings.mlflow.tracking_uri,
            extra_filter="tags.data_source = 'nominal'",
        )
        if run is None:
            missing.append(channel)
            continue
        try:
            raw = download_artifact_bytes(
                run.info.run_id, "errors.npy", settings.mlflow.tracking_uri
            )
        except (OSError, Exception):
            missing.append(channel)
            continue
        out[channel] = bytes_to_errors(raw)

    if missing:
        log.warning(
            "tune.nominal_baseline.missing",
            mission=mission,
            n_missing=len(missing),
            channels=missing[:10],
            note="FP penalty skipped for these channels until a nominal-tagged "
            "scoring run exists — run `ray score --mission <mission>` "
            "(no --injected) to backfill.",
        )
    return out


def _clamp_to_search_space(
    config: dict[str, Any], search_space: dict[str, Any]
) -> dict[str, Any]:
    """Clip each value in *config* into its search-space bounds.

    Used to make the untuned Settings.model defaults a legal candidate in the
    baseline guard (run_hpo_sweep) — a default outside the sweep's bounds
    (e.g. threshold_min_anomaly_len above the ISS K cap, or threshold_window
    above _ISS_MAX_THRESHOLD_WINDOW) could otherwise "win" the baseline
    comparison with a value the search space deems invalid for live serving.

    randint domains have an exclusive upper bound; uniform domains (only
    threshold_z here) have an inclusive upper bound. Config values are float
    only for threshold_z, so isinstance(value, float) is sufficient to select
    the right bound without inspecting the Ray domain type.
    """
    clamped: dict[str, Any] = {}
    for key, value in config.items():
        domain = search_space[key]
        lower, upper = domain.lower, domain.upper
        if isinstance(value, float):
            clamped[key] = min(max(value, lower), upper)
        else:
            clamped[key] = min(max(value, lower), upper - 1)
    return clamped


# ---------------------------------------------------------------------------
# Trial function — pure numpy, no torch dependency
# ---------------------------------------------------------------------------

def _scoring_trial(
    config: dict[str, Any],
    *,
    channel_data: dict[str, tuple[np.ndarray[Any, Any], np.ndarray[Any, np.dtype[np.bool_]]]],
    nominal_errors: dict[str, np.ndarray[Any, Any]],
    fp_penalty_weight: float,
) -> dict[str, float]:
    """Ray Tune trial: score all channels in a subsystem group with config params.

    Each trial applies the scoring pipeline with the sampled hyperparameters
    over pre-loaded immutable channel data and returns mean F0.5, mean nominal
    false-positive rate, and the combined objective the sweep actually
    optimizes (see module docstring for the rationale).

    No model re-training occurs — only the numpy scoring pass is repeated.
    Returns a dict rather than calling ray.train.report() so that the function
    body has no ray imports. Ray Tune 2.x records a function trainable's return
    value as the trial's final metrics, so get_best_result() works correctly.

    Args:
        config:   Dict of sampled hyperparameter values from SEARCH_SPACE.
        channel_data: Mapping channel -> (errors, labels), pre-validated and
            pre-sliced to the HPO portion by ``_prepare_channel_data``.  Never
            contains the held-out eval tail.
        nominal_errors: Mapping channel -> smoothed-error array from that
            channel's nominal (un-injected) scoring run.  Channels absent here
            (no nominal-tagged run yet) contribute no penalty term.
        fp_penalty_weight: Weight on mean nominal false-positive rate,
            subtracted from mean seg_f0_5 to form "objective".
    """
    from spacecraft_telemetry.model.scoring import (
        dynamic_threshold,
        evaluate,
        evaluate_overlap,
        flag_anomalies,
        smooth_errors,
    )

    f0_5_scores: list[float] = []
    seg_f0_5_scores: list[float] = []
    for errors, labels in channel_data.values():

        # Un-pruned pipeline — identical to the headline path in score_channel()
        # and to what the online serving engine produces. No prune step here:
        # see SEARCH_SPACE note on train/serve parity.
        smoothed = smooth_errors(errors, int(config["error_smoothing_window"]))
        threshold = dynamic_threshold(
            smoothed,
            int(config["threshold_window"]),
            float(config["threshold_z"]),
        )
        flags = flag_anomalies(smoothed, threshold, int(config["threshold_min_anomaly_len"]))
        f0_5_scores.append(evaluate(labels, flags)["f0_5"])
        seg_f0_5_scores.append(evaluate_overlap(labels, flags)["seg_f0_5"])

    fp_rates: list[float] = []
    for nom_errors in nominal_errors.values():
        nom_smoothed = smooth_errors(nom_errors, int(config["error_smoothing_window"]))
        nom_threshold = dynamic_threshold(
            nom_smoothed,
            int(config["threshold_window"]),
            float(config["threshold_z"]),
        )
        nom_flags = flag_anomalies(
            nom_smoothed, nom_threshold, int(config["threshold_min_anomaly_len"])
        )
        fp_rates.append(float(nom_flags.mean()) if len(nom_flags) else 0.0)

    # seg_f0_5 (segment-overlap) measures fault recall/precision against
    # injected labels — it matches how long ESA anomalies should be scored and
    # what serving produces. point-level f0_5 is reported alongside for
    # observability. nominal_fp_rate measures how often this config fires on
    # data with no real anomalies (the live-replay failure mode). "objective"
    # is what the sweep actually optimizes (see run_hpo_sweep).
    mean_f0_5 = float(np.mean(f0_5_scores)) if f0_5_scores else 0.0
    mean_seg_f0_5 = float(np.mean(seg_f0_5_scores)) if seg_f0_5_scores else 0.0
    mean_fp_rate = float(np.mean(fp_rates)) if fp_rates else 0.0
    objective = mean_seg_f0_5 - fp_penalty_weight * mean_fp_rate
    return {
        "f0_5": mean_f0_5,
        "seg_f0_5": mean_seg_f0_5,
        "nominal_fp_rate": mean_fp_rate,
        "objective": objective,
    }


# ---------------------------------------------------------------------------
# Single subsystem sweep
# ---------------------------------------------------------------------------

def _resilient_mlflow_callback(**kwargs: Any) -> Any:
    """Build an MLflowLoggerCallback that can't crash the sweep on a transient
    MLflow failure.

    The per-trial MLflow runs this callback creates are observability only: the
    sweep's outcome comes from each trial's returned ``objective`` (recorded by
    Ray, not MLflow), and ``run_hpo_sweep`` re-queries MLflow for the best
    trial's run_id afterwards, tolerating ``None``. Upstream Ray's callback
    (``ray.air.integrations.mlflow``) does an unguarded ``self._trial_runs[trial]``
    in ``log_trial_end``; when ``start_run()`` failed at trial start — a token
    refresh or network blip against a remote (Cloud Run) tracking server —
    the trial is never registered, so a single blip raises ``KeyError`` inside
    the trial-error handler and kills the whole RayJob. Swallow-and-log each
    hook so one flaky trial can't take down a 50-trial sweep.

    Lazy import keeps ``ray.air`` off the module-import path (matching the other
    Ray imports, which live inside ``run_hpo_sweep``).
    """
    from ray.air.integrations.mlflow import MLflowLoggerCallback

    class _ResilientMLflowLoggerCallback(MLflowLoggerCallback):
        # Broad excepts are intentional: MLflow logging is observability only and
        # must never propagate into Ray's trial-lifecycle handling.
        #
        # NOTE: Use get_logger(__name__) lazily inside each method rather than
        # closing over the module-level `log` variable.  When cloudpickle
        # serialises this locally-defined class it captures globals referenced
        # by the method bodies; the module-level structlog BoundLogger is not
        # picklable under pytest (sys.stdout is replaced by the capture
        # fixture, so PrintLogger.__reduce__ raises PicklingError).  Calling
        # get_logger() at call-time avoids including a logger instance in the
        # pickled state — only the importable get_logger function reference is
        # captured instead.
        def log_trial_start(self, trial: Any) -> None:
            try:
                super().log_trial_start(trial)
            except Exception as exc:
                get_logger(__name__).warning(
                    "tune.mlflow_callback.start_failed", trial=str(trial), error=str(exc)
                )

        def log_trial_result(self, iteration: int, trial: Any, result: dict[str, Any]) -> None:
            try:
                super().log_trial_result(iteration, trial, result)
            except Exception as exc:
                get_logger(__name__).warning(
                    "tune.mlflow_callback.result_failed", trial=str(trial), error=str(exc)
                )

        def log_trial_end(self, trial: Any, failed: bool = False) -> None:
            try:
                super().log_trial_end(trial, failed=failed)
            except Exception as exc:
                get_logger(__name__).warning(
                    "tune.mlflow_callback.end_failed", trial=str(trial), error=str(exc)
                )

    return _ResilientMLflowLoggerCallback(**kwargs)


def run_hpo_sweep(
    subsystem: str,
    channels: list[str],
    settings: Settings,
    mission: str,
    *,
    search_space: dict[str, Any] | None = None,
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
        - ``"config"``           — best scoring-param dict (4 keys from SEARCH_SPACE).
        - ``"seg_f0_5"``         — segment-overlap F0.5 of the selected trial over
          the HPO portion (un-pruned; fault-recall component only).
        - ``"nominal_fp_rate"``  — mean fraction of nominal-data windows the
          selected trial's config flags as anomalous (0.0 if no channel in this
          subsystem has a nominal-tagged baseline run yet).
        - ``"objective"``        — ``seg_f0_5 - fp_penalty_weight * nominal_fp_rate``,
          the actual quantity the sweep optimizes.
        - ``"run_id"``           — MLflow run ID of the best trial. None if
          unavailable, or if the baseline guard kept the untuned
          Settings.model defaults instead of the sweep's best trial (see
          module docstring) — in that case ``"config"`` holds the defaults
          and no MLflow trial backs them.
    """
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
    channel_data, scoring_run_ids = _prepare_channel_data(settings_abs, mission, channels)
    nominal_errors = _load_nominal_errors(settings_abs, mission, channels)

    trial_fn = tune.with_parameters(
        _scoring_trial,
        channel_data=channel_data,
        nominal_errors=nominal_errors,
        fp_penalty_weight=settings.tune.fp_penalty_weight,
    )

    # Capture the wall-clock time just before launching the sweep.  Used below
    # to scope the MLflow run search to trials from *this* sweep only, so that
    # re-runs don't accidentally return the best run from a previous sweep.
    _sweep_start_ms = int(time.time() * 1000)

    _search_space = search_space if search_space is not None else SEARCH_SPACE
    tuner = tune.Tuner(
        trial_fn,
        param_space=_search_space,
        tune_config=tune.TuneConfig(
            metric="objective",
            mode="max",
            num_samples=settings.tune.num_samples,
            max_concurrent_trials=settings.tune.max_concurrent_trials,
            search_alg=HyperOptSearch(metric="objective", mode="max"),
            scheduler=FIFOScheduler(),  # type: ignore[no-untyped-call]
        ),
        run_config=tune.RunConfig(
            name=f"hpo-{subsystem}",
            verbose=0,
            callbacks=[
                _resilient_mlflow_callback(
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
    best = results.get_best_result(metric="objective", mode="max")

    best_config: dict[str, Any] = best.config or {}
    best_metrics = best.metrics or {}
    best_seg_f0_5: float = best_metrics.get("seg_f0_5", 0.0)
    best_nominal_fp_rate: float = best_metrics.get("nominal_fp_rate", 0.0)
    best_objective: float = best_metrics.get("objective", best_seg_f0_5)

    # Baseline guard (see module docstring): the sweep optimizes hpo_portion,
    # but a winning trial there can still score worse than the untuned
    # Settings.model defaults on the objective (ISS Phase 15: tuned K landed
    # at 7-10, killing recall on injected faults). Evaluate the defaults —
    # clamped into this sweep's search-space bounds, e.g. the ISS
    # threshold_window AOS cap — with the same objective, and keep them
    # whenever they are >= the sweep's best trial, so a sweep can never make
    # detection worse than doing nothing.
    baseline_config = _clamp_to_search_space(
        {
            "error_smoothing_window": cfg.error_smoothing_window,
            "threshold_window": cfg.threshold_window,
            "threshold_z": cfg.threshold_z,
            "threshold_min_anomaly_len": cfg.threshold_min_anomaly_len,
        },
        _search_space,
    )
    baseline_metrics = _scoring_trial(
        baseline_config,
        channel_data=channel_data,
        nominal_errors=nominal_errors,
        fp_penalty_weight=settings.tune.fp_penalty_weight,
    )
    used_baseline = baseline_metrics["objective"] >= best_objective
    if used_baseline:
        log.info(
            "tune.sweep.baseline_kept",
            subsystem=subsystem,
            baseline_objective=round(baseline_metrics["objective"], 4),
            tuned_objective=round(best_objective, 4),
            note="untuned defaults matched or beat the HPO sweep on the held-out "
            "objective — keeping defaults for this subsystem instead of the "
            "sweep's best trial.",
        )
        best_config = baseline_config
        best_seg_f0_5 = baseline_metrics["seg_f0_5"]
        best_nominal_fp_rate = baseline_metrics["nominal_fp_rate"]
        best_objective = baseline_metrics["objective"]

    # Look up the MLflow run ID for the best trial so score_channel can record
    # lineage via the tuned_from_run tag (Step 5 / runner.py).
    # Scoped to runs from this sweep only (start_time >= _sweep_start_ms) so
    # that re-runs don't pick up the all-time best from a previous sweep.
    # Skipped when the baseline guard kept the untuned defaults — no MLflow
    # trial backs them, and a None run_id is the correct "not tuned" signal
    # for score_channel, not a lookup failure worth warning about.
    best_run_id: str | None = None
    if not used_baseline:
        with suppress(Exception):
            import mlflow as _mlflow
            _client = _mlflow.MlflowClient(tracking_uri=settings.mlflow.tracking_uri)
            _exp = _client.get_experiment_by_name(_exp_name)
            if _exp:
                _runs = _client.search_runs(
                    [_exp.experiment_id],
                    filter_string=(
                        f"tags.subsystem = '{subsystem}'"
                        f" and attributes.start_time >= {_sweep_start_ms}"
                    ),
                    order_by=["metrics.objective DESC"],
                    max_results=1,
                )
                if _runs:
                    best_run_id = _runs[0].info.run_id
        if best_run_id is None:
            log.warning("tune.sweep.best_run_id_missing", subsystem=subsystem,
                        note="lineage tag tuned_from_run will be absent on scoring runs")

    # Log the channel manifest (which channels and their scoring run IDs fed
    # this sweep) as an artifact on the best HPO run.  Makes the input lineage
    # of the sweep browseable from the Run Artifacts tab without a separate
    # query.  Best-effort: a failure here must not abort the sweep.
    if best_run_id is not None and scoring_run_ids:
        with suppress(Exception):
            import mlflow as _mlflow
            _manifest_client = _mlflow.MlflowClient(
                tracking_uri=settings.mlflow.tracking_uri
            )
            _manifest_client.log_dict(
                run_id=best_run_id,
                dictionary={
                    "subsystem": subsystem,
                    "hpo_eval_fraction": settings.tune.hpo_eval_fraction,
                    "channels": {
                        ch: {"scoring_run_id": rid}
                        for ch, rid in scoring_run_ids.items()
                    },
                },
                artifact_file="hpo_channel_manifest.json",
            )

    # threshold_z pegged near the search-space floor is the signature of an
    # optimizer chasing undetectable faults at the expense of nominal-data
    # precision (see module docstring) — worth a loud signal even though the
    # penalty above should now make this rare. Not applicable when the
    # baseline guard kept the untuned defaults — there's no optimizer trial
    # to diagnose.
    _z_floor = _search_space["threshold_z"].lower
    if not used_baseline and best_config.get("threshold_z", _z_floor) <= _z_floor + 0.1:
        log.warning(
            "tune.sweep.z_pegged_to_floor",
            subsystem=subsystem,
            threshold_z=best_config.get("threshold_z"),
            nominal_fp_rate=round(best_nominal_fp_rate, 4),
            note="best config sits at the threshold_z search-space floor — "
            "likely chasing undetectable faults rather than a real threshold.",
        )

    log.info(
        "tune.sweep.end",
        subsystem=subsystem,
        best_objective=round(best_objective, 4),
        best_seg_f0_5=round(best_seg_f0_5, 4),
        best_nominal_fp_rate=round(best_nominal_fp_rate, 4),
        best_config=best_config,
        best_run_id=best_run_id,
        used_baseline=used_baseline,
    )
    return {
        "config": best_config,
        "seg_f0_5": best_seg_f0_5,
        "nominal_fp_rate": best_nominal_fp_rate,
        "objective": best_objective,
        "run_id": best_run_id,
    }


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
    scored by Phase 4 (e.g. local dev with 1-5 channels).

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

    # ISS uses a narrowed space: threshold_window capped (bounds the serving
    # warm-start priming requirement) and threshold_min_anomaly_len capped
    # (short injected faults cannot sustain long consecutive-exceedance runs).
    # See the comments on _ISS_MAX_THRESHOLD_WINDOW / ISS_SEARCH_SPACE.
    # ESA uses the default space.
    _space = ISS_SEARCH_SPACE if mission.startswith("ISS") else SEARCH_SPACE
    if mission.startswith("ISS"):
        log.info(
            "tune.all_sweeps.iss_search_space",
            max_threshold_window=_ISS_MAX_THRESHOLD_WINDOW,
        )

    def _to_entry(sweep_result: dict[str, Any]) -> dict[str, Any]:
        """Convert run_hpo_sweep result to the on-disk tuned_configs entry."""
        return {
            **sweep_result.get("config", {}),
            "_meta": {
                "run_id": sweep_result.get("run_id"),
                "seg_f0_5": sweep_result.get("seg_f0_5", 0.0),
                "nominal_fp_rate": sweep_result.get("nominal_fp_rate", 0.0),
                "objective": sweep_result.get("objective", sweep_result.get("seg_f0_5", 0.0)),
            },
        }

    sweep_results: dict[str, dict[str, Any]] = {}
    # Keep the GCP ID token fresh for the whole sweep. MLflowLoggerCallback logs
    # every trial from inside the blocking tuner.fit() calls below, with no
    # per-trial hook to refresh — a multi-subsystem sweep can exceed the 60-min
    # token lifetime, after which the tail 401s (the same failure mode the
    # per-epoch refresh fixed for training). The background refresher covers both
    # the parallel (ThreadPoolExecutor) and sequential paths; the env var is
    # process-global so all worker threads see the refreshed token.
    with _keep_mlflow_auth_fresh():
        if settings.tune.parallel_subsystems and len(eligible) > 1:
            max_workers = min(settings.tune.max_parallel_subsystems, len(eligible))
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        run_hpo_sweep, sub, sub_channels, settings, mission,
                        search_space=_space,
                    ): sub
                    for sub, sub_channels in eligible.items()
                }
                for future in concurrent.futures.as_completed(futures):
                    subsystem = futures[future]
                    sweep_results[subsystem] = _to_entry(future.result())
        else:
            for sub, sub_channels in eligible.items():
                sweep_results[sub] = _to_entry(
                    run_hpo_sweep(sub, sub_channels, settings, mission, search_space=_space)
                )

    write_tuned_configs(sweep_results, output)

    # Open a summary MLflow run to make tuned_configs.json visible in the
    # HPO experiment's Runs UI — the on-disk file is the authoritative copy;
    # this is a read-only mirror for discoverability.  Best-effort: failures
    # here must not abort the pipeline.
    with suppress(Exception):
        _configure_mlflow(settings)
        _hpo_exp = _mlflow_experiment_name(settings.model.model_type, "hpo", mission)
        with _open_run(
            experiment=_hpo_exp,
            run_name="tuned-configs-summary",
            tags={
                "model_type": settings.model.model_type,
                "mission_id": mission,
                "phase": "hpo",
            },
        ):
            _log_artifact_bytes(output.read_bytes(), "tuned_configs.json")

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
                "_meta": {
                    "run_id": "abc123...", "seg_f0_5": 0.72,
                    "nominal_fp_rate": 0.01, "objective": 0.67
                }
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
