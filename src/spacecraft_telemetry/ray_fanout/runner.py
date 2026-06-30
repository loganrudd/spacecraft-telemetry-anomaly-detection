"""Ray Core fan-out runner for parallel channel training and scoring.

Public API
----------
discover_channels(settings, mission) -> list[str]
    Scan processed-data dirs for available channel IDs.

train_all_channels(settings, mission, channels, *, max_channels=None) -> list[dict]
    Fan out train_channel across all channels using Ray Core.

score_all_channels(settings, mission, channels, *, max_channels=None,
                   tuned_configs=None, data_source="nominal") -> list[dict]
    Fan out score_channel across all channels, optionally applying per-subsystem
    scoring param overrides from Phase 5 HPO output (tuned_configs.json).
    data_source tags each run "nominal" or "injected" (Phase 15) so HPO can
    later locate a channel's nominal baseline run for the FP-rate penalty.

tuned_configs schema (Phase 5 writes, score_all_channels reads)
---------------------------------------------------------------
A dict keyed by subsystem name. Each entry contains scoring-param overrides
and a ``_meta`` block written by run_all_sweeps:

    {
        "subsystem_1": {
            "threshold_z": 2.8, "threshold_window": 200,
            "_meta": {"run_id": "abc123...", "f0_5": 0.72}
        }
    }

``_meta`` is stripped before applying overrides to ModelConfig (not in
_TUNABLE_SCORING_FIELDS). ``_meta.run_id`` is passed to score_channel as
``parent_hpo_run_id`` to set the ``tuned_from_run`` MLflow lineage tag.
Channels whose subsystem has no entry in tuned_configs (or when tuned_configs
is None) use the unmodified base settings (Hundman defaults).
"""

from __future__ import annotations

from typing import Any

from spacecraft_telemetry.core.config import Settings
from spacecraft_telemetry.core.logging import get_logger
from spacecraft_telemetry.core.metadata import load_channel_subsystem_map
from spacecraft_telemetry.core.paths import absolutize_if_local, to_upath

log = get_logger(__name__)

# Scoring fields that Phase 5 is allowed to tune; any other key in tuned_configs
# is ignored. This prevents an accidental override of architecture params.
# prune_min_decrease is intentionally absent: pruning is offline-only (the
# online serving path cannot replicate it), so a tuned config must not carry it
# — that keeps the served params identical to what HPO optimized. See
# ray_fanout/tune.py SEARCH_SPACE and docs/architecture/online-pruning-investigation.md.
_TUNABLE_SCORING_FIELDS = frozenset(
    {"threshold_z", "threshold_window", "error_smoothing_window", "threshold_min_anomaly_len"}
)


def discover_channels(settings: Settings, mission: str) -> list[str]:
    """Return sorted channel IDs found in the preprocessed train output for a mission.

    Scans:
        {preprocess.processed_data_dir}/{mission}/train/mission_id={mission}/channel_id=*/

    Returns an empty list (not an exception) if no channels have been preprocessed
    yet — lets the caller print a clear "no channels found" message rather than
    crashing.

    Args:
        settings: Resolved Settings.
        mission:  Mission name, e.g. "ESA-Mission1".

    Returns:
        Sorted list of channel ID strings.
    """
    base = (
        to_upath(settings.preprocess.processed_data_dir)
        / mission
        / "train"
        / f"mission_id={mission}"
    )
    if not base.exists():
        return []
    return sorted(
        p.name.removeprefix("channel_id=")
        for p in base.iterdir()
        if p.name.startswith("channel_id=")
    )


# load_channel_subsystem_map is defined in core.metadata and re-exported here
# for backward-compatibility with callers that import from ray_fanout.runner.
# (ray_fanout/__init__.py and cli.py import it from here.)
__all_runner_exports__ = ["load_channel_subsystem_map"]


def _with_abs_paths(settings: Settings) -> Settings:
    """Return a copy of settings with relative paths resolved to absolute.

    Ray workers run from a temp directory (Ray's session dir), so any relative
    paths in settings would fail to resolve. Resolving to absolute in the main
    process (where CWD is correct) before ray.put() fixes this.
    """
    return settings.model_copy(
        update={
            "preprocess": settings.preprocess.model_copy(
                update={
                    "processed_data_dir": absolutize_if_local(
                        settings.preprocess.processed_data_dir
                    )
                }
            ),
            "model": settings.model.model_copy(
                update={"artifacts_dir": absolutize_if_local(settings.model.artifacts_dir)}
            ),
            "data": settings.data.model_copy(
                update={"raw_data_dir": absolutize_if_local(settings.data.raw_data_dir)}
            ),
        }
    )


def _ensure_mlflow_experiments(settings: Settings, mission: str, phases: list[str]) -> None:
    """Pre-create MLflow experiments in the driver process before Ray tasks run.

    When the experiment doesn't yet exist, multiple Ray workers calling
    mlflow.set_experiment() concurrently hit a TOCTOU race: each worker calls
    get_experiment_by_name (returns None) then create_experiment, but only one
    creation succeeds.  MLflow's client does not always retry gracefully on the
    "already exists" error, so the losing workers may fall back to the "Default"
    experiment and log runs there instead of the intended one.

    Creating the experiment once in the driver (serial, no race) ensures it
    exists before any worker tries to use it.  Workers then always take the
    "experiment exists → get ID" branch, which is idempotent and race-free.

    Failures are suppressed — a missing experiment is non-fatal; workers have
    their own fallback behaviour.
    """
    from contextlib import suppress

    import mlflow

    from spacecraft_telemetry.mlflow_tracking.conventions import experiment_name
    from spacecraft_telemetry.mlflow_tracking.runs import configure_mlflow

    with suppress(Exception):
        configure_mlflow(settings)
    for phase in phases:
        name = experiment_name("telemanom", phase, mission)
        with suppress(Exception):
            mlflow.set_experiment(name)
            log.debug("mlflow.experiment.ensured", name=name)


def train_all_channels(
    settings: Settings,
    mission: str,
    channels: list[str],
    *,
    max_channels: int | None = None,
) -> list[dict[str, Any]]:
    """Fan out train_channel across channels using Ray Core.

    Uses ray.put(settings) to share the settings object once rather than
    serialising it once per task. Partial failures do not abort the sweep —
    failed tasks return status="error" with a traceback in error_msg.

    Args:
        settings:     Fully resolved Settings.
        mission:      Mission name, e.g. "ESA-Mission1".
        channels:     Ordered list of channel IDs to train.
        max_channels: Cap the sweep at this many channels (for local smoke tests).

    Returns:
        List of per-channel result dicts in the same order as the channels input.
    """
    import ray

    from spacecraft_telemetry.ray_fanout.tasks import make_train_task

    work = channels[:max_channels] if max_channels is not None else channels
    if not work:
        log.warning("ray.train.no_channels", mission=mission)
        return []

    _ensure_mlflow_experiments(settings, mission, ["training"])
    log.info("ray.train.sweep.start", mission=mission, n_channels=len(work))

    train_task = make_train_task(
        num_gpus=settings.ray.num_gpus_per_task,
        max_retries=settings.ray.max_retries,
    )
    settings_ref = ray.put(_with_abs_paths(settings))
    futures = [train_task.remote(settings_ref, mission, ch) for ch in work]
    results: list[dict[str, Any]] = ray.get(futures)

    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_err = len(results) - n_ok
    log.info("ray.train.sweep.end", mission=mission, n_ok=n_ok, n_error=n_err)

    return results


def score_all_channels(
    settings: Settings,
    mission: str,
    channels: list[str],
    *,
    max_channels: int | None = None,
    tuned_configs: dict[str, dict[str, Any]] | None = None,
    eval_split: str = "final_portion",
    data_source: str = "nominal",
) -> list[dict[str, Any]]:
    """Fan out score_channel across channels using Ray Core.

    Optionally applies per-subsystem scoring param overrides from Phase 5 HPO
    output. Each channel is mapped to its subsystem via channels.csv; if the
    subsystem has a tuned config, the scoring params are overridden before
    dispatching the task.

    One ray.put() per unique settings variant (at most one per subsystem), not
    one per channel — avoids redundant object-store entries for channels sharing
    the same tuned config.

    Args:
        settings:      Fully resolved Settings.
        mission:       Mission name, e.g. "ESA-Mission1".
        channels:      Ordered list of channel IDs to score.
        max_channels:  Cap the sweep at this many channels.
        tuned_configs: Optional dict mapping subsystem name → scoring param
                       overrides. Channels with no matching subsystem entry
                       receive the base settings (Hundman defaults). Keys must
                       be a subset of: threshold_z, threshold_window,
                       error_smoothing_window, threshold_min_anomaly_len.
        data_source:   "nominal" (default) or "injected" — tags each scoring
                       run so ray_fanout.tune can find a channel's nominal
                       baseline run independently of its injected-data run for
                       the HPO false-positive-rate penalty. Pass "injected"
                       when ``settings.preprocess.processed_data_dir`` points
                       at the fault-injected dataset (Phase 15).

    Returns:
        List of per-channel result dicts in the same order as the channels input.
    """
    import ray

    from spacecraft_telemetry.ray_fanout.tasks import make_score_task

    work = channels[:max_channels] if max_channels is not None else channels
    if not work:
        log.warning("ray.score.no_channels", mission=mission)
        return []

    _ensure_mlflow_experiments(settings, mission, ["scoring"])
    log.info("ray.score.sweep.start", mission=mission, n_channels=len(work),
             tuned=tuned_configs is not None)

    # Resolve relative paths to absolute before ray.put() — Ray workers run
    # from Ray's session dir, not the project root.
    abs_settings = _with_abs_paths(settings)

    # Build channel → subsystem map once (reads channels.csv).
    ch_to_sub: dict[str, str] = {}
    if tuned_configs:
        ch_to_sub = load_channel_subsystem_map(abs_settings, mission)
        if not ch_to_sub:
            processed_map_path = (
                to_upath(abs_settings.preprocess.processed_data_dir)
                / mission
                / "metadata"
                / "channel_subsystems.json"
            )
            raw_map_path = to_upath(abs_settings.data.raw_data_dir) / mission / "channels.csv"
            raise ValueError(
                "tuned_configs were provided, but no channel->subsystem map was found. "
                "Expected either processed metadata at "
                f"{processed_map_path} "
                "or raw metadata at "
                f"{raw_map_path}."
            )

    # Build one settings variant per unique subsystem tuned config.
    # Cache as ray object refs to avoid re-putting identical objects.
    settings_refs: dict[str, Any] = {}
    base_ref = ray.put(abs_settings)

    def _get_settings_ref(channel: str) -> Any:
        if not tuned_configs:
            return base_ref
        subsystem = ch_to_sub.get(channel)
        overrides = tuned_configs.get(subsystem, {}) if subsystem else {}
        if not overrides:
            return base_ref
        # Filter to only recognised scoring fields for safety.
        safe_overrides = {k: v for k, v in overrides.items() if k in _TUNABLE_SCORING_FIELDS}
        if not safe_overrides:
            return base_ref
        # subsystem is not None here: if it were, overrides would be {} and we'd
        # have returned base_ref above.
        assert subsystem is not None
        if subsystem not in settings_refs:
            tuned_settings = abs_settings.model_copy(
                update={"model": abs_settings.model.model_copy(update=safe_overrides)}
            )
            settings_refs[subsystem] = ray.put(tuned_settings)
        return settings_refs[subsystem]

    def _get_hpo_run_id(channel: str) -> str | None:
        """Return the HPO MLflow run_id for this channel's subsystem, or None."""
        if not tuned_configs:
            return None
        subsystem = ch_to_sub.get(channel)
        if not subsystem:
            return None
        meta = tuned_configs.get(subsystem, {}).get("_meta")
        if not meta or not isinstance(meta, dict):
            return None
        run_id = meta.get("run_id")
        return str(run_id) if run_id is not None else None

    # eval_split (default "final_portion") selects which temporal slice the
    # reported metrics cover. "final_portion" is the held-out last 40%: HPO only
    # saw hpo_portion (first hpo_eval_fraction), so this keeps baseline-vs-tuned
    # F0.5 apples-to-apples and leakage-free. Pass "full_test" for a coverage /
    # deployment-readiness view that scores every test window (mildly optimistic
    # for tuned params, which saw the first 60%) — use it to evaluate channels
    # whose anomalies fall outside the held-out slice, not for the HPO comparison.

    # predict() is a torch LSTM forward pass — the heavy part of scoring (~20
    # min/channel on CPU for large channels); only the downstream anomaly scoring
    # is pure numpy. Honour num_gpus_per_task so a cloud score job can run the
    # forward pass on the L4 (cluster_score.yaml sets NUM_GPUS=0.125 → 8-way pack).
    # Defaults to 0.0 (CPU) locally and anywhere the knob is unset, so local dev
    # and the CPU path are unaffected. resolve_device("auto") in score_channel
    # picks CUDA automatically once Ray exposes a GPU to the task.
    score_task = make_score_task(
        num_gpus=settings.ray.num_gpus_per_task,
        max_retries=settings.ray.max_retries,
    )
    futures = [
        score_task.remote(
            _get_settings_ref(ch), mission, ch, eval_split, _get_hpo_run_id(ch), data_source
        )
        for ch in work
    ]
    results: list[dict[str, Any]] = ray.get(futures)

    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_skipped = sum(1 for r in results if r["status"] == "skipped")
    n_err = len(results) - n_ok - n_skipped
    log.info(
        "ray.score.sweep.end",
        mission=mission,
        n_ok=n_ok,
        n_skipped=n_skipped,
        n_error=n_err,
    )

    return results
