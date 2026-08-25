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

train_all_subsystems(settings, mission, channels, *, max_subsystems=None) -> list[dict]
score_all_subsystems(settings, mission, channels, *, max_subsystems=None,
                     tuned_configs=None, eval_split="final_portion",
                     data_source="nominal") -> list[dict]
    Multivariate fan-out (docs/plans/021-multivariate-telemanom.md): the same
    two sweeps, but the Ray task boundary is the GROUP, not the channel.
    Channels are grouped via load_channel_group_map — a measured group id where
    the mission has one (docs/plans/023), else the subsystem name; each task
    trains/scores one joint model on settings.model.input_channels=<that group>.
    ``channels`` is still the flat input list to group — a channel with no
    entry is dropped with a warning, never silently included in the wrong
    group.

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
from spacecraft_telemetry.core.metadata import (
    load_channel_group_map,
    load_channel_subsystem_map,
)
from spacecraft_telemetry.core.paths import absolutize_if_local, output_path, to_upath

log = get_logger(__name__)

# Scoring fields that Phase 5 is allowed to tune; any other key in tuned_configs
# is ignored. This prevents an accidental override of architecture params.
# prune_min_decrease is intentionally absent: pruning is offline-only (the
# online serving path cannot replicate it), so a tuned config must not carry it
# — that keeps the served params identical to what HPO optimized. See
# ray_fanout/tune.py SEARCH_SPACE and docs/architecture/online-pruning-investigation.md.
#
# min_error_value IS present, despite also being called "pruning" in the
# ESA-ADB paper. The two are unrelated mechanisms: prune_min_decrease is
# Hundman §3.3 (ranks all flagged sequences against each other — retrospective,
# batch-only), whereas min_error_value is an absolute per-window floor that the
# streaming engine applies identically (api/inference.py). It is therefore a
# legitimate tuned param with no train/serve parity cost.
_TUNABLE_SCORING_FIELDS = frozenset(
    {
        "threshold_z",
        "threshold_window",
        "error_smoothing_window",
        "threshold_min_anomaly_len",
        "min_error_value",
    }
)


def _tuned_meta(entry: dict[str, Any] | None) -> tuple[str | None, str | None]:
    """Read ``(hpo_run_id, tuned_source)`` out of a tuned_configs entry's ``_meta``.

    Both fan-outs (score_all_channels and score_all_subsystems) need this, and
    both had their own copy — the subsystem one carrying a comment reading
    "See _get_tuned_source in score_all_channels", which is an acknowledgement
    of the duplication rather than a fix (docs/reviews/021, item B4).

    The two fields are deliberately independent, not alternatives:

    - ``run_id`` is present when a Ray Tune trial produced the config, and
      becomes score_channel's ``parent_hpo_run_id`` (the ``tuned_from_run``
      lineage tag).
    - ``source`` is present when an exhaustive grid produced it
      (scripts/threshold_ceiling.py), which has NO HPO run to point at —
      fabricating one would corrupt the lineage tag.

    A run carrying neither tag is read downstream as an untuned Hundman-defaults
    baseline, which is why a grid-produced config must still carry provenance:
    otherwise esa_adb's report files it as the protocol-matched untuned row.

    Returns ``(None, None)`` for a missing or malformed ``_meta``.
    """
    if not entry:
        return None, None
    meta = entry.get("_meta")
    if not isinstance(meta, dict):
        return None, None
    run_id = meta.get("run_id")
    source = meta.get("source")
    return (
        str(run_id) if run_id is not None else None,
        str(source) if source is not None else None,
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
    base = output_path(
        settings.preprocess.processed_data_dir, mission, settings.variant,
        "train", f"mission_id={mission}",
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
        name = experiment_name("telemanom", phase, mission, settings.variant)
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
            processed_map_path = output_path(
                abs_settings.preprocess.processed_data_dir,
                mission,
                abs_settings.variant,
                "metadata",
                "channel_subsystems.json",
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

    def _channel_meta(channel: str) -> tuple[str | None, str | None]:
        """(hpo_run_id, tuned_source) for this channel's subsystem entry.

        Only the channel->subsystem indirection is local; the ``_meta`` reading
        itself is shared with score_all_subsystems via _tuned_meta.
        """
        if not tuned_configs:
            return None, None
        subsystem = ch_to_sub.get(channel)
        if not subsystem:
            return None, None
        return _tuned_meta(tuned_configs.get(subsystem))

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
    futures = []
    for ch in work:
        _hpo_run_id, _tuned_source = _channel_meta(ch)
        futures.append(
            score_task.remote(
                _get_settings_ref(ch), mission, ch, eval_split, _hpo_run_id, data_source,
                _tuned_source,
            )
        )
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


# ---------------------------------------------------------------------------
# Multivariate fan-out (docs/plans/021-multivariate-telemanom.md)
# ---------------------------------------------------------------------------


def _group_channels_by_subsystem(
    settings: Settings, mission: str, channels: list[str]
) -> dict[str, list[str]]:
    """Group ``channels`` into multivariate groups, preserving each group's order.

    The grouping key comes from ``load_channel_group_map``: a measured group id
    when the mission has one (docs/plans/023), otherwise the subsystem name,
    which is what every mission used before that plan and what ISS still uses.
    Joint modelling needs members that share a timestamp grid, and that is a
    property of the data — the natural ESA group crosses subsystem boundaries —
    so the subsystem name is the fallback, not the definition.

    A group's order is the order its members appear in ``channels`` — this
    becomes the model's persisted ``input_channels`` order (model/io.py), so
    it must be deterministic and caller-controlled, not the map's (dict)
    iteration order.

    Channels with no entry are dropped with a warning rather than silently
    grouped under a sentinel key — an unmapped channel in a multivariate group
    is exactly the "silently reordered/wrong input" failure mode plan 021 calls
    out.
    """
    ch_to_group = load_channel_group_map(settings, mission)
    groups: dict[str, list[str]] = {}
    unmapped: list[str] = []
    for ch in channels:
        group = ch_to_group.get(ch)
        if group is None:
            unmapped.append(ch)
            continue
        groups.setdefault(group, []).append(ch)
    if unmapped:
        log.warning(
            "ray.subsystem_group.unmapped_channels", mission=mission, channels=unmapped
        )
    return groups


def _resolve_tuned_entry(
    tuned_configs: dict[str, dict[str, Any]] | None,
    group_key: str,
    group_channels: list[str],
    ch_to_sub: dict[str, str],
) -> dict[str, Any] | None:
    """Find the tuned_configs entry for one multivariate group.

    The model's key and the tuning key are deliberately at different
    granularities (docs/plans/023 stage .4). Models are keyed by channel GROUP,
    because ESA normalised values per group specifically to preserve
    cross-channel dependencies within one. HPO stays keyed by SUBSYSTEM, because
    the dataset paper states subsystem names are consistent across missions so
    cross-mission models are possible, while group numbers are mission-local —
    so a subsystem-tuned config is the reusable artifact.

    Resolution order:
    1. an entry keyed by the group itself — a deliberate per-group override;
    2. the entry for the subsystem the group belongs to.

    Without step 2 this lookup silently misses: ``run_all_sweeps`` writes
    ``{"subsystem_6": {...}}`` while the group key is ``subsystem_6_g03``, so
    ``.get(group_key)`` returns None, no overrides are applied, and a "tuned"
    scoring run is byte-identical to the untuned baseline — with no error
    anywhere to say so. That is the whole reason this function exists.

    The subsystem is resolved from the group's MEMBERS, never by parsing the
    group key: the key is an arbitrary registry string, and a naming convention
    is not a data structure. A group spanning multiple subsystems returns None
    with a warning rather than picking one — silently tuning a group with
    another subsystem's thresholds is the failure this is guarding against.
    """
    if not tuned_configs:
        return None
    entry = tuned_configs.get(group_key)
    if entry is not None:
        return entry
    subsystems = {ch_to_sub[c] for c in group_channels if c in ch_to_sub}
    if len(subsystems) == 1:
        return tuned_configs.get(subsystems.pop())
    log.warning(
        "ray.score.tuned_config_unresolved",
        group=group_key,
        subsystems=sorted(subsystems),
        reason="group spans multiple subsystems (or none); scoring with untuned defaults",
    )
    return None


def train_all_subsystems(
    settings: Settings,
    mission: str,
    channels: list[str],
    *,
    max_subsystems: int | None = None,
) -> list[dict[str, Any]]:
    """Fan out train_channel across SUBSYSTEM groups using Ray Core.

    The multivariate counterpart to train_all_channels: each Ray task trains
    one joint model per subsystem (settings.model.input_channels set to that
    subsystem's member channels), so the fan-out unit is the subsystem
    (~4-8 tasks for ESA-Mission1) rather than the channel (~100) — see the
    "breaks a locked architectural decision" section of docs/plans/021.

    One ray.put() per subsystem (not one shared settings object): each
    task's settings differs in input_channels/target_channels, unlike
    train_all_channels where every task shares identical settings. With
    O(10) subsystems this is cheap.

    Args:
        settings:       Fully resolved Settings.
        mission:        Mission name, e.g. "ESA-Mission1".
        channels:       Flat channel IDs to group by subsystem and train.
        max_subsystems: Cap the sweep at this many subsystems (smoke tests).

    Returns:
        List of per-subsystem result dicts (the "channel" key holds the
        subsystem name — see ray_fanout/tasks.py's result schema), sorted by
        subsystem name.
    """
    import ray

    from spacecraft_telemetry.ray_fanout.tasks import make_train_task

    abs_settings = _with_abs_paths(settings)
    groups = _group_channels_by_subsystem(abs_settings, mission, channels)
    subsystems = sorted(groups)[:max_subsystems] if max_subsystems is not None else sorted(groups)
    if not subsystems:
        log.warning("ray.train.no_subsystems", mission=mission)
        return []

    _ensure_mlflow_experiments(settings, mission, ["training"])
    log.info("ray.train.subsystem_sweep.start", mission=mission, n_subsystems=len(subsystems))

    train_task = make_train_task(
        num_gpus=settings.ray.num_gpus_per_task,
        max_retries=settings.ray.max_retries,
    )
    futures = []
    for subsystem in subsystems:
        group_channels = groups[subsystem]
        sub_settings = abs_settings.model_copy(
            update={
                "model": abs_settings.model.model_copy(
                    update={"input_channels": group_channels, "target_channels": group_channels}
                )
            }
        )
        futures.append(train_task.remote(ray.put(sub_settings), mission, subsystem))
    results: list[dict[str, Any]] = ray.get(futures)

    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_err = len(results) - n_ok
    log.info(
        "ray.train.subsystem_sweep.end", mission=mission, n_ok=n_ok, n_error=n_err
    )

    return results


def score_all_subsystems(
    settings: Settings,
    mission: str,
    channels: list[str],
    *,
    max_subsystems: int | None = None,
    tuned_configs: dict[str, dict[str, Any]] | None = None,
    eval_split: str = "final_portion",
    data_source: str = "nominal",
) -> list[dict[str, Any]]:
    """Fan out score_channel across SUBSYSTEM groups using Ray Core.

    The multivariate counterpart to score_all_channels. ``tuned_configs`` is
    already subsystem-keyed (Phase 5 HPO groups by subsystem for univariate
    too), so applying it here needs no channel->subsystem indirection at
    dispatch time — it applies directly to each group.

    Args:
        settings:       Fully resolved Settings.
        mission:        Mission name, e.g. "ESA-Mission1".
        channels:       Flat channel IDs to group by subsystem and score.
        max_subsystems: Cap the sweep at this many subsystems.
        tuned_configs:  Optional dict mapping subsystem name -> scoring param
                        overrides (see score_all_channels's tuned_configs schema).
        eval_split:     Which temporal slice of the test set to evaluate.
        data_source:    "nominal" (default) or "injected" — see score_all_channels.

    Returns:
        List of per-subsystem result dicts, sorted by subsystem name. Each
        multivariate result's metric fields hold score_channel's
        {channel_id: metrics_dict, ...} return, not a flat dict — see
        model/scoring.py's score_channel docstring.
    """
    import ray

    from spacecraft_telemetry.ray_fanout.tasks import make_score_task

    abs_settings = _with_abs_paths(settings)
    groups = _group_channels_by_subsystem(abs_settings, mission, channels)
    subsystems = sorted(groups)[:max_subsystems] if max_subsystems is not None else sorted(groups)
    if not subsystems:
        log.warning("ray.score.no_subsystems", mission=mission)
        return []

    _ensure_mlflow_experiments(settings, mission, ["scoring"])
    log.info(
        "ray.score.subsystem_sweep.start",
        mission=mission,
        n_subsystems=len(subsystems),
        tuned=tuned_configs is not None,
    )

    score_task = make_score_task(
        num_gpus=settings.ray.num_gpus_per_task,
        max_retries=settings.ray.max_retries,
    )
    ch_to_sub = load_channel_subsystem_map(abs_settings, mission)
    futures = []
    for subsystem in subsystems:
        group_channels = groups[subsystem]
        overrides: dict[str, Any] = {}
        entry = _resolve_tuned_entry(tuned_configs, subsystem, group_channels, ch_to_sub)
        if entry:
            overrides = {k: v for k, v in entry.items() if k in _TUNABLE_SCORING_FIELDS}
        # tuned_configs is already subsystem-keyed, so unlike score_all_channels
        # there is no channel->subsystem indirection to do first.
        hpo_run_id, tuned_source = _tuned_meta(entry)
        sub_settings = abs_settings.model_copy(
            update={
                "model": abs_settings.model.model_copy(
                    update={
                        "input_channels": group_channels,
                        "target_channels": group_channels,
                        **overrides,
                    }
                )
            }
        )
        futures.append(
            score_task.remote(
                ray.put(sub_settings), mission, subsystem, eval_split, hpo_run_id, data_source,
                tuned_source,
            )
        )
    results: list[dict[str, Any]] = ray.get(futures)

    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_skipped = sum(1 for r in results if r["status"] == "skipped")
    n_err = len(results) - n_ok - n_skipped
    log.info(
        "ray.score.subsystem_sweep.end",
        mission=mission,
        n_ok=n_ok,
        n_skipped=n_skipped,
        n_error=n_err,
    )

    return results
