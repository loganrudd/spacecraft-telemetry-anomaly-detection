"""Ray Core fan-out runner for parallel channel training and scoring.

Public API
----------
discover_channels(settings, mission) -> list[str]
    Scan Spark processed-data dirs for available channel IDs.

train_all_channels(settings, mission, channels, *, max_channels=None) -> list[dict]
    Fan out train_channel across all channels using Ray Core.

score_all_channels(settings, mission, channels, *, max_channels=None,
                   tuned_configs=None) -> list[dict]
    Fan out score_channel across all channels, optionally applying per-subsystem
    scoring param overrides from Phase 6 HPO output (tuned_configs.json).

tuned_configs schema (Phase 6 writes, score_all_channels reads)
---------------------------------------------------------------
A dict keyed by subsystem name mapping to a dict of ModelConfig scoring field
overrides. Any subset of the four scoring fields is accepted:

    {
        "subsystem_1": {"threshold_z": 2.8, "threshold_window": 200},
        "subsystem_6": {"threshold_z": 3.2, "error_smoothing_window": 40}
    }

Channels whose subsystem has no entry in tuned_configs (or when tuned_configs
is None) use the unmodified base settings (Hundman defaults).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from spacecraft_telemetry.core.config import Settings
from spacecraft_telemetry.core.logging import get_logger

log = get_logger(__name__)

# Scoring fields that Phase 6 is allowed to tune; any other key in tuned_configs
# is ignored. This prevents an accidental override of architecture params.
_TUNABLE_SCORING_FIELDS = frozenset(
    {"threshold_z", "threshold_window", "error_smoothing_window", "threshold_min_anomaly_len"}
)


def discover_channels(settings: Settings, mission: str) -> list[str]:
    """Return sorted channel IDs found in the Spark train output for a mission.

    Scans:
        {spark.processed_data_dir}/{mission}/train/mission_id={mission}/channel_id=*/

    Returns an empty list (not an exception) if no channels have been preprocessed
    yet — lets the caller print a clear "no channels found" message rather than
    crashing.

    Note: gs:// URIs are not supported here; scan is local-filesystem only.
    Use --channels explicitly when running against a cloud Spark output directory.

    Args:
        settings: Resolved Settings.
        mission:  Mission name, e.g. "ESA-Mission1".

    Returns:
        Sorted list of channel ID strings.
    """
    base = (
        Path(str(settings.spark.processed_data_dir))
        / mission
        / "train"
        / f"mission_id={mission}"
    )
    if not base.exists():
        return []
    return sorted(
        p.name.removeprefix("channel_id=")
        for p in base.iterdir()
        if p.is_dir() and p.name.startswith("channel_id=")
    )


def _load_channel_subsystem_map(settings: Settings, mission: str) -> dict[str, str]:
    """Read channels.csv and return a mapping of channel_id -> subsystem name.

    Returns an empty dict if channels.csv is missing — callers fall back to
    Hundman defaults gracefully.

    The CSV is expected at: {data.raw_data_dir}/{mission}/channels.csv
    with columns: Channel, Subsystem, Physical Unit, Group, Target
    """
    csv_path = Path(str(settings.data.raw_data_dir)) / mission / "channels.csv"
    if not csv_path.exists():
        log.warning("channels.csv not found; tuned_configs will not be applied", path=str(csv_path))
        return {}
    mapping: dict[str, str] = {}
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            channel = row.get("Channel", "").strip()
            subsystem = row.get("Subsystem", "").strip()
            if channel and subsystem:
                mapping[channel] = subsystem
    return mapping


def _with_abs_paths(settings: Settings) -> Settings:
    """Return a copy of settings with relative paths resolved to absolute.

    Ray workers run from a temp directory (Ray's session dir), so any relative
    paths in settings would fail to resolve. Resolving to absolute in the main
    process (where CWD is correct) before ray.put() fixes this.
    """
    return settings.model_copy(
        update={
            "spark": settings.spark.model_copy(
                update={"processed_data_dir": Path(str(settings.spark.processed_data_dir)).resolve()}
            ),
            "model": settings.model.model_copy(
                update={"artifacts_dir": Path(str(settings.model.artifacts_dir)).resolve()}
            ),
            "data": settings.data.model_copy(
                update={"raw_data_dir": Path(str(settings.data.raw_data_dir)).resolve()}
            ),
        }
    )


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

    from spacecraft_telemetry.ray_training.tasks import make_train_task

    work = channels[:max_channels] if max_channels is not None else channels
    if not work:
        log.warning("ray.train.no_channels", mission=mission)
        return []

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
) -> list[dict[str, Any]]:
    """Fan out score_channel across channels using Ray Core.

    Optionally applies per-subsystem scoring param overrides from Phase 6 HPO
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

    Returns:
        List of per-channel result dicts in the same order as the channels input.
    """
    import ray

    from spacecraft_telemetry.ray_training.tasks import make_score_task

    work = channels[:max_channels] if max_channels is not None else channels
    if not work:
        log.warning("ray.score.no_channels", mission=mission)
        return []

    log.info("ray.score.sweep.start", mission=mission, n_channels=len(work),
             tuned=tuned_configs is not None)

    # Resolve relative paths to absolute before ray.put() — Ray workers run
    # from Ray's session dir, not the project root.
    abs_settings = _with_abs_paths(settings)

    # Build channel → subsystem map once (reads channels.csv).
    ch_to_sub: dict[str, str] = {}
    if tuned_configs:
        ch_to_sub = _load_channel_subsystem_map(abs_settings, mission)

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

    score_task = make_score_task(
        num_gpus=settings.ray.num_gpus_per_task,
        max_retries=settings.ray.max_retries,
    )
    futures = [score_task.remote(_get_settings_ref(ch), mission, ch) for ch in work]
    results: list[dict[str, Any]] = ray.get(futures)

    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_err = len(results) - n_ok
    log.info("ray.score.sweep.end", mission=mission, n_ok=n_ok, n_error=n_err)

    return results
