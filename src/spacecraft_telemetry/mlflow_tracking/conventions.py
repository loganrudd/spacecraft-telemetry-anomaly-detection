"""MLflow naming conventions — model-type-agnostic.

All strings that appear in MLflow (experiment names, registered model names,
run tags) are produced here.  Nothing in this module knows what Telemanom is.
The model_type / phase / mission triples keep experiments scoped and searchable
without hard-coded strings scattered across callers.
"""

from __future__ import annotations

from typing import Any


def experiment_name(model_type: str, phase: str, mission: str) -> str:
    """Return the canonical experiment name for a (model_type, phase, mission) triple.

    Examples:
        experiment_name("telemanom", "training", "ESA-Mission1")
        → "telemanom-training-ESA-Mission1"

        experiment_name("dc_vae", "hpo", "ESA-Mission2")
        → "dc_vae-hpo-ESA-Mission2"
    """
    return f"{model_type}-{phase}-{mission}"


def registered_model_name(model_type: str, mission: str, key: str) -> str:
    """Return the canonical registered-model name.

    ``key`` is whatever uniquely identifies one trained model within a mission
    for the given model type:
      - Telemanom: channel_id  (one model per channel)
      - DC-VAE: subsystem or mission-level group (one model per group)

    Examples:
        registered_model_name("telemanom", "ESA-Mission1", "channel_1")
        → "telemanom-ESA-Mission1-channel_1"

        registered_model_name("dc_vae", "ESA-Mission1", "power")
        → "dc_vae-ESA-Mission1-power"
    """
    return f"{model_type}-{mission}-{key}"


def common_tags(
    *,
    model_type: str,
    mission: str,
    phase: str,
    training_data_hash: str | None = None,
    channel: str | None = None,
    subsystem: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Return the standard MLflow tag dict for a run.

    All values are coerced to str (MLflow requires string tag values).
    Keys whose value is None are omitted so the UI stays uncluttered.

    Args:
        model_type:          Identifies the model family ("telemanom", "dc_vae", …).
        mission:             Mission identifier, e.g. "ESA-Mission1".
        phase:               Pipeline stage ("training", "scoring", "hpo", …).
        training_data_hash:  Fingerprint of the train-split Parquet (optional).
        channel:             Channel ID for per-channel models (optional).
        subsystem:           Spacecraft subsystem name (optional).
        extra:               Any additional tags to merge in (optional).

    Returns:
        Dict[str, str] suitable for passing directly to mlflow.start_run(tags=…).
    """
    tags: dict[str, str] = {
        "model_type": model_type,
        "mission_id": mission,
        "phase": phase,
    }
    if training_data_hash is not None:
        tags["training_data_hash"] = training_data_hash
    if channel is not None:
        tags["channel_id"] = channel
    if subsystem is not None:
        tags["subsystem"] = subsystem
    if extra:
        tags.update({k: str(v) for k, v in extra.items()})
    return tags
