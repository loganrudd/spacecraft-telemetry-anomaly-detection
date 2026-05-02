"""ray_training: Ray Core parallel training + Ray Tune HPO (Phase 5/6).

Public API
----------
discover_channels    Scan Spark output dirs for available channel IDs.
train_all_channels   Fan out train_channel across channels via Ray Core.
score_all_channels   Fan out score_channel across channels via Ray Core.
make_train_task      Factory: returns a @ray.remote train task.
make_score_task      Factory: returns a @ray.remote score task.
SEARCH_SPACE         Ray Tune search space over the 4 scoring parameters.
run_hpo_sweep        Run one Tune experiment for a named subsystem.
run_all_sweeps       Group channels by subsystem, run all sweeps, write JSON.
write_tuned_configs  Persist subsystem → best_config mapping as JSON.
"""

from spacecraft_telemetry.ray_training.runner import (
    discover_channels,
    load_channel_subsystem_map,
    score_all_channels,
    train_all_channels,
)
from spacecraft_telemetry.ray_training.tasks import make_score_task, make_train_task
from spacecraft_telemetry.ray_training.tune import (
    SEARCH_SPACE,
    run_all_sweeps,
    run_hpo_sweep,
    write_tuned_configs,
)

__all__ = [
    "SEARCH_SPACE",
    "discover_channels",
    "load_channel_subsystem_map",
    "make_score_task",
    "make_train_task",
    "run_all_sweeps",
    "run_hpo_sweep",
    "score_all_channels",
    "train_all_channels",
    "write_tuned_configs",
]
