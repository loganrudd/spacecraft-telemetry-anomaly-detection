"""ray_training: Ray Core parallel training + Ray Tune HPO (Phase 5/6).

Public API
----------
discover_channels    Scan Spark output dirs for available channel IDs.
train_all_channels   Fan out train_channel across channels via Ray Core.
score_all_channels   Fan out score_channel across channels via Ray Core.
make_train_task      Factory: returns a @ray.remote train task.
make_score_task      Factory: returns a @ray.remote score task.
"""

from spacecraft_telemetry.ray_training.runner import (
    discover_channels,
    score_all_channels,
    train_all_channels,
)
from spacecraft_telemetry.ray_training.tasks import make_score_task, make_train_task

__all__ = [
    "discover_channels",
    "make_score_task",
    "make_train_task",
    "score_all_channels",
    "train_all_channels",
]
