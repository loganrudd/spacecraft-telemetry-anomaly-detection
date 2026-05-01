"""Ray remote task definitions for Phase 5 parallel training.

Wraps model.training.train_channel and model.scoring.score_channel as
@ray.remote tasks. Phase 4 functions are intentionally untouched — the remote
variants live here exclusively so that:
  - The CLI `model train` / `model score` commands keep working without Ray.
  - Phase 9 (FastAPI) can import scoring functions without pulling in Ray.
  - GPU resource allocation can be chosen at call time (CPU locally, 0.25 GPU
    on cloud T4s) via the factory pattern.

Usage:
    train_task = make_train_task(num_gpus=settings.ray.num_gpus_per_task)
    score_task = make_score_task(num_gpus=settings.ray.num_gpus_per_task)
    settings_ref = ray.put(settings)
    futures = [train_task.remote(settings_ref, mission, ch) for ch in channels]
    results = ray.get(futures)

Task result dict schema
-----------------------
Both train and score tasks return a flat dict with these keys:

    channel     str             channel ID
    status      "ok" | "error"
    error_msg   str | None      traceback string on error, else None

Train-only keys (None on error):
    best_epoch      int | None
    best_val_loss   float | None
    epochs_run      int | None

Score-only keys (absent or None on error):
    precision   float | None
    recall      float | None
    f1          float | None
    f0_5        float | None
    n_true_positive_labels      int | None
    n_predicted_positive_labels int | None
"""

from __future__ import annotations

import traceback
from typing import Any


def make_train_task(num_gpus: float) -> Any:
    """Return a @ray.remote train_channel task with the given GPU allocation.

    Args:
        num_gpus: GPU fraction per task. 0.0 = CPU-only (local dev).
                  0.25 on T4 clouds packs 4 models per physical GPU.

    Returns:
        A ray.remote-decorated callable. Invoke with .remote(settings_ref, mission, channel).
    """
    import ray

    from spacecraft_telemetry.core.config import Settings
    from spacecraft_telemetry.core.logging import get_logger

    @ray.remote(num_cpus=1, num_gpus=num_gpus)
    def _train(settings: Any, mission: str, channel: str) -> dict[str, Any]:
        log = get_logger(__name__)
        # Ray auto-dereferences ObjectRefs passed to .remote() — settings is
        # already the Settings object, not a ref.
        typed_settings: Settings = settings
        try:
            from spacecraft_telemetry.model.training import train_channel

            result = train_channel(typed_settings, mission, channel)
            return {
                "channel": channel,
                "status": "ok",
                "error_msg": None,
                "best_epoch": result.best_epoch,
                "best_val_loss": result.best_val_loss,
                "epochs_run": result.epochs_run,
            }
        except Exception:  # noqa: BLE001
            tb = traceback.format_exc()
            log.error("ray.train.task.failed", channel=channel, traceback=tb)
            return {
                "channel": channel,
                "status": "error",
                "error_msg": tb,
                "best_epoch": None,
                "best_val_loss": None,
                "epochs_run": None,
            }

    return _train


def make_score_task(num_gpus: float) -> Any:
    """Return a @ray.remote score_channel task with the given GPU allocation.

    Args:
        num_gpus: GPU fraction per task. Same as make_train_task.

    Returns:
        A ray.remote-decorated callable. Invoke with .remote(settings_ref, mission, channel).
    """
    import ray

    from spacecraft_telemetry.core.config import Settings
    from spacecraft_telemetry.core.logging import get_logger

    @ray.remote(num_cpus=1, num_gpus=num_gpus)
    def _score(settings: Any, mission: str, channel: str) -> dict[str, Any]:
        log = get_logger(__name__)
        # Ray auto-dereferences ObjectRefs passed to .remote() — settings is
        # already the Settings object, not a ref.
        typed_settings: Settings = settings
        try:
            from spacecraft_telemetry.model.scoring import score_channel

            metrics = score_channel(typed_settings, mission, channel)
            return {
                "channel": channel,
                "status": "ok",
                "error_msg": None,
                **metrics,
            }
        except Exception:  # noqa: BLE001
            tb = traceback.format_exc()
            log.error("ray.score.task.failed", channel=channel, traceback=tb)
            return {
                "channel": channel,
                "status": "error",
                "error_msg": tb,
            }

    return _score
