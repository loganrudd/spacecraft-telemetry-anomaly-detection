"""Ray remote task definitions for Phase 4 parallel training.

Wraps model.training.train_channel and model.scoring.score_channel as
@ray.remote tasks. Phase 3 functions are intentionally untouched — the remote
variants live here exclusively so that:
  - The CLI `model train` / `model score` commands keep working without Ray.
  - Phase 8 (FastAPI) can import scoring functions without pulling in Ray.
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
    status      "ok" | "error" | "skipped"
                                "skipped" = no trained model for this channel
                                (expected for untrained channels); distinct from
                                "error" so real failures stay visible.
    error_msg   str | None      traceback on error / reason on skip, else None

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


def make_train_task(num_gpus: float, max_retries: int = 3) -> Any:
    """Return a @ray.remote train_channel task with the given GPU allocation.

    Args:
        num_gpus:    GPU fraction per task. 0.0 = CPU-only (local dev).
                     0.25 on T4 clouds packs 4 models per physical GPU.
        max_retries: Number of times Ray retries a failed task before marking
                     it as error. Set from RayConfig.max_retries — critical for
                     preemptible-VM resilience on cloud.

    Returns:
        A ray.remote-decorated callable. Invoke with .remote(settings_ref, mission, channel).
    """
    import ray

    from spacecraft_telemetry.core.config import Settings
    from spacecraft_telemetry.core.logging import get_logger

    @ray.remote(num_cpus=1, num_gpus=num_gpus, max_retries=max_retries)
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
        except Exception:
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


def make_score_task(num_gpus: float, max_retries: int = 3) -> Any:
    """Return a @ray.remote score_channel task with the given GPU allocation.

    Args:
        num_gpus:    GPU fraction per task. Same as make_train_task.
        max_retries: Number of times Ray retries a failed task. See make_train_task.

    Returns:
        A ray.remote-decorated callable. Invoke with .remote(settings_ref, mission, channel).
    """
    import ray

    from spacecraft_telemetry.core.config import Settings
    from spacecraft_telemetry.core.logging import get_logger

    @ray.remote(num_cpus=1, num_gpus=num_gpus, max_retries=max_retries)
    def _score(
        settings: Any,
        mission: str,
        channel: str,
        eval_split: str = "full_test",
        parent_hpo_run_id: str | None = None,
    ) -> dict[str, Any]:
        log = get_logger(__name__)
        # Ray auto-dereferences ObjectRefs passed to .remote() — settings is
        # already the Settings object, not a ref.
        typed_settings: Settings = settings
        from spacecraft_telemetry.model.io import ModelNotFoundError
        try:
            from typing import Literal, cast

            from spacecraft_telemetry.model.scoring import score_channel
            _split = cast(
                Literal["full_test", "hpo_portion", "final_portion"], eval_split
            )
            metrics = score_channel(
                typed_settings, mission, channel,
                eval_split=_split,
                parent_hpo_run_id=parent_hpo_run_id,
            )
            return {
                "channel": channel,
                "status": "ok",
                "error_msg": None,
                **metrics,
            }
        except ModelNotFoundError as exc:
            # Expected for untrained channels (e.g. partial/smoke-test training).
            # Not an error — keeps real failures distinguishable in the summary.
            log.info("ray.score.task.skipped", channel=channel, reason=str(exc))
            return _null_score_result(channel, status="skipped", error_msg=str(exc))
        except Exception:
            tb = traceback.format_exc()
            log.error("ray.score.task.failed", channel=channel, traceback=tb)
            return _null_score_result(channel, status="error", error_msg=tb)

    return _score


def _null_score_result(channel: str, *, status: str, error_msg: str | None) -> dict[str, Any]:
    """Result dict with all metric fields None — shared by skipped/error cases."""
    return {
        "channel": channel,
        "status": status,
        "error_msg": error_msg,
        "precision": None,
        "recall": None,
        "f1": None,
        "f0_5": None,
        "n_true_positive_labels": None,
        "n_predicted_positive_labels": None,
        "seg_precision": None,
        "seg_recall": None,
        "seg_f1": None,
        "seg_f0_5": None,
        "n_true_seqs": None,
        "n_pred_seqs": None,
        "pruned_seg_precision": None,
        "pruned_seg_recall": None,
        "pruned_seg_f0_5": None,
        "pruned_n_pred_seqs": None,
    }
