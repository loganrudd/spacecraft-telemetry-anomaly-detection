"""Unit tests for ray_training/tasks.py.

Tests that factory functions return correct callable objects without requiring
Ray to be initialised (import-time only), and that the task result schema
matches the contract described in tasks.py.
"""

from __future__ import annotations

from typing import Any, cast

import pytest


# ---------------------------------------------------------------------------
# Factory smoke tests (no Ray init required)
# ---------------------------------------------------------------------------


def test_make_train_task_returns_callable() -> None:
    """make_train_task should return a remote-decorated callable."""
    pytest.importorskip("ray")
    from spacecraft_telemetry.ray_training.tasks import make_train_task

    task = make_train_task(num_gpus=0.0)
    # Ray wraps the function — it should have a .remote attribute.
    assert hasattr(task, "remote"), "make_train_task must return a ray.remote-wrapped callable"


def test_make_score_task_returns_callable() -> None:
    """make_score_task should return a remote-decorated callable."""
    pytest.importorskip("ray")
    from spacecraft_telemetry.ray_training.tasks import make_score_task

    task = make_score_task(num_gpus=0.0)
    assert hasattr(task, "remote"), "make_score_task must return a ray.remote-wrapped callable"


def test_tasks_distinct_for_different_gpu_fractions() -> None:
    """Each make_*_task call with a different GPU fraction should produce a new task."""
    pytest.importorskip("ray")
    from spacecraft_telemetry.ray_training.tasks import make_train_task

    task_cpu = make_train_task(num_gpus=0.0)
    task_gpu = make_train_task(num_gpus=0.25)
    # They should be different objects (different ray remote descriptors).
    assert task_cpu is not task_gpu


# ---------------------------------------------------------------------------
# Integration: tasks execute end-to-end with a Ray local cluster
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_train_task_ok(ray_local, ray_series_parquet) -> None:
    """train task should return status='ok' with expected keys on valid data."""
    pytest.importorskip("ray")
    import ray

    from spacecraft_telemetry.ray_training.tasks import make_train_task

    settings = ray_series_parquet
    task = make_train_task(num_gpus=0.0)
    settings_ref = ray.put(settings)
    result = cast(dict[str, Any], ray.get(task.remote(settings_ref, "ESA-Mission1", "channel_1")))

    assert result["status"] == "ok", f"Expected ok, got: {result.get('error_msg')}"
    assert result["channel"] == "channel_1"
    assert result["error_msg"] is None
    assert isinstance(result["best_epoch"], int)
    assert isinstance(result["best_val_loss"], float)
    assert isinstance(result["epochs_run"], int)


@pytest.mark.slow
def test_train_task_error_on_missing_data(ray_local) -> None:
    """train task should return status='error' gracefully when data is missing."""
    pytest.importorskip("ray")
    import ray

    from spacecraft_telemetry.core.config import load_settings
    from spacecraft_telemetry.ray_training.tasks import make_train_task

    settings = load_settings("test")
    task = make_train_task(num_gpus=0.0)
    settings_ref = ray.put(settings)
    result = cast(dict[str, Any], ray.get(task.remote(settings_ref, "ESA-Mission1", "nonexistent_channel")))

    assert result["status"] == "error"
    assert result["error_msg"] is not None
    assert result["channel"] == "nonexistent_channel"
    assert result["best_epoch"] is None
    assert result["best_val_loss"] is None


@pytest.mark.slow
def test_score_task_ok(ray_local, pretrained_channel) -> None:
    """score task should return status='ok' after a channel has been trained."""
    pytest.importorskip("ray")
    import ray

    from spacecraft_telemetry.ray_training.tasks import make_score_task

    settings = pretrained_channel
    task = make_score_task(num_gpus=0.0)
    settings_ref = ray.put(settings)
    result = cast(dict[str, Any], ray.get(task.remote(settings_ref, "ESA-Mission1", "channel_1")))

    assert result["status"] == "ok", f"Expected ok, got: {result.get('error_msg')}"
    assert result["channel"] == "channel_1"
    assert result["error_msg"] is None
    for key in ("precision", "recall", "f1", "f0_5"):
        assert key in result
        assert 0.0 <= result[key] <= 1.0, f"{key} out of range"


@pytest.mark.slow
def test_score_task_error_on_missing_model(ray_local, ray_series_parquet) -> None:
    """score task should return status='error' when no model artifact exists."""
    pytest.importorskip("ray")
    import ray

    from spacecraft_telemetry.ray_training.tasks import make_score_task

    settings = ray_series_parquet
    task = make_score_task(num_gpus=0.0)
    settings_ref = ray.put(settings)
    result = cast(dict[str, Any], ray.get(task.remote(settings_ref, "ESA-Mission1", "channel_99")))

    assert result["status"] == "error"
    assert result["error_msg"] is not None
