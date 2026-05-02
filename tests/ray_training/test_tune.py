"""Unit and integration tests for ray_training/tune.py."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest

from spacecraft_telemetry.core.config import load_settings


def test_write_tuned_configs_writes_json(tmp_path: Path) -> None:
    """write_tuned_configs writes a stable JSON mapping to disk."""
    from spacecraft_telemetry.ray_training.tune import write_tuned_configs

    out = tmp_path / "models" / "ESA-Mission1" / "tuned_configs.json"
    payload = {
        "subsystem_1": {
            "threshold_z": 2.8,
            "threshold_window": 200,
            "error_smoothing_window": 25,
            "threshold_min_anomaly_len": 3,
        }
    }

    write_tuned_configs(payload, out)

    assert out.exists()
    assert json.loads(out.read_text()) == payload


def test_validate_trial_inputs_shape_mismatch_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Preflight validation should fail fast on labels/errors shape mismatch."""
    from spacecraft_telemetry.ray_training.tune import _validate_trial_inputs

    settings = load_settings("test")

    class _Paths:
        errors = "fake-errors.npy"

    def _fake_artifact_paths(*_args, **_kwargs):
        return _Paths()

    def _fake_read_bytes(*_args, **_kwargs):
        arr = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        buf = io.BytesIO()
        np.save(buf, arr)
        return buf.getvalue()

    def _fake_labels(*_args, **_kwargs):
        return np.array([True, False], dtype=np.bool_)

    monkeypatch.setattr("spacecraft_telemetry.model.io.artifact_paths", _fake_artifact_paths)
    monkeypatch.setattr("spacecraft_telemetry.model.io._read_bytes", _fake_read_bytes)
    monkeypatch.setattr("spacecraft_telemetry.model.dataset.load_window_labels", _fake_labels)

    with pytest.raises(ValueError, match="input mismatch"):
        _validate_trial_inputs(settings, "ESA-Mission1", ["channel_1"])


def test_scoring_trial_returns_metric(monkeypatch: pytest.MonkeyPatch) -> None:
    """_scoring_trial returns a final metrics dict containing f0_5."""
    from spacecraft_telemetry.ray_training.tune import _scoring_trial

    settings = load_settings("test")

    class _Paths:
        errors = "fake-errors.npy"

    def _fake_artifact_paths(*_args, **_kwargs):
        return _Paths()

    def _fake_read_bytes(*_args, **_kwargs):
        arr = np.array([0.1, 0.2, 0.5, 0.9], dtype=np.float64)
        buf = io.BytesIO()
        np.save(buf, arr)
        return buf.getvalue()

    def _fake_labels(*_args, **_kwargs):
        return np.array([False, False, True, True], dtype=np.bool_)

    monkeypatch.setattr("spacecraft_telemetry.model.io.artifact_paths", _fake_artifact_paths)
    monkeypatch.setattr("spacecraft_telemetry.model.io._read_bytes", _fake_read_bytes)
    monkeypatch.setattr("spacecraft_telemetry.model.dataset.load_window_labels", _fake_labels)

    result = _scoring_trial(
        {
            "error_smoothing_window": 5,
            "threshold_window": 3,
            "threshold_z": 2.0,
            "threshold_min_anomaly_len": 1,
        },
        settings=settings,
        mission="ESA-Mission1",
        channels=["channel_1"],
    )

    assert "f0_5" in result
    assert isinstance(result["f0_5"], float)


@pytest.mark.slow
def test_run_hpo_sweep_smoke(ray_local, ray_series_parquet, tmp_path: Path) -> None:
    """run_hpo_sweep runs end-to-end on one channel with tiny sample count."""
    pytest.importorskip("ray")

    from spacecraft_telemetry.model.scoring import score_channel
    from spacecraft_telemetry.model.training import train_channel
    from spacecraft_telemetry.ray_training.tune import run_hpo_sweep

    mission = "ESA-Mission1"
    channel = "channel_1"

    settings = ray_series_parquet.model_copy(
        update={
            "tune": ray_series_parquet.tune.model_copy(
                update={
                    "num_samples": 2,
                    "max_concurrent_trials": 1,
                    "mlflow_tracking_uri": str(tmp_path / "mlruns"),
                    "mlflow_experiment_prefix": "test-hpo",
                }
            )
        }
    )

    train_channel(settings, mission, channel)
    score_channel(settings, mission, channel)

    best = run_hpo_sweep("subsystem_1", [channel], settings, mission)
    assert set(best.keys()) == {
        "error_smoothing_window",
        "threshold_window",
        "threshold_z",
        "threshold_min_anomaly_len",
    }
