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


def test_write_tuned_configs_roundtrips_meta(tmp_path: Path) -> None:
    """write_tuned_configs preserves the _meta block for HPO lineage tracking."""
    from spacecraft_telemetry.ray_training.tune import write_tuned_configs

    out = tmp_path / "tuned_configs.json"
    payload = {
        "subsystem_1": {
            "threshold_z": 2.8,
            "threshold_window": 200,
            "error_smoothing_window": 25,
            "threshold_min_anomaly_len": 3,
            "_meta": {"run_id": "abc123", "f0_5": 0.72},
        }
    }
    write_tuned_configs(payload, out)
    loaded = json.loads(out.read_text())
    assert loaded["subsystem_1"]["_meta"]["run_id"] == "abc123"
    assert loaded["subsystem_1"]["_meta"]["f0_5"] == pytest.approx(0.72)


def test_prepare_channel_data_shape_mismatch_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Preparation should fail fast on labels/errors shape mismatch."""
    from spacecraft_telemetry.ray_training.tune import _prepare_channel_data

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
        _prepare_channel_data(settings, "ESA-Mission1", ["channel_1"])


def test_scoring_trial_returns_metric(monkeypatch: pytest.MonkeyPatch) -> None:
    """_scoring_trial returns a final metrics dict containing f0_5."""
    from spacecraft_telemetry.ray_training.tune import _scoring_trial

    _ = monkeypatch

    channel_data = {
        "channel_1": (
            np.array([0.1, 0.2, 0.5, 0.9], dtype=np.float64),
            np.array([False, False, True, True], dtype=np.bool_),
        )
    }

    result = _scoring_trial(
        {
            "error_smoothing_window": 5,
            "threshold_window": 3,
            "threshold_z": 2.0,
            "threshold_min_anomaly_len": 1,
        },
        channel_data=channel_data,
    )

    assert "f0_5" in result
    assert isinstance(result["f0_5"], float)


def test_run_hpo_sweep_requires_initialized_ray(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_hpo_sweep should fail fast if caller does not own Ray session."""
    import ray

    from spacecraft_telemetry.ray_training.tune import run_hpo_sweep

    monkeypatch.setattr(ray, "is_initialized", lambda: False)
    settings = load_settings("test")

    with pytest.raises(RuntimeError, match="Ray is not initialized"):
        run_hpo_sweep("subsystem_1", ["channel_1"], settings, "ESA-Mission1")


def test_run_all_sweeps_no_eligible_writes_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """run_all_sweeps should write empty JSON when no scored channels are eligible."""
    import ray

    from spacecraft_telemetry.ray_training.tune import run_all_sweeps

    settings = load_settings("test").model_copy(
        update={
            "model": load_settings("test").model.model_copy(
                update={"artifacts_dir": tmp_path / "models"}
            )
        }
    )

    monkeypatch.setattr(ray, "is_initialized", lambda: True)
    monkeypatch.setattr(
        "spacecraft_telemetry.ray_training.tune.load_channel_subsystem_map",
        lambda *_args, **_kwargs: {"channel_1": "subsystem_1"},
    )

    out = run_all_sweeps(settings, "ESA-Mission1", ["channel_1"])
    assert out.exists()
    assert json.loads(out.read_text()) == {}


def test_run_all_sweeps_filters_and_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """run_all_sweeps should run only eligible scored channels per subsystem."""
    import ray

    from spacecraft_telemetry.model.io import artifact_paths
    from spacecraft_telemetry.ray_training.tune import run_all_sweeps

    base_settings = load_settings("test")
    settings = base_settings.model_copy(
        update={
            "model": base_settings.model.model_copy(update={"artifacts_dir": tmp_path / "models"}),
            "tune": base_settings.tune.model_copy(update={"parallel_subsystems": False}),
        }
    )

    # Only channel_1 is scored.
    ch1 = artifact_paths(settings, "ESA-Mission1", "channel_1").errors
    Path(str(ch1)).parent.mkdir(parents=True, exist_ok=True)
    Path(str(ch1)).write_bytes(b"not_used_by_this_test")

    monkeypatch.setattr(ray, "is_initialized", lambda: True)
    monkeypatch.setattr(
        "spacecraft_telemetry.ray_training.tune.load_channel_subsystem_map",
        lambda *_args, **_kwargs: {
            "channel_1": "subsystem_1",
            "channel_2": "subsystem_1",
            "channel_3": "subsystem_6",
        },
    )

    calls: list[tuple[str, list[str]]] = []

    def _fake_run_hpo_sweep(subsystem: str, channels: list[str], *_args, **_kwargs):
        calls.append((subsystem, channels))
        return {
            "config": {
                "error_smoothing_window": 10,
                "threshold_window": 100,
                "threshold_z": 2.5,
                "threshold_min_anomaly_len": 2,
            },
            "f0_5": 0.75,
            "run_id": "fake-run-id-abc",
        }

    monkeypatch.setattr(
        "spacecraft_telemetry.ray_training.tune.run_hpo_sweep",
        _fake_run_hpo_sweep,
    )

    out = run_all_sweeps(settings, "ESA-Mission1", ["channel_1", "channel_2", "channel_3"])
    assert out.exists()
    loaded = json.loads(out.read_text())
    assert list(loaded) == ["subsystem_1"]
    assert calls == [("subsystem_1", ["channel_1"])]
    entry = loaded["subsystem_1"]
    assert entry["error_smoothing_window"] == 10
    assert entry["threshold_z"] == 2.5
    assert "_meta" in entry
    assert entry["_meta"]["run_id"] == "fake-run-id-abc"
    assert entry["_meta"]["f0_5"] == pytest.approx(0.75)


def test_hpo_portion_slicing(monkeypatch: pytest.MonkeyPatch) -> None:
    """_prepare_channel_data returns slices of length floor(N * hpo_eval_fraction)."""
    from spacecraft_telemetry.ray_training.tune import _prepare_channel_data

    settings = load_settings("test")  # hpo_eval_fraction = 0.6
    n_total = 10
    expected_n = int(n_total * settings.tune.hpo_eval_fraction)  # floor(10 * 0.6) = 6

    class _Paths:
        errors = "fake-errors.npy"

    def _fake_artifact_paths(*_args, **_kwargs):
        return _Paths()

    def _fake_read_bytes(*_args, **_kwargs):
        buf = io.BytesIO()
        np.save(buf, np.ones(n_total, dtype=np.float64))
        return buf.getvalue()

    def _fake_labels(*_args, **_kwargs):
        return np.ones(n_total, dtype=np.bool_)  # anomalies in both halves

    monkeypatch.setattr("spacecraft_telemetry.model.io.artifact_paths", _fake_artifact_paths)
    monkeypatch.setattr("spacecraft_telemetry.model.io._read_bytes", _fake_read_bytes)
    monkeypatch.setattr("spacecraft_telemetry.model.dataset.load_window_labels", _fake_labels)

    result = _prepare_channel_data(settings, "ESA-Mission1", ["channel_1"])

    assert "channel_1" in result
    errors, labels = result["channel_1"]
    assert len(errors) == expected_n
    assert len(labels) == expected_n


def test_warns_when_held_out_has_no_anomalies(monkeypatch: pytest.MonkeyPatch) -> None:
    """_prepare_channel_data warns when held-out portion contains no anomalies."""
    from spacecraft_telemetry.ray_training.tune import _prepare_channel_data

    settings = load_settings("test")  # hpo_eval_fraction = 0.6
    n_total = 10  # HPO: first 6, held-out: last 4

    class _Paths:
        errors = "fake-errors.npy"

    def _fake_artifact_paths(*_args, **_kwargs):
        return _Paths()

    def _fake_read_bytes(*_args, **_kwargs):
        buf = io.BytesIO()
        np.save(buf, np.ones(n_total, dtype=np.float64))
        return buf.getvalue()

    def _fake_labels(*_args, **_kwargs):
        labels = np.zeros(n_total, dtype=np.bool_)
        labels[:6] = True  # all anomalies in HPO portion, none in held-out tail
        return labels

    monkeypatch.setattr("spacecraft_telemetry.model.io.artifact_paths", _fake_artifact_paths)
    monkeypatch.setattr("spacecraft_telemetry.model.io._read_bytes", _fake_read_bytes)
    monkeypatch.setattr("spacecraft_telemetry.model.dataset.load_window_labels", _fake_labels)

    with pytest.warns(UserWarning, match="Held-out portion"):
        _prepare_channel_data(settings, "ESA-Mission1", ["channel_1"])


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
                }
            ),
            "mlflow": ray_series_parquet.mlflow.model_copy(
                update={"tracking_uri": f"sqlite:///{tmp_path}/mlflow.db"}
            ),
        }
    )

    train_channel(settings, mission, channel)
    score_channel(settings, mission, channel)

    best = run_hpo_sweep("subsystem_1", [channel], settings, mission)
    assert set(best.keys()) == {"config", "f0_5", "run_id"}
    config = best["config"]
    assert set(config.keys()) == {
        "error_smoothing_window",
        "threshold_window",
        "threshold_z",
        "threshold_min_anomaly_len",
    }
    assert isinstance(config["error_smoothing_window"], int)
    assert isinstance(config["threshold_window"], int)
    assert isinstance(config["threshold_min_anomaly_len"], int)
    assert isinstance(config["threshold_z"], float)
    assert 5 <= config["error_smoothing_window"] <= 100
    assert 50 <= config["threshold_window"] <= 500
    assert 1 <= config["threshold_min_anomaly_len"] <= 10
    assert 1.5 <= config["threshold_z"] <= 5.0
    assert isinstance(best["f0_5"], float)
    assert best["run_id"] is None or isinstance(best["run_id"], str)
