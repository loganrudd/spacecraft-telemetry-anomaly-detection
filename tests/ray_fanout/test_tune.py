"""Unit and integration tests for ray_fanout/tune.py."""

from __future__ import annotations

import json
import typing
from pathlib import Path

import numpy as np
import pytest

from spacecraft_telemetry.core.config import load_settings


def test_write_tuned_configs_writes_json(tmp_path: Path) -> None:
    """write_tuned_configs writes a stable JSON mapping to disk."""
    from spacecraft_telemetry.ray_fanout.tune import write_tuned_configs

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
    from spacecraft_telemetry.ray_fanout.tune import write_tuned_configs

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
    from spacecraft_telemetry.ray_fanout.tune import _prepare_channel_data

    settings = load_settings("test")

    # 3-element errors array.
    _errors_bytes = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    import io as _io
    _buf = _io.BytesIO()
    np.save(_buf, _errors_bytes)
    _raw = _buf.getvalue()

    class _FakeRun:
        class info:
            run_id = "fake-run-id"

    monkeypatch.setattr(
        "spacecraft_telemetry.ray_fanout.tune.find_latest_run_for_channel",
        lambda *_args, **_kwargs: _FakeRun(),
    )
    monkeypatch.setattr(
        "spacecraft_telemetry.ray_fanout.tune.download_artifact_bytes",
        lambda *_args, **_kwargs: _raw,
    )
    monkeypatch.setattr(
        "spacecraft_telemetry.model.dataset.load_window_labels",
        # 2-element labels — shape mismatch with 3-element errors.
        lambda *_args, **_kwargs: np.array([True, False], dtype=np.bool_),
    )

    with pytest.raises(ValueError, match="input mismatch"):
        _prepare_channel_data(settings, "ESA-Mission1", ["channel_1"])


def test_scoring_trial_returns_metric(monkeypatch: pytest.MonkeyPatch) -> None:
    """_scoring_trial returns a final metrics dict containing f0_5 and objective."""
    from spacecraft_telemetry.ray_fanout.tune import _scoring_trial

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
        nominal_errors={},
        fp_penalty_weight=5.0,
    )

    assert "f0_5" in result
    assert isinstance(result["f0_5"], float)
    assert "seg_f0_5" in result
    assert isinstance(result["seg_f0_5"], float)
    # No nominal_errors provided -> zero penalty -> objective == seg_f0_5.
    assert result["nominal_fp_rate"] == 0.0
    assert result["objective"] == result["seg_f0_5"]


def test_scoring_trial_nominal_fp_penalizes_objective(monkeypatch: pytest.MonkeyPatch) -> None:
    """A config that fires constantly on nominal data is penalized in objective."""
    from spacecraft_telemetry.ray_fanout.tune import _scoring_trial

    _ = monkeypatch

    channel_data = {
        "channel_1": (
            np.array([0.1, 0.2, 0.5, 0.9], dtype=np.float64),
            np.array([False, False, True, True], dtype=np.bool_),
        )
    }
    # Monotonically increasing nominal errors -> smoothed value always exceeds
    # the prior window's (mean + tiny z*std) -> near-1.0 fp_rate with z~0.
    nominal_errors = {"channel_1": np.linspace(0.0, 5.0, 20, dtype=np.float64)}

    config = {
        "error_smoothing_window": 5,
        "threshold_window": 3,
        "threshold_z": 0.01,
        "threshold_min_anomaly_len": 1,
    }

    result = _scoring_trial(
        config, channel_data=channel_data, nominal_errors=nominal_errors, fp_penalty_weight=5.0
    )

    assert result["nominal_fp_rate"] > 0.0
    assert result["objective"] < result["seg_f0_5"]


def test_resilient_mlflow_callback_swallows_unregistered_trial() -> None:
    """The resilient callback must not raise when a trial was never registered.

    Reproduces the upstream Ray bug: log_trial_start's start_run() failed (so
    the trial is absent from _trial_runs), then the trial errors and
    on_trial_error -> log_trial_end does self._trial_runs[trial] -> KeyError,
    which killed the whole RayJob. The subclass should log-and-continue instead.
    """
    pytest.importorskip("ray")
    from spacecraft_telemetry.ray_fanout.tune import _resilient_mlflow_callback

    cb = _resilient_mlflow_callback(experiment_name="telemanom-hpo-ISS", tags={})
    # setup() would connect to MLflow; emulate a post-setup state where a trial
    # failed at start_run() and so is absent from _trial_runs.
    cb._trial_runs = {}
    cb.should_save_artifact = False

    class _FakeTrial:
        config: typing.ClassVar[dict[str, object]] = {}

        def __str__(self) -> str:
            return "_scoring_trial_deadbeef"

    # Both the error path (log_trial_end failed=True) and a stray result must
    # not propagate — upstream these would KeyError.
    cb.log_trial_end(_FakeTrial(), failed=True)
    cb.log_trial_result(0, _FakeTrial(), {"training_iteration": 1, "objective": 0.1})


def test_run_hpo_sweep_requires_initialized_ray(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_hpo_sweep should fail fast if caller does not own Ray session."""
    import ray

    from spacecraft_telemetry.ray_fanout.tune import run_hpo_sweep

    monkeypatch.setattr(ray, "is_initialized", lambda: False)
    settings = load_settings("test")

    with pytest.raises(RuntimeError, match="Ray is not initialized"):
        run_hpo_sweep("subsystem_1", ["channel_1"], settings, "ESA-Mission1")


def test_run_all_sweeps_no_eligible_writes_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """run_all_sweeps should write empty JSON when no scored channels are eligible."""
    import ray

    from spacecraft_telemetry.ray_fanout.tune import run_all_sweeps

    settings = load_settings("test").model_copy(
        update={
            "model": load_settings("test").model.model_copy(
                update={"artifacts_dir": tmp_path / "models"}
            )
        }
    )

    monkeypatch.setattr(ray, "is_initialized", lambda: True)
    monkeypatch.setattr(
        "spacecraft_telemetry.ray_fanout.tune.load_channel_subsystem_map",
        lambda *_args, **_kwargs: {"channel_1": "subsystem_1"},
    )
    # No scoring run exists for any channel → all channels ineligible.
    monkeypatch.setattr(
        "spacecraft_telemetry.ray_fanout.tune.find_latest_run_for_channel",
        lambda *_args, **_kwargs: None,
    )

    out = run_all_sweeps(settings, "ESA-Mission1", ["channel_1"])
    assert out.exists()
    assert json.loads(out.read_text()) == {}


def test_run_all_sweeps_filters_and_runs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """run_all_sweeps should run only eligible scored channels per subsystem."""
    import ray

    from spacecraft_telemetry.ray_fanout.tune import run_all_sweeps

    base_settings = load_settings("test")
    settings = base_settings.model_copy(
        update={
            "model": base_settings.model.model_copy(update={"artifacts_dir": tmp_path / "models"}),
            "tune": base_settings.tune.model_copy(update={"parallel_subsystems": False}),
        }
    )

    # channel_1 has a scoring run in MLflow; channel_2 and channel_3 do not.
    class _FakeRun:
        class info:
            run_id = "fake-scored-run-id"

    def _fake_find_latest_run(exp: str, ch: str, uri: str):
        return _FakeRun() if ch == "channel_1" else None

    monkeypatch.setattr(ray, "is_initialized", lambda: True)
    monkeypatch.setattr(
        "spacecraft_telemetry.ray_fanout.tune.find_latest_run_for_channel",
        _fake_find_latest_run,
    )
    monkeypatch.setattr(
        "spacecraft_telemetry.ray_fanout.tune.load_channel_subsystem_map",
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
            "seg_f0_5": 0.75,
            "run_id": "fake-run-id-abc",
        }

    monkeypatch.setattr(
        "spacecraft_telemetry.ray_fanout.tune.run_hpo_sweep",
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
    assert entry["_meta"]["seg_f0_5"] == pytest.approx(0.75)


@pytest.mark.parametrize(
    ("mission", "variant", "expected_space_name", "threshold_z_bounds"),
    [
        ("ESA-Mission1", None, "ESA_M1_SEARCH_SPACE", (2.5, 8.0)),
        # A real ESA-Mission1 variant (mission unchanged, variant set) keeps
        # the widened space — the selector is mission-keyed, not name-keyed.
        ("ESA-Mission1", "adb-24m", "ESA_M1_SEARCH_SPACE", (2.5, 8.0)),
        # A legacy un-migrated ESA-Mission1-ADB* pseudo-mission is a DIFFERENT
        # `mission` string entirely and correctly falls through to the default
        # space — this is the plan 020 fix (was `mission.startswith(...)`, a
        # pseudo-mission prefix test); migrate via plan 020 stage 020.5 to
        # regain the widened space under mission=ESA-Mission1 + a variant.
        ("ESA-Mission1-ADB", None, "SEARCH_SPACE", (2.5, 5.0)),
        ("ISS", None, "ISS_SEARCH_SPACE", (2.5, 5.0)),
        ("ESA-Mission2", None, "SEARCH_SPACE", (2.5, 5.0)),
    ],
)
def test_run_all_sweeps_selects_search_space_by_mission(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mission: str,
    variant: str | None,
    expected_space_name: str,
    threshold_z_bounds: tuple[float, float],
) -> None:
    """The search-space selector must route each (mission, variant) to the right space.

    The ISS row is the valuable half: ISS_SEARCH_SPACE = {**SEARCH_SPACE, ...},
    so the 2026-08-14 widening of the base SEARCH_SPACE for ESA-Mission1 would
    silently widen ISS too if `mission.startswith("ISS")` were ever checked
    after (or dropped in favour of) the ESA-Mission1 branch.
    """
    import ray

    from spacecraft_telemetry.ray_fanout import tune as tune_module

    expected_space = getattr(tune_module, expected_space_name)

    base_settings = load_settings("test")
    settings = base_settings.model_copy(
        update={
            "model": base_settings.model.model_copy(update={"artifacts_dir": tmp_path / "models"}),
            "tune": base_settings.tune.model_copy(update={"parallel_subsystems": False}),
            "variant": variant,
        }
    )

    class _FakeRun:
        class info:
            run_id = "fake-scored-run-id"

    monkeypatch.setattr(ray, "is_initialized", lambda: True)
    monkeypatch.setattr(
        "spacecraft_telemetry.ray_fanout.tune.load_channel_subsystem_map",
        lambda *_args, **_kwargs: {"channel_1": "subsystem_1"},
    )
    monkeypatch.setattr(
        "spacecraft_telemetry.ray_fanout.tune.find_latest_run_for_channel",
        lambda *_args, **_kwargs: _FakeRun(),
    )

    captured: dict[str, object] = {}

    def _fake_run_hpo_sweep(subsystem, channels, settings, mission, *, search_space=None):
        captured["search_space"] = search_space
        return {
            "config": {
                "error_smoothing_window": 10,
                "threshold_window": 100,
                "threshold_z": 2.5,
                "threshold_min_anomaly_len": 2,
            },
            "seg_f0_5": 0.5,
            "run_id": "fake-run-id",
        }

    monkeypatch.setattr(tune_module, "run_hpo_sweep", _fake_run_hpo_sweep)

    tune_module.run_all_sweeps(settings, mission, ["channel_1"])

    assert captured["search_space"] is expected_space
    z = captured["search_space"]["threshold_z"]
    assert (z.lower, z.upper) == threshold_z_bounds


def _pinned_params(
    config: dict[str, float], search_space: dict[str, object], epsilon: float = 0.05
) -> list[str]:
    """Return the names of any ``config`` params within ``epsilon`` of their search-space bound.

    Skips params whose search-space entry is a fixed value rather than a Ray
    Tune distribution (e.g. ISS_SEARCH_SPACE's ``min_error_value: 0.0``) —
    those have no bound to pin against. A pinned optimum means the search
    space is a confound rather than a genuine optimum (see ESA_M1_SEARCH_SPACE's
    docstring) — this is a cheap assertion on a tuned-config fixture, not a
    live-sweep test.
    """
    pinned = []
    for name, value in config.items():
        dist = search_space.get(name)
        lower = getattr(dist, "lower", None)
        upper = getattr(dist, "upper", None)
        if lower is None or upper is None:
            continue
        if abs(value - lower) <= epsilon or abs(value - upper) <= epsilon:
            pinned.append(name)
    return pinned


class TestPinnedParamDetection:
    """Regression coverage for the 2026-08-15 evening re-tune finding.

    subsystem_5's threshold_z landed at 2.511 against ESA_M1_SEARCH_SPACE's
    lower bound of 2.5, and subsystem_3's min_error_value landed at 0.5685
    against the upper bound of 0.6 (see docs/reviews/019-esa-adb-comparable-eval.md).
    Both are cheap fixtures pinned here so a future re-discovery is a single
    failing assertion instead of a fresh investigation.
    """

    def test_flags_param_pinned_to_lower_bound(self) -> None:
        from spacecraft_telemetry.ray_fanout.tune import ESA_M1_SEARCH_SPACE

        config = {"threshold_z": 2.511, "min_error_value": 0.1}
        assert _pinned_params(config, ESA_M1_SEARCH_SPACE) == ["threshold_z"]

    def test_flags_param_pinned_to_upper_bound(self) -> None:
        from spacecraft_telemetry.ray_fanout.tune import ESA_M1_SEARCH_SPACE

        config = {"threshold_z": 4.0, "min_error_value": 0.5685}
        assert _pinned_params(config, ESA_M1_SEARCH_SPACE) == ["min_error_value"]

    def test_interior_config_flags_nothing(self) -> None:
        from spacecraft_telemetry.ray_fanout.tune import ESA_M1_SEARCH_SPACE

        config = {"threshold_z": 4.0, "min_error_value": 0.3}
        assert _pinned_params(config, ESA_M1_SEARCH_SPACE) == []

    def test_fixed_iss_min_error_value_is_not_flagged(self) -> None:
        """ISS_SEARCH_SPACE pins min_error_value=0.0 (a fixed value, no distribution)."""
        from spacecraft_telemetry.ray_fanout.tune import ISS_SEARCH_SPACE

        config = {"min_error_value": 0.0, "threshold_z": 2.5}
        assert _pinned_params(config, ISS_SEARCH_SPACE) == ["threshold_z"]


def test_hpo_portion_slicing(monkeypatch: pytest.MonkeyPatch) -> None:
    """_prepare_channel_data returns slices of length floor(N * hpo_eval_fraction)."""
    from spacecraft_telemetry.ray_fanout.tune import _prepare_channel_data

    settings = load_settings("test")  # hpo_eval_fraction = 0.6
    n_total = 10
    expected_n = int(n_total * settings.tune.hpo_eval_fraction)  # floor(10 * 0.6) = 6

    import io as _io
    _buf = _io.BytesIO()
    np.save(_buf, np.ones(n_total, dtype=np.float64))
    _raw = _buf.getvalue()

    class _FakeRun:
        class info:
            run_id = "fake-run-id"

    monkeypatch.setattr(
        "spacecraft_telemetry.ray_fanout.tune.find_latest_run_for_channel",
        lambda *_args, **_kwargs: _FakeRun(),
    )
    monkeypatch.setattr(
        "spacecraft_telemetry.ray_fanout.tune.download_artifact_bytes",
        lambda *_args, **_kwargs: _raw,
    )
    monkeypatch.setattr(
        "spacecraft_telemetry.model.dataset.load_window_labels",
        lambda *_args, **_kwargs: np.ones(n_total, dtype=np.bool_),
    )

    result = _prepare_channel_data(settings, "ESA-Mission1", ["channel_1"])

    channel_data, scoring_run_ids = result
    assert "channel_1" in channel_data
    errors, labels = channel_data["channel_1"]
    assert len(errors) == expected_n
    assert len(labels) == expected_n
    assert scoring_run_ids.get("channel_1") == "fake-run-id"


def test_warns_when_held_out_has_no_anomalies(monkeypatch: pytest.MonkeyPatch) -> None:
    """_prepare_channel_data warns when held-out portion contains no anomalies."""
    from spacecraft_telemetry.ray_fanout.tune import _prepare_channel_data

    settings = load_settings("test")  # hpo_eval_fraction = 0.6
    n_total = 10  # HPO: first 6, held-out: last 4

    import io as _io
    _buf = _io.BytesIO()
    np.save(_buf, np.ones(n_total, dtype=np.float64))
    _raw = _buf.getvalue()

    class _FakeRun:
        class info:
            run_id = "fake-run-id"

    def _fake_labels(*_args, **_kwargs):
        labels = np.zeros(n_total, dtype=np.bool_)
        labels[:6] = True  # all anomalies in HPO portion, none in held-out tail
        return labels

    monkeypatch.setattr(
        "spacecraft_telemetry.ray_fanout.tune.find_latest_run_for_channel",
        lambda *_args, **_kwargs: _FakeRun(),
    )
    monkeypatch.setattr(
        "spacecraft_telemetry.ray_fanout.tune.download_artifact_bytes",
        lambda *_args, **_kwargs: _raw,
    )
    monkeypatch.setattr("spacecraft_telemetry.model.dataset.load_window_labels", _fake_labels)

    with pytest.warns(UserWarning, match="Held-out portion"):
        _prepare_channel_data(settings, "ESA-Mission1", ["channel_1"])


@pytest.mark.slow
def test_run_hpo_sweep_smoke(ray_local, ray_series_parquet, tmp_path: Path) -> None:
    """run_hpo_sweep runs end-to-end on one channel with tiny sample count."""
    pytest.importorskip("ray")

    from spacecraft_telemetry.model.scoring import score_channel
    from spacecraft_telemetry.model.training import train_channel
    from spacecraft_telemetry.ray_fanout.tune import run_hpo_sweep

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
    assert set(best.keys()) == {
        "config", "seg_f0_5", "nominal_fp_rate", "objective", "run_id",
    }
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
    assert isinstance(best["seg_f0_5"], float)
    assert best["run_id"] is None or isinstance(best["run_id"], str)
