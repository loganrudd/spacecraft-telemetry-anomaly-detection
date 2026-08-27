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


# ---------------------------------------------------------------------------
# _find_channel_errors_run / _find_multivariate_scoring_run (docs/plans/021)
# ---------------------------------------------------------------------------


class _FakeRun:
    def __init__(self, run_id: str, tags: dict[str, str] | None = None) -> None:
        class _Info:
            def __init__(self, rid: str) -> None:
                self.run_id = rid

        class _Data:
            def __init__(self, t: dict[str, str]) -> None:
                self.tags = t

        self.info = _Info(run_id)
        self.data = _Data(tags or {})


def test_find_channel_errors_run_prefers_univariate(monkeypatch: pytest.MonkeyPatch) -> None:
    """The univariate per-channel run wins when one exists — no fallback call."""
    from spacecraft_telemetry.ray_fanout.tune import _find_channel_errors_run

    settings = load_settings("test")
    monkeypatch.setattr(
        "spacecraft_telemetry.ray_fanout.tune.find_latest_run_for_channel",
        lambda *_a, **_k: _FakeRun("uni-run"),
    )

    def _fail(*_a: object, **_k: object) -> None:
        raise AssertionError("multivariate fallback should not be called")

    monkeypatch.setattr(
        "spacecraft_telemetry.ray_fanout.tune._find_multivariate_scoring_run", _fail
    )

    found = _find_channel_errors_run(settings, "ESA-Mission1", "channel_1", "exp")
    assert found is not None
    run, artifact_path = found
    assert run.info.run_id == "uni-run"
    assert artifact_path == "errors.npy"


def test_find_channel_errors_run_falls_back_to_multivariate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No univariate run -> the multivariate subsystem run is used instead,
    reading its per-channel errors/{channel}.npy artifact."""
    from spacecraft_telemetry.ray_fanout.tune import _find_channel_errors_run

    settings = load_settings("test")
    monkeypatch.setattr(
        "spacecraft_telemetry.ray_fanout.tune.find_latest_run_for_channel",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        "spacecraft_telemetry.ray_fanout.tune.load_channel_group_map",
        lambda *_a, **_k: {"channel_41": "subsystem_1"},
    )
    monkeypatch.setattr(
        "spacecraft_telemetry.ray_fanout.tune.find_latest_run_by_tag",
        lambda *_a, **_k: _FakeRun("mv-run", tags={"channels": "channel_41,channel_42"}),
    )

    found = _find_channel_errors_run(settings, "ESA-Mission1", "channel_41", "exp")
    assert found is not None
    run, artifact_path = found
    assert run.info.run_id == "mv-run"
    assert artifact_path == "errors/channel_41.npy"


def test_find_multivariate_scoring_run_rejects_non_member_channel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A subsystem-tagged run that doesn't actually list this channel must
    not be trusted — same subsystem name, different (or narrower) group."""
    from spacecraft_telemetry.ray_fanout.tune import _find_channel_errors_run

    settings = load_settings("test")
    monkeypatch.setattr(
        "spacecraft_telemetry.ray_fanout.tune.find_latest_run_for_channel",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        "spacecraft_telemetry.ray_fanout.tune.load_channel_group_map",
        lambda *_a, **_k: {"channel_99": "subsystem_1"},
    )
    monkeypatch.setattr(
        "spacecraft_telemetry.ray_fanout.tune.find_latest_run_by_tag",
        lambda *_a, **_k: _FakeRun("mv-run", tags={"channels": "channel_41,channel_42"}),
    )

    found = _find_channel_errors_run(settings, "ESA-Mission1", "channel_99", "exp")
    assert found is None


def test_find_channel_errors_run_swallows_fallback_exceptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broken subsystem-map/tracking-backend lookup degrades to 'not found',
    same as a genuine absence — never propagates and aborts the caller."""
    from spacecraft_telemetry.ray_fanout.tune import _find_channel_errors_run

    settings = load_settings("test")
    monkeypatch.setattr(
        "spacecraft_telemetry.ray_fanout.tune.find_latest_run_for_channel",
        lambda *_a, **_k: None,
    )

    def _raise(*_a: object, **_k: object) -> None:
        raise RuntimeError("tracking backend unreachable")

    monkeypatch.setattr(
        "spacecraft_telemetry.ray_fanout.tune.load_channel_subsystem_map", _raise
    )

    found = _find_channel_errors_run(settings, "ESA-Mission1", "channel_1", "exp")
    assert found is None


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

        # Real mlflow Runs always carry .data.tags; a univariate run simply has
        # no `channels` tag, which is what makes it univariate to the loader.
        class data:
            tags: typing.ClassVar[dict[str, str]] = {}

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


def test_prepare_channel_data_shape_mismatch_names_multivariate_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Running HPO with multivariate settings must name the ACTUAL cause.

    HPO consumes saved error arrays and never loads a model, so it has to run
    without input_channels. If someone sets them anyway, load_window_labels
    returns 2-D (M, C) labels against the 1-D (M,) per-channel errors a
    multivariate scoring run saved. The generic message blames
    window_size/prediction_horizon, which would send a reader hunting the
    wrong problem entirely (docs/plans/021-multivariate-telemanom.md).
    """
    import io as _io

    from spacecraft_telemetry.ray_fanout.tune import _prepare_channel_data

    channels = ["channel_41", "channel_42"]
    settings = load_settings("test")
    settings = settings.model_copy(
        update={
            "model": settings.model.model_copy(
                update={"input_channels": channels, "target_channels": channels}
            )
        }
    )

    _buf = _io.BytesIO()
    np.save(_buf, np.array([0.1, 0.2, 0.3], dtype=np.float64))  # 1-D (3,) errors
    _raw = _buf.getvalue()

    class _FakeRun:
        class info:
            run_id = "fake-run-id"

        # Real mlflow Runs always carry .data.tags; a univariate run simply has
        # no `channels` tag, which is what makes it univariate to the loader.
        class data:
            tags: typing.ClassVar[dict[str, str]] = {}

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
        # (3, 2) per-channel labels — what the multivariate path really returns.
        lambda *_args, **_kwargs: np.zeros((3, 2), dtype=np.bool_),
    )

    with pytest.raises(ValueError, match="input_channels is set"):
        _prepare_channel_data(settings, "ESA-Mission1", ["channel_41"])


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


# ---------------------------------------------------------------------------
# _scoring_trial — mission-level metric (docs/plans/021-multivariate-telemanom.md 021.4b)
# ---------------------------------------------------------------------------


def test_scoring_trial_omits_mission_metrics_by_default() -> None:
    """Without channel_timestamps/mission_events/mission_timeline_hpo, the
    return dict is exactly the pre-021.4b key set — no behaviour change for
    any existing caller."""
    from spacecraft_telemetry.ray_fanout.tune import _scoring_trial

    channel_data = {
        "channel_1": (
            np.array([0.1, 0.2, 0.5, 0.9], dtype=np.float64),
            np.array([False, False, True, True], dtype=np.bool_),
        )
    }
    config = {
        "error_smoothing_window": 2, "threshold_window": 2,
        "threshold_z": 1.0, "threshold_min_anomaly_len": 1,
    }

    result = _scoring_trial(
        config, channel_data=channel_data, nominal_errors={}, fp_penalty_weight=5.0
    )

    assert set(result.keys()) == {"f0_5", "seg_f0_5", "nominal_fp_rate", "objective"}


def test_scoring_trial_computes_mission_metrics_when_provided() -> None:
    """With ground truth supplied, the trial ALSO reports mission_precision/
    mission_recall/mission_f0_5 via esa_adb.metrics.corrected_event_wise —
    without changing objective (still seg_f0_5-based, risk 6)."""
    import pandas as pd

    from spacecraft_telemetry.esa_adb.events import Event
    from spacecraft_telemetry.ray_fanout.tune import _scoring_trial

    t0 = pd.Timestamp("2000-01-01", tz="UTC")
    # 5 windows, 90s apart; an anomaly flagged in windows 2-3 should be
    # matched by a ground-truth event spanning the same span. tz-naive, like
    # window_target_timestamps' real return (PyArrow's to_numpy() drops the
    # tz label — see _flags_to_intervals, which re-localizes to UTC).
    timestamps = pd.DatetimeIndex(
        [t0.tz_localize(None) + pd.Timedelta(seconds=90 * i) for i in range(5)]
    ).values
    channel_data = {
        "channel_1": (
            np.array([0.0, 0.0, 5.0, 5.0, 0.0], dtype=np.float64),
            np.array([False, False, True, True, False], dtype=np.bool_),
        )
    }
    channel_timestamps = {"channel_1": timestamps}
    mission_timeline_hpo = [(t0, t0 + pd.Timedelta(seconds=90 * 5))]
    mission_events = [
        Event(
            event_id="E1",
            category="Anomaly",
            intervals=((t0 + pd.Timedelta(seconds=180), t0 + pd.Timedelta(seconds=360)),),
            channels=frozenset({"channel_1"}),
        )
    ]
    config = {
        "error_smoothing_window": 1, "threshold_window": 2,
        "threshold_z": 1.0, "threshold_min_anomaly_len": 1,
    }

    result = _scoring_trial(
        config,
        channel_data=channel_data,
        nominal_errors={},
        fp_penalty_weight=5.0,
        channel_timestamps=channel_timestamps,
        mission_events=mission_events,
        mission_timeline_hpo=mission_timeline_hpo,
    )

    assert {"mission_precision", "mission_recall", "mission_f0_5"} <= set(result.keys())
    for key in ("mission_precision", "mission_recall", "mission_f0_5"):
        assert 0.0 <= result[key] <= 1.0
    # "objective" must stay seg_f0_5-based — the mission metric is additive,
    # never folded into what the sweep actually selects on.
    assert result["objective"] == pytest.approx(result["seg_f0_5"])


def test_run_hpo_sweep_trial_tags_unchanged_at_variant_none() -> None:
    """run_hpo_sweep's trial-run tags were refactored from an inline dict to
    common_tags() (docs/reviews/020-experiment-variant-axis.md §3.8) -- pin
    that the emitted tag set at variant=None is byte-identical to the
    pre-refactor dict, since run_hpo_sweep itself is too heavy (a real Tune
    loop) to assert on directly in a unit test.
    """
    from spacecraft_telemetry.mlflow_tracking.conventions import common_tags

    tags = common_tags(
        model_type="telemanom",
        mission="ESA-Mission1",
        phase="hpo",
        variant=None,
        subsystem="subsystem_1",
        extra={"eval_split": "hpo_portion"},
    )
    assert tags == {
        "model_type": "telemanom",
        "mission_id": "ESA-Mission1",
        "phase": "hpo",
        "subsystem": "subsystem_1",
        "eval_split": "hpo_portion",
    }


def test_run_hpo_sweep_trial_tags_include_variant_when_set() -> None:
    from spacecraft_telemetry.mlflow_tracking.conventions import common_tags

    tags = common_tags(
        model_type="telemanom",
        mission="ESA-Mission1",
        phase="hpo",
        variant="adb-24m",
        subsystem="subsystem_1",
        extra={"eval_split": "hpo_portion"},
    )
    assert tags == {
        "model_type": "telemanom",
        "mission_id": "ESA-Mission1",
        "phase": "hpo",
        "subsystem": "subsystem_1",
        "eval_split": "hpo_portion",
        "variant": "adb-24m",
    }


def test_run_all_sweeps_summary_run_tags_unchanged_at_variant_none() -> None:
    """Same pin as above for run_all_sweeps's tuned-configs-summary run tags."""
    from spacecraft_telemetry.mlflow_tracking.conventions import common_tags

    tags = common_tags(model_type="telemanom", mission="ESA-Mission1", phase="hpo", variant=None)
    assert tags == {
        "model_type": "telemanom",
        "mission_id": "ESA-Mission1",
        "phase": "hpo",
    }


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

    def _fake_find_latest_run(exp: str, ch: str, uri: str, extra_filter: str | None = None):
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
    meta = entry["_meta"]
    assert meta["run_id"] == "fake-run-id-abc"
    assert meta["seg_f0_5"] == pytest.approx(0.75)
    # docs/plans/022 stage 022.2 schema — shared with the exhaustive-grid
    # writer in scripts/threshold_ceiling.py.
    assert meta["provenance"] == "ray_tune"
    assert meta["objective_name"] == "mean_per_channel_seg_f0_5_minus_fp_penalty"
    # _fake_run_hpo_sweep's result has no "objective" key, so this falls back
    # to seg_f0_5 — same fallback _to_entry always applied.
    assert meta["objective_value"] == pytest.approx(0.75)
    assert meta["selected_on"] == "hpo_portion"
    assert meta["hpo_eval_fraction"] == settings.tune.hpo_eval_fraction
    assert meta["outer_split"] == "chronological_50_50"
    assert meta["error_smoothing_window"] == 10
    assert meta["threshold_window"] == 100
    assert meta["min_run_length"] == 2
    assert meta["axes"] is None
    assert meta["expansions"] is None
    assert meta["interior"] is None
    assert "objective" not in meta


@pytest.mark.parametrize(
    ("mission", "variant", "expected_space_name", "threshold_z_bounds"),
    [
        ("ESA-Mission1", None, "ESA_M1_SEARCH_SPACE", (1.0, 8.0)),
        # A real ESA-Mission1 variant (mission unchanged, variant set) keeps
        # the widened space — the selector is mission-keyed, not name-keyed.
        ("ESA-Mission1", "adb-24m", "ESA_M1_SEARCH_SPACE", (1.0, 8.0)),
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
    then-lower bound of 2.5, and subsystem_3's min_error_value landed at 0.5685
    against the upper bound of 0.6 (see docs/reviews/019-esa-adb-comparable-eval.md).
    Both are cheap fixtures pinned here so a future re-discovery is a single
    failing assertion instead of a fresh investigation.

    The threshold_z lower bound was subsequently dropped 2.5 -> 1.0 for ESA
    (docs/plans/021 stage 021.5b) precisely because configs kept landing on it.
    """

    def test_flags_param_pinned_to_lower_bound(self) -> None:
        from spacecraft_telemetry.ray_fanout.tune import ESA_M1_SEARCH_SPACE

        config = {"threshold_z": 1.02, "min_error_value": 0.1}
        assert _pinned_params(config, ESA_M1_SEARCH_SPACE) == ["threshold_z"]

    def test_historical_2_511_is_no_longer_pinned(self) -> None:
        """The 2026-08-15 finding, re-asserted as a *fix*: subsystem_5's
        threshold_z=2.511 sat on the old 2.5 floor. With the ESA floor now at
        1.0 that same value is comfortably interior, which is the whole point
        of the 021.5b change — the optimizer is no longer wall-limited there."""
        from spacecraft_telemetry.ray_fanout.tune import ESA_M1_SEARCH_SPACE

        config = {"threshold_z": 2.511, "min_error_value": 0.1}
        assert _pinned_params(config, ESA_M1_SEARCH_SPACE) == []

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


class TestIssKeepsItsThresholdZFloor:
    """The trap guarding the 021.5b change.

    ESA's threshold_z floor was lowered 2.5 -> 1.0 because both plan-021 arms'
    optima sat on the 2.5 bound. That floor exists for an ISS reason: during
    Phase 15 the optimizer drove z to ~1.6 chasing undetectable drift faults and
    fired on nominal noise in replay. ISS_SEARCH_SPACE is built as
    `{**SEARCH_SPACE, ...}` and does NOT override threshold_z, so the lowering
    had to go in ESA_M1_SEARCH_SPACE — putting it in the base would silently
    re-expose ISS to exactly that failure.

    ESA can afford a low z because its optimum pairs it with a high
    min_error_value (0.4-0.5); ISS pins min_error_value to 0.0 and so has no
    compensating suppression.
    """

    def test_iss_floor_is_unchanged(self) -> None:
        from spacecraft_telemetry.ray_fanout.tune import ISS_SEARCH_SPACE

        assert ISS_SEARCH_SPACE["threshold_z"].lower == pytest.approx(2.5)

    def test_base_floor_is_unchanged(self) -> None:
        """The base space feeds ISS and every non-Mission1 ESA mission."""
        from spacecraft_telemetry.ray_fanout.tune import SEARCH_SPACE

        assert SEARCH_SPACE["threshold_z"].lower == pytest.approx(2.5)

    def test_only_esa_m1_got_the_lower_floor(self) -> None:
        from spacecraft_telemetry.ray_fanout.tune import (
            ESA_M1_SEARCH_SPACE,
            ISS_SEARCH_SPACE,
            SEARCH_SPACE,
        )

        assert ESA_M1_SEARCH_SPACE["threshold_z"].lower == pytest.approx(1.0)
        assert ESA_M1_SEARCH_SPACE["threshold_z"].lower < SEARCH_SPACE["threshold_z"].lower
        assert ESA_M1_SEARCH_SPACE["threshold_z"].lower < ISS_SEARCH_SPACE["threshold_z"].lower

    def test_iss_still_pins_min_error_value_to_zero(self) -> None:
        """Documents *why* ISS cannot inherit a low z: it has no error floor to
        compensate with. If this ever becomes tunable for ISS, the z floor
        decision must be revisited together with it."""
        from spacecraft_telemetry.ray_fanout.tune import ISS_SEARCH_SPACE

        assert ISS_SEARCH_SPACE["min_error_value"] == 0.0


class TestFlagPeggedParams:
    """docs/plans/024 stage .0 — the bound-truncation guard.

    A sweep can land a winning config exactly on (or within an epsilon of) a
    search-space bound; that is not a converged optimum. Until now nothing
    noticed automatically — three prior recurrences, each caught by a human
    reading a config (see docs/plans/024's "truncation, three times over").
    ``_flag_pegged_params`` is the automatic version.
    """

    def test_interior_config_flags_nothing(self) -> None:
        from spacecraft_telemetry.ray_fanout.tune import (
            ESA_M1_SEARCH_SPACE,
            _flag_pegged_params,
        )

        config = {"threshold_z": 4.0, "min_error_value": 0.3}
        assert _flag_pegged_params(config, ESA_M1_SEARCH_SPACE) == {}

    def test_replays_plan_019_arm_a(self) -> None:
        """arm A: threshold_z=4.783 against the then-(2.5, 5.0) ceiling.

        Predates ESA_M1_SEARCH_SPACE's existence — that widening event IS
        this finding. Replayed against a literal (2.5, 5.0) domain, matching
        the base SEARCH_SPACE's current threshold_z bounds (unchanged since
        the ESA-Mission1 widening branched off it as ESA_M1_SEARCH_SPACE).
        """
        from ray import tune

        from spacecraft_telemetry.ray_fanout.tune import _flag_pegged_params

        space = {"threshold_z": tune.uniform(2.5, 5.0)}
        flagged = _flag_pegged_params({"threshold_z": 4.783}, space)
        assert "threshold_z" in flagged
        assert flagged["threshold_z"]["bound_type"] == "upper"

    def test_replays_plan_023_subsystem_1_sighting(self) -> None:
        """Fresh, post-widening sighting: subsystem_1 pegged threshold_z at
        exactly 8.000 against today's ESA_M1_SEARCH_SPACE 8.0 ceiling.

        This matters more than the arm A replay: it proves the guard is
        needed at TODAY's bounds, not only in hindsight.
        """
        from spacecraft_telemetry.ray_fanout.tune import (
            ESA_M1_SEARCH_SPACE,
            _flag_pegged_params,
        )

        flagged = _flag_pegged_params({"threshold_z": 8.000}, ESA_M1_SEARCH_SPACE)
        assert "threshold_z" in flagged
        assert flagged["threshold_z"]["bound_type"] == "upper"
        assert flagged["threshold_z"]["bound"] == pytest.approx(8.0)

    def test_lower_bound_is_flagged(self) -> None:
        from spacecraft_telemetry.ray_fanout.tune import (
            ESA_M1_SEARCH_SPACE,
            _flag_pegged_params,
        )

        flagged = _flag_pegged_params({"threshold_z": 1.05}, ESA_M1_SEARCH_SPACE)
        assert flagged["threshold_z"]["bound_type"] == "lower"

    def test_randint_param_at_bound_is_not_flagged(self) -> None:
        """Discrete (tune.randint) params are out of scope — pegging at a
        discrete bound is common and usually meaningful."""
        from spacecraft_telemetry.ray_fanout.tune import (
            ESA_M1_SEARCH_SPACE,
            _flag_pegged_params,
        )

        config = {"threshold_window": 500, "threshold_min_anomaly_len": 10}
        assert _flag_pegged_params(config, ESA_M1_SEARCH_SPACE) == {}

    def test_fixed_iss_min_error_value_is_not_flagged(self) -> None:
        """ISS_SEARCH_SPACE pins min_error_value=0.0 as a plain float, not a
        Ray Tune domain — no bound to pin against."""
        from spacecraft_telemetry.ray_fanout.tune import (
            ISS_SEARCH_SPACE,
            _flag_pegged_params,
        )

        assert _flag_pegged_params({"min_error_value": 0.0}, ISS_SEARCH_SPACE) == {}


class TestRunAllSweepsPeggedParamGuard:
    """Integration coverage: the guard must actually reach tuned_configs.json
    and the logs from run_all_sweeps, not just work in isolation."""

    def _run(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        *,
        winning_threshold_z: float,
    ) -> dict:
        import ray

        from spacecraft_telemetry.ray_fanout.tune import run_all_sweeps

        base_settings = load_settings("test")
        settings = base_settings.model_copy(
            update={
                "model": base_settings.model.model_copy(
                    update={"artifacts_dir": tmp_path / "models"}
                ),
                "tune": base_settings.tune.model_copy(update={"parallel_subsystems": False}),
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

        def _fake_run_hpo_sweep(subsystem, channels, *_args, **_kwargs):
            return {
                "config": {
                    "error_smoothing_window": 10,
                    "threshold_window": 100,
                    "threshold_z": winning_threshold_z,
                    "min_error_value": 0.3,
                    "threshold_min_anomaly_len": 2,
                },
                "seg_f0_5": 0.75,
                "run_id": "fake-run-id-abc",
            }

        monkeypatch.setattr(
            "spacecraft_telemetry.ray_fanout.tune.run_hpo_sweep",
            _fake_run_hpo_sweep,
        )

        out = run_all_sweeps(settings, "ESA-Mission1", ["channel_1"])
        return json.loads(out.read_text())["subsystem_1"]["_meta"]

    def test_pegged_config_recorded_in_meta_and_logged(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        meta = self._run(monkeypatch, tmp_path, winning_threshold_z=8.000)

        assert meta["pegged_params"] is not None
        assert meta["pegged_params"]["threshold_z"]["bound_type"] == "upper"
        assert meta["pegged_params"]["threshold_z"]["value"] == pytest.approx(8.000)

        stdout = capsys.readouterr().out
        assert "tune.sweep.param_pegged" in stdout
        assert "subsystem_1" in stdout

    def test_interior_config_records_no_pegged_params(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        meta = self._run(monkeypatch, tmp_path, winning_threshold_z=4.0)
        assert meta["pegged_params"] is None


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

        # Real mlflow Runs always carry .data.tags; a univariate run simply has
        # no `channels` tag, which is what makes it univariate to the loader.
        class data:
            tags: typing.ClassVar[dict[str, str]] = {}

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

        # Real mlflow Runs always carry .data.tags; a univariate run simply has
        # no `channels` tag, which is what makes it univariate to the loader.
        class data:
            tags: typing.ClassVar[dict[str, str]] = {}

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
    from spacecraft_telemetry.ray_fanout.tune import SEARCH_SPACE, run_hpo_sweep

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
    # Mirrors SEARCH_SPACE exactly — min_error_value joined it in 583a850
    # (the ESA-ADB absolute error floor). Keep this set and the bounds below
    # in sync with SEARCH_SPACE; an exhaustive == is deliberate, so adding a
    # tunable without deciding what it should smoke-test fails loudly here.
    assert set(config.keys()) == set(SEARCH_SPACE.keys())
    assert set(config.keys()) == {
        "error_smoothing_window",
        "threshold_window",
        "threshold_z",
        "threshold_min_anomaly_len",
        "min_error_value",
    }
    assert isinstance(config["error_smoothing_window"], int)
    assert isinstance(config["threshold_window"], int)
    assert isinstance(config["threshold_min_anomaly_len"], int)
    assert isinstance(config["threshold_z"], float)
    assert isinstance(config["min_error_value"], float)
    assert 5 <= config["error_smoothing_window"] <= 100
    assert 50 <= config["threshold_window"] <= 500
    assert 1 <= config["threshold_min_anomaly_len"] <= 10
    assert 0.0 <= config["min_error_value"] <= 0.3
    assert 1.5 <= config["threshold_z"] <= 5.0
    assert isinstance(best["seg_f0_5"], float)
    assert best["run_id"] is None or isinstance(best["run_id"], str)


@pytest.mark.slow
def test_run_hpo_sweep_logs_mission_metric_per_trial(
    ray_local, ray_series_parquet_multichannel, tmp_path: Path
) -> None:
    """A real sweep with ESA-ADB ground truth available logs mission_f0_5/
    mission_precision/mission_recall on its per-trial MLflow runs.

    This is the integration risk the fast _scoring_trial unit tests can't
    cover: whether run_hpo_sweep's tune.with_parameters wiring and the
    MLflowLoggerCallback actually surface the new dict keys end-to-end, not
    just whether _scoring_trial computes them correctly in isolation.
    """
    pytest.importorskip("ray")
    import mlflow

    from spacecraft_telemetry.ray_fanout.runner import score_all_subsystems, train_all_subsystems
    from spacecraft_telemetry.ray_fanout.tune import run_hpo_sweep

    mission = "ESA-Mission1"
    channels = ["channel_41", "channel_42"]

    sample_dir = tmp_path / "sample" / mission
    sample_dir.mkdir(parents=True)
    (sample_dir / "labels.csv").write_text(
        "ID,Channel,StartTime,EndTime\n"
        "E1,channel_41,2000-01-01T00:03:00Z,2000-01-01T00:06:00Z\n"
    )
    (sample_dir / "anomaly_types.csv").write_text(
        "ID,Category,Class,Subclass,Dimensionality\nE1,Anomaly,Rare,Rare,Univariate\n"
    )

    settings = ray_series_parquet_multichannel.model_copy(
        update={
            "data": ray_series_parquet_multichannel.data.model_copy(
                update={"sample_data_dir": str(tmp_path / "sample")}
            ),
            "tune": ray_series_parquet_multichannel.tune.model_copy(
                update={"num_samples": 2, "max_concurrent_trials": 1}
            ),
            "mlflow": ray_series_parquet_multichannel.mlflow.model_copy(
                update={"tracking_uri": f"sqlite:///{tmp_path}/mlflow.db"}
            ),
        }
    )

    train_all_subsystems(settings, mission, channels)
    score_all_subsystems(settings, mission, channels)
    run_hpo_sweep("subsystem_1", channels, settings, mission)

    # Search across every experiment rather than assuming the HPO one by
    # name: this test's Ray/MLflow test harness routes trial runs by ambient
    # ray.tune context, not strictly by the experiment_name passed to
    # _resilient_mlflow_callback, so the reliable check is "some run,
    # somewhere, carries the new keys" — not which experiment holds it.
    client = mlflow.tracking.MlflowClient(tracking_uri=settings.mlflow.tracking_uri)
    with_mission_metric = [
        r
        for exp in client.search_experiments()
        for r in client.search_runs([exp.experiment_id])
        if "mission_f0_5" in r.data.metrics
    ]
    assert with_mission_metric, (
        "no run anywhere logged mission_f0_5 — expected every HPO trial to, "
        "since ESA-ADB ground truth (labels.csv/anomaly_types.csv) was available"
    )
    m = with_mission_metric[0].data.metrics
    assert 0.0 <= m["mission_precision"] <= 1.0
    assert 0.0 <= m["mission_recall"] <= 1.0
    assert 0.0 <= m["mission_f0_5"] <= 1.0


@pytest.mark.slow
def test_run_hpo_sweep_finds_multivariate_errors(
    ray_local, ray_series_parquet_multichannel, tmp_path: Path
) -> None:
    """run_hpo_sweep finds a channel's errors even when it was scored as part
    of a multivariate subsystem group, not individually (docs/plans/021).

    Without the fallback in _find_channel_errors_run, this would raise
    "run_hpo_sweep has no usable channels" — no channel in a multivariate
    group has an individual channel_id-tagged scoring run.
    """
    pytest.importorskip("ray")

    from spacecraft_telemetry.ray_fanout.runner import score_all_subsystems, train_all_subsystems
    from spacecraft_telemetry.ray_fanout.tune import run_hpo_sweep

    mission = "ESA-Mission1"
    channels = ["channel_41", "channel_42"]

    settings = ray_series_parquet_multichannel.model_copy(
        update={
            "tune": ray_series_parquet_multichannel.tune.model_copy(
                update={"num_samples": 2, "max_concurrent_trials": 1}
            ),
            "mlflow": ray_series_parquet_multichannel.mlflow.model_copy(
                update={"tracking_uri": f"sqlite:///{tmp_path}/mlflow.db"}
            ),
        }
    )

    train_all_subsystems(settings, mission, channels)
    score_all_subsystems(settings, mission, channels)

    best = run_hpo_sweep("subsystem_1", channels, settings, mission)
    assert isinstance(best["seg_f0_5"], float)
    config = best["config"]
    assert set(config.keys()) == {
        "error_smoothing_window", "threshold_window",
        "threshold_z", "threshold_min_anomaly_len", "min_error_value",
    }


def test_find_multivariate_scoring_run_looks_up_by_group_key_not_subsystem(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The run's `subsystem` tag holds the key the model was SCORED under.

    Since docs/plans/023 stage .4 that is the channel group
    (``subsystem_6_g09``), not the subsystem (``subsystem_6``). Querying the
    subsystem finds nothing, every channel looks unscored, and the entire HPO
    sweep is skipped with only a "no errors.npy" note — a silent no-op that
    burns the whole tuning budget. This pins the tag actually queried.
    """
    from spacecraft_telemetry.ray_fanout.tune import _find_channel_errors_run

    settings = load_settings("test")
    monkeypatch.setattr(
        "spacecraft_telemetry.ray_fanout.tune.find_latest_run_for_channel",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        "spacecraft_telemetry.ray_fanout.tune.load_channel_group_map",
        lambda *_a, **_k: {"channel_47": "subsystem_6_g09"},
    )
    queried: list[str] = []

    def _capture(_exp, _tag, value, *_a, **_k):
        queried.append(value)
        return _FakeRun("mv-run", tags={"channels": "channel_47,channel_48,channel_49"})

    monkeypatch.setattr(
        "spacecraft_telemetry.ray_fanout.tune.find_latest_run_by_tag", _capture
    )

    found = _find_channel_errors_run(settings, "ESA-Mission1", "channel_47", "exp")
    assert found is not None
    assert queried == ["subsystem_6_g09"], (
        f"looked up {queried!r}; must query the group key the run is tagged with"
    )
    assert found[1] == "errors/channel_47.npy"


class TestWindowLabelsMatchTheScoringRunsIndex:
    """A multivariate run's per-channel errors are windowed over the group's
    JOINT (intersected) index, so labels must be built the same way.

    Building them from the channel's own series yields an array shorter or
    longer by however many rows the intersection dropped, and _prepare_channel_data
    rejects the channel on the shape check. Invisible for a group that
    intersects at 100% (channels 41-46, the only group ever tuned before), which
    is why it survived until plan 023's 99.999%-aligned groups.
    """

    class _Run:
        def __init__(self, tags: dict[str, str]) -> None:
            self.data = type("D", (), {"tags": tags})()
            self.info = type("I", (), {"run_id": "r"})()

    def test_multivariate_selects_this_channels_column(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import numpy as np

        from spacecraft_telemetry.ray_fanout import tune as _tune

        group = ["channel_a", "channel_b", "channel_c"]
        # (M, C) joint-index labels; each column distinct so a wrong column shows.
        joint = np.array([[True, False, False], [False, True, False]], dtype=bool)
        captured: dict[str, object] = {}

        def _fake_load(settings, _mission, key):
            captured["input_channels"] = settings.model.input_channels
            captured["key"] = key
            return joint

        monkeypatch.setattr(
            "spacecraft_telemetry.model.dataset.load_window_labels", _fake_load
        )
        run = self._Run({"channels": ",".join(group), "subsystem": "grp_01"})
        got = _tune._window_labels_matching_run(
            load_settings("test"), "ESA-Mission1", "channel_b", run, {}
        )

        assert got.tolist() == [False, True], "must take channel_b's column"
        assert captured["input_channels"] == group, "labels must be built for the GROUP"
        assert captured["key"] == "grp_01"

    def test_group_labels_are_loaded_once_per_group(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Without the cache every member re-reads the whole group's parquet.
        import numpy as np

        from spacecraft_telemetry.ray_fanout import tune as _tune

        group = ["channel_a", "channel_b", "channel_c"]
        calls: list[int] = []

        def _fake_load(_settings, _mission, _key):
            calls.append(1)
            return np.zeros((4, 3), dtype=bool)

        monkeypatch.setattr(
            "spacecraft_telemetry.model.dataset.load_window_labels", _fake_load
        )
        run = self._Run({"channels": ",".join(group)})
        cache: dict = {}
        for ch in group:
            _tune._window_labels_matching_run(
                load_settings("test"), "ESA-Mission1", ch, run, cache
            )
        assert len(calls) == 1, f"loaded {len(calls)}x for one group"

    def test_univariate_run_uses_the_per_channel_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import numpy as np

        from spacecraft_telemetry.ray_fanout import tune as _tune

        captured: dict[str, object] = {}

        def _fake_load(settings, _mission, key):
            captured["input_channels"] = settings.model.input_channels
            captured["key"] = key
            return np.array([True, False], dtype=bool)

        monkeypatch.setattr(
            "spacecraft_telemetry.model.dataset.load_window_labels", _fake_load
        )
        run = self._Run({})  # no `channels` tag -> univariate
        got = _tune._window_labels_matching_run(
            load_settings("test"), "ESA-Mission1", "channel_a", run, {}
        )
        assert got.tolist() == [True, False]
        assert captured["input_channels"] is None, "univariate path must stay untouched"
        assert captured["key"] == "channel_a"
