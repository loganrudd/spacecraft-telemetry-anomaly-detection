"""Tests for scripts/threshold_ceiling.py's pure config-building helpers.

docs/reviews/022-tuning-layer.md, stage 1.1/1.2: `_build_tuned_config` and
`_make_sweep_fn` were extracted from main() so the 022.2 `_meta` schema and
the sweep_fn selection are testable without a network round-trip.
"""

from __future__ import annotations

import importlib.util
import json
import types
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from spacecraft_telemetry.core.config import load_settings
from spacecraft_telemetry.ray_fanout.threshold_search import NonConvergenceError, WideningResult

_SCRIPT_PATH = Path(__file__).parent.parent.parent / "scripts" / "threshold_ceiling.py"


def _load_script_module() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("threshold_ceiling", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def script_module() -> types.ModuleType:
    return _load_script_module()


def _widening(
    *, expansions: int = 0, best_z: float = 3.0, best_floor: float = 0.2
) -> WideningResult:
    grid = {(best_z, best_floor): 0.75}
    return WideningResult(
        grid=grid,
        axes={"threshold_z": [1.0, best_z, 5.0], "min_error_value": [0.0, best_floor, 0.4]},
        axis_order=["threshold_z", "min_error_value"],
        best_point=(best_z, best_floor),
        best_score=0.75,
        expansions=expansions,
    )


# ---------------------------------------------------------------------------
# _build_tuned_config
# ---------------------------------------------------------------------------


class TestBuildTunedConfig:
    def test_emits_full_022_2_meta_key_set(self, script_module: types.ModuleType) -> None:
        widening = _widening(expansions=2, best_z=6.0, best_floor=0.3)
        entry = script_module._build_tuned_config(
            "subsystem_5",
            best_z=6.0,
            best_floor=0.3,
            threshold_window=315,
            min_run_length=2,
            smoothing_window=30,
            objective="mission",
            select_on="hpo_portion",
            hpo_eval_fraction=0.6,
            widening=widening,
        )
        config = entry["subsystem_5"]
        meta = config["_meta"]

        assert config["threshold_z"] == 6.0
        assert config["min_error_value"] == 0.3
        assert config["threshold_window"] == 315
        assert config["threshold_min_anomaly_len"] == 2
        assert config["error_smoothing_window"] == 30

        assert meta["provenance"] == "exhaustive_grid"
        assert meta["run_id"] is None
        assert meta["interior"] is True
        assert meta["source"] == "scripts/threshold_ceiling.py exhaustive grid (mission)"
        assert meta["objective_name"] == "mission_corrected_event_wise_f0_5"
        assert meta["objective_value"] == pytest.approx(0.75)
        assert meta["selected_on"] == "hpo_portion"
        assert meta["hpo_eval_fraction"] == 0.6
        assert meta["outer_split"] == "chronological_50_50"
        assert meta["error_smoothing_window"] == 30
        assert meta["threshold_window"] == 315
        assert meta["min_run_length"] == 2
        assert meta["axes"] == {
            "threshold_z": [1.0, 6.0, 5.0],
            "min_error_value": [0.0, 0.3, 0.4],
        }
        assert meta["expansions"] == 2

    def test_per_channel_objective_names_the_mean_seg_f0_5_metric(
        self, script_module: types.ModuleType
    ) -> None:
        widening = _widening()
        entry = script_module._build_tuned_config(
            "subsystem_1",
            best_z=3.0,
            best_floor=0.2,
            threshold_window=100,
            min_run_length=2,
            smoothing_window=25,
            objective="per_channel",
            select_on="final_portion",
            hpo_eval_fraction=0.6,
            widening=widening,
        )
        meta = entry["subsystem_1"]["_meta"]
        assert meta["objective_name"] == "mean_per_channel_seg_f0_5"
        assert meta["source"] == "scripts/threshold_ceiling.py exhaustive grid (per_channel)"


# ---------------------------------------------------------------------------
# Key-set parity across both tuned_configs.json writers (A4)
# ---------------------------------------------------------------------------


def _tune_meta_via_run_all_sweeps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> dict[str, Any]:
    """Real _meta dict from ray_fanout.tune's run_all_sweeps' _to_entry.

    Mirrors tests/ray_fanout/test_tune.py::test_run_all_sweeps_filters_and_runs'
    mocking pattern — the tune-side half of the parity check this test proves.
    """
    import ray

    from spacecraft_telemetry.ray_fanout.tune import run_all_sweeps

    base_settings = load_settings("test")
    settings = base_settings.model_copy(
        update={
            "model": base_settings.model.model_copy(update={"artifacts_dir": tmp_path / "models"}),
            "tune": base_settings.tune.model_copy(update={"parallel_subsystems": False}),
        }
    )

    class _FakeRun:
        class info:
            run_id = "fake-scored-run-id"

    monkeypatch.setattr(ray, "is_initialized", lambda: True)
    monkeypatch.setattr(
        "spacecraft_telemetry.ray_fanout.tune.find_latest_run_for_channel",
        lambda exp, ch, uri, extra_filter=None: _FakeRun() if ch == "channel_1" else None,
    )
    monkeypatch.setattr(
        "spacecraft_telemetry.ray_fanout.tune.load_channel_subsystem_map",
        lambda *_args, **_kwargs: {"channel_1": "subsystem_1"},
    )
    monkeypatch.setattr(
        "spacecraft_telemetry.ray_fanout.tune.run_hpo_sweep",
        lambda subsystem, channels, *_args, **_kwargs: {
            "config": {
                "error_smoothing_window": 10,
                "threshold_window": 100,
                "threshold_z": 2.5,
                "threshold_min_anomaly_len": 2,
            },
            "seg_f0_5": 0.75,
            "run_id": "fake-run-id-abc",
        },
    )

    out = run_all_sweeps(settings, "ESA-Mission1", ["channel_1"])
    loaded = json.loads(out.read_text())
    result: dict[str, Any] = loaded["subsystem_1"]["_meta"]
    return result


class TestMetaKeySetParity:
    # Pre-existing gap, uncovered before this test: tune.py's _to_entry emits
    # two Ray-Tune-only diagnostic keys ("seg_f0_5", "nominal_fp_rate") not in
    # write_tuned_configs' documented schema and not read by anything
    # downstream (_tuned_meta reads only run_id/source; score_all_channels
    # filters _meta out entirely) — but tests/ray_fanout/test_tune.py asserts
    # on their presence directly, so removing them is a real behaviour change
    # outside docs/reviews/022 stage 2.1's declared one-line scope (the
    # "source" string only). Documented and permitted here rather than
    # silently narrowed to a passing-by-luck equality check.
    _TUNE_ONLY_DIAGNOSTIC_KEYS = frozenset({"seg_f0_5", "nominal_fp_rate"})

    def test_grid_and_tune_writers_emit_the_same_meta_key_set(
        self, script_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        grid_meta = script_module._build_tuned_config(
            "subsystem_5",
            best_z=3.0,
            best_floor=0.2,
            threshold_window=100,
            min_run_length=2,
            smoothing_window=25,
            objective="per_channel",
            select_on="hpo_portion",
            hpo_eval_fraction=0.6,
            widening=_widening(),
        )["subsystem_5"]["_meta"]

        tune_meta = _tune_meta_via_run_all_sweeps(monkeypatch, tmp_path)

        assert set(grid_meta) == set(tune_meta) - self._TUNE_ONLY_DIAGNOSTIC_KEYS

    def test_parity_check_bites_on_a_missing_key(
        self, script_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A parity test that passes vacuously is worse than none — prove this
        one actually fails when the key sets genuinely diverge."""
        grid_meta = script_module._build_tuned_config(
            "subsystem_5",
            best_z=3.0,
            best_floor=0.2,
            threshold_window=100,
            min_run_length=2,
            smoothing_window=25,
            objective="per_channel",
            select_on="hpo_portion",
            hpo_eval_fraction=0.6,
            widening=_widening(),
        )["subsystem_5"]["_meta"]
        del grid_meta["axes"]

        tune_meta = _tune_meta_via_run_all_sweeps(monkeypatch, tmp_path)

        assert set(grid_meta) != set(tune_meta)


# ---------------------------------------------------------------------------
# _make_sweep_fn
# ---------------------------------------------------------------------------


class TestMakeSweepFnPerChannel:
    def test_binds_z_and_floor_values_onto_sweep_group(
        self, script_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, object] = {}

        def _fake_sweep_group(
            per_channel: object,
            *,
            z_values: list[float],
            floor_values: list[float],
            threshold_window: int,
            min_run_length: int,
            eval_slice: slice | None = None,
            prepared: dict[str, object] | None = None,
        ) -> dict[tuple[float, float], float]:
            captured["z_values"] = z_values
            captured["floor_values"] = floor_values
            captured["threshold_window"] = threshold_window
            captured["min_run_length"] = min_run_length
            captured["prepared"] = prepared
            return {(z, f): 0.0 for z in z_values for f in floor_values}

        monkeypatch.setattr(script_module, "sweep_group", _fake_sweep_group)

        settings = load_settings("test")
        sweep_fn = script_module._make_sweep_fn(
            "per_channel",
            settings=settings,
            mission="ESA-Mission1",
            channels=["channel_41"],
            per_channel={"channel_41": (np.zeros(5), np.zeros(5, dtype=bool))},
            eval_slice=slice(None),
            metadata_by_channel={},
            select_on="hpo_portion",
            threshold_window=100,
            min_run_length=2,
        )
        grid = sweep_fn({"threshold_z": [1.0, 2.0], "min_error_value": [0.0, 0.5]})

        assert captured["z_values"] == [1.0, 2.0]
        assert captured["floor_values"] == [0.0, 0.5]
        assert captured["threshold_window"] == 100
        assert captured["min_run_length"] == 2
        assert set(grid) == {(1.0, 0.0), (1.0, 0.5), (2.0, 0.0), (2.0, 0.5)}
        # P2: precomputed rolling terms are bound in, not left for
        # sweep_group to recompute per round.
        prepared = captured["prepared"]
        assert isinstance(prepared, dict) and set(prepared) == {"channel_41"}


class TestMakeSweepFnMission:
    def _patch_mission_prep(
        self,
        script_module: types.ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        *,
        fake_sweep_group_mission_level: object,
    ) -> dict[str, int]:
        """Stub _prepare_mission_sweep's five real dependencies + the swept
        function, and return a call-count dict keyed by function name — used
        both to prove the keyword-binding contract and (P1) that prep runs
        exactly once no matter how many times the returned sweep_fn is
        invoked. Stubbed at their OWN modules (not threshold_ceiling's
        namespace), since _prepare_mission_sweep imports them locally — this
        stays a network-free unit test of the binding contract, not an
        integration test of mission_timeline/hpo_cutoff/load_events
        themselves.
        """
        counts: dict[str, int] = {
            "mission_timeline": 0, "hpo_cutoff": 0, "load_events": 0, "group_events": 0,
            "window_target_timestamps_from_metadata": 0,
        }

        def _counted(name: str, fn: object) -> object:
            def _wrapped(*args: object, **kwargs: object) -> object:
                counts[name] += 1
                return fn(*args, **kwargs)  # type: ignore[operator]

            return _wrapped

        monkeypatch.setattr(
            script_module, "sweep_group_mission_level", fake_sweep_group_mission_level
        )
        monkeypatch.setattr(
            "spacecraft_telemetry.esa_adb.timeline.mission_timeline",
            _counted("mission_timeline", lambda *_a, **_kw: []),
        )
        monkeypatch.setattr(
            "spacecraft_telemetry.esa_adb.report.hpo_cutoff",
            _counted("hpo_cutoff", lambda *_a, **_kw: pd.Timestamp.min.tz_localize("UTC")),
        )
        monkeypatch.setattr(
            "spacecraft_telemetry.esa_adb.events.load_events",
            _counted("load_events", lambda *_a, **_kw: pd.DataFrame()),
        )
        monkeypatch.setattr(
            "spacecraft_telemetry.esa_adb.events.group_events",
            _counted("group_events", lambda *_a, **_kw: []),
        )
        monkeypatch.setattr(
            "spacecraft_telemetry.model.dataset.window_target_timestamps_from_metadata",
            _counted(
                "window_target_timestamps_from_metadata", lambda _settings, *_a, **_kw: np.zeros(5)
            ),
        )
        return counts

    def test_reaches_sweep_group_mission_level_with_bound_z_and_floor_values(
        self, script_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, object] = {}

        def _fake_sweep_group_mission_level(
            per_channel: object,
            channel_timestamps: object,
            mission_events: object,
            mission_timeline: object,
            *,
            z_values: list[float],
            floor_values: list[float],
            threshold_window: int,
            min_run_length: int,
            eval_slice: slice | None = None,
            prepared: dict[str, object] | None = None,
        ) -> dict[tuple[float, float], float]:
            captured["z_values"] = z_values
            captured["floor_values"] = floor_values
            captured["prepared"] = prepared
            return {(z, f): 0.0 for z in z_values for f in floor_values}

        self._patch_mission_prep(
            script_module, monkeypatch,
            fake_sweep_group_mission_level=_fake_sweep_group_mission_level,
        )

        settings = load_settings("test")
        per_channel = {"channel_41": (np.zeros(5), np.zeros(5, dtype=bool))}
        metadata_by_channel = {
            "channel_41": (np.zeros(5, dtype=np.int32), np.zeros(5, dtype=bool), np.zeros(5))
        }
        sweep_fn = script_module._make_sweep_fn(
            "mission",
            settings=settings,
            mission="ESA-Mission1",
            channels=["channel_41"],
            per_channel=per_channel,
            eval_slice=slice(None),
            metadata_by_channel=metadata_by_channel,
            select_on="hpo_portion",
            threshold_window=100,
            min_run_length=2,
        )
        grid = sweep_fn({"threshold_z": [3.0, 4.0], "min_error_value": [0.1]})

        assert captured["z_values"] == [3.0, 4.0]
        assert captured["floor_values"] == [0.1]
        assert set(grid) == {(3.0, 0.1), (4.0, 0.1)}

    def test_mission_prep_runs_once_across_multiple_sweep_fn_calls(
        self, script_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """P1 regression: a widening driver invokes the returned sweep_fn
        once per round. _prepare_mission_sweep's five dependencies (timeline,
        cutoff, events, target timestamps) must run exactly once — at
        _make_sweep_fn() call time, not once per sweep_fn() invocation.
        """

        def _fake_sweep_group_mission_level(
            *_args: object, z_values: list[float], floor_values: list[float], **_kwargs: object
        ) -> dict[tuple[float, float], float]:
            return {(z, f): 0.0 for z in z_values for f in floor_values}

        counts = self._patch_mission_prep(
            script_module, monkeypatch,
            fake_sweep_group_mission_level=_fake_sweep_group_mission_level,
        )

        settings = load_settings("test")
        per_channel = {"channel_41": (np.zeros(5), np.zeros(5, dtype=bool))}
        metadata_by_channel = {
            "channel_41": (np.zeros(5, dtype=np.int32), np.zeros(5, dtype=bool), np.zeros(5))
        }
        sweep_fn = script_module._make_sweep_fn(
            "mission",
            settings=settings,
            mission="ESA-Mission1",
            channels=["channel_41"],
            per_channel=per_channel,
            eval_slice=slice(None),
            metadata_by_channel=metadata_by_channel,
            select_on="hpo_portion",
            threshold_window=100,
            min_run_length=2,
        )
        # Simulate three widening rounds calling the same sweep_fn.
        sweep_fn({"threshold_z": [3.0], "min_error_value": [0.1]})
        sweep_fn({"threshold_z": [4.0], "min_error_value": [0.1]})
        sweep_fn({"threshold_z": [5.0], "min_error_value": [0.1]})

        assert counts == {
            "mission_timeline": 1,
            "hpo_cutoff": 1,
            "load_events": 1,
            "group_events": 1,
            "window_target_timestamps_from_metadata": 1,
        }


# ---------------------------------------------------------------------------
# NonConvergenceError -> SystemExit
# ---------------------------------------------------------------------------


class TestNonConvergenceBecomesSystemExit:
    def test_non_convergence_error_raises_system_exit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """main()'s widen_to_convergence call wraps NonConvergenceError in a
        SystemExit with remediation guidance — exercised directly against the
        same try/except shape main() uses, without running main() end to end.
        """
        from spacecraft_telemetry.ray_fanout.threshold_search import widen_to_convergence

        def _sweep_fn(axes: dict[str, list[float]]) -> dict[tuple[float, ...], float]:
            return {(a, b): a - b for a in axes["threshold_z"] for b in axes["min_error_value"]}

        with pytest.raises(SystemExit):
            try:
                widen_to_convergence(
                    _sweep_fn,
                    {"threshold_z": [1.0, 2.0], "min_error_value": [0.0, 0.1]},
                    natural_bounds={},
                    max_expansions=1,
                )
            except NonConvergenceError as exc:
                raise SystemExit(
                    f"Threshold search did not converge: {exc}\n"
                    "Widen --z-values / --floor-values, or investigate why the "
                    "objective keeps improving toward the edge."
                ) from None


class TestNExpandMaxExpansionsCLI:
    """docs/reviews/022, item P3: --n-expand/--max-expansions expose
    widen_to_convergence's tuning knobs rather than hardcoding one value for
    both the cheap per_channel objective and the materially slower mission
    one."""

    _MISSION = "ESA-Mission1-ThresholdCeilingCLITest"
    _CHANNEL = "channel_41"
    _WINDOW_SIZE = 3
    _PREDICTION_HORIZON = 1
    _N_ROWS = 20  # -> 20 - (3+1) + 1 = 17 windows

    def _setup(self, tmp_path: Path) -> Any:
        """Real MLflow scoring run + processed series — everything main()'s
        per_channel-objective path reads before it ever calls
        widen_to_convergence. threshold_ceiling.py's _load_channel needs the
        full score_channel() param set (threshold_window,
        error_smoothing_window, threshold_min_anomaly_len) — a heavier
        fixture than tests/esa_adb/test_report.py's, which only needs
        threshold_min_anomaly_len for its own (different) read path.
        """
        import mlflow
        import pyarrow as pa
        import pyarrow.parquet as pq

        from spacecraft_telemetry.mlflow_tracking import (
            common_tags,
            experiment_name,
            log_artifact_bytes,
            log_params,
            open_run,
        )
        from spacecraft_telemetry.model.io import errors_to_bytes

        processed_dir = tmp_path / "processed"
        partition_dir = (
            processed_dir / self._MISSION / "test"
            / f"mission_id={self._MISSION}" / f"channel_id={self._CHANNEL}"
        )
        partition_dir.mkdir(parents=True, exist_ok=True)
        n_windows = self._N_ROWS - (self._WINDOW_SIZE + self._PREDICTION_HORIZON) + 1
        table = pa.table({
            "telemetry_timestamp": pa.array(
                pd.date_range("2000-01-01", periods=self._N_ROWS, freq="90s", tz="UTC")
            ),
            "value_normalized": pa.array([0.0] * self._N_ROWS, type=pa.float32()),
            "segment_id": pa.array([0] * self._N_ROWS, type=pa.int32()),
            "is_anomaly": pa.array([False] * self._N_ROWS, type=pa.bool_()),
        })
        pq.write_table(table, partition_dir / "part.parquet")

        mlflow_uri = f"sqlite:///{tmp_path}/mlflow.db"
        mlflow.set_tracking_uri(mlflow_uri)
        base_settings = load_settings("test")
        settings = base_settings.model_copy(update={
            "preprocess": base_settings.preprocess.model_copy(
                update={"processed_data_dir": str(processed_dir)}
            ),
            "model": base_settings.model.model_copy(update={
                "window_size": self._WINDOW_SIZE, "prediction_horizon": self._PREDICTION_HORIZON,
            }),
            "mlflow": base_settings.mlflow.model_copy(update={"tracking_uri": mlflow_uri}),
        })

        exp = experiment_name(settings.model.model_type, "scoring", self._MISSION)
        tags = common_tags(
            model_type=settings.model.model_type, mission=self._MISSION, phase="scoring",
            channel=self._CHANNEL, extra={"eval_split": "full_test", "tuned_from_run": "fake-hpo"},
        )
        with open_run(experiment=exp, run_name=self._CHANNEL, tags=tags) as run:
            assert run is not None
            log_params({
                "threshold_window": 10,
                "threshold_min_anomaly_len": 1,
                "error_smoothing_window": 5,
            })
            log_artifact_bytes(errors_to_bytes(np.zeros(n_windows)), "errors.npy")
        return settings

    def _fake_widen(
        self, captured: dict[str, object]
    ) -> Any:
        def _widen(sweep_fn: object, axes: dict[str, list[float]], **kwargs: object) -> Any:
            captured.update(kwargs)
            z_values = axes["threshold_z"]
            floor_values = axes["min_error_value"]
            grid = {(z, f): 0.5 for z in z_values for f in floor_values}
            return WideningResult(
                grid=grid,
                axes=axes,
                axis_order=list(axes),
                best_point=(z_values[0], floor_values[0]),
                best_score=0.5,
                expansions=0,
            )

        return _widen

    def test_flags_are_forwarded_to_widen_to_convergence(
        self, script_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import sys

        settings = self._setup(tmp_path)
        captured: dict[str, object] = {}
        monkeypatch.setattr(script_module, "load_settings", lambda _env: settings)
        monkeypatch.setattr(script_module, "widen_to_convergence", self._fake_widen(captured))
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "threshold_ceiling.py",
                "--env", "test",
                "--mission", self._MISSION,
                "--channels", self._CHANNEL,
                "--n-expand", "5",
                "--max-expansions", "7",
            ],
        )

        script_module.main()

        assert captured["n_expand"] == 5
        assert captured["max_expansions"] == 7

    def test_defaults_match_the_prior_hardcoded_values(
        self, script_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        import sys

        settings = self._setup(tmp_path)
        captured: dict[str, object] = {}
        monkeypatch.setattr(script_module, "load_settings", lambda _env: settings)
        monkeypatch.setattr(script_module, "widen_to_convergence", self._fake_widen(captured))
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "threshold_ceiling.py",
                "--env", "test",
                "--mission", self._MISSION,
                "--channels", self._CHANNEL,
            ],
        )

        script_module.main()

        assert captured["n_expand"] == 3
        assert captured["max_expansions"] == 3
