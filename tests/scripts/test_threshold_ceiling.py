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
    # two Ray-Tune-only diagnostic keys not in write_tuned_configs' documented
    # schema and not read by anything downstream (_tuned_meta reads only
    # run_id/source; score_all_channels filters _meta out entirely). Stage 1
    # must not change tune.py's behaviour, so this pins today's actual gap
    # rather than asserting a false equality; docs/reviews/022 stage 2.1
    # removes them from tune.py in the same commit that tightens this to
    # exact equality, mirroring the C4 provenance test's pin-then-flip shape.
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
        ) -> dict[tuple[float, float], float]:
            captured["z_values"] = z_values
            captured["floor_values"] = floor_values
            captured["threshold_window"] = threshold_window
            captured["min_run_length"] = min_run_length
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


class TestMakeSweepFnMission:
    def test_reaches_sweep_group_mission_level_with_bound_z_and_floor_values(
        self, script_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_make_sweep_fn("mission", ...) must route through _sweep_mission's
        real prep glue to sweep_group_mission_level, with z_values=/
        floor_values= bound correctly. The prep functions are stubbed at
        their own modules (not threshold_ceiling's namespace, since
        _sweep_mission imports them locally) so this stays a network-free
        unit test of the keyword-binding contract, not an integration test
        of mission_timeline/hpo_cutoff/load_events themselves.
        """
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
        ) -> dict[tuple[float, float], float]:
            captured["z_values"] = z_values
            captured["floor_values"] = floor_values
            return {(z, f): 0.0 for z in z_values for f in floor_values}

        monkeypatch.setattr(
            script_module, "sweep_group_mission_level", _fake_sweep_group_mission_level
        )
        monkeypatch.setattr(
            "spacecraft_telemetry.esa_adb.timeline.mission_timeline",
            lambda *_a, **_kw: [],
        )
        monkeypatch.setattr(
            "spacecraft_telemetry.esa_adb.report.hpo_cutoff",
            lambda *_a, **_kw: pd.Timestamp.min.tz_localize("UTC"),
        )
        monkeypatch.setattr(
            "spacecraft_telemetry.esa_adb.events.load_events",
            lambda *_a, **_kw: pd.DataFrame(),
        )
        monkeypatch.setattr(
            "spacecraft_telemetry.esa_adb.events.group_events",
            lambda *_a, **_kw: [],
        )
        monkeypatch.setattr(
            "spacecraft_telemetry.model.dataset.window_target_timestamps_from_metadata",
            lambda _settings, *_a, **_kw: np.zeros(5),
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
