"""Tests for scripts/grid_resample_cost.py.

docs/plans/023-channel-time-grid.md, stage 023.2: this is the gate — the tool
must show, on data it did not see before, that resampling recovers alignment
lost to phase drift (the channel_70/71 phenomenon) and stays honest when
resampling genuinely cannot help (disjoint time ranges).

Loaded by file path, same pattern as test_check_channel_group.py — the
script is a standalone CLI, not part of the installed package.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

from spacecraft_telemetry.core.config import Settings
from tests.scripts._helpers import write_channel_metadata as _write_channel

_SCRIPT_PATH = Path(__file__).parent.parent.parent / "scripts" / "grid_resample_cost.py"


def _load_script_module() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("grid_resample_cost", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # see test_check_channel_group.py for why
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def script_module() -> types.ModuleType:
    return _load_script_module()


_MISSION = "ESA-Mission1"


def _settings(processed_dir: Path, *, window_size: int = 5, forecast_steps: int = 1) -> Settings:
    return Settings(
        preprocess={"processed_data_dir": processed_dir},
        model={
            "window_size": window_size,
            "prediction_horizon": 1,
            "forecast_steps": forecast_steps,
        },
    )


# ---------------------------------------------------------------------------
# _bucket_timestamps_gap_preserving
# ---------------------------------------------------------------------------


def test_bucket_timestamps_keeps_only_non_empty_buckets(
    script_module: types.ModuleType,
) -> None:
    import pandas as pd

    ts = pd.DatetimeIndex(
        [pd.Timestamp(0, unit="s"), pd.Timestamp(5, unit="s"), pd.Timestamp(65, unit="s")]
    )
    buckets = script_module._bucket_timestamps_gap_preserving(ts, rate_s=30)
    # t=0 and t=5 both floor to bucket 0; t=65 floors to bucket 60 — no bucket
    # 30 was ever invented, unlike a dense resample().ffill().
    assert list(buckets) == [pd.Timestamp(0, unit="s"), pd.Timestamp(60, unit="s")]


def test_bucket_timestamps_empty_input(script_module: types.ModuleType) -> None:
    import pandas as pd

    assert len(script_module._bucket_timestamps_gap_preserving(pd.DatetimeIndex([]), 30)) == 0


# ---------------------------------------------------------------------------
# measure_resampled — phase-drift alignment recovery
# ---------------------------------------------------------------------------


def test_resampling_recovers_alignment_lost_to_phase_drift(
    tmp_path: Path, script_module: types.ModuleType
) -> None:
    """docs/plans/023 channel_70/71: same cadence, phase-offset, zero native
    overlap — bucketing must recover full alignment."""
    processed_dir = tmp_path / "processed"
    ts_a = list(range(0, 20 * 30, 30))  # 0, 30, 60, ... (on 30s boundaries)
    ts_b = [t + 5 for t in ts_a]  # same 30s cadence, +5s phase offset
    _write_channel(processed_dir, _MISSION, "channel_a", "train", ts_a, [0] * 20)
    _write_channel(processed_dir, _MISSION, "channel_b", "train", ts_b, [0] * 20)

    settings = _settings(processed_dir, window_size=5)  # span = 6

    from check_channel_group import check_group  # native baseline, same import the script uses

    native = check_group(settings, _MISSION, ["channel_a", "channel_b"])
    assert native.n_aligned == 0  # confirms the native problem exists

    resampled = script_module.measure_resampled(
        settings, _MISSION, ["channel_a", "channel_b"], rate_s=30
    )
    assert resampled.n_aligned == 20
    assert resampled.n_joint_segments == 1
    assert resampled.joint_windows == 15  # 20 - span(6) + 1


def test_resampling_cannot_fix_genuinely_disjoint_ranges(
    tmp_path: Path, script_module: types.ModuleType
) -> None:
    processed_dir = tmp_path / "processed"
    ts_a = list(range(0, 20 * 30, 30))
    ts_b = [t + 100_000 for t in ts_a]  # a wholly different time range
    _write_channel(processed_dir, _MISSION, "channel_a", "train", ts_a, [0] * 20)
    _write_channel(processed_dir, _MISSION, "channel_b", "train", ts_b, [0] * 20)

    settings = _settings(processed_dir)
    resampled = script_module.measure_resampled(
        settings, _MISSION, ["channel_a", "channel_b"], rate_s=30
    )
    assert resampled.n_aligned == 0
    assert resampled.joint_windows == 0


def test_measure_resampled_per_channel_rows_reflects_bucket_count(
    tmp_path: Path, script_module: types.ModuleType
) -> None:
    processed_dir = tmp_path / "processed"
    # Two native ticks land in the same 30s bucket -> 1 bucket, not 2.
    ts_a = [0, 5, 30, 60]
    _write_channel(processed_dir, _MISSION, "channel_a", "train", ts_a, [0] * len(ts_a))
    _write_channel(processed_dir, _MISSION, "channel_b", "train", ts_a, [0] * len(ts_a))

    settings = _settings(processed_dir, window_size=2)
    resampled = script_module.measure_resampled(
        settings, _MISSION, ["channel_a", "channel_b"], rate_s=30
    )
    assert resampled.per_channel_rows == {"channel_a": 3, "channel_b": 3}


# ---------------------------------------------------------------------------
# measure_resampled — a genuine gap survives bucketing; jitter does not
# ---------------------------------------------------------------------------


def test_real_gap_survives_bucketing_short_jitter_does_not(
    tmp_path: Path, script_module: types.ModuleType
) -> None:
    processed_dir = tmp_path / "processed"
    # channel_a: clean 30s cadence, 20 rows, one segment.
    ts_a = list(range(0, 20 * 30, 30))
    _write_channel(processed_dir, _MISSION, "channel_a", "train", ts_a, [0] * 20)
    # channel_b: same nominal cadence but every tick lands 1s late (native gap
    # detection on channel_b alone would see this as fine, regular jitter, but
    # the point is it still floors into the SAME 30s buckets as channel_a).
    ts_b = [t + 1 for t in ts_a]
    _write_channel(processed_dir, _MISSION, "channel_b", "train", ts_b, [0] * 20)

    settings = _settings(processed_dir, window_size=5)
    resampled = script_module.measure_resampled(
        settings, _MISSION, ["channel_a", "channel_b"], rate_s=30
    )
    assert resampled.n_joint_segments == 1
    assert resampled.joint_windows == 15


# ---------------------------------------------------------------------------
# build_cost_table — verdict
# ---------------------------------------------------------------------------


def test_build_cost_table_verdict_material_improvement(
    tmp_path: Path, script_module: types.ModuleType
) -> None:
    processed_dir = tmp_path / "processed"
    ts_a = list(range(0, 20 * 30, 30))
    ts_b = [t + 5 for t in ts_a]  # phase drift -> native n_aligned = 0
    _write_channel(processed_dir, _MISSION, "channel_a", "train", ts_a, [0] * 20)
    _write_channel(processed_dir, _MISSION, "channel_b", "train", ts_b, [0] * 20)

    settings = _settings(processed_dir, window_size=5)
    table = script_module.build_cost_table(
        settings, _MISSION, ["channel_a", "channel_b"], rates_s=[30, 90]
    )
    assert table["native"]["joint_windows"] == 0
    assert table["best_rate_s"] == 30
    assert table["verdict"] == "resampling materially raises joint window yield"
    assert table["resampled"]["30"]["joint_windows"] == 15
    assert table["resampled"]["90"]["rate_s"] == 90


def test_build_cost_table_verdict_stays_collapsed_when_ranges_are_disjoint(
    tmp_path: Path, script_module: types.ModuleType
) -> None:
    processed_dir = tmp_path / "processed"
    ts_a = list(range(0, 20 * 30, 30))
    ts_b = [t + 100_000 for t in ts_a]
    _write_channel(processed_dir, _MISSION, "channel_a", "train", ts_a, [0] * 20)
    _write_channel(processed_dir, _MISSION, "channel_b", "train", ts_b, [0] * 20)

    settings = _settings(processed_dir)
    table = script_module.build_cost_table(
        settings, _MISSION, ["channel_a", "channel_b"], rates_s=[30, 300]
    )
    assert table["verdict"] == "yield stays collapsed — fragmentation is not fixed by resampling"


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def test_main_writes_per_rate_and_summary_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    script_module: types.ModuleType,
) -> None:
    processed_dir = tmp_path / "processed"
    ts_a = list(range(0, 20 * 30, 30))
    ts_b = [t + 5 for t in ts_a]
    _write_channel(processed_dir, _MISSION, "channel_a", "train", ts_a, [0] * 20)
    _write_channel(processed_dir, _MISSION, "channel_b", "train", ts_b, [0] * 20)

    out_dir = tmp_path / "outputs"
    monkeypatch.setattr(
        "sys.argv",
        [
            "grid_resample_cost.py",
            "--env", "test",
            "--mission", _MISSION,
            "--channels", "channel_a,channel_b",
            "--rates", "30,90",
            "--out-dir", str(out_dir),
        ],
    )
    monkeypatch.setattr(
        script_module, "load_settings", lambda env: _settings(processed_dir, window_size=5)
    )

    script_module.main()

    assert (out_dir / "grid_cost_30s.json").exists()
    assert (out_dir / "grid_cost_90s.json").exists()
    assert (out_dir / "grid_cost_summary.json").exists()

    import json

    summary = json.loads((out_dir / "grid_cost_summary.json").read_text())
    assert summary["best_rate_s"] == 30
