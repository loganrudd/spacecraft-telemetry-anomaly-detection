"""Tests for scripts/check_channel_group.py.

docs/plans/023-channel-time-grid.md, stage 023.1: this is the gate — the tool
must reproduce known alignment/fragmentation/window-yield numbers on data it
did not see before, using the same join (`_align_multi_channel`, via
`load_multichannel_series_metadata`) training itself uses.

The script is a standalone CLI (not part of the installed package), so it's
loaded directly by file path — same pattern as
tests/scripts/test_build_reference_profiles.py and
tests/scripts/test_threshold_ceiling.py.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from spacecraft_telemetry.core.config import Settings

_SCRIPT_PATH = Path(__file__).parent.parent.parent / "scripts" / "check_channel_group.py"


def _load_script_module() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("check_channel_group", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec: the module defines a dataclass under `from
    # __future__ import annotations`, whose field resolution looks itself up
    # in sys.modules by __module__ name — without this it resolves to None
    # and crashes on `.list[str]` field type strings.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def script_module() -> types.ModuleType:
    return _load_script_module()


_MISSION = "ESA-Mission1"


def _write_channel(
    processed_dir: Path,
    mission: str,
    channel: str,
    split: str,
    timestamps_s: list[int],
    segment_ids: list[int],
) -> None:
    """Write one channel's metadata columns at explicit timestamps/segment ids.

    Full control over both, unlike test_dataset.py's evenly-spaced synthetic
    writer, so tests can construct phase offsets and fragmentation directly.
    """
    assert len(timestamps_s) == len(segment_ids)
    n = len(timestamps_s)
    table = pa.table(
        {
            "telemetry_timestamp": pa.array(
                [
                    pa.scalar(t, type=pa.timestamp("s", tz="UTC")).cast(
                        pa.timestamp("us", tz="UTC")
                    )
                    for t in timestamps_s
                ],
                type=pa.timestamp("us", tz="UTC"),
            ),
            "value_normalized": pa.array([0.0] * n, type=pa.float32()),
            "segment_id": pa.array(segment_ids, type=pa.int32()),
            "is_anomaly": pa.array([False] * n, type=pa.bool_()),
        }
    )
    part_dir = (
        processed_dir / mission / split / f"mission_id={mission}" / f"channel_id={channel}"
    )
    part_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, part_dir / "part.parquet")


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
# check_group — full overlap
# ---------------------------------------------------------------------------


def test_check_group_full_overlap_reports_zero_loss_and_expected_windows(
    tmp_path: Path, script_module: types.ModuleType
) -> None:
    processed_dir = tmp_path / "processed"
    ts = list(range(0, 20 * 30, 30))  # 20 rows, 30s cadence
    _write_channel(processed_dir, _MISSION, "channel_a", "train", ts, [0] * 20)
    _write_channel(processed_dir, _MISSION, "channel_b", "train", ts, [0] * 20)

    settings = _settings(processed_dir, window_size=5)  # span = 5 + 1 = 6
    report = script_module.check_group(settings, _MISSION, ["channel_a", "channel_b"])

    assert report.per_channel_rows == {"channel_a": 20, "channel_b": 20}
    assert report.n_aligned == 20
    assert report.alignment_loss_frac == pytest.approx(0.0)
    assert report.n_joint_segments == 1
    assert report.max_joint_segment_len == 20
    assert report.joint_windows == 15  # 20 - span(6) + 1


def test_check_group_as_dict_rounds_loss_frac(
    tmp_path: Path, script_module: types.ModuleType
) -> None:
    processed_dir = tmp_path / "processed"
    ts_a = list(range(0, 20 * 30, 30))
    ts_b = ts_a[3:]  # channel_b missing its first 3 rows -> real alignment loss
    _write_channel(processed_dir, _MISSION, "channel_a", "train", ts_a, [0] * len(ts_a))
    _write_channel(processed_dir, _MISSION, "channel_b", "train", ts_b, [0] * len(ts_b))

    settings = _settings(processed_dir, window_size=5)
    report = script_module.check_group(settings, _MISSION, ["channel_a", "channel_b"])

    assert report.n_aligned == 17
    assert report.alignment_loss_frac == pytest.approx(3 / 20)
    d = report.as_dict()
    assert d["alignment_loss_frac"] == round(3 / 20, 4)
    assert d["channels"] == ["channel_a", "channel_b"]


# ---------------------------------------------------------------------------
# check_group — zero overlap (phase offset, docs/plans/023 channel_70/71 case)
# ---------------------------------------------------------------------------


def test_check_group_zero_overlap_reports_rather_than_raises(
    tmp_path: Path, script_module: types.ModuleType
) -> None:
    processed_dir = tmp_path / "processed"
    ts_a = list(range(0, 20 * 30, 30))
    ts_b = [t + 15 for t in ts_a]  # same 30s cadence, 15s phase offset -> no shared row
    _write_channel(processed_dir, _MISSION, "channel_a", "train", ts_a, [0] * 20)
    _write_channel(processed_dir, _MISSION, "channel_b", "train", ts_b, [0] * 20)

    settings = _settings(processed_dir)
    report = script_module.check_group(settings, _MISSION, ["channel_a", "channel_b"])

    assert report.n_aligned == 0
    assert report.alignment_loss_frac == pytest.approx(1.0)
    assert report.n_joint_segments == 0
    assert report.joint_windows == 0
    # Per-channel rows are still reported even though the join is empty.
    assert report.per_channel_rows == {"channel_a": 20, "channel_b": 20}


# ---------------------------------------------------------------------------
# check_group — fragmentation (docs/plans/023 family-1 phenomenon)
# ---------------------------------------------------------------------------


def test_check_group_full_alignment_can_still_fragment_to_zero_windows(
    tmp_path: Path, script_module: types.ModuleType
) -> None:
    """100% row alignment, but one member's gaps slice every joint segment
    to length 1 -> zero windows survive, even though nothing is "missing"."""
    processed_dir = tmp_path / "processed"
    n = 20
    ts = list(range(0, n * 30, 30))
    _write_channel(processed_dir, _MISSION, "channel_a", "train", ts, [0] * n)
    # channel_b's OWN gap detection drew a boundary at every row.
    _write_channel(processed_dir, _MISSION, "channel_b", "train", ts, list(range(n)))

    settings = _settings(processed_dir, window_size=5)
    report = script_module.check_group(settings, _MISSION, ["channel_a", "channel_b"])

    assert report.n_aligned == n
    assert report.alignment_loss_frac == pytest.approx(0.0)
    assert report.n_joint_segments == n
    assert report.max_joint_segment_len == 1
    assert report.joint_windows == 0


def test_check_group_partial_fragmentation_reduces_but_does_not_zero_windows(
    tmp_path: Path, script_module: types.ModuleType
) -> None:
    processed_dir = tmp_path / "processed"
    n = 20
    ts = list(range(0, n * 30, 30))
    _write_channel(processed_dir, _MISSION, "channel_a", "train", ts, [0] * n)
    # One boundary at the midpoint -> two joint segments of 10 rows each.
    seg_b = [0] * 10 + [1] * 10
    _write_channel(processed_dir, _MISSION, "channel_b", "train", ts, seg_b)

    settings = _settings(processed_dir, window_size=5)  # span = 6
    report = script_module.check_group(settings, _MISSION, ["channel_a", "channel_b"])

    assert report.n_joint_segments == 2
    assert report.max_joint_segment_len == 10
    # Each 10-row segment: 10 - 6 + 1 = 5 windows -> 10 total, vs 15 unfragmented.
    assert report.joint_windows == 10


def test_check_group_joint_windows_shrinks_with_forecast_steps(
    tmp_path: Path, script_module: types.ModuleType
) -> None:
    processed_dir = tmp_path / "processed"
    n = 20
    ts = list(range(0, n * 30, 30))
    _write_channel(processed_dir, _MISSION, "channel_a", "train", ts, [0] * n)
    _write_channel(processed_dir, _MISSION, "channel_b", "train", ts, [0] * n)

    settings = _settings(processed_dir, window_size=5, forecast_steps=4)  # span = 5+1+4-1=9
    report = script_module.check_group(settings, _MISSION, ["channel_a", "channel_b"])
    assert report.joint_windows == 12  # 20 - 9 + 1


# ---------------------------------------------------------------------------
# enumerate_families
# ---------------------------------------------------------------------------


def test_enumerate_families_groups_aligned_channels_and_separates_phase_offset(
    tmp_path: Path, script_module: types.ModuleType
) -> None:
    processed_dir = tmp_path / "processed"
    ts = list(range(0, 20 * 30, 30))
    _write_channel(processed_dir, _MISSION, "channel_a", "train", ts, [0] * 20)
    _write_channel(processed_dir, _MISSION, "channel_b", "train", ts, [0] * 20)
    ts_offset = [t + 15 for t in ts]
    _write_channel(processed_dir, _MISSION, "channel_c", "train", ts_offset, [0] * 20)

    settings = _settings(processed_dir)
    families = script_module.enumerate_families(
        settings, _MISSION, ["channel_a", "channel_b", "channel_c"], probe_days=1
    )

    assert sorted(families) == [["channel_a", "channel_b"], ["channel_c"]]


def test_enumerate_families_singleton_when_no_overlap_anywhere(
    tmp_path: Path, script_module: types.ModuleType
) -> None:
    processed_dir = tmp_path / "processed"
    ts = list(range(0, 20 * 30, 30))
    for i, ch in enumerate(["channel_a", "channel_b", "channel_c"]):
        _write_channel(
            processed_dir, _MISSION, ch, "train", [t + i * 10 for t in ts], [0] * 20
        )
    settings = _settings(processed_dir)
    families = script_module.enumerate_families(
        settings, _MISSION, ["channel_a", "channel_b", "channel_c"], probe_days=1
    )
    assert sorted(len(f) for f in families) == [1, 1, 1]


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def test_main_requires_channels_unless_enumerate_families(
    monkeypatch: pytest.MonkeyPatch, script_module: types.ModuleType
) -> None:
    monkeypatch.setattr("sys.argv", ["check_channel_group.py", "--mission", "ESA-Mission1"])
    with pytest.raises(SystemExit):
        script_module.main()


def test_main_writes_json_report_to_out_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    script_module: types.ModuleType,
) -> None:
    processed_dir = tmp_path / "processed"
    ts = list(range(0, 20 * 30, 30))
    _write_channel(processed_dir, _MISSION, "channel_a", "train", ts, [0] * 20)
    _write_channel(processed_dir, _MISSION, "channel_b", "train", ts, [0] * 20)

    out_path = tmp_path / "report.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "check_channel_group.py",
            "--env", "test",
            "--mission", _MISSION,
            "--channels", "channel_a,channel_b",
            "--out", str(out_path),
        ],
    )
    monkeypatch.setattr(
        script_module,
        "load_settings",
        lambda env: _settings(processed_dir, window_size=5),
    )

    script_module.main()

    import json

    written = json.loads(out_path.read_text())
    assert written["n_aligned"] == 20
    assert written["joint_windows"] == 15
