"""Tests for esa_adb.offline — run-map loading for the tracking-server-free path."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spacecraft_telemetry.esa_adb.offline import OfflineRunMap, RunSpec, load_run_map


def _write_map(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "run_map.json"
    path.write_text(json.dumps(data))
    return path


def _valid_map() -> dict:
    return {
        "mission": "ESA-Mission1",
        "runs_dir": "data/adb_offline/runs",
        "channels": {
            "channel_41": {
                "baseline": {"run_id": "aaa", "threshold_min_anomaly_len": 3},
                "tuned": {"run_id": "bbb", "threshold_min_anomaly_len": 7},
            },
            "channel_42": {
                "baseline": {"run_id": "ccc", "threshold_min_anomaly_len": 3},
            },
        },
    }


# ---------------------------------------------------------------------------
# load_run_map
# ---------------------------------------------------------------------------


class TestLoadRunMap:
    def test_loads_mission_and_channels(self, tmp_path: Path) -> None:
        path = _write_map(tmp_path, _valid_map())
        run_map = load_run_map(path)
        assert run_map.mission == "ESA-Mission1"
        assert run_map.channels() == ["channel_41", "channel_42"]

    def test_builds_artifact_paths_from_runs_dir(self, tmp_path: Path) -> None:
        path = _write_map(tmp_path, _valid_map())
        run_map = load_run_map(path)
        spec = run_map.get("channel_41", tuned=False)
        assert spec.run_id == "aaa"
        assert spec.errors_path == "data/adb_offline/runs/aaa/errors.npy"
        assert spec.threshold_path == "data/adb_offline/runs/aaa/threshold.npy"
        assert spec.threshold_min_anomaly_len == 3

    def test_runs_dir_trailing_slash_stripped(self, tmp_path: Path) -> None:
        data = _valid_map()
        data["runs_dir"] = "data/adb_offline/runs/"
        path = _write_map(tmp_path, data)
        run_map = load_run_map(path)
        spec = run_map.get("channel_41", tuned=False)
        assert spec.errors_path == "data/adb_offline/runs/aaa/errors.npy"

    def test_channel_with_only_baseline_has_no_tuned_entry(self, tmp_path: Path) -> None:
        path = _write_map(tmp_path, _valid_map())
        run_map = load_run_map(path)
        assert "channel_42" not in run_map.tuned
        assert "channel_42" in run_map.baseline

    def test_missing_top_level_key_raises(self, tmp_path: Path) -> None:
        data = _valid_map()
        del data["runs_dir"]
        path = _write_map(tmp_path, data)
        with pytest.raises(ValueError, match="runs_dir"):
            load_run_map(path)

    def test_empty_channels_raises(self, tmp_path: Path) -> None:
        data = _valid_map()
        data["channels"] = {}
        path = _write_map(tmp_path, data)
        with pytest.raises(ValueError, match="empty"):
            load_run_map(path)

    def test_unknown_variant_raises(self, tmp_path: Path) -> None:
        data = _valid_map()
        data["channels"]["channel_41"]["bogus"] = {"run_id": "x", "threshold_min_anomaly_len": 1}
        path = _write_map(tmp_path, data)
        with pytest.raises(ValueError, match="unknown variant"):
            load_run_map(path)

    def test_missing_run_id_raises(self, tmp_path: Path) -> None:
        data = _valid_map()
        del data["channels"]["channel_41"]["baseline"]["run_id"]
        path = _write_map(tmp_path, data)
        with pytest.raises(ValueError, match="run_id"):
            load_run_map(path)

    def test_missing_threshold_min_anomaly_len_raises(self, tmp_path: Path) -> None:
        data = _valid_map()
        del data["channels"]["channel_41"]["baseline"]["threshold_min_anomaly_len"]
        path = _write_map(tmp_path, data)
        with pytest.raises(ValueError, match="threshold_min_anomaly_len"):
            load_run_map(path)

    def test_extra_top_level_keys_are_ignored(self, tmp_path: Path) -> None:
        """A _provenance-style metadata key must not break loading."""
        data = _valid_map()
        data["_provenance"] = {"note": "recovered from GCS"}
        path = _write_map(tmp_path, data)
        run_map = load_run_map(path)
        assert run_map.mission == "ESA-Mission1"


# ---------------------------------------------------------------------------
# OfflineRunMap.get
# ---------------------------------------------------------------------------


class TestOfflineRunMapGet:
    def test_get_missing_channel_raises_key_error(self, tmp_path: Path) -> None:
        path = _write_map(tmp_path, _valid_map())
        run_map = load_run_map(path)
        with pytest.raises(KeyError, match="channel_99"):
            run_map.get("channel_99", tuned=False)

    def test_get_missing_tuned_variant_raises_key_error(self, tmp_path: Path) -> None:
        path = _write_map(tmp_path, _valid_map())
        run_map = load_run_map(path)
        with pytest.raises(KeyError, match="tuned"):
            run_map.get("channel_42", tuned=True)

    def test_direct_construction(self) -> None:
        spec = RunSpec(
            run_id="x",
            errors_path="e.npy",
            threshold_path="t.npy",
            threshold_min_anomaly_len=5,
        )
        run_map = OfflineRunMap(mission="M", baseline={"c": spec}, tuned={})
        assert run_map.get("c", tuned=False) is spec
        assert run_map.channels() == ["c"]
