"""Smoke test for scripts/esa_adb_report.py's CLI wiring.

Reuses the fixture builders from tests/esa_adb/test_report.py (series
Parquet + labels.csv/anomaly_types.csv + fake MLflow scoring runs) rather
than duplicating them — this test only proves argparse -> build_report ->
JSON output wiring works, not the metric math (already covered by
tests/esa_adb/test_metrics.py and test_report.py).
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pytest

from spacecraft_telemetry.model.io import errors_to_bytes, threshold_to_bytes
from tests.esa_adb.test_report import (
    _CHANNEL,
    _MISSION,
    _log_scoring_run,
    _settings,
    _write_labels,
    _write_series,
)

_SCRIPT_PATH = Path(__file__).parent.parent.parent / "scripts" / "esa_adb_report.py"


def _load_script_module() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("esa_adb_report", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def script_module() -> types.ModuleType:
    return _load_script_module()


def test_main_writes_expected_report_json(
    script_module: types.ModuleType, mocker: Any, tmp_path: Path
) -> None:
    processed_dir = tmp_path / "processed"
    sample_dir = tmp_path / "sample"
    _write_series(processed_dir)
    _write_labels(sample_dir)
    mlflow_uri = f"sqlite:///{tmp_path}/mlflow.db"
    # _log_scoring_run uses mlflow's global current tracking URI (via
    # open_run -> mlflow.start_run), not settings.mlflow.tracking_uri
    # directly — mirror the mlflow_uri fixture (tests/esa_adb/conftest.py)
    # by setting it explicitly; the root isolate_mlflow_globals autouse
    # fixture resets it to "" before and after this test.
    mlflow.set_tracking_uri(mlflow_uri)
    settings = _settings(processed_dir, sample_dir, mlflow_uri)

    _log_scoring_run(settings, tuned=False)
    _log_scoring_run(settings, tuned=True)

    mocker.patch.object(script_module, "load_settings", return_value=settings)
    out_path = tmp_path / "report.json"
    mocker.patch.object(
        sys,
        "argv",
        [
            "esa_adb_report.py",
            "--env",
            "test",
            "--mission",
            _MISSION,
            "--channels",
            _CHANNEL,
            "--out",
            str(out_path),
        ],
    )

    script_module.main()

    assert out_path.exists()
    report = json.loads(out_path.read_text())
    assert report["mission"] == _MISSION
    assert report["channels"] == [_CHANNEL]
    assert len(report["rows"]) == 8
    assert len(report["footnotes"]) > 0


def test_main_offline_run_map_never_touches_mlflow(
    script_module: types.ModuleType, mocker: Any, tmp_path: Path
) -> None:
    """--run-map must produce a full report without any MLflow server reachable.

    settings.mlflow.tracking_uri is deliberately left pointing at a bogus
    sqlite path that was never created — if the offline path accidentally
    fell through to querying MLflow, this would fail loudly rather than
    silently succeed via the wrong backend.
    """
    processed_dir = tmp_path / "processed"
    sample_dir = tmp_path / "sample"
    _write_series(processed_dir)
    _write_labels(sample_dir)
    bogus_uri = f"sqlite:///{tmp_path}/does-not-exist/mlflow.db"
    settings = _settings(processed_dir, sample_dir, bogus_uri)

    runs_dir = tmp_path / "runs"
    smoothed = np.zeros(17, dtype=np.float64)
    smoothed[14] = 5.0  # matches the label event window (see test_report.py)
    threshold = np.ones(17, dtype=np.float64)
    for run_id in ("offline-baseline", "offline-tuned"):
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "errors.npy").write_bytes(errors_to_bytes(smoothed))
        (run_dir / "threshold.npy").write_bytes(threshold_to_bytes(threshold))

    run_map_path = tmp_path / "run_map.json"
    run_map_path.write_text(
        json.dumps(
            {
                "mission": _MISSION,
                "runs_dir": str(runs_dir),
                "channels": {
                    _CHANNEL: {
                        "baseline": {
                            "run_id": "offline-baseline",
                            "threshold_min_anomaly_len": 1,
                        },
                        "tuned": {
                            "run_id": "offline-tuned",
                            "threshold_min_anomaly_len": 1,
                        },
                    }
                },
            }
        )
    )

    mocker.patch.object(script_module, "load_settings", return_value=settings)
    out_path = tmp_path / "report.json"
    mocker.patch.object(
        sys,
        "argv",
        [
            "esa_adb_report.py",
            "--env",
            "test",
            "--mission",
            _MISSION,
            "--channels",
            _CHANNEL,
            "--run-map",
            str(run_map_path),
            "--out",
            str(out_path),
        ],
    )

    script_module.main()

    report = json.loads(out_path.read_text())
    assert report["mission"] == _MISSION
    ours_rows = [r for r in report["rows"] if r["label"].startswith("ours")]
    assert len(ours_rows) == 4
    assert all(r["n_events"] is not None for r in ours_rows)
