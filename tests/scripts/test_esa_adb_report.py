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
import pytest

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
