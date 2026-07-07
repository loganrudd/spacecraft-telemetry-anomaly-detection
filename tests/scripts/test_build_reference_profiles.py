"""Tests for scripts/build_reference_profiles.py's --split argument wiring.

The script is a standalone CLI (not part of the installed package), so it's
loaded directly by file path rather than imported as a module.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

_SCRIPT_PATH = Path(__file__).parent.parent.parent / "scripts" / "build_reference_profiles.py"


def _load_script_module() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("build_reference_profiles", _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def script_module() -> types.ModuleType:
    return _load_script_module()


def _stub_io(script_module: types.ModuleType, mocker: Any, tmp_path: Path) -> Any:
    """Mock every external I/O call so main() only exercises argument wiring."""
    mocker.patch.object(script_module, "discover_channels", return_value=["ch-a"])
    mock_build = mocker.patch.object(
        script_module,
        "build_reference_profile",
        return_value=pd.DataFrame({"value_normalized": [0.0]}),
    )
    mocker.patch.object(
        script_module, "reference_profile_path", return_value=tmp_path / "ref.parquet"
    )
    mocker.patch.object(script_module, "save_reference_profile")
    return mock_build


class TestSplitArgument:
    def test_default_split_is_test(
        self, script_module: types.ModuleType, mocker: Any, tmp_path: Path
    ) -> None:
        mock_build = _stub_io(script_module, mocker, tmp_path)
        mocker.patch.object(
            sys, "argv",
            ["build_reference_profiles.py", "--mission", "ESA-Mission1", "--env", "test"],
        )

        script_module.main()

        mock_build.assert_called_once()
        _args, kwargs = mock_build.call_args
        assert kwargs["split"] == "test"

    def test_split_train_is_threaded_through(
        self, script_module: types.ModuleType, mocker: Any, tmp_path: Path
    ) -> None:
        mock_build = _stub_io(script_module, mocker, tmp_path)
        mocker.patch.object(
            sys, "argv",
            ["build_reference_profiles.py", "--mission", "ISS",
             "--env", "test", "--split", "train"],
        )

        script_module.main()

        mock_build.assert_called_once()
        _args, kwargs = mock_build.call_args
        assert kwargs["split"] == "train"

    def test_invalid_split_rejected(
        self, script_module: types.ModuleType, mocker: Any
    ) -> None:
        mocker.patch.object(
            sys, "argv",
            ["build_reference_profiles.py", "--mission", "ISS", "--split", "bogus"],
        )
        with pytest.raises(SystemExit):
            script_module.main()
