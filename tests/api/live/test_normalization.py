"""Tests for the normalization loader and normalize() function."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spacecraft_telemetry.api.live.normalization import (
    load_normalization_params,
    normalize,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def norm_params_file(tmp_path: Path) -> Path:
    """Write a normalization_params.json fixture and return processed_dir."""
    mission = "ISS"
    mission_dir = tmp_path / mission
    mission_dir.mkdir()
    params = {
        "S1000003": {"mean": 20.5, "std": 3.2},
        "P4000001": {"mean": 135.0, "std": 8.7},
        "USLAB000018": {"mean": 0.0, "std": 1.0},
    }
    (mission_dir / "normalization_params.json").write_text(json.dumps(params))
    return tmp_path


# ---------------------------------------------------------------------------
# load_normalization_params
# ---------------------------------------------------------------------------


def test_load_returns_all_channels(norm_params_file: Path) -> None:
    params = load_normalization_params(norm_params_file, "ISS")
    assert set(params.keys()) == {"S1000003", "P4000001", "USLAB000018"}


def test_load_values_are_float(norm_params_file: Path) -> None:
    params = load_normalization_params(norm_params_file, "ISS")
    for ch, v in params.items():
        assert isinstance(v["mean"], float), f"{ch}: mean not float"
        assert isinstance(v["std"], float), f"{ch}: std not float"


def test_load_mean_std_correct(norm_params_file: Path) -> None:
    params = load_normalization_params(norm_params_file, "ISS")
    assert params["S1000003"]["mean"] == pytest.approx(20.5)
    assert params["S1000003"]["std"] == pytest.approx(3.2)
    assert params["P4000001"]["mean"] == pytest.approx(135.0)
    assert params["P4000001"]["std"] == pytest.approx(8.7)


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_normalization_params(tmp_path, "ISS")


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------


def test_normalize_zero_mean_unit_std() -> None:
    """normalize with mean=0, std=1 is identity."""
    params = {"ch": {"mean": 0.0, "std": 1.0}}
    assert normalize("ch", 3.5, params) == pytest.approx(3.5)


def test_normalize_known_values() -> None:
    """(raw - mean) / std for known inputs."""
    params = {"S1000003": {"mean": 20.5, "std": 3.2}}
    result = normalize("S1000003", 23.7, params)
    assert result == pytest.approx((23.7 - 20.5) / 3.2)


def test_normalize_negative_result() -> None:
    """Value below mean produces negative normalized value."""
    params = {"ch": {"mean": 10.0, "std": 2.0}}
    assert normalize("ch", 6.0, params) == pytest.approx(-2.0)


def test_normalize_round_trip() -> None:
    """Denormalizing the result recovers the original value."""
    params = {"ch": {"mean": 50.0, "std": 5.0}}
    raw = 47.3
    normalized = normalize("ch", raw, params)
    recovered = normalized * params["ch"]["std"] + params["ch"]["mean"]
    assert recovered == pytest.approx(raw)


def test_normalize_missing_channel_raises() -> None:
    params = {"ch": {"mean": 0.0, "std": 1.0}}
    with pytest.raises(KeyError):
        normalize("nonexistent", 1.0, params)


def test_normalize_zero_std_raises() -> None:
    params = {"ch": {"mean": 5.0, "std": 0.0}}
    with pytest.raises(ZeroDivisionError):
        normalize("ch", 5.0, params)
