"""Tests for model.device.resolve_device."""

import pytest

torch = pytest.importorskip("torch")

from spacecraft_telemetry.model.device import resolve_device  # noqa: E402


def test_cpu_returns_cpu_device() -> None:
    device = resolve_device("cpu")
    assert device == torch.device("cpu")


def test_auto_returns_a_device() -> None:
    device = resolve_device("auto")
    assert device.type in ("cpu", "mps", "cuda")


def test_unknown_setting_raises() -> None:
    with pytest.raises(ValueError, match="Unknown device setting"):
        resolve_device("tpu")  # type: ignore[arg-type]


def test_cuda_raises_when_unavailable() -> None:
    if torch.cuda.is_available():
        pytest.skip("CUDA is available on this machine")
    with pytest.raises(ValueError, match="CUDA is not available"):
        resolve_device("cuda")


def test_mps_raises_when_unavailable() -> None:
    if torch.backends.mps.is_available():
        pytest.skip("MPS is available on this machine")
    with pytest.raises(ValueError, match="MPS is not available"):
        resolve_device("mps")
