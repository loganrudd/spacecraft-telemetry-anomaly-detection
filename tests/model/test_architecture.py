"""Tests for model.architecture — TelemanomLSTM."""

import pytest

torch = pytest.importorskip("torch")

from spacecraft_telemetry.core.config import ModelConfig  # noqa: E402
from spacecraft_telemetry.model.architecture import TelemanomLSTM, build_model  # noqa: E402


@pytest.fixture
def default_model() -> TelemanomLSTM:
    return TelemanomLSTM()


def test_forward_shape(default_model: TelemanomLSTM) -> None:
    x = torch.zeros(4, 250, 1)
    out = default_model(x)
    assert out.shape == (4, 1)


def test_forward_shape_single_sample(default_model: TelemanomLSTM) -> None:
    x = torch.zeros(1, 250, 1)
    out = default_model(x)
    assert out.shape == (1, 1)


def test_param_count_in_range(default_model: TelemanomLSTM) -> None:
    n_params = sum(p.numel() for p in default_model.parameters())
    assert 50_000 <= n_params <= 500_000, f"Unexpected param count: {n_params}"


def test_deterministic_with_seed() -> None:
    x = torch.ones(2, 10, 1)
    torch.manual_seed(0)
    m1 = TelemanomLSTM(hidden_dim=16, num_layers=2)
    torch.manual_seed(0)
    m2 = TelemanomLSTM(hidden_dim=16, num_layers=2)
    # eval() disables dropout so the comparison is deterministic given equal weights
    m1.eval()
    m2.eval()
    with torch.no_grad():
        assert torch.equal(m1(x), m2(x))


def test_build_model_uses_config() -> None:
    cfg = ModelConfig(hidden_dim=32, num_layers=1, dropout=0.0)
    model = build_model(cfg)
    assert model.hidden_dim == 32
    assert model.num_layers == 1


def test_build_model_forward_shape() -> None:
    cfg = ModelConfig(hidden_dim=16, num_layers=2)
    model = build_model(cfg)
    x = torch.zeros(3, 10, 1)
    assert model(x).shape == (3, 1)
