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


# ---------------------------------------------------------------------------
# Multivariate (docs/plans/021-multivariate-telemanom.md)
# ---------------------------------------------------------------------------


def test_n_channels_default_matches_univariate() -> None:
    """n_channels=1 (the default) is the pre-021 architecture exactly."""
    m1 = TelemanomLSTM(hidden_dim=16, num_layers=2)
    m2 = TelemanomLSTM(hidden_dim=16, num_layers=2, n_channels=1)
    assert m1.lstm.input_size == m2.lstm.input_size == 1
    assert m1.fc.out_features == m2.fc.out_features == 1


def test_forward_shape_multivariate() -> None:
    model = TelemanomLSTM(hidden_dim=16, num_layers=2, n_channels=6)
    x = torch.zeros(4, 250, 6)
    out = model(x)
    assert out.shape == (4, 6)


def test_param_count_scales_with_n_channels() -> None:
    """Only input_size and the output Linear widen with C — hidden_dim is fixed,
    so growth is linear in C via the LSTM's first-layer input weights + fc."""
    uni = TelemanomLSTM(hidden_dim=80, num_layers=2)
    multi = TelemanomLSTM(hidden_dim=80, num_layers=2, n_channels=6)
    n_uni = sum(p.numel() for p in uni.parameters())
    n_multi = sum(p.numel() for p in multi.parameters())
    assert n_multi > n_uni
    # hidden_dim/num_layers are Hundman-locked (.claude/rules/pytorch.md) —
    # the multivariate model must not silently drift into a bigger LSTM.
    assert multi.hidden_dim == uni.hidden_dim == 80
    assert multi.num_layers == uni.num_layers == 2


def test_build_model_derives_n_channels_from_input_channels() -> None:
    cfg = ModelConfig(
        hidden_dim=16, num_layers=1,
        input_channels=["channel_41", "channel_42", "channel_43"],
        target_channels=["channel_41", "channel_42", "channel_43"],
    )
    model = build_model(cfg)
    assert model.n_channels == 3
    x = torch.zeros(2, 10, 3)
    assert model(x).shape == (2, 3)


def test_build_model_input_channels_none_is_univariate() -> None:
    cfg = ModelConfig(hidden_dim=16, num_layers=1)
    model = build_model(cfg)
    assert model.n_channels == 1
