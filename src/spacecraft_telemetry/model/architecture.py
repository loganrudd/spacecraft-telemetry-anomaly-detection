"""Telemanom LSTM architecture (Hundman et al. 2018).

2-layer LSTM forecaster.
Univariate (default):    Input (B, window_size, 1)  -> Output (B, 1)
Multivariate (docs/plans/021-multivariate-telemanom.md):
                          Input (B, window_size, C)  -> Output (B, C)
Multi-step (021.7):       Input (B, window_size, C)  -> Output (B, C, H)

Architecture is intentionally off-the-shelf. Do NOT modify without a plan revision.
Phase 5 (Ray Tune) varies hidden_dim / num_layers / dropout via ModelConfig overrides.
Plan 021 is the one sanctioned exception to "off-the-shelf": ``n_channels`` and
``forecast_steps`` widen ``input_size`` and the output ``Linear`` to match the
ESA-ADB comparator's 6-in/6-out, 10-step-ahead config — everything else (2 layers,
hidden_dim=80, dropout=0.3) is unchanged.

Output rank is deliberately 2-D when ``forecast_steps == 1`` and 3-D otherwise,
rather than always 3-D with a trailing 1. Every pre-021.7 caller (training loss,
predict(), the serving engine) consumes (B, C) or (B, 1), and a gratuitous
trailing axis would break them all silently via broadcasting rather than loudly
via a shape error.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from spacecraft_telemetry.core.config import ModelConfig


class TelemanomLSTM(nn.Module):
    """Two-layer LSTM forecaster faithful to Hundman et al. 2018 defaults."""

    def __init__(
        self,
        hidden_dim: int = 80,
        num_layers: int = 2,
        dropout: float = 0.3,
        n_channels: int = 1,
        forecast_steps: int = 1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.n_channels = n_channels
        self.forecast_steps = forecast_steps
        # dropout is applied between LSTM layers; ignored when num_layers == 1
        self.lstm = nn.LSTM(
            input_size=n_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        # One flat head emitting C*H, reshaped on the way out — NOT H separate
        # heads. A single Linear keeps the parameter count and initialisation
        # identical to the C-output model when H == 1, so forecast_steps=1 is
        # provably the same architecture rather than merely a similar one.
        self.fc = nn.Linear(hidden_dim, n_channels * forecast_steps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, W, n_channels)
        lstm_out, _ = self.lstm(x)          # (B, W, hidden_dim)
        last_hidden = lstm_out[:, -1, :]    # (B, hidden_dim)
        out: torch.Tensor = self.fc(last_hidden)   # (B, n_channels * H)
        if self.forecast_steps > 1:
            # (B, C, H) — channel-major so out[:, i, :] is channel i's forecast
            # trajectory, matching how scoring extracts a per-channel series.
            return out.view(-1, self.n_channels, self.forecast_steps)
        return out                          # (B, n_channels), pre-021.7 shape


def build_model(model_config: ModelConfig) -> TelemanomLSTM:
    """Construct a TelemanomLSTM from a ModelConfig.

    Keeps Settings out of the architecture module so it can be imported
    independently in Phase 8 (FastAPI serving) without the full config stack.

    ``n_channels`` is derived from ``model_config.input_channels`` — None (the
    default) is the univariate n_channels=1 case, byte-identical to pre-021.
    ``forecast_steps`` defaults to 1, the pre-021.7 scalar-target head.
    """
    n_channels = len(model_config.input_channels) if model_config.input_channels else 1
    return TelemanomLSTM(
        hidden_dim=model_config.hidden_dim,
        num_layers=model_config.num_layers,
        dropout=model_config.dropout,
        n_channels=n_channels,
        forecast_steps=model_config.forecast_steps,
    )
