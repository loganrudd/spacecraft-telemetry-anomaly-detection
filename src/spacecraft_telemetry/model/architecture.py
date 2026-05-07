"""Telemanom LSTM architecture (Hundman et al. 2018).

2-layer LSTM forecaster for univariate telemetry channels.
Input:  (B, window_size, 1)
Output: (B, 1)  — one-step-ahead prediction

Architecture is intentionally off-the-shelf. Do NOT modify without a plan revision.
Phase 5 (Ray Tune) varies hidden_dim / num_layers / dropout via ModelConfig overrides.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from spacecraft_telemetry.core.config import ModelConfig


class TelemanomLSTM(nn.Module):
    """Two-layer LSTM forecaster faithful to Hundman et al. 2018 defaults."""

    def __init__(self, hidden_dim: int = 80, num_layers: int = 2, dropout: float = 0.3) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # dropout is applied between LSTM layers; ignored when num_layers == 1
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, W, 1)
        lstm_out, _ = self.lstm(x)          # (B, W, hidden_dim)
        last_hidden = lstm_out[:, -1, :]    # (B, hidden_dim)
        out: torch.Tensor = self.fc(last_hidden)
        return out                          # (B, 1)


def build_model(model_config: ModelConfig) -> TelemanomLSTM:
    """Construct a TelemanomLSTM from a ModelConfig.

    Keeps Settings out of the architecture module so it can be imported
    independently in Phase 8 (FastAPI serving) without the full config stack.
    """
    return TelemanomLSTM(
        hidden_dim=model_config.hidden_dim,
        num_layers=model_config.num_layers,
        dropout=model_config.dropout,
    )
