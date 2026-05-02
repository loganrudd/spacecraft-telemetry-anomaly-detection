"""Device resolution for Telemanom LSTM training and inference.

Always use resolve_device(setting) — never hardcode torch.device(...) in callers.
"""

from __future__ import annotations

from typing import Literal


def resolve_device(
    setting: Literal["auto", "cpu", "mps", "cuda"] | str,
) -> torch.device:  # type: ignore[name-defined]  # noqa: F821
    """Resolve a device setting string to a torch.device.

    Args:
        setting: One of "auto", "cpu", "mps", "cuda".
                 "auto" selects CUDA > MPS > CPU in priority order.

    Returns:
        A torch.device for the resolved backend.

    Raises:
        ValueError: If an explicit backend is requested but not available.
    """
    import torch

    if setting == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if setting == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(
                "device='cuda' requested but CUDA is not available on this machine."
            )
        return torch.device("cuda")

    if setting == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError(
                "device='mps' requested but MPS is not available on this machine. "
                "MPS requires macOS 12.3+ with an Apple Silicon or AMD GPU."
            )
        return torch.device("mps")

    if setting == "cpu":
        return torch.device("cpu")

    raise ValueError(
        f"Unknown device setting {setting!r}. "
        "Expected one of: 'auto', 'cpu', 'mps', 'cuda'."
    )
