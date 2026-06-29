"""Normalization parameter loader and z-score normalizer for the live pump.

The preprocessing pipeline writes normalization_params.json to
``{processed_dir}/{mission}/normalization_params.json`` with schema::

    {channel_id: {"mean": float, "std": float}, ...}

The live pump loads this once at startup and calls normalize() per tick to
apply the exact same z-score transform used during training, preventing
train/serve skew.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from upath import UPath

NormalizationParams = dict[str, dict[str, float]]


def load_normalization_params(
    processed_dir: Path | UPath | str,
    mission: str,
) -> NormalizationParams:
    """Load normalization_params.json for *mission* from *processed_dir*.

    Works with both local paths and GCS URIs (``gs://...``).

    Args:
        processed_dir: Root of the preprocessed data tree.
        mission:       Mission identifier (e.g. ``"ISS"``).

    Returns:
        Dict mapping channel_id → ``{"mean": float, "std": float}``.
    """
    path = UPath(str(processed_dir)) / mission / "normalization_params.json"
    with path.open("r") as fh:
        raw: dict[str, Any] = json.load(fh)
    return {
        channel: {"mean": float(v["mean"]), "std": float(v["std"])} for channel, v in raw.items()
    }


def normalize(channel: str, raw: float, params: NormalizationParams) -> float:
    """Apply z-score normalization: ``(raw - mean) / std``.

    Args:
        channel: Channel identifier used to look up params.
        raw:     Raw physical-unit value from the live feed.
        params:  Dict returned by :func:`load_normalization_params`.

    Returns:
        Normalized value suitable for ``ChannelInferenceEngine.step()``.

    Raises:
        KeyError: If *channel* is absent from *params*.
        ZeroDivisionError: If ``std == 0`` for *channel*.
    """
    p = params[channel]
    return (raw - p["mean"]) / p["std"]
