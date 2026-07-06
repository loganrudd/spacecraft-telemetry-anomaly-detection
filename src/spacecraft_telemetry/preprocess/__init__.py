"""Pandas + PyArrow + Ray Core preprocessing pipeline.

Pure-Python (no JVM): pandas/PyArrow transforms with a @ray.remote fan-out
across channels.

Public API
----------
ESA pipeline:
    run_preprocessing(settings, mission, channels=None, parallel=True) -> dict

ISS pipeline:
    run_iss_preprocessing(settings, channels=None, parallel=True) -> dict

I/O:
    read_channel(path, channel_id, mission_id) -> pd.DataFrame
    read_labels(path) -> pd.DataFrame
    write_series(df, output_path) -> None
    read_iss_ticks(raw_ticks_dir, channel_id) -> pd.DataFrame
    discover_iss_channels(raw_ticks_dir, exclude=None) -> list[str]

Transforms (all pure pandas, single-channel scope):
    handle_nulls, detect_gaps, normalize,
    temporal_train_test_split, label_timesteps
    resample_to_grid, compute_los_mask, augment_with_los
"""

from typing import TYPE_CHECKING, Any

from spacecraft_telemetry.preprocess.io import (
    discover_iss_channels,
    read_channel,
    read_iss_ticks,
    read_labels,
    write_series,
)
from spacecraft_telemetry.preprocess.transforms import (
    augment_with_los,
    compute_los_mask,
    detect_gaps,
    handle_nulls,
    label_timesteps,
    normalize,
    resample_to_grid,
    temporal_train_test_split,
)

# `pipeline` imports ray at module top (the @ray.remote fan-out lives there), but
# ray is deliberately absent from the slim serving image (deploy/api/Dockerfile
# installs [serving,tracking,gcp,inference], not [ml]). Import it lazily via PEP
# 562 so pulling in a pure-pandas sibling like `transforms` (e.g. the API's
# los_stats needs compute_los_mask) does not drag ray into the serving process.
# The two entrypoints below only run in the full [ml] image (CLI / RayJob).
if TYPE_CHECKING:
    from spacecraft_telemetry.preprocess.pipeline import (
        run_iss_preprocessing,
        run_preprocessing,
    )

_LAZY_PIPELINE_EXPORTS = frozenset({"run_preprocessing", "run_iss_preprocessing"})


def __getattr__(name: str) -> Any:
    if name in _LAZY_PIPELINE_EXPORTS:
        from spacecraft_telemetry.preprocess import pipeline

        return getattr(pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "augment_with_los",
    "compute_los_mask",
    "detect_gaps",
    "discover_iss_channels",
    "handle_nulls",
    "label_timesteps",
    "normalize",
    "read_channel",
    "read_iss_ticks",
    "read_labels",
    "resample_to_grid",
    "run_iss_preprocessing",
    "run_preprocessing",
    "temporal_train_test_split",
    "write_series",
]
