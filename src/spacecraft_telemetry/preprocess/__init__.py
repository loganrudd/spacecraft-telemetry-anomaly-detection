"""Pandas + PyArrow + Ray Core preprocessing pipeline.

Pure-Python (no JVM): pandas/PyArrow transforms with a @ray.remote fan-out
across channels.

Public API
----------
run_preprocessing(settings, mission, channels=None, parallel=True) -> dict
    Full pipeline for one mission: read → clean → normalize → label → split → write.

read_channel(path, channel_id, mission_id) -> pd.DataFrame
read_labels(path) -> pd.DataFrame
write_series(df, output_path) -> None

Transforms (all pure pandas, single-channel scope):
    handle_nulls, detect_gaps, normalize,
    temporal_train_test_split, label_timesteps
"""

from spacecraft_telemetry.preprocess.io import read_channel, read_labels, write_series
from spacecraft_telemetry.preprocess.pipeline import run_preprocessing
from spacecraft_telemetry.preprocess.transforms import (
    detect_gaps,
    handle_nulls,
    label_timesteps,
    normalize,
    temporal_train_test_split,
)

__all__ = [
    "detect_gaps",
    "handle_nulls",
    "label_timesteps",
    "normalize",
    "read_channel",
    "read_labels",
    "run_preprocessing",
    "temporal_train_test_split",
    "write_series",
]
