"""Train→score seam at forecast_steps > 1 (docs/reviews/021, item C2).

Every H=10 shape contract is verified on ONE side of the train/score boundary
or the other — the dataset builds (C, F) targets, the architecture emits a
(C, H) head, collapse_forecast_errors reduces (N, C, H) — but nothing crossed
the seam. That is exactly where 021.7's real defect lived: the horizon reached
the training job and not the scoring job, so a correctly-trained H=10 model was
scored under H=1 settings and reported a plausible number.

This runs the whole path on a tiny fixture: train_channel → score_channel with
``forecast_steps=3`` over a multivariate group of 2, then asserts the two
properties that only hold if BOTH sides agree about the horizon:

1. Per-channel metrics exist for every channel in the group.
2. The saved error arrays are COLLAPSED — one error per window per channel,
   length equal to the window count the H=3 span produces, not the H=1 span's.

.claude/rules/testing.md permits slow full-training tests and excludes them
from the default run, which is where this belongs: it is a real training loop,
not a unit test of an internal.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("torch")

import pyarrow as pa
import pyarrow.parquet as pq

_MISSION = "ESA-Mission1"
_SUBSYSTEM = "subsystem_1"
_CHANNELS = ["channel_41", "channel_42"]
_WINDOW_SIZE = 8
_FORECAST_STEPS = 3
_N_TRAIN = 80
_N_TEST = 50
_ANOMALY_TAIL = 6


def _write_partition(
    processed_dir: Path, split: str, channel: str, n: int, *, anomaly_tail: int = 0
) -> None:
    """One channel partition on a fixed 90s grid, deterministic values.

    Both channels share the timestamp grid so the multivariate inner-join
    aligns every row — an alignment loss here would change the window count
    and make the length assertions test the join instead of the horizon.
    """
    base = datetime(2000, 1, 1, tzinfo=UTC).timestamp()
    timestamps = [
        pa.scalar(base + i * 90, type=pa.timestamp("s", tz="UTC")).cast(
            pa.timestamp("us", tz="UTC")
        )
        for i in range(n)
    ]
    # A repeating ramp: learnable enough that training is not degenerate, and
    # identical across runs so the test cannot flake on RNG.
    offset = float(_CHANNELS.index(channel))
    values = [float((i % 7) + offset) for i in range(n)]
    table = pa.table({
        "telemetry_timestamp": pa.array(timestamps, type=pa.timestamp("us", tz="UTC")),
        "value_normalized": pa.array(values, type=pa.float32()),
        "segment_id": pa.array(np.zeros(n, dtype=np.int32)),
        "is_anomaly": pa.array([False] * (n - anomaly_tail) + [True] * anomaly_tail),
    })
    part_dir = (
        processed_dir / _MISSION / split / f"mission_id={_MISSION}" / f"channel_id={channel}"
    )
    part_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, part_dir / "part.parquet")


@pytest.fixture()
def multistep_settings(mlflow_uri: str, tmp_path: Path) -> Any:
    """Settings for a multivariate H=3 model over a 2-channel group."""
    from spacecraft_telemetry.core.config import load_settings

    processed_dir = tmp_path / "processed"
    for channel in _CHANNELS:
        _write_partition(processed_dir, "train", channel, _N_TRAIN)
        _write_partition(
            processed_dir, "test", channel, _N_TEST, anomaly_tail=_ANOMALY_TAIL
        )

    norm_file = processed_dir / _MISSION / "normalization_params.json"
    norm_file.parent.mkdir(parents=True, exist_ok=True)
    norm_file.write_text(
        json.dumps({ch: {"mean": 0.0, "std": 1.0} for ch in _CHANNELS})
    )

    base = load_settings("test")
    return base.model_copy(update={
        "mlflow": base.mlflow.model_copy(update={"tracking_uri": mlflow_uri}),
        "preprocess": base.preprocess.model_copy(
            update={"processed_data_dir": processed_dir}
        ),
        "model": base.model.model_copy(update={
            "artifacts_dir": tmp_path / "models",
            "window_size": _WINDOW_SIZE,
            "forecast_steps": _FORECAST_STEPS,
            "input_channels": _CHANNELS,
            "target_channels": _CHANNELS,
            "hidden_dim": 4,
            "num_layers": 1,
            "dropout": 0.0,
            "batch_size": 4,
            "epochs": 2,
            "error_smoothing_window": 3,
            "threshold_window": 8,
        }),
    })


@pytest.mark.slow
def test_multistep_multivariate_train_then_score(
    multistep_settings: Any, mlflow_uri: str
) -> None:
    """The seam: one trained H=3 group model, scored by the real scorer."""
    from mlflow.tracking import MlflowClient

    from spacecraft_telemetry.mlflow_tracking.conventions import experiment_name
    from spacecraft_telemetry.model.dataset import load_window_labels
    from spacecraft_telemetry.model.io import bytes_to_errors, download_artifact_bytes
    from spacecraft_telemetry.model.scoring import score_channel
    from spacecraft_telemetry.model.training import train_channel

    settings = multistep_settings
    train_channel(settings, _MISSION, _SUBSYSTEM)
    metrics = score_channel(settings, _MISSION, _SUBSYSTEM)

    # 1. Per-channel breakdown — the multivariate return contract.
    assert set(metrics) == set(_CHANNELS)
    for channel in _CHANNELS:
        assert 0.0 <= metrics[channel]["f0_5"] <= 1.0
        assert 0.0 <= metrics[channel]["seg_f0_5"] <= 1.0

    # 2. The saved arrays are collapsed to one error per window per channel.
    #    The expected length comes from load_window_labels, which derives the
    #    window index from the SAME forecast_steps — so if either side of the
    #    seam disagreed about the horizon, these lengths would differ.
    expected_windows = len(load_window_labels(settings, _MISSION, _SUBSYSTEM))
    assert expected_windows > 0

    client = MlflowClient(tracking_uri=mlflow_uri)
    exp = client.get_experiment_by_name(
        experiment_name(settings.model.model_type, "scoring", _MISSION)
    )
    assert exp is not None
    runs = client.search_runs([exp.experiment_id])
    assert len(runs) == 1, "one run per group model, not one per channel"
    run_id = runs[0].info.run_id

    for channel in _CHANNELS:
        errors = bytes_to_errors(
            download_artifact_bytes(run_id, f"errors/{channel}.npy", mlflow_uri)
        )
        threshold = bytes_to_errors(
            download_artifact_bytes(run_id, f"threshold/{channel}.npy", mlflow_uri)
        )
        assert errors.ndim == 1, (
            f"{channel}: saved errors are {errors.shape}, not collapsed to one "
            "value per window — the horizon axis survived into the scorer"
        )
        assert errors.shape == (expected_windows,)
        assert threshold.shape == (expected_windows,)

    # 3. The horizon is recorded on the run, so the two sides can be compared
    #    after the fact rather than inferred.
    assert runs[0].data.params.get("forecast_steps") == str(_FORECAST_STEPS)


@pytest.mark.slow
def test_multistep_window_count_is_shorter_than_single_step(
    multistep_settings: Any,
) -> None:
    """A longer horizon consumes more timesteps per window, so FEWER windows
    fit. Pins the direction: if the two sides of the seam ever silently
    disagreed about forecast_steps, this is the difference that would show up
    as a length mismatch rather than as wrong numbers."""
    from spacecraft_telemetry.model.dataset import load_window_labels

    settings = multistep_settings
    multi = len(load_window_labels(settings, _MISSION, _SUBSYSTEM))

    single = settings.model_copy(update={
        "model": settings.model.model_copy(update={"forecast_steps": 1})
    })
    one_step = len(load_window_labels(single, _MISSION, _SUBSYSTEM))

    assert multi == one_step - (_FORECAST_STEPS - 1), (
        f"H={_FORECAST_STEPS} produced {multi} windows and H=1 produced "
        f"{one_step}; the span differs by forecast_steps - 1"
    )
