"""Pandas + Ray Core preprocessing pipeline.

Orchestrates: read → null-fill → gap-detect → normalize → label → split → write.
Replaces spark/pipeline.py — same public API, same on-disk contract, no JVM.

Parallelism model: each channel is processed by a @ray.remote task (one per CPU).
The labels DataFrame is shared via ray.put() — serialized once, read by all workers.

Sequential mode (parallel=False) runs the same per-channel logic inline without Ray,
used by unit tests to avoid Ray cold-start cost.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import ray

from spacecraft_telemetry.core.config import Settings
from spacecraft_telemetry.core.logging import get_logger
from spacecraft_telemetry.preprocess.io import read_channel, read_labels, write_series
from spacecraft_telemetry.preprocess.transforms import (
    detect_gaps,
    handle_nulls,
    label_timesteps,
    normalize,
    temporal_train_test_split,
)

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Per-channel core logic (called by both parallel and sequential paths)
# ---------------------------------------------------------------------------


def _preprocess_channel(
    settings: Settings,
    mission: str,
    channel: str,
    train_out: Path,
    test_out: Path,
    labels_df: pd.DataFrame | None,
) -> dict[str, Any]:
    """Preprocess a single channel and write train/test Parquet partitions.

    Returns a result dict for the caller to accumulate into the pipeline summary.
    """
    data_dir = Path(str(settings.data.sample_data_dir))
    channel_path = data_dir / mission / "channels" / f"{channel}.parquet"

    raw_df = read_channel(channel_path, channel, mission)
    cleaned = handle_nulls(raw_df)
    gapped = detect_gaps(cleaned, gap_multiplier=settings.preprocess.gap_multiplier)
    normalized, params = normalize(gapped, method=settings.preprocess.normalization)

    if labels_df is not None:
        labeled = label_timesteps(normalized, labels_df)
    else:
        labeled = normalized.copy()
        labeled["is_anomaly"] = False

    series_cols = [
        "telemetry_timestamp", "value_normalized",
        "channel_id", "mission_id", "segment_id", "is_anomaly",
    ]
    series_df = labeled[[c for c in series_cols if c in labeled.columns]]

    train_series, test_series = temporal_train_test_split(
        series_df, train_fraction=settings.preprocess.train_fraction
    )

    train_count = len(train_series)
    test_count = len(test_series)

    write_series(train_series, train_out)
    write_series(test_series, test_out)

    return {
        "channel_id": channel,
        "rows_in": len(raw_df),
        "train_rows": train_count,
        "test_rows": test_count,
        "params": params if train_count > 0 else {},
    }


# ---------------------------------------------------------------------------
# Ray remote wrapper
# ---------------------------------------------------------------------------


@ray.remote(num_cpus=1, max_retries=3)
def _preprocess_channel_remote(
    settings: Settings,
    mission: str,
    channel: str,
    train_out_str: str,
    test_out_str: str,
    labels_df: "pd.DataFrame | None",
) -> dict[str, Any]:
    """Ray remote wrapper around _preprocess_channel.

    Ray auto-dereferences ObjectRef arguments before calling the function body,
    so labels_df arrives as an actual DataFrame (not an ObjectRef). The ray.put()
    in the driver still avoids per-task re-serialisation — one copy in the object
    store is fetched by each worker transparently.
    """
    return _preprocess_channel(
        settings, mission, channel,
        Path(train_out_str), Path(test_out_str),
        labels_df,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_preprocessing(
    settings: Settings,
    mission: str,
    channels: list[str] | None = None,
    parallel: bool = True,
) -> dict[str, int]:
    """Run the full preprocessing pipeline for one mission.

    Reads raw channel Parquet from {data.sample_data_dir}/{mission}/channels/
    and labels from {data.sample_data_dir}/{mission}/labels.csv (optional).
    Writes output to {preprocess.processed_data_dir}/{mission}/:
      - train/   — partitioned by mission_id/channel_id
      - test/    — partitioned by mission_id/channel_id
      - normalization_params.json — per-channel mean/std for inference

    Output directories are cleared before processing so re-runs are idempotent.

    Args:
        settings:  Resolved Settings object.
        mission:   Mission name, e.g. "ESA-Mission1".
        channels:  Explicit channel list; discovers from data_dir if None.
        parallel:  Use Ray Core for channel fan-out (default True).
                   Pass False in unit tests to skip Ray initialisation.

    Returns:
        Summary dict: {channels_processed, rows_in, train_rows, test_rows}.

    Raises:
        FileNotFoundError: If no channel Parquet files found.
    """
    cfg = settings.preprocess
    data_dir = Path(str(settings.data.sample_data_dir))
    output_dir = Path(str(cfg.processed_data_dir))

    channel_dir = data_dir / mission / "channels"
    labels_path = data_dir / mission / "labels.csv"

    # Discover channels if not explicitly provided.
    if channels is None:
        channel_paths = sorted(channel_dir.glob("*.parquet"))
        if not channel_paths:
            raise FileNotFoundError(f"No channel Parquet files found in {channel_dir}")
        channel_list = [p.stem for p in channel_paths]
    else:
        channel_list = channels
        # Validate that requested channels exist.
        for ch in channel_list:
            ch_path = channel_dir / f"{ch}.parquet"
            if not ch_path.exists():
                raise FileNotFoundError(f"Channel Parquet not found: {ch_path}")
        if not channel_list:
            raise FileNotFoundError(f"No channel Parquet files found in {channel_dir}")

    # Read labels once; shared across all channels.
    labels_df: pd.DataFrame | None = None
    if labels_path.exists():
        labels_df = read_labels(labels_path)

    # Clear output dirs: re-runs must not accumulate duplicates.
    train_out = output_dir / mission / "train"
    test_out = output_dir / mission / "test"
    for out_dir in (train_out, test_out):
        if out_dir.exists():
            shutil.rmtree(out_dir)

    log.info(
        "pipeline.start",
        mission=mission,
        n_channels=len(channel_list),
        parallel=parallel,
    )

    if parallel:
        results = _run_parallel(settings, mission, channel_list, train_out, test_out, labels_df)
    else:
        results = _run_sequential(settings, mission, channel_list, train_out, test_out, labels_df)

    # Merge normalization params from all channels that produced train rows.
    normalization_params: dict[str, dict[str, float]] = {}
    total_rows_in = 0
    total_train_rows = 0
    total_test_rows = 0

    for result in results:
        total_rows_in += result["rows_in"]
        total_train_rows += result["train_rows"]
        total_test_rows += result["test_rows"]
        if result["train_rows"] > 0:
            normalization_params.update(result["params"])
        else:
            log.warning(
                "pipeline.channel.no_train_rows",
                channel_id=result["channel_id"],
                mission=mission,
            )

    params_path = output_dir / mission / "normalization_params.json"
    params_path.parent.mkdir(parents=True, exist_ok=True)
    params_path.write_text(json.dumps(normalization_params, indent=2))

    summary: dict[str, int] = {
        "channels_processed": len(channel_list),
        "rows_in": total_rows_in,
        "train_rows": total_train_rows,
        "test_rows": total_test_rows,
    }
    log.info("pipeline.complete", mission=mission, **summary)
    return summary


def _run_sequential(
    settings: Settings,
    mission: str,
    channels: list[str],
    train_out: Path,
    test_out: Path,
    labels_df: pd.DataFrame | None,
) -> list[dict[str, Any]]:
    results = []
    for channel in channels:
        log.info("pipeline.channel.start", channel_id=channel, mission=mission)
        result = _preprocess_channel(settings, mission, channel, train_out, test_out, labels_df)
        log.info("pipeline.channel.done", **{k: v for k, v in result.items() if k != "params"})
        results.append(result)
    return results


def _run_parallel(
    settings: Settings,
    mission: str,
    channels: list[str],
    train_out: Path,
    test_out: Path,
    labels_df: pd.DataFrame | None,
) -> list[dict[str, Any]]:
    # Resolve relative paths before passing to Ray workers.
    abs_settings = settings.model_copy(
        update={
            "preprocess": settings.preprocess.model_copy(
                update={
                    "processed_data_dir": Path(
                        str(settings.preprocess.processed_data_dir)
                    ).resolve()
                }
            ),
            "data": settings.data.model_copy(
                update={
                    "sample_data_dir": Path(str(settings.data.sample_data_dir)).resolve()
                }
            ),
        }
    )

    labels_ref = ray.put(labels_df) if labels_df is not None else None
    settings_ref = ray.put(abs_settings)

    abs_train_out = str(train_out.resolve())
    abs_test_out = str(test_out.resolve())

    futures = [
        _preprocess_channel_remote.remote(
            settings_ref, mission, channel,
            abs_train_out, abs_test_out,
            labels_ref,
        )
        for channel in channels
    ]

    log.info("pipeline.parallel.submitted", n_tasks=len(futures))
    results: list[dict[str, Any]] = ray.get(futures)
    return results
