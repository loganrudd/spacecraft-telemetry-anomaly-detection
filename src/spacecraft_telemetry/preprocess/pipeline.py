"""Pandas + Ray Core preprocessing pipeline.

Orchestrates: read → null-fill → gap-detect → normalize → label → split → write.
Pure-Python, no JVM.

Parallelism model: each channel is processed by a @ray.remote task (one per CPU).
The labels DataFrame is shared via ray.put() — serialized once, read by all workers.

Sequential mode (parallel=False) runs the same per-channel logic inline without Ray,
used by unit tests to avoid Ray cold-start cost.
"""

from __future__ import annotations

import json
import resource
from typing import Any

import pandas as pd
import ray
from upath import UPath

from spacecraft_telemetry.core.config import Settings
from spacecraft_telemetry.core.logging import get_logger
from spacecraft_telemetry.core.paths import absolutize_if_local, to_upath
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
    train_out: UPath,
    test_out: UPath,
    labels_df: pd.DataFrame | None,
) -> dict[str, Any]:
    """Preprocess a single channel and write train/test Parquet partitions.

    Returns a result dict for the caller to accumulate into the pipeline summary.
    """
    data_dir = to_upath(settings.data.sample_data_dir)
    channel_path = data_dir / mission / "channels" / f"{channel}.parquet"

    try:
        raw_df = read_channel(channel_path, channel, mission)
    except TypeError as exc:
        log.warning("pipeline.channel.skip", channel_id=channel, mission=mission, reason=str(exc))
        return {"channel_id": channel, "rows_in": 0, "train_rows": 0, "test_rows": 0, "params": {}}
    rows_in = len(raw_df)
    cleaned = handle_nulls(raw_df)
    del raw_df
    gapped = detect_gaps(cleaned, gap_multiplier=settings.preprocess.gap_multiplier)
    del cleaned
    normalized, params = normalize(gapped, method=settings.preprocess.normalization)
    del gapped

    if labels_df is not None:
        labeled = label_timesteps(normalized, labels_df)
    else:
        labeled = normalized.copy()
        labeled["is_anomaly"] = False
    del normalized

    series_cols = [
        "telemetry_timestamp", "value_normalized",
        "channel_id", "mission_id", "segment_id", "is_anomaly",
    ]
    series_df = labeled[series_cols]
    del labeled

    train_series, test_series = temporal_train_test_split(
        series_df,
        train_fraction=settings.preprocess.train_fraction,
        train_lookback=settings.preprocess.train_lookback,
    )
    del series_df

    train_count = len(train_series)
    test_count = len(test_series)

    write_series(train_series, train_out)
    del train_series
    write_series(test_series, test_out)

    # ru_maxrss unit: KB on Linux (GKE workers), bytes on macOS.
    # Values over 1 billion are bytes (macOS); otherwise kilobytes (Linux).
    _rss_raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    _peak_mb = _rss_raw / 1024 / 1024 if _rss_raw > 1_000_000_000 else _rss_raw / 1024
    log.info("pipeline.channel.peak_rss_mb", channel_id=channel, peak_rss_mb=round(_peak_mb, 1))

    return {
        "channel_id": channel,
        "rows_in": rows_in,
        "train_rows": train_count,
        "test_rows": test_count,
        "params": params if train_count > 0 else {},
    }


# ---------------------------------------------------------------------------
# Ray remote wrapper
# ---------------------------------------------------------------------------


@ray.remote(num_cpus=1, max_calls=1, max_retries=3)
def _preprocess_channel_remote(
    settings: Settings,
    mission: str,
    channel: str,
    train_out_str: str,  # serialized as str across Ray boundary; reconstructed as UPath
    test_out_str: str,
    labels_df: pd.DataFrame | None,
) -> dict[str, Any]:
    """Ray remote wrapper around _preprocess_channel.

    Ray auto-dereferences ObjectRef arguments before calling the function body,
    so labels_df arrives as an actual DataFrame (not an ObjectRef). The ray.put()
    in the driver still avoids per-task re-serialisation — one copy in the object
    store is fetched by each worker transparently.
    """
    return _preprocess_channel(
        settings, mission, channel,
        UPath(train_out_str), UPath(test_out_str),
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
    data_dir = to_upath(settings.data.sample_data_dir)
    output_dir = to_upath(cfg.processed_data_dir)

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

    # Filter against suitability manifest if one exists. Channels flagged as
    # skip (empty/flat/constant) are excluded here rather than failing mid-run.
    # Pass the raw sample_data_dir string straight through — suitability_manifest_path
    # calls to_upath internally. Wrapping in pathlib.Path first would collapse a
    # gs:// URI to gs:/ and break the bucket parse.
    from spacecraft_telemetry.preprocess.profiler import (
        filter_channels as _filter_channels,
        suitability_manifest_path as _manifest_path,
    )

    _manifest = _manifest_path(str(settings.data.sample_data_dir), mission)
    channel_list, _skipped = _filter_channels(channel_list, _manifest)
    if _skipped:
        log.info(
            "pipeline.channels.skipped",
            mission=mission,
            n_skipped=len(_skipped),
            channels=_skipped,
            reason="channel_suitability manifest",
        )
    if not channel_list:
        raise FileNotFoundError(
            f"All channels for {mission} were filtered by the suitability manifest. "
            "Run 'preprocess profile' to inspect, or delete channel_suitability.json to reset."
        )

    # Read labels once; shared across all channels.
    labels_df: pd.DataFrame | None = None
    if labels_path.exists():
        labels_df = read_labels(labels_path)

    # Clear output dirs: re-runs must not accumulate duplicates.
    # Use the underlying fsspec fs.rm() so this works for both local and gs://.
    train_out = output_dir / mission / "train"
    test_out = output_dir / mission / "test"
    for out_dir in (train_out, test_out):
        if out_dir.exists():
            out_dir.fs.rm(str(out_dir), recursive=True)

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

    mission_out = output_dir / mission
    if not str(mission_out).startswith("gs://"):
        mission_out.mkdir(parents=True, exist_ok=True)

    params_path = mission_out / "normalization_params.json"
    params_path.write_text(json.dumps(normalization_params, indent=2))

    # Write a channels.txt manifest consumed by --channels-from in cloud-train/tune.
    successful_channels = [
        r["channel_id"] for r in results if r["train_rows"] > 0
    ]
    channels_txt = mission_out / "channels.txt"
    channels_txt.write_text("\n".join(successful_channels) + "\n")
    log.info("pipeline.channels_manifest", path=str(channels_txt), n=len(successful_channels))

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
    train_out: UPath,
    test_out: UPath,
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
    train_out: UPath,
    test_out: UPath,
    labels_df: pd.DataFrame | None,
) -> list[dict[str, Any]]:
    # Resolve relative local paths to absolute before ray.put() — Ray workers
    # run from Ray's session dir. Cloud URIs (gs://) pass through unchanged.
    abs_settings = settings.model_copy(
        update={
            "preprocess": settings.preprocess.model_copy(
                update={
                    "processed_data_dir": absolutize_if_local(
                        settings.preprocess.processed_data_dir
                    )
                }
            ),
            "data": settings.data.model_copy(
                update={
                    "sample_data_dir": absolutize_if_local(settings.data.sample_data_dir)
                }
            ),
        }
    )

    labels_ref = ray.put(labels_df) if labels_df is not None else None
    settings_ref = ray.put(abs_settings)

    abs_train_out = str(absolutize_if_local(train_out))
    abs_test_out = str(absolutize_if_local(test_out))

    # Large channels (>150MB compressed Parquet, ~3.5GB peak RSS) get num_cpus=4
    # so Ray monopolizes the entire 4-CPU node — no concurrent task can be
    # scheduled alongside them. max_calls=1 on the decorator ensures the worker
    # process exits after each task, returning all RSS to the OS before the next
    # task starts (Python's allocator does not release memory to the OS on del).
    # Small channels (~950MB peak) pack 4 per node (4×950MB=3.8GB + 0.3GB Ray
    # daemons = 4.1GB, well within the 5.7GB kill threshold on 6Gi workers).
    channel_dir = to_upath(abs_settings.data.sample_data_dir) / mission / "channels"
    _LARGE_THRESHOLD = 150 * 1024 * 1024  # 150MB

    futures = []
    n_large = 0
    for channel in channels:
        try:
            size = (channel_dir / f"{channel}.parquet").stat().st_size
        except Exception:
            size = _LARGE_THRESHOLD + 1   # unknown → conservative/safe default
        is_large = size > _LARGE_THRESHOLD
        n_large += is_large
        task = (
            _preprocess_channel_remote.options(num_cpus=4)
            if is_large
            else _preprocess_channel_remote
        )
        futures.append(task.remote(
            settings_ref, mission, channel,
            abs_train_out, abs_test_out,
            labels_ref,
        ))

    log.info(
        "pipeline.parallel.submitted",
        n_tasks=len(futures),
        n_large=n_large,
        n_small=len(futures) - n_large,
    )
    results: list[dict[str, Any]] = ray.get(futures)
    return results
