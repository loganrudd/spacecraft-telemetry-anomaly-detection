"""Spark preprocessing pipeline orchestration.

Orchestrates: read → null-fill → gap-detect → normalize → label_timesteps → split:
  Series: temporal_train_test_split → write_series train/ + test/

Processes channels sequentially to respect the M1 RAM constraint (512 m driver).
Windowing is deferred to the PyTorch DataLoader (Plan 002.5).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from spacecraft_telemetry.core.config import Settings
from spacecraft_telemetry.core.logging import get_logger
from spacecraft_telemetry.spark.io import read_channel, read_labels, write_series
from spacecraft_telemetry.spark.transforms import (
    detect_gaps,
    handle_nulls,
    label_timesteps,
    normalize,
    temporal_train_test_split,
)

log = get_logger(__name__)


def run_preprocessing(
    spark: SparkSession,
    settings: Settings,
    mission: str,
) -> dict[str, int]:
    """Run the full preprocessing pipeline for one mission.

    Reads channel Parquet files from {data_dir}/{mission}/channels/ and labels from
    {data_dir}/{mission}/labels.csv (optional). Writes output to {output_dir}/{mission}/:
      - train/     — LSTM training windows, anomalies excluded, partitioned
      - test/      — LSTM evaluation windows (includes labeled anomaly windows), partitioned
      - normalization_params.json — per-channel mean/std for inference-time normalization

    Output directories are cleared before processing so re-runs are idempotent.
    Channels are processed sequentially to respect the M1 RAM constraint.

    Args:
        spark: Active SparkSession.
        settings: Settings object (SparkConfig + DataConfig govern all pipeline parameters).
        mission: Mission name matching a subdirectory in the data dir (e.g. "ESA-Mission1").

    Returns:
        Summary dict: {channels_processed, rows_in, train_rows, test_rows}.

    Raises:
        FileNotFoundError: If no channel Parquet files are found in the channels directory.
    """
    cfg = settings.spark
    data_dir = Path(str(settings.data.sample_data_dir))
    output_dir = Path(str(cfg.processed_data_dir))

    channel_dir = data_dir / mission / "channels"
    labels_path = data_dir / mission / "labels.csv"

    channel_paths = sorted(channel_dir.glob("*.parquet"))
    if not channel_paths:
        raise FileNotFoundError(f"No channel Parquet files found in {channel_dir}")

    # Read labels once — shared across all channels.
    # label_timesteps applies F.broadcast() so Spark ships the small labels
    # table to each executor rather than shuffling the large series DataFrame.
    labels_df = None
    if labels_path.exists():
        labels_df = read_labels(spark, labels_path)

    # Feature definitions restricted to the configured window sizes.
    # (Kept for reference by Evidently monitoring — not used in this pipeline.)

    # Clear output dirs: re-runs must not accumulate duplicates across channels.
    train_out = output_dir / mission / "train"
    test_out = output_dir / mission / "test"
    for out_dir in (train_out, test_out):
        if out_dir.exists():
            shutil.rmtree(out_dir)

    normalization_params: dict[str, dict[str, float]] = {}
    total_rows_in = 0
    total_train_rows = 0
    total_test_rows = 0

    for channel_path in channel_paths:
        channel_id = channel_path.stem
        log.info("pipeline.channel.start", channel_id=channel_id, mission=mission)

        # Step 1-4: read → clean → gap detect → normalize
        raw_df = read_channel(spark, channel_path, channel_id, mission)
        cleaned = handle_nulls(raw_df)
        gapped = detect_gaps(cleaned, gap_multiplier=cfg.gap_multiplier)
        normalized, params = normalize(gapped, method=cfg.normalization)

        # Cache the normalized result — series branch builds from here.
        # count() forces materialisation into the cache so subsequent reads are fast.
        normalized = normalized.cache()
        n_rows = normalized.count()
        total_rows_in += n_rows

        # Attach per-timestep anomaly flags before branching.
        if labels_df is not None:
            labeled = label_timesteps(normalized, labels_df)
        else:
            # No labels file: mark everything nominal so the schema stays consistent.
            labeled = normalized.withColumn("is_anomaly", F.lit(False))

        # Series branch: temporal split → write per-timestep rows.
        # Anomaly exclusion is a DataLoader concern (skip_anomalous_windows flag),
        # so the full labeled series (including anomalous rows) is written to train/.
        series_cols = [
            "telemetry_timestamp", "value_normalized",
            "channel_id", "mission_id", "segment_id", "is_anomaly",
        ]
        train_series, test_series = temporal_train_test_split(
            labeled.select(*series_cols), train_fraction=cfg.train_fraction
        )

        train_count = train_series.count()
        test_count = test_series.count()
        total_train_rows += train_count
        total_test_rows += test_count

        if train_count == 0:
            # Channel produced no training rows — no model will be trained.
            # Omit from normalization_params so Phase 8 ignores this channel.
            log.warning(
                "pipeline.channel.no_train_rows",
                channel_id=channel_id,
                mission=mission,
                test_rows=test_count,
            )
        else:
            normalization_params.update(params)

        write_series(train_series, train_out, mode="append")
        write_series(test_series, test_out, mode="append")

        normalized.unpersist()
        log.info(
            "pipeline.channel.done",
            channel_id=channel_id,
            rows=n_rows,
            train_rows=train_count,
            test_rows=test_count,
        )

    # Persist normalization params — required at inference time (Phase 8) to apply
    # the identical z-score transform to incoming telemetry. Only channels that
    # produced training windows are included (others have no trained model).
    params_path = output_dir / mission / "normalization_params.json"
    params_path.parent.mkdir(parents=True, exist_ok=True)
    params_path.write_text(json.dumps(normalization_params, indent=2))

    summary: dict[str, int] = {
        "channels_processed": len(channel_paths),
        "rows_in": total_rows_in,
        "train_rows": total_train_rows,
        "test_rows": total_test_rows,
    }
    log.info("pipeline.complete", mission=mission, **summary)
    return summary
