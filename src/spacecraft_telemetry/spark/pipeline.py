"""Spark preprocessing pipeline orchestration.

Orchestrates: read → null-fill → gap-detect → normalize → branch:
  Features: add_rolling_features → write features/
  Windows:  create_windows → temporal_split → label_join → exclude_anomalies → write train/ + test/

Processes channels sequentially to respect the M1 RAM constraint (512 m driver).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from spacecraft_telemetry.core.config import Settings
from spacecraft_telemetry.core.logging import get_logger
from spacecraft_telemetry.features.definitions import FEATURE_DEFINITIONS, FeatureDefinition
from spacecraft_telemetry.spark.io import read_channel, read_labels, write_features, write_windows
from spacecraft_telemetry.spark.transforms import (
    add_rolling_features,
    create_windows,
    detect_gaps,
    exclude_anomalies_from_train,
    handle_nulls,
    join_anomaly_labels,
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
      - features/  — one row per timestamp, partitioned by mission_id + channel_id
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
        Summary dict: {channels_processed, rows_in, windows_out, feature_rows_out,
        train_windows, test_windows}.

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

    # Read labels once — shared across all channels, cached to avoid re-reading per channel.
    labels_df = None
    if labels_path.exists():
        labels_df = read_labels(spark, labels_path).cache()

    # Feature definitions restricted to the configured window sizes.
    feature_defs: list[FeatureDefinition] = [
        fd
        for fd in FEATURE_DEFINITIONS
        if fd.name == "rate_of_change" or fd.window_size in cfg.feature_windows
    ]
    feature_output_cols = [
        "telemetry_timestamp",
        "channel_id",
        "mission_id",
        "value_normalized",
    ] + [fd.name for fd in feature_defs]

    # Clear output dirs: re-runs must not accumulate duplicates across channels.
    features_out = output_dir / mission / "features"
    train_out = output_dir / mission / "train"
    test_out = output_dir / mission / "test"
    for out_dir in (features_out, train_out, test_out):
        if out_dir.exists():
            shutil.rmtree(out_dir)

    normalization_params: dict[str, dict[str, float]] = {}
    total_rows_in = 0

    for channel_path in channel_paths:
        channel_id = channel_path.stem
        log.info("pipeline.channel.start", channel_id=channel_id, mission=mission)

        # Step 1-4: read → clean → gap detect → normalize
        raw_df = read_channel(spark, channel_path, channel_id, mission)
        cleaned = handle_nulls(raw_df)
        gapped = detect_gaps(cleaned, gap_multiplier=cfg.gap_multiplier)
        normalized, params = normalize(gapped, method=cfg.normalization)
        normalization_params.update(params)

        # Cache the normalized result — both branches build from here.
        # count() forces materialisation into the cache so subsequent reads are fast.
        normalized = normalized.cache()
        n_rows = normalized.count()
        total_rows_in += n_rows

        # Features branch: rolling stats → write for Feast (Phase 3)
        features_df = add_rolling_features(normalized, feature_defs=feature_defs)
        cols_to_write = [c for c in feature_output_cols if c in features_df.columns]
        write_features(features_df.select(*cols_to_write), features_out, mode="append")

        # Windows branch: split first so no window straddles the train/test boundary.
        train_norm, test_norm = temporal_train_test_split(
            normalized, train_fraction=cfg.train_fraction
        )
        train_w = create_windows(
            train_norm,
            window_size=cfg.window_size,
            prediction_horizon=cfg.prediction_horizon,
        )
        test_w = create_windows(
            test_norm,
            window_size=cfg.window_size,
            prediction_horizon=cfg.prediction_horizon,
        )

        if labels_df is not None:
            train_w = join_anomaly_labels(train_w, labels_df)
            test_w = join_anomaly_labels(test_w, labels_df)
        else:
            # No labels file: mark everything nominal so the schema stays consistent.
            train_w = train_w.withColumn("is_anomaly", F.lit(False))
            test_w = test_w.withColumn("is_anomaly", F.lit(False))

        # Telemanom trains on nominal data only.
        clean_train = exclude_anomalies_from_train(train_w)

        write_windows(clean_train, train_out, mode="append")
        write_windows(test_w, test_out, mode="append")

        normalized.unpersist()
        log.info("pipeline.channel.done", channel_id=channel_id, rows=n_rows)

    if labels_df is not None:
        labels_df.unpersist()

    # Persist normalization params — required at inference time (Phase 9) to apply
    # the identical z-score transform to incoming telemetry.
    params_path = output_dir / mission / "normalization_params.json"
    params_path.parent.mkdir(parents=True, exist_ok=True)
    params_path.write_text(json.dumps(normalization_params, indent=2))

    # Count final outputs by reading from the written Parquets (one scan each).
    def _count_parquet(path: Path) -> int:
        return spark.read.parquet(str(path)).count() if path.exists() else 0

    feature_rows_out = _count_parquet(features_out)
    train_windows = _count_parquet(train_out)
    test_windows = _count_parquet(test_out)

    summary: dict[str, int] = {
        "channels_processed": len(channel_paths),
        "rows_in": total_rows_in,
        "windows_out": train_windows + test_windows,
        "feature_rows_out": feature_rows_out,
        "train_windows": train_windows,
        "test_windows": test_windows,
    }
    log.info("pipeline.complete", mission=mission, **summary)
    return summary
