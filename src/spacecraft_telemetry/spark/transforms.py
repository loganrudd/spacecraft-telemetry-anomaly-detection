"""PySpark preprocessing transforms: null handling, gap detection, normalization,
feature engineering.

Each function is a pure DataFrame → DataFrame transform (or DataFrame → (DataFrame, dict)).
Composable and stateless — no SparkSession dependency.
"""

from __future__ import annotations

from typing import Literal

from pyspark.sql import Column, DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, FloatType
from pyspark.sql.window import WindowSpec

from spacecraft_telemetry.core.logging import get_logger
from spacecraft_telemetry.features.definitions import (
    FEATURE_DEFINITIONS,
    FeatureDefinition,
)

log = get_logger(__name__)

# Maps the operation segment of a rolling feature name to its Spark aggregation function.
# Keyed by the middle token of "rolling_{op}_{n}" names (e.g. "mean", "std").
_ROLLING_AGG = {
    "mean": F.avg,
    "std": F.stddev,
    "min": F.min,
    "max": F.max,
}


def handle_nulls(
    df: DataFrame, strategy: Literal["forward_fill"] = "forward_fill"
) -> DataFrame:
    """Forward-fill null values within each channel, dropping any leading nulls.

    Nulls are filled using the last non-null value in timestamp order within the
    same channel. Rows with no prior non-null (leading nulls) are dropped entirely
    because there is no valid value to carry forward.

    Args:
        df: DataFrame with 'value', 'channel_id', 'telemetry_timestamp' columns.
        strategy: Currently only "forward_fill" is supported.

    Returns:
        DataFrame with no null values in the 'value' column.
    """
    # Cheap check: stop at the first null row rather than counting all of them.
    if df.filter(F.col("value").isNull()).rdd.isEmpty():
        log.info("handle_nulls.skipped", strategy=strategy)
        return df

    w = (
        Window.partitionBy("channel_id")
        .orderBy("telemetry_timestamp")
        .rowsBetween(Window.unboundedPreceding, 0)
    )
    df = df.withColumn("value", F.last("value", ignorenulls=True).over(w))
    df = df.filter(F.col("value").isNotNull())

    if df.rdd.isEmpty():
        log.warning("handle_nulls.all_rows_dropped", strategy=strategy)
    else:
        log.info("handle_nulls", strategy=strategy)

    return df


def detect_gaps(df: DataFrame, gap_multiplier: float = 3.0) -> DataFrame:
    """Detect time gaps and assign contiguous segment IDs.

    A gap is an interval between consecutive rows exceeding gap_multiplier *
    the median sampling interval for that channel. The first row of each channel
    is never a gap.

    Adds two columns:
    - is_gap (BooleanType): True on the first row after a gap.
    - segment_id (IntegerType): 0-based, increments at each gap boundary.

    Args:
        df: DataFrame with 'value', 'channel_id', 'telemetry_timestamp' columns.
        gap_multiplier: Factor above median interval that triggers a gap flag.

    Returns:
        DataFrame with added 'is_gap' and 'segment_id' columns.
    """
    w_ts = Window.partitionBy("channel_id").orderBy("telemetry_timestamp")

    df = df.withColumn("_prev_ts", F.lag("telemetry_timestamp", 1).over(w_ts)).withColumn(
        "_interval_s",
        (F.unix_timestamp("telemetry_timestamp") - F.unix_timestamp("_prev_ts")).cast("double"),
    )

    median_df = (
        df.filter(F.col("_interval_s").isNotNull())
        .groupBy("channel_id")
        .agg(F.percentile_approx("_interval_s", 0.5).alias("_median_interval"))
    )
    df = df.join(median_df, on="channel_id", how="left")

    df = df.withColumn(
        "is_gap",
        F.when(F.col("_prev_ts").isNull(), F.lit(False))
        .when(
            F.col("_interval_s") > gap_multiplier * F.col("_median_interval"),
            F.lit(True),
        )
        .otherwise(F.lit(False)),
    )

    w_cumsum = (
        Window.partitionBy("channel_id")
        .orderBy("telemetry_timestamp")
        .rowsBetween(Window.unboundedPreceding, 0)
    )
    df = df.withColumn(
        "segment_id",
        F.sum(F.col("is_gap").cast("int")).over(w_cumsum).cast("int"),
    )

    log.info("detect_gaps", gap_multiplier=gap_multiplier)
    return df.drop("_prev_ts", "_interval_s", "_median_interval")


def normalize(
    df: DataFrame,
    method: Literal["z-score"] = "z-score",
) -> tuple[DataFrame, dict[str, dict[str, float]]]:
    """Add a value_normalized column using per-channel z-score normalization.

    Computes mean and std per channel from the DataFrame, then adds
    value_normalized = (value - mean) / std. Channels with std=0 (constant
    signal) are normalized to 0.0 rather than NaN.

    The formula matches features.definitions.normalize_value() exactly — that
    function is the reference implementation tested for train-serve equivalence.
    Phase 9 (FastAPI) calls normalize_value() directly on incoming telemetry.

    The returned params dict must be persisted by the pipeline (Step 8) as
    normalization_params.json — these values are required at inference time
    (Phase 9) to apply the identical transform to incoming telemetry.

    Args:
        df: DataFrame with 'value' and 'channel_id' columns.
        method: Normalization method — currently only "z-score" is supported.

    Returns:
        (normalized_df, params) where params = {channel_id: {"mean": ..., "std": ...}}.
    """
    stats_df = df.groupBy("channel_id").agg(
        F.mean("value").alias("_mean"),
        F.stddev("value").alias("_std"),
    )

    params: dict[str, dict[str, float]] = {
        row["channel_id"]: {
            "mean": float(row["_mean"]),
            "std": float(row["_std"]) if row["_std"] is not None else 0.0,
        }
        for row in stats_df.collect()
    }

    df = df.join(stats_df, on="channel_id", how="left")
    df = df.withColumn(
        "value_normalized",
        F.when(
            F.col("_std").isNull() | (F.col("_std") == 0.0),
            F.lit(0.0),
        )
        .otherwise((F.col("value") - F.col("_mean")) / F.col("_std"))
        .cast(FloatType()),
    )

    log.info("normalize", method=method, channels=list(params.keys()))

    return df.drop("_mean", "_std"), params


# ---------------------------------------------------------------------------
# Private helpers for add_rolling_features
# ---------------------------------------------------------------------------


def _rolling_col(fd: FeatureDefinition, w_base: WindowSpec) -> Column:
    """Spark Column for a rolling-window statistic.

    Returns null for the first window_size-1 rows of each partition so that
    behaviour matches the numpy reference implementation (which also returns
    NaN when the buffer is shorter than the requested window).
    """
    n = fd.window_size
    op = fd.name.split("_")[1]  # "mean" | "std" | "min" | "max"
    agg_fn = _ROLLING_AGG[op]
    w = w_base.rowsBetween(-(n - 1), 0)
    agg = agg_fn("value_normalized").over(w)
    return F.when(F.col("_rn") < n, F.lit(None).cast(FloatType())).otherwise(agg.cast(FloatType()))


def _rate_of_change_col(w_base: WindowSpec) -> Column:
    """Spark Column for the first derivative: Δvalue / Δtime (seconds).

    Returns null on the first row of each partition (no previous value) and
    when consecutive timestamps are identical (division by zero).
    """
    prev_val = F.lag("value_normalized", 1).over(w_base)
    prev_ts_unix = F.lag(F.unix_timestamp("telemetry_timestamp"), 1).over(w_base)
    dt = F.unix_timestamp("telemetry_timestamp").cast("double") - prev_ts_unix.cast("double")
    return F.when(prev_val.isNull() | (dt == 0.0), F.lit(None).cast(FloatType())).otherwise(
        ((F.col("value_normalized") - prev_val) / dt).cast(FloatType())
    )


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def add_rolling_features(
    df: DataFrame,
    feature_defs: list[FeatureDefinition] | None = None,
) -> DataFrame:
    """Add rolling feature columns derived from value_normalized.

    Iterates FEATURE_DEFINITIONS (or a caller-supplied subset) and adds one
    Spark column per feature. Windows are partitioned by (channel_id, segment_id)
    so they never cross gap boundaries detected by detect_gaps().

    The rolling statistics match the numpy reference in features.definitions:
    - Rows with fewer than window_size predecessors within the segment → null
      (same as the numpy reference returning NaN for a short buffer)
    - rate_of_change → null for the first row and when Δtime == 0

    These columns are written to data/processed/{mission}/features/ and
    consumed by Phase 3 (Feast offline store) and Phase 8 (Evidently drift).

    Args:
        df: DataFrame with columns: telemetry_timestamp, value_normalized,
            channel_id, mission_id, segment_id.
        feature_defs: Feature registry to iterate. Defaults to FEATURE_DEFINITIONS.
            Pass a subset for testing or targeted pipelines.

    Returns:
        DataFrame with one new FloatType column per FeatureDefinition.
    """
    if feature_defs is None:
        feature_defs = FEATURE_DEFINITIONS

    # Base window: partition by channel + segment, order by time.
    # segment_id prevents rolling windows from crossing gap boundaries.
    w_base = Window.partitionBy("channel_id", "segment_id").orderBy("telemetry_timestamp")

    # Row number within each (channel, segment) partition.
    # Used by _rolling_col to null-out the first window_size-1 rows.
    df = df.withColumn("_rn", F.row_number().over(w_base))

    for fd in feature_defs:
        col = (
            _rate_of_change_col(w_base)
            if fd.name == "rate_of_change"
            else _rolling_col(fd, w_base)
        )
        df = df.withColumn(fd.name, col)

    log.info("add_rolling_features", features=[fd.name for fd in feature_defs])
    return df.drop("_rn")


# ---------------------------------------------------------------------------
# Windowing for LSTM input
# ---------------------------------------------------------------------------


def create_windows(
    df: DataFrame,
    window_size: int = 250,
    prediction_horizon: int = 1,
) -> DataFrame:
    """Create sliding-window sequences for Telemanom LSTM input.

    Each output row is one training or evaluation example:
      values         — Array<Float> of window_size consecutive value_normalized
                       readings in ascending timestamp order (oldest first)
      target         — value_normalized at window_end + prediction_horizon steps
      window_start_ts / window_end_ts — timestamps bracketing the input window

    Windows are strictly contained within a single (channel_id, segment_id)
    partition, so they never span gap boundaries from detect_gaps().

    Segments with fewer than (window_size + prediction_horizon) rows produce
    zero windows and are silently excluded (count logged at INFO).

    Number of windows per segment of length N:
        max(0, N - window_size - prediction_horizon + 1)

    Args:
        df: DataFrame with columns: telemetry_timestamp, value_normalized,
            channel_id, mission_id, segment_id.
        window_size: Length of each input sequence (Telemanom default: 250).
        prediction_horizon: Steps ahead to forecast (default: 1 = next reading).

    Returns:
        DataFrame with columns: window_id (LongType), channel_id, mission_id,
        segment_id, window_start_ts (TimestampType), window_end_ts (TimestampType),
        values (Array<Float>, length == window_size), target (FloatType).
        is_anomaly is added separately in join_anomaly_labels().
    """
    w = Window.partitionBy("channel_id", "segment_id").orderBy("telemetry_timestamp")
    vals_w = w.rowsBetween(-(window_size - 1), 0)

    df = (
        df.withColumn("values", F.collect_list("value_normalized").over(vals_w))
        .withColumn("window_end_ts", F.col("telemetry_timestamp"))
        .withColumn("window_start_ts", F.min("telemetry_timestamp").over(vals_w))
        .withColumn("target", F.lead("value_normalized", prediction_horizon).over(w))
    )

    # Drop rows where the trailing window is shorter than window_size (early rows
    # in a segment) or where no target exists (final prediction_horizon rows).
    df = df.filter(F.size("values") == window_size).filter(F.col("target").isNotNull())

    df = df.withColumn("window_id", F.monotonically_increasing_id())

    windows_count = df.count()
    log.info(
        "create_windows",
        window_size=window_size,
        prediction_horizon=prediction_horizon,
        windows_count=windows_count,
    )

    return df.select(
        "window_id",
        "channel_id",
        "mission_id",
        "segment_id",
        "window_start_ts",
        "window_end_ts",
        F.col("values").cast(ArrayType(FloatType())).alias("values"),
        F.col("target").cast(FloatType()).alias("target"),
    )


# ---------------------------------------------------------------------------
# Train/test split and anomaly label join
# ---------------------------------------------------------------------------


def temporal_train_test_split(
    df: DataFrame,
    train_fraction: float = 0.8,
) -> tuple[DataFrame, DataFrame]:
    """Split a DataFrame into train and test sets using a per-channel timestamp cutoff.

    The cutoff for each channel is:
        min_ts + train_fraction * (max_ts - min_ts)

    Rows at or before the cutoff → train; rows after → test. This is a temporal
    (non-random) split that preserves ordering: the model trains on the earlier
    portion of each channel's history and evaluates on the later portion.

    Args:
        df: DataFrame with 'channel_id' and 'telemetry_timestamp' columns.
        train_fraction: Fraction of the time range assigned to training (default 0.8).

    Returns:
        (train_df, test_df)
    """
    w = Window.partitionBy("channel_id")
    df = (
        df.withColumn("_min_ts", F.min(F.unix_timestamp("telemetry_timestamp")).over(w))
        .withColumn("_max_ts", F.max(F.unix_timestamp("telemetry_timestamp")).over(w))
        .withColumn(
            "_cutoff_ts",
            F.col("_min_ts") + train_fraction * (F.col("_max_ts") - F.col("_min_ts")),
        )
    )

    train = df.filter(F.unix_timestamp("telemetry_timestamp") <= F.col("_cutoff_ts")).drop(
        "_min_ts", "_max_ts", "_cutoff_ts"
    )

    test = df.filter(F.unix_timestamp("telemetry_timestamp") > F.col("_cutoff_ts")).drop(
        "_min_ts", "_max_ts", "_cutoff_ts"
    )

    log.info("temporal_train_test_split", train_fraction=train_fraction)
    return train, test


def join_anomaly_labels(df: DataFrame, labels_df: DataFrame) -> DataFrame:
    """Add is_anomaly boolean column by joining window time ranges against anomaly labels.

    A window is anomalous if its [window_start_ts, window_end_ts) overlaps any label
    interval [start_time, end_time) for the same channel_id. Channels absent from
    labels_df are treated as fully nominal (all False).

    Args:
        df: Windows DataFrame with 'window_id', 'channel_id', 'window_start_ts',
            'window_end_ts' columns.
        labels_df: Labels DataFrame with 'channel_id', 'start_time', 'end_time'.

    Returns:
        DataFrame with 'is_anomaly' (BooleanType) column added.
    """
    win = df.alias("win")
    lbl = labels_df.alias("lbl")

    # Left join: one row per (window, overlapping label) pair.
    # Overlap condition: window_end > label_start AND window_start < label_end.
    matched = win.join(
        lbl,
        on=(
            (F.col("win.channel_id") == F.col("lbl.channel_id"))
            & (F.col("win.window_end_ts") > F.col("lbl.start_time"))
            & (F.col("win.window_start_ts") < F.col("lbl.end_time"))
        ),
        how="left",
    ).select(
        F.col("win.window_id"),
        F.col("lbl.start_time").isNotNull().alias("_matched"),
    )

    # Collapse: any match → is_anomaly=True for that window_id.
    flags = matched.groupBy("window_id").agg(F.max("_matched").cast("boolean").alias("is_anomaly"))

    result = df.join(flags, on="window_id", how="left").withColumn(
        "is_anomaly",
        F.coalesce(F.col("is_anomaly"), F.lit(False)),
    )

    anomaly_count = result.filter(F.col("is_anomaly")).count()
    log.info("join_anomaly_labels", anomaly_count=anomaly_count)
    return result


def exclude_anomalies_from_train(train_df: DataFrame) -> DataFrame:
    """Remove anomalous windows from the training set.

    Telemanom trains on nominal data only — it learns the normal pattern and flags
    deviations at inference time. Anomaly windows are kept in the test set for
    evaluation.

    Args:
        train_df: Windows DataFrame with 'is_anomaly' (BooleanType) column.

    Returns:
        DataFrame with all is_anomaly=True rows removed.
    """
    total = train_df.count()
    result = train_df.filter(~F.col("is_anomaly"))
    excluded = total - result.count()
    log.info("exclude_anomalies_from_train", excluded=excluded)
    return result
