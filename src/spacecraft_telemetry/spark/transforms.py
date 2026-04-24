"""PySpark preprocessing transforms: null handling, gap detection, normalization,
feature engineering.

Each function is a pure DataFrame → DataFrame transform (or DataFrame → (DataFrame, dict)).
Composable and stateless — no SparkSession dependency.
"""

from __future__ import annotations

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.window import WindowSpec
from pyspark.sql.types import FloatType

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


def handle_nulls(df: DataFrame, strategy: str = "forward_fill") -> DataFrame:
    """Forward-fill null values within each channel, dropping any leading nulls.

    Nulls are filled using the last non-null value in timestamp order within the
    same channel. Rows with no prior non-null (leading nulls) are dropped entirely
    because there is no valid value to carry forward.

    Args:
        df: DataFrame with 'value', 'channel_id', 'telemetry_timestamp' columns.
        strategy: Currently only "forward_fill" is supported.

    Returns:
        DataFrame with no null values in the 'value' column.

    Raises:
        ValueError: If strategy is not "forward_fill".
    """
    if strategy != "forward_fill":
        raise ValueError(
            f"Unsupported null-handling strategy {strategy!r}. "
            "Only 'forward_fill' is supported."
        )

    null_count = df.filter(F.col("value").isNull()).count()
    if null_count == 0:
        log.info("handle_nulls", null_count=0, rows_dropped=0)
        return df

    w = (
        Window.partitionBy("channel_id")
        .orderBy("telemetry_timestamp")
        .rowsBetween(Window.unboundedPreceding, 0)
    )
    df = df.withColumn("value", F.last("value", ignorenulls=True).over(w))

    rows_before = df.count()
    df = df.filter(F.col("value").isNotNull())
    rows_dropped = rows_before - df.count()

    log.info(
        "handle_nulls",
        nulls_filled=null_count - rows_dropped,
        rows_dropped=rows_dropped,
    )
    return df


def detect_gaps(df: DataFrame, gap_multiplier: float = 3.0) -> DataFrame:
    """Detect time gaps and assign contiguous segment IDs.

    A gap is an interval between consecutive rows exceeding gap_multiplier ×
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

    df = df.withColumn(
        "_prev_ts", F.lag("telemetry_timestamp", 1).over(w_ts)
    ).withColumn(
        "_interval_s",
        (F.unix_timestamp("telemetry_timestamp") - F.unix_timestamp("_prev_ts")).cast(
            "double"
        ),
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

    gap_count = df.filter(F.col("is_gap")).count()
    log.info("detect_gaps", gap_count=gap_count, gap_multiplier=gap_multiplier)

    return df.drop("_prev_ts", "_interval_s", "_median_interval")


def normalize(
    df: DataFrame,
    method: str = "z-score",
) -> tuple[DataFrame, dict[str, dict[str, float]]]:
    """Add a value_normalized column using per-channel z-score normalization.

    Computes mean and std per channel from the DataFrame, then adds
    value_normalized = (value - mean) / std. Channels with std=0 (constant
    signal) are normalized to 0.0 rather than NaN.

    The returned params dict must be persisted by the pipeline (Step 8) as
    normalization_params.json — these values are required at inference time
    (Phase 9) to apply the identical transform to incoming telemetry.

    Args:
        df: DataFrame with 'value' and 'channel_id' columns.
        method: Currently only "z-score" is supported.

    Returns:
        (normalized_df, params) where params = {channel_id: {"mean": ..., "std": ...}}.

    Raises:
        ValueError: If method is not "z-score".
    """
    if method != "z-score":
        raise ValueError(
            f"Unsupported normalization method {method!r}. Only 'z-score' is supported."
        )

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
    return (
        F.when(F.col("_rn") < n, F.lit(None).cast(FloatType()))
        .otherwise(agg.cast(FloatType()))
    )


def _rate_of_change_col(w_base: WindowSpec) -> Column:
    """Spark Column for the first derivative: Δvalue / Δtime (seconds).

    Returns null on the first row of each partition (no previous value) and
    when consecutive timestamps are identical (division by zero).
    """
    prev_val = F.lag("value_normalized", 1).over(w_base)
    prev_ts_unix = F.lag(F.unix_timestamp("telemetry_timestamp"), 1).over(w_base)
    dt = F.unix_timestamp("telemetry_timestamp").cast("double") - prev_ts_unix.cast("double")
    return (
        F.when(prev_val.isNull() | (dt == 0.0), F.lit(None).cast(FloatType()))
        .otherwise(((F.col("value_normalized") - prev_val) / dt).cast(FloatType()))
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
    w_base = (
        Window.partitionBy("channel_id", "segment_id")
        .orderBy("telemetry_timestamp")
    )

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
