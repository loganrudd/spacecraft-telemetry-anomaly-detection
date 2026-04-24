"""PySpark preprocessing transforms: null handling, gap detection, normalization.

Each function is a pure DataFrame → DataFrame transform (or DataFrame → (DataFrame, dict)).
Composable and stateless — no SparkSession dependency.
"""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.types import FloatType

from spacecraft_telemetry.core.logging import get_logger

log = get_logger(__name__)


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
