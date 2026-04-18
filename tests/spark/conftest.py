"""Shared fixtures for PySpark preprocessing tests.

SparkSession is session-scoped — started once per test run, shared across all
Spark tests. Individual data fixtures are function-scoped so tests stay isolated.

If PySpark cannot start (wrong Java version, missing JDK), all Spark tests are
skipped automatically via the `spark_session` fixture. Install JDK 17 or 21 to
run Spark tests: `brew install openjdk@21`
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# SparkSession — one per test process
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def spark_session():
    """Start a minimal local SparkSession. Skip all Spark tests if JVM fails."""
    try:
        from pyspark.sql import SparkSession

        session = (
            SparkSession.builder.appName("spacecraft-telemetry-test")
            .master("local[1]")
            .config("spark.driver.memory", "512m")
            .config("spark.ui.enabled", "false")
            .config(
                "spark.hadoop.fs.viewfs.overload.scheme.target.file.impl",
                "org.apache.hadoop.fs.LocalFileSystem",
            )
            .getOrCreate()
        )
        session.sparkContext.setLogLevel("ERROR")
        yield session
        session.stop()
    except Exception as exc:
        pytest.skip(f"SparkSession could not start — install JDK 17 or 21. ({exc})")


# ---------------------------------------------------------------------------
# Pandas DataFrames — mimic raw channel Parquet files
# ---------------------------------------------------------------------------
# These match the format written by ingest/sample.py:
#   - DatetimeIndex named "datetime"
#   - One float32 column named after the channel


@pytest.fixture()
def sample_channel_pd() -> pd.DataFrame:
    """100 rows, regular 90-second intervals, no nulls.

    Mirrors channel_1.parquet from the real ESA-Mission1 sample.
    """
    index = pd.date_range(start="2000-01-01", periods=100, freq="90s", name="datetime")
    values = pd.array(
        [float(i % 10) * 0.1 for i in range(100)],
        dtype="float32",
    )
    return pd.DataFrame({"channel_1": values}, index=index)


@pytest.fixture()
def irregular_channel_pd() -> pd.DataFrame:
    """100 rows with irregular intervals: one 20-minute gap at row 50.

    Useful for testing gap detection and segment splitting.
    Rows 0-49: regular 90s. Row 50 jumps ahead by 20 min. Rows 51-99: regular 90s.
    """
    base = pd.Timestamp("2000-01-01")
    timestamps: list[pd.Timestamp] = []
    for i in range(50):
        timestamps.append(base + pd.Timedelta(seconds=90 * i))
    gap_start = timestamps[-1] + pd.Timedelta(minutes=20)
    for i in range(50):
        timestamps.append(gap_start + pd.Timedelta(seconds=90 * i))

    index = pd.DatetimeIndex(timestamps, name="datetime")
    values = pd.array(
        [float(i % 10) * 0.1 for i in range(100)],
        dtype="float32",
    )
    return pd.DataFrame({"channel_2": values}, index=index)


@pytest.fixture()
def channel_with_nulls_pd() -> pd.DataFrame:
    """100 rows, regular 90-second intervals, 5 null values at known positions."""
    index = pd.date_range(start="2000-01-01", periods=100, freq="90s", name="datetime")
    values: list[float | None] = [float(i % 10) * 0.1 for i in range(100)]
    null_positions = [5, 20, 35, 60, 80]
    for pos in null_positions:
        values[pos] = None
    return pd.DataFrame(
        {"channel_3": pd.array(values, dtype="Float32")},  # nullable float
        index=index,
    )


@pytest.fixture()
def labels_pd() -> pd.DataFrame:
    """3 anomaly segments matching the sample_channel_pd time range.

    Segment 1: rows 10-14 (2000-01-01 00:15:00 → 00:21:00)
    Segment 2: rows 40-44 (2000-01-01 01:00:00 → 01:06:00)
    Segment 3: rows 70-74 (2000-01-01 01:45:00 → 01:51:00)
    """
    base = pd.Timestamp("2000-01-01")
    return pd.DataFrame({
        "ID": ["id_1", "id_1", "id_2"],
        "Channel": ["channel_1", "channel_1", "channel_1"],
        "StartTime": [
            (base + pd.Timedelta(seconds=90 * 10)).isoformat() + "Z",
            (base + pd.Timedelta(seconds=90 * 40)).isoformat() + "Z",
            (base + pd.Timedelta(seconds=90 * 70)).isoformat() + "Z",
        ],
        "EndTime": [
            (base + pd.Timedelta(seconds=90 * 14)).isoformat() + "Z",
            (base + pd.Timedelta(seconds=90 * 44)).isoformat() + "Z",
            (base + pd.Timedelta(seconds=90 * 74)).isoformat() + "Z",
        ],
    })


# ---------------------------------------------------------------------------
# On-disk Parquet and CSV fixtures — for io.py tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_channel_parquet(tmp_path: Path, sample_channel_pd: pd.DataFrame) -> Path:
    """Write sample_channel_pd to a Parquet file and return its path."""
    path = tmp_path / "channel_1.parquet"
    sample_channel_pd.to_parquet(path)
    return path


@pytest.fixture()
def irregular_channel_parquet(tmp_path: Path, irregular_channel_pd: pd.DataFrame) -> Path:
    """Write irregular_channel_pd to a Parquet file and return its path."""
    path = tmp_path / "channel_2.parquet"
    irregular_channel_pd.to_parquet(path)
    return path


@pytest.fixture()
def labels_csv(tmp_path: Path, labels_pd: pd.DataFrame) -> Path:
    """Write labels_pd to a CSV file and return its path."""
    path = tmp_path / "labels.csv"
    labels_pd.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Spark DataFrame fixtures — standardised schema for transform tests
# ---------------------------------------------------------------------------
# These represent data AFTER io.py has read and normalised the raw Parquet,
# so they use the standard column names (telemetry_timestamp, value, etc.).


@pytest.fixture()
def sample_spark_df(spark_session, sample_channel_pd: pd.DataFrame):
    """Spark DF in raw-channel schema: telemetry_timestamp, value, channel_id, mission_id."""
    pdf = sample_channel_pd.reset_index()
    pdf = pdf.rename(columns={"datetime": "telemetry_timestamp", "channel_1": "value"})
    pdf["channel_id"] = "channel_1"
    pdf["mission_id"] = "ESA-Mission1"
    pdf["value"] = pdf["value"].astype("float32")
    return spark_session.createDataFrame(pdf)


@pytest.fixture()
def irregular_spark_df(spark_session, irregular_channel_pd: pd.DataFrame):
    """Spark DF with a 20-minute gap — for gap detection tests."""
    pdf = irregular_channel_pd.reset_index()
    pdf = pdf.rename(columns={"datetime": "telemetry_timestamp", "channel_2": "value"})
    pdf["channel_id"] = "channel_2"
    pdf["mission_id"] = "ESA-Mission1"
    pdf["value"] = pdf["value"].astype("float32")
    return spark_session.createDataFrame(pdf)


@pytest.fixture()
def nulls_spark_df(spark_session, channel_with_nulls_pd: pd.DataFrame):
    """Spark DF with 5 null values at known positions — for null handling tests."""
    pdf = channel_with_nulls_pd.reset_index()
    pdf = pdf.rename(columns={"datetime": "telemetry_timestamp", "channel_3": "value"})
    pdf["channel_id"] = "channel_3"
    pdf["mission_id"] = "ESA-Mission1"
    # Float32 with nulls — keep as float64 for Spark (Spark nullable FloatType)
    pdf["value"] = pdf["value"].astype("float64")
    return spark_session.createDataFrame(pdf)
