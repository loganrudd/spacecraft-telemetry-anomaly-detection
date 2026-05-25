"""Shared fixtures for pandas preprocessing tests.

All fixtures are pure pandas/PyArrow — no SparkSession dependency.
The pipeline_input_dir layout mirrors what ingest/sample.py produces:
    {root}/ESA-Mission1/channels/channel_1.parquet
    {root}/ESA-Mission1/labels.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_parquet_micros(df: pd.DataFrame, path: Path) -> None:
    """Write a pandas DataFrame to Parquet with microsecond UTC timestamps.

    Ensures the DatetimeIndex is UTC-aware and microsecond precision so PyArrow
    writes TIMESTAMP(MICROS, true) — matching what the pipeline reads back.
    """
    df = df.copy()
    if isinstance(df.index, pd.DatetimeIndex):
        idx = df.index
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        df.index = idx.as_unit("us")
    df.to_parquet(path)


# ---------------------------------------------------------------------------
# Base pandas DataFrames — raw channel format (DatetimeIndex + float32 column)
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_channel_pd() -> pd.DataFrame:
    """100 rows, regular 90-second intervals, no nulls."""
    index = pd.date_range(start="2000-01-01", periods=100, freq="90s", name="datetime")
    values = pd.array(
        [float(i % 10) * 0.1 for i in range(100)],
        dtype="float32",
    )
    return pd.DataFrame({"channel_1": values}, index=index)


@pytest.fixture()
def irregular_channel_pd() -> pd.DataFrame:
    """100 rows with a 20-minute gap at row 50 — for gap detection tests."""
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
    for pos in [5, 20, 35, 60, 80]:
        values[pos] = None
    return pd.DataFrame(
        {"channel_3": pd.array(values, dtype="Float32")},
        index=index,
    )


@pytest.fixture()
def labels_pd() -> pd.DataFrame:
    """3 anomaly segments covering rows 10-14, 40-44, 70-74 of sample_channel_pd."""
    base = pd.Timestamp("2000-01-01")
    return pd.DataFrame(
        {
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
        }
    )


# ---------------------------------------------------------------------------
# On-disk fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_channel_parquet(tmp_path: Path, sample_channel_pd: pd.DataFrame) -> Path:
    """Write sample_channel_pd to a Parquet file and return its path."""
    path = tmp_path / "channel_1.parquet"
    _write_parquet_micros(sample_channel_pd, path)
    return path


@pytest.fixture()
def irregular_channel_parquet(tmp_path: Path, irregular_channel_pd: pd.DataFrame) -> Path:
    """Write irregular_channel_pd to a Parquet file and return its path."""
    path = tmp_path / "channel_2.parquet"
    _write_parquet_micros(irregular_channel_pd, path)
    return path


@pytest.fixture()
def labels_csv(tmp_path: Path, labels_pd: pd.DataFrame) -> Path:
    """Write labels_pd to a CSV file and return its path."""
    path = tmp_path / "labels.csv"
    labels_pd.to_csv(path, index=False)
    return path


@pytest.fixture()
def pipeline_input_dir(
    tmp_path: Path, sample_channel_pd: pd.DataFrame, labels_pd: pd.DataFrame
) -> Path:
    """Build the directory structure expected by run_preprocessing.

    Layout:
        {tmp_path}/input/ESA-Mission1/channels/channel_1.parquet
        {tmp_path}/input/ESA-Mission1/labels.csv

    Returns the root input directory ({tmp_path}/input).
    """
    mission = "ESA-Mission1"
    channels_dir = tmp_path / "input" / mission / "channels"
    channels_dir.mkdir(parents=True)
    _write_parquet_micros(sample_channel_pd, channels_dir / "channel_1.parquet")
    labels_pd.to_csv(tmp_path / "input" / mission / "labels.csv", index=False)
    return tmp_path / "input"


@pytest.fixture()
def settings(pipeline_input_dir: Path, tmp_path: Path):
    """Settings pointing at the synthetic input + a fresh output directory."""
    from spacecraft_telemetry.core.config import DataConfig, PreprocessingConfig, Settings

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return Settings(
        data=DataConfig(sample_data_dir=pipeline_input_dir),
        preprocess=PreprocessingConfig(
            processed_data_dir=output_dir,
            train_fraction=0.8,
            feature_windows=[10],
        ),
    )
