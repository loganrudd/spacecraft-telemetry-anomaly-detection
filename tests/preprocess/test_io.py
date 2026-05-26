"""Tests for preprocess/io.py — read_channel, read_labels, write_series."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from spacecraft_telemetry.preprocess.io import read_channel, read_labels, write_series
from spacecraft_telemetry.preprocess.schemas import SERIES_FILE_SCHEMA

# ---------------------------------------------------------------------------
# read_channel
# ---------------------------------------------------------------------------


class TestReadChannel:
    def test_returns_standard_columns(self, sample_channel_parquet: Path) -> None:
        df = read_channel(sample_channel_parquet, "channel_1", "ESA-Mission1")
        assert set(df.columns) == {"telemetry_timestamp", "value", "channel_id", "mission_id"}

    def test_channel_id_column_populated(self, sample_channel_parquet: Path) -> None:
        df = read_channel(sample_channel_parquet, "channel_1", "ESA-Mission1")
        assert (df["channel_id"] == "channel_1").all()

    def test_mission_id_column_populated(self, sample_channel_parquet: Path) -> None:
        df = read_channel(sample_channel_parquet, "channel_1", "ESA-Mission1")
        assert (df["mission_id"] == "ESA-Mission1").all()

    def test_value_is_float32(self, sample_channel_parquet: Path) -> None:
        df = read_channel(sample_channel_parquet, "channel_1", "ESA-Mission1")
        assert df["value"].dtype == np.float32

    def test_timestamp_is_utc_aware(self, sample_channel_parquet: Path) -> None:
        df = read_channel(sample_channel_parquet, "channel_1", "ESA-Mission1")
        assert df["telemetry_timestamp"].dt.tz is not None
        assert str(df["telemetry_timestamp"].dt.tz) == "UTC"

    def test_row_count_preserved(self, sample_channel_parquet: Path) -> None:
        df = read_channel(sample_channel_parquet, "channel_1", "ESA-Mission1")
        assert len(df) == 100

    def test_raises_on_missing_channel_column(
        self, sample_channel_parquet: Path
    ) -> None:
        with pytest.raises(ValueError, match="channel_9"):
            read_channel(sample_channel_parquet, "channel_9", "ESA-Mission1")

    def test_raises_on_missing_datetime_column(self, tmp_path: Path) -> None:
        bad_parquet = tmp_path / "bad.parquet"
        pd.DataFrame({"channel_1": [1.0]}).to_parquet(bad_parquet)
        with pytest.raises(ValueError, match="datetime"):
            read_channel(bad_parquet, "channel_1", "ESA-Mission1")


# ---------------------------------------------------------------------------
# read_labels
# ---------------------------------------------------------------------------


class TestReadLabels:
    def test_returns_standard_columns(self, labels_csv: Path) -> None:
        df = read_labels(labels_csv)
        assert set(df.columns) == {"anomaly_id", "channel_id", "start_time", "end_time"}

    def test_row_count(self, labels_csv: Path) -> None:
        df = read_labels(labels_csv)
        assert len(df) == 3

    def test_start_time_is_utc_aware(self, labels_csv: Path) -> None:
        df = read_labels(labels_csv)
        assert str(df["start_time"].dt.tz) == "UTC"

    def test_end_time_is_utc_aware(self, labels_csv: Path) -> None:
        df = read_labels(labels_csv)
        assert str(df["end_time"].dt.tz) == "UTC"

    def test_start_before_end(self, labels_csv: Path) -> None:
        df = read_labels(labels_csv)
        assert (df["start_time"] < df["end_time"]).all()

    def test_subsecond_timestamps_stripped(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "sub.csv"
        csv_path.write_text(
            "ID,Channel,StartTime,EndTime\n"
            "id_1,channel_1,2000-01-01T00:00:00.429Z,2000-01-01T01:00:00.999Z\n"
        )
        df = read_labels(csv_path)
        assert len(df) == 1
        assert df["start_time"].iloc[0] == pd.Timestamp("2000-01-01T00:00:00Z")


# ---------------------------------------------------------------------------
# write_series
# ---------------------------------------------------------------------------


class TestWriteSeries:
    def _make_series_df(self) -> pd.DataFrame:
        ts = pd.date_range("2000-01-01", periods=10, freq="90s", tz="UTC")
        return pd.DataFrame(
            {
                "telemetry_timestamp": ts,
                "value_normalized": np.zeros(10, dtype=np.float32),
                "channel_id": "channel_1",
                "mission_id": "ESA-Mission1",
                "segment_id": np.zeros(10, dtype=np.int32),
                "is_anomaly": [False] * 10,
            }
        )

    def test_creates_partition_directory(self, tmp_path: Path) -> None:
        write_series(self._make_series_df(), tmp_path)
        partition = tmp_path / "mission_id=ESA-Mission1" / "channel_id=channel_1"
        assert partition.is_dir()

    def test_writes_parquet_file(self, tmp_path: Path) -> None:
        write_series(self._make_series_df(), tmp_path)
        partition = tmp_path / "mission_id=ESA-Mission1" / "channel_id=channel_1"
        assert any(partition.glob("*.parquet"))

    def test_output_has_series_file_schema_columns(self, tmp_path: Path) -> None:
        write_series(self._make_series_df(), tmp_path)
        partition = tmp_path / "mission_id=ESA-Mission1" / "channel_id=channel_1"
        # partitioning=None prevents PyArrow 18+ from auto-adding Hive partition columns.
        table = pq.read_table(next(partition.glob("*.parquet")), partitioning=None)
        assert set(table.schema.names) == set(SERIES_FILE_SCHEMA.names)

    def test_partition_columns_not_in_file(self, tmp_path: Path) -> None:
        write_series(self._make_series_df(), tmp_path)
        partition = tmp_path / "mission_id=ESA-Mission1" / "channel_id=channel_1"
        # partitioning=None prevents PyArrow 18+ from auto-adding Hive partition columns.
        table = pq.read_table(next(partition.glob("*.parquet")), partitioning=None)
        assert "channel_id" not in table.schema.names
        assert "mission_id" not in table.schema.names

    def test_row_count_preserved(self, tmp_path: Path) -> None:
        write_series(self._make_series_df(), tmp_path)
        partition = tmp_path / "mission_id=ESA-Mission1" / "channel_id=channel_1"
        table = pq.read_table(next(partition.glob("*.parquet")), partitioning=None)
        assert table.num_rows == 10

    def test_output_sorted_by_timestamp(self, tmp_path: Path) -> None:
        df = self._make_series_df()
        # Shuffle before writing — write_series must sort.
        write_series(df.sample(frac=1, random_state=0).reset_index(drop=True), tmp_path)
        partition = tmp_path / "mission_id=ESA-Mission1" / "channel_id=channel_1"
        result = pq.read_table(next(partition.glob("*.parquet")), partitioning=None).to_pandas()
        ts = result["telemetry_timestamp"]
        assert (ts.diff().iloc[1:] >= pd.Timedelta(0)).all()
