"""Tests for ingest.explore."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from rich.console import Console

from spacecraft_telemetry.ingest.explore import (
    DataExplorer,
    _detect_time_column,
    _estimate_interval_s,
    _first_match,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _write_channel(
    sample_dir: Path,
    mission: str,
    channel: str,
    n_rows: int = 100,
    freq: str = "1s",
    include_timestamp: bool = True,
    nulls: bool = False,
) -> Path:
    rng = np.random.default_rng(0)
    data: dict = {"value": rng.random(n_rows)}
    if include_timestamp:
        data["timestamp"] = pd.date_range("2020-01-01", periods=n_rows, freq=freq)
    if nulls:
        data["value"][::10] = float("nan")
    df = pd.DataFrame(data)
    path = sample_dir / mission / "channels" / f"{channel}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def _write_labels(sample_dir: Path, mission: str, rows: list[dict]) -> Path:
    path = sample_dir / mission / "labels.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _capture_console() -> Console:
    """Return a rich Console that writes to a StringIO buffer."""
    return Console(file=StringIO(), width=120)


# ---------------------------------------------------------------------------
# _detect_time_column
# ---------------------------------------------------------------------------


class TestDetectTimeColumn:
    def test_detects_datetime64_column(self) -> None:
        df = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=3), "v": [1, 2, 3]})
        assert _detect_time_column(df) == "timestamp"

    def test_detects_by_name_when_dtype_is_object(self) -> None:
        df = pd.DataFrame({"time": ["2020-01-01", "2020-01-02"], "v": [1, 2]})
        assert _detect_time_column(df) == "time"

    def test_returns_none_when_no_time_column(self) -> None:
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        assert _detect_time_column(df) is None

    def test_prefers_datetime64_over_name_match(self) -> None:
        # Both columns match by name, but 'ts' has actual datetime dtype
        df = pd.DataFrame(
            {
                "time": ["2020-01-01", "2020-01-02"],
                "ts": pd.date_range("2020-01-01", periods=2),
            }
        )
        assert _detect_time_column(df) == "ts"


# ---------------------------------------------------------------------------
# _estimate_interval_s
# ---------------------------------------------------------------------------


class TestEstimateIntervalS:
    def test_one_second_interval(self) -> None:
        df = pd.DataFrame({"ts": pd.date_range("2020-01-01", periods=10, freq="1s")})
        assert _estimate_interval_s(df, "ts") == pytest.approx(1.0)

    def test_returns_none_for_single_row(self) -> None:
        df = pd.DataFrame({"ts": pd.date_range("2020-01-01", periods=1)})
        assert _estimate_interval_s(df, "ts") is None

    def test_handles_irregular_spacing(self) -> None:
        # Median of [1s, 1s, 10s, 1s, 1s] = 1s
        times = pd.to_datetime(
            [
                "2020-01-01 00:00:00",
                "2020-01-01 00:00:01",
                "2020-01-01 00:00:02",
                "2020-01-01 00:00:12",
                "2020-01-01 00:00:13",
                "2020-01-01 00:00:14",
            ]
        )
        df = pd.DataFrame({"ts": times})
        assert _estimate_interval_s(df, "ts") == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _first_match
# ---------------------------------------------------------------------------


class TestFirstMatch:
    def test_returns_first_match(self) -> None:
        cols = pd.Index(["channel", "start", "end"])
        assert _first_match(cols, ("channel", "chan")) == "channel"

    def test_returns_none_if_no_match(self) -> None:
        cols = pd.Index(["foo", "bar"])
        assert _first_match(cols, ("channel", "chan")) is None

    def test_respects_priority_order(self) -> None:
        cols = pd.Index(["chan", "channel"])
        assert _first_match(cols, ("channel", "chan")) == "channel"


# ---------------------------------------------------------------------------
# mission_report
# ---------------------------------------------------------------------------


class TestMissionReport:
    def test_counts_channels_and_rows(self, tmp_path: Path) -> None:
        _write_channel(tmp_path, "M1", "A-1", n_rows=100)
        _write_channel(tmp_path, "M1", "B-1", n_rows=200)

        report = DataExplorer(tmp_path).mission_report("M1")

        assert report.mission == "M1"
        assert report.n_channels == 2
        assert report.channel_names == ["A-1", "B-1"]
        assert report.total_rows == 300

    def test_detects_time_range(self, tmp_path: Path) -> None:
        _write_channel(tmp_path, "M1", "A-1", n_rows=10, freq="1s")

        report = DataExplorer(tmp_path).mission_report("M1")

        assert report.time_range is not None
        start, end = report.time_range
        assert "2020-01-01" in start
        assert start < end

    def test_time_range_is_none_when_no_timestamp(self, tmp_path: Path) -> None:
        _write_channel(tmp_path, "M1", "A-1", include_timestamp=False)

        report = DataExplorer(tmp_path).mission_report("M1")

        assert report.time_range is None

    def test_estimates_sampling_interval(self, tmp_path: Path) -> None:
        _write_channel(tmp_path, "M1", "A-1", n_rows=10, freq="2s")

        report = DataExplorer(tmp_path).mission_report("M1")

        assert report.sampling_interval_s == pytest.approx(2.0)

    def test_raises_if_no_parquet_files(self, tmp_path: Path) -> None:
        (tmp_path / "M1" / "channels").mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="No Parquet files"):
            DataExplorer(tmp_path).mission_report("M1")

    def test_channel_names_sorted_alphabetically(self, tmp_path: Path) -> None:
        for ch in ("C-1", "A-1", "B-1"):
            _write_channel(tmp_path, "M1", ch)

        report = DataExplorer(tmp_path).mission_report("M1")

        assert report.channel_names == ["A-1", "B-1", "C-1"]


# ---------------------------------------------------------------------------
# channel_summary
# ---------------------------------------------------------------------------


class TestChannelSummary:
    def test_basic_fields(self, tmp_path: Path) -> None:
        _write_channel(tmp_path, "M1", "channel_1", n_rows=50)

        summary = DataExplorer(tmp_path).channel_summary("M1", "1")

        assert summary.channel == "1"
        assert summary.n_rows == 50
        assert summary.n_columns == 2  # timestamp + value

    def test_value_stats_present_for_numeric_columns(self, tmp_path: Path) -> None:
        _write_channel(tmp_path, "M1", "channel_1", n_rows=100)

        summary = DataExplorer(tmp_path).channel_summary("M1", "1")

        assert "value" in summary.value_stats
        stats = summary.value_stats["value"]
        assert stats["min"] <= stats["mean"] <= stats["max"]
        assert stats["std"] >= 0

    def test_null_counts_reported(self, tmp_path: Path) -> None:
        _write_channel(tmp_path, "M1", "channel_1", n_rows=100, nulls=True)

        summary = DataExplorer(tmp_path).channel_summary("M1", "1")

        # Every 10th row has a null, so ~10 nulls in 100 rows
        assert summary.null_counts["value"] > 0

    def test_time_range_detected(self, tmp_path: Path) -> None:
        _write_channel(tmp_path, "M1", "channel_1", n_rows=10, freq="1s")

        summary = DataExplorer(tmp_path).channel_summary("M1", "1")

        assert summary.time_range is not None
        assert summary.time_range[0] < summary.time_range[1]

    def test_time_range_none_without_timestamp(self, tmp_path: Path) -> None:
        _write_channel(tmp_path, "M1", "channel_1", include_timestamp=False)

        summary = DataExplorer(tmp_path).channel_summary("M1", "1")

        assert summary.time_range is None

    def test_raises_if_file_missing(self, tmp_path: Path) -> None:
        (tmp_path / "M1" / "channels").mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="Channel file not found"):
            DataExplorer(tmp_path).channel_summary("M1", "99")

    def test_dtypes_recorded(self, tmp_path: Path) -> None:
        _write_channel(tmp_path, "M1", "channel_1")

        summary = DataExplorer(tmp_path).channel_summary("M1", "1")

        assert "value" in summary.dtypes
        assert "float" in summary.dtypes["value"]


# ---------------------------------------------------------------------------
# label_report
# ---------------------------------------------------------------------------


class TestLabelReport:
    def test_counts_segments_and_channels(self, tmp_path: Path) -> None:
        _write_labels(
            tmp_path,
            "M1",
            [
                {"channel": "A-1", "start": 0, "end": 10},
                {"channel": "A-1", "start": 50, "end": 60},
                {"channel": "B-1", "start": 5, "end": 15},
            ],
        )

        report = DataExplorer(tmp_path).label_report("M1")

        assert report.n_anomaly_segments == 3
        assert report.n_labeled_channels == 2
        assert set(report.channels_with_labels) == {"A-1", "B-1"}

    def test_counts_anomaly_types(self, tmp_path: Path) -> None:
        _write_labels(
            tmp_path,
            "M1",
            [
                {"channel": "A-1", "start": 0, "end": 5, "anomaly_type": "spike"},
                {"channel": "A-1", "start": 10, "end": 20, "anomaly_type": "drift"},
                {"channel": "B-1", "start": 0, "end": 5, "anomaly_type": "spike"},
            ],
        )

        report = DataExplorer(tmp_path).label_report("M1")

        assert report.anomaly_types == {"spike": 2, "drift": 1}

    def test_returns_empty_report_when_no_labels_csv(self, tmp_path: Path) -> None:
        (tmp_path / "M1").mkdir(parents=True)

        report = DataExplorer(tmp_path).label_report("M1")

        assert report.n_labeled_channels == 0
        assert report.n_anomaly_segments == 0

    def test_returns_empty_report_for_empty_csv(self, tmp_path: Path) -> None:
        _write_labels(tmp_path, "M1", [])

        report = DataExplorer(tmp_path).label_report("M1")

        assert report.n_anomaly_segments == 0

    def test_anomaly_types_empty_when_no_type_column(self, tmp_path: Path) -> None:
        _write_labels(
            tmp_path,
            "M1",
            [
                {"channel": "A-1", "start": 0, "end": 10},
            ],
        )

        report = DataExplorer(tmp_path).label_report("M1")

        assert report.anomaly_types == {}


# ---------------------------------------------------------------------------
# print_report (smoke test — just verify it doesn't raise)
# ---------------------------------------------------------------------------


class TestPrintReport:
    def test_prints_without_error(self, tmp_path: Path) -> None:
        _write_channel(tmp_path, "M1", "channel_1", n_rows=20)
        _write_labels(tmp_path, "M1", [{"channel": "channel_1", "start": 0, "end": 5}])

        con = _capture_console()
        DataExplorer(tmp_path).print_report("M1", console=con)

        output = con.file.getvalue()  # type: ignore[union-attr]
        assert "M1" in output
        assert "channel_1" in output

    def test_prints_error_message_for_missing_mission(self, tmp_path: Path) -> None:
        con = _capture_console()
        DataExplorer(tmp_path).print_report("NonExistent", console=con)

        output = con.file.getvalue()  # type: ignore[union-attr]
        assert "not found" in output.lower()

    def test_handles_mission_with_no_labels(self, tmp_path: Path) -> None:
        _write_channel(tmp_path, "M1", "channel_1", n_rows=10)

        con = _capture_console()
        DataExplorer(tmp_path).print_report("M1", console=con)

        output = con.file.getvalue()  # type: ignore[union-attr]
        assert "0" in output  # labeled channels = 0
