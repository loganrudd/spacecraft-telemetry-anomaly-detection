"""Tests for ingest.sample."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spacecraft_telemetry.ingest.sample import (
    SampleCreator,
    SampleManifest,
    _channel_name,
    _detect_channel_column,
)

# ---------------------------------------------------------------------------
# Test fixtures / helpers
# ---------------------------------------------------------------------------


def _make_df(n_rows: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=n_rows, freq="1s"),
            "value": rng.random(n_rows),
        }
    )


def _write_raw_mission(
    raw_dir: Path,
    mission: str,
    channel_names: list[str],
    n_rows: int = 1000,
    labels_rows: list[dict] | None = None,
) -> Path:
    """Create a minimal raw mission directory with pickle files."""
    mission_dir = raw_dir / mission
    channel_dir = mission_dir / "channels"
    channel_dir.mkdir(parents=True)

    for name in channel_names:
        df = _make_df(n_rows)
        with (channel_dir / f"{name}.pkl").open("wb") as fh:
            pickle.dump(df, fh)

    if labels_rows is not None:
        labels_path = mission_dir / "labels.csv"
        if labels_rows:
            pd.DataFrame(labels_rows).to_csv(labels_path, index=False)
        else:
            labels_path.write_text("channel,start,end\n")

    return mission_dir


# ---------------------------------------------------------------------------
# _channel_name
# ---------------------------------------------------------------------------


class TestChannelName:
    def test_strips_pkl_suffix(self) -> None:
        assert _channel_name(Path("A-1.pkl")) == "A-1"

    def test_strips_pkl_zip_suffix(self) -> None:
        assert _channel_name(Path("B-2.pkl.zip")) == "B-2"

    def test_falls_back_to_stem(self) -> None:
        assert _channel_name(Path("channel.csv")) == "channel"


# ---------------------------------------------------------------------------
# _detect_channel_column
# ---------------------------------------------------------------------------


class TestDetectChannelColumn:
    def test_detects_channel(self) -> None:
        df = pd.DataFrame({"channel": ["A-1"], "start": [0], "end": [10]})
        assert _detect_channel_column(df) == "channel"

    def test_detects_channel_id(self) -> None:
        df = pd.DataFrame({"channel_id": ["A-1"]})
        assert _detect_channel_column(df) == "channel_id"

    def test_returns_none_if_unknown(self) -> None:
        df = pd.DataFrame({"foo": [1], "bar": [2]})
        assert _detect_channel_column(df) is None


# ---------------------------------------------------------------------------
# _select_channels
# ---------------------------------------------------------------------------


class TestSelectChannels:
    def test_returns_alphabetical_first_n(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw"
        _write_raw_mission(raw, "M1", ["C-1", "A-1", "B-1"])

        creator = SampleCreator(raw, tmp_path / "sample", sample_channels=2)
        assert creator._select_channels("M1") == ["A-1", "B-1"]

    def test_returns_all_when_fewer_than_n(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw"
        _write_raw_mission(raw, "M1", ["A-1", "B-1"])

        creator = SampleCreator(raw, tmp_path / "sample", sample_channels=10)
        assert creator._select_channels("M1") == ["A-1", "B-1"]

    def test_raises_if_channel_dir_missing(self, tmp_path: Path) -> None:
        creator = SampleCreator(tmp_path / "raw", tmp_path / "sample")
        with pytest.raises(FileNotFoundError, match="Channel directory not found"):
            creator._select_channels("M1")

    def test_raises_if_no_pickle_files(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw"
        (raw / "M1" / "channels").mkdir(parents=True)

        creator = SampleCreator(raw, tmp_path / "sample")
        with pytest.raises(FileNotFoundError, match="No pickle files found"):
            creator._select_channels("M1")


# ---------------------------------------------------------------------------
# _take_first_n_rows
# ---------------------------------------------------------------------------


class TestTakeFirstNRows:
    def test_returns_correct_fraction(self, tmp_path: Path) -> None:
        creator = SampleCreator(tmp_path, tmp_path, sample_fraction=0.1)
        df = pd.DataFrame({"v": range(1000)})
        assert len(creator._take_first_n_rows(df)) == 100

    def test_returns_at_least_one_row(self, tmp_path: Path) -> None:
        creator = SampleCreator(tmp_path, tmp_path, sample_fraction=0.001)
        df = pd.DataFrame({"v": range(5)})
        assert len(creator._take_first_n_rows(df)) >= 1

    def test_slice_starts_at_row_zero(self, tmp_path: Path) -> None:
        creator = SampleCreator(tmp_path, tmp_path, sample_fraction=0.1)
        df = pd.DataFrame({"v": range(100)})
        sampled = creator._take_first_n_rows(df)
        assert sampled["v"].iloc[0] == 0
        assert sampled["v"].iloc[-1] == 9


# ---------------------------------------------------------------------------
# create_sample — integration
# ---------------------------------------------------------------------------


class TestCreateSample:
    def test_writes_parquet_for_selected_channels_only(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw"
        sample = tmp_path / "sample"
        _write_raw_mission(raw, "M1", ["A-1", "B-1", "C-1"])

        SampleCreator(raw, sample, sample_fraction=0.1, sample_channels=2).create_sample("M1")

        assert (sample / "M1" / "channels" / "A-1.parquet").exists()
        assert (sample / "M1" / "channels" / "B-1.parquet").exists()
        assert not (sample / "M1" / "channels" / "C-1.parquet").exists()

    def test_parquet_has_correct_row_count(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw"
        sample = tmp_path / "sample"
        _write_raw_mission(raw, "M1", ["A-1"], n_rows=1000)

        SampleCreator(raw, sample, sample_fraction=0.1, sample_channels=1).create_sample("M1")

        df = pd.read_parquet(sample / "M1" / "channels" / "A-1.parquet")
        assert len(df) == 100

    def test_parquet_preserves_first_rows(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw"
        sample = tmp_path / "sample"
        _write_raw_mission(raw, "M1", ["A-1"], n_rows=500)

        original: pd.DataFrame = pickle.loads((raw / "M1" / "channels" / "A-1.pkl").read_bytes())

        SampleCreator(raw, sample, sample_fraction=0.2, sample_channels=1).create_sample("M1")

        sampled = pd.read_parquet(sample / "M1" / "channels" / "A-1.parquet")
        assert sampled["value"].iloc[0] == pytest.approx(original["value"].iloc[0])

    def test_returns_sample_manifest_instance(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw"
        sample = tmp_path / "sample"
        _write_raw_mission(raw, "M1", ["A-1"])

        manifest = SampleCreator(
            raw, sample, sample_fraction=1.0, sample_channels=1
        ).create_sample("M1")

        assert isinstance(manifest, SampleManifest)
        assert manifest.mission == "M1"

    def test_manifest_json_is_written(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw"
        sample = tmp_path / "sample"
        _write_raw_mission(raw, "M1", ["A-1", "B-1"])

        SampleCreator(raw, sample, sample_fraction=0.5, sample_channels=2).create_sample("M1")

        data = json.loads((sample / "M1" / "manifest.json").read_text())
        assert data["mission"] == "M1"
        assert data["sample_fraction"] == 0.5
        assert data["channels"] == ["A-1", "B-1"]

    def test_manifest_row_counts_match_parquet(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw"
        sample = tmp_path / "sample"
        _write_raw_mission(raw, "M1", ["A-1"], n_rows=1000)

        manifest = SampleCreator(
            raw, sample, sample_fraction=0.1, sample_channels=1
        ).create_sample("M1")

        assert manifest.row_counts["A-1"] == 100

    def test_manifest_source_and_sample_dirs(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw"
        sample = tmp_path / "sample"
        _write_raw_mission(raw, "M1", ["A-1"])

        manifest = SampleCreator(
            raw, sample, sample_fraction=1.0, sample_channels=1
        ).create_sample("M1")

        assert manifest.source_dir == str(raw / "M1")
        assert manifest.sample_dir == str(sample / "M1")


# ---------------------------------------------------------------------------
# Labels filtering
# ---------------------------------------------------------------------------


class TestWriteLabels:
    def test_filters_to_selected_channels(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw"
        sample = tmp_path / "sample"
        _write_raw_mission(
            raw,
            "M1",
            ["A-1", "B-1", "C-1"],
            labels_rows=[
                {"channel": "A-1", "start": 0, "end": 10},
                {"channel": "B-1", "start": 5, "end": 15},
                {"channel": "C-1", "start": 2, "end": 8},
            ],
        )

        SampleCreator(raw, sample, sample_fraction=1.0, sample_channels=2).create_sample("M1")

        labels = pd.read_csv(sample / "M1" / "labels.csv")
        assert set(labels["channel"].tolist()) == {"A-1", "B-1"}

    def test_no_labels_csv_does_not_raise(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw"
        sample = tmp_path / "sample"
        _write_raw_mission(raw, "M1", ["A-1"])  # no labels_rows → no labels.csv

        SampleCreator(raw, sample, sample_fraction=1.0, sample_channels=1).create_sample("M1")

        assert not (sample / "M1" / "labels.csv").exists()

    def test_empty_labels_csv_is_copied(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw"
        sample = tmp_path / "sample"
        _write_raw_mission(raw, "M1", ["A-1"], labels_rows=[])

        SampleCreator(raw, sample, sample_fraction=1.0, sample_channels=1).create_sample("M1")

        labels = pd.read_csv(sample / "M1" / "labels.csv")
        assert len(labels) == 0

    def test_labels_without_channel_column_copied_as_is(self, tmp_path: Path) -> None:
        raw = tmp_path / "raw"
        sample = tmp_path / "sample"
        _write_raw_mission(raw, "M1", ["A-1"])
        # Write labels with an unrecognised column name
        (raw / "M1" / "labels.csv").write_text("start,end\n0,10\n")

        SampleCreator(raw, sample, sample_fraction=1.0, sample_channels=1).create_sample("M1")

        labels = pd.read_csv(sample / "M1" / "labels.csv")
        assert len(labels) == 1
