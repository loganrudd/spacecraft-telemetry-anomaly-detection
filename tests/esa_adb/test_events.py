"""Tests for esa_adb.events — labels.csv x anomaly_types.csv join and event grouping."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from spacecraft_telemetry.core.config import Settings, load_settings
from spacecraft_telemetry.esa_adb.events import group_events, load_events

_MISSION = "ESA-Mission1-Test"


def _write_fixture(base: Path, labels_rows: list[dict], types_rows: list[dict]) -> None:
    mission_dir = base / _MISSION
    mission_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(labels_rows).to_csv(mission_dir / "labels.csv", index=False)
    pd.DataFrame(types_rows).to_csv(mission_dir / "anomaly_types.csv", index=False)


def _settings(base: Path) -> Settings:
    base_settings = load_settings("test")
    return base_settings.model_copy(
        update={"data": base_settings.data.model_copy(update={"sample_data_dir": str(base)})}
    )


# ---------------------------------------------------------------------------
# load_events
# ---------------------------------------------------------------------------


class TestLoadEvents:
    def test_joins_labels_to_anomaly_types(self, tmp_path: Path) -> None:
        _write_fixture(
            tmp_path,
            labels_rows=[
                {
                    "ID": "id_1",
                    "Channel": "channel_41",
                    "StartTime": "2000-01-01T00:00:00Z",
                    "EndTime": "2000-01-01T01:00:00Z",
                },
            ],
            types_rows=[
                {
                    "ID": "id_1",
                    "Class": "class_1",
                    "Subclass": "subclass_1",
                    "Category": "Anomaly",
                    "Dimensionality": "Univariate",
                    "Locality": "Local",
                    "Length": "Subsequence",
                },
            ],
        )
        df = load_events(_settings(tmp_path), _MISSION)
        assert len(df) == 1
        row = df.iloc[0]
        assert row["event_id"] == "id_1"
        assert row["channel_id"] == "channel_41"
        assert row["category"] == "Anomaly"
        assert row["class_"] == "class_1"
        assert row["subclass"] == "subclass_1"

    def test_raises_on_event_id_missing_from_anomaly_types(self, tmp_path: Path) -> None:
        _write_fixture(
            tmp_path,
            labels_rows=[
                {
                    "ID": "id_orphan",
                    "Channel": "channel_41",
                    "StartTime": "2000-01-01T00:00:00Z",
                    "EndTime": "2000-01-01T01:00:00Z",
                },
            ],
            types_rows=[
                {
                    "ID": "id_1",
                    "Class": "class_1",
                    "Subclass": "subclass_1",
                    "Category": "Anomaly",
                    "Dimensionality": "Univariate",
                    "Locality": "Local",
                    "Length": "Subsequence",
                },
            ],
        )
        with pytest.raises(ValueError, match="id_orphan"):
            load_events(_settings(tmp_path), _MISSION)

    def test_multiple_fragments_same_event_and_channel(self, tmp_path: Path) -> None:
        """One event can appear as several non-contiguous fragments on one channel."""
        _write_fixture(
            tmp_path,
            labels_rows=[
                {
                    "ID": "id_1",
                    "Channel": "channel_41",
                    "StartTime": "2000-01-01T00:00:00Z",
                    "EndTime": "2000-01-01T01:00:00Z",
                },
                {
                    "ID": "id_1",
                    "Channel": "channel_41",
                    "StartTime": "2000-01-01T02:00:00Z",
                    "EndTime": "2000-01-01T03:00:00Z",
                },
            ],
            types_rows=[
                {
                    "ID": "id_1",
                    "Class": "class_1",
                    "Subclass": "subclass_1",
                    "Category": "Anomaly",
                    "Dimensionality": "Univariate",
                    "Locality": "Local",
                    "Length": "Subsequence",
                },
            ],
        )
        df = load_events(_settings(tmp_path), _MISSION)
        assert len(df) == 2
        assert df["event_id"].nunique() == 1


# ---------------------------------------------------------------------------
# group_events
# ---------------------------------------------------------------------------


def _events_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["start_time"] = pd.to_datetime(df["start_time"], utc=True)
    df["end_time"] = pd.to_datetime(df["end_time"], utc=True)
    return df


_FULL_TIMELINE = [(pd.Timestamp("2000-01-01T00:00:00Z"), pd.Timestamp("2000-01-02T00:00:00Z"))]


class TestGroupEvents:
    def test_merges_fragments_of_same_event(self) -> None:
        df = _events_df(
            [
                {
                    "event_id": "id_1",
                    "channel_id": "channel_41",
                    "start_time": "2000-01-01T00:00:00Z",
                    "end_time": "2000-01-01T01:00:00Z",
                    "category": "Anomaly",
                },
                {
                    "event_id": "id_1",
                    "channel_id": "channel_41",
                    "start_time": "2000-01-01T02:00:00Z",
                    "end_time": "2000-01-01T03:00:00Z",
                    "category": "Anomaly",
                },
            ]
        )
        events = group_events(df, channels=["channel_41"], timeline=_FULL_TIMELINE)
        assert len(events) == 1
        assert events[0].event_id == "id_1"
        assert len(events[0].intervals) == 2, "non-contiguous fragments stay as 2 intervals"

    def test_point_event_is_not_dropped(self) -> None:
        """Zero-duration (StartTime == EndTime) annotations must survive grouping.

        Half-open [t, t) is the empty set, so without widening these events
        intersect nothing and vanish from the ground truth. On Mission1's
        lightweight test split that silently removed 9 of 65 events — exactly
        the paper's "Point" count in Supplementary Table 11.
        """
        df = _events_df(
            [
                {
                    "event_id": "id_point",
                    "channel_id": "channel_41",
                    "start_time": "2000-01-01T06:00:00Z",
                    "end_time": "2000-01-01T06:00:00Z",
                    "category": "Anomaly",
                }
            ]
        )
        events = group_events(df, channels=["channel_41"], timeline=_FULL_TIMELINE)
        assert len(events) == 1, "point event was dropped"
        (start, end), = events[0].intervals
        assert start == pd.Timestamp("2000-01-01T06:00:00Z")
        assert end > start, "point event must have non-zero width to be detectable"
        assert end - start == pd.Timedelta(1, "ns"), "widening must stay minimal"

    def test_point_event_detected_by_covering_interval(self) -> None:
        """A detection spanning the instant must overlap the widened point event."""
        from spacecraft_telemetry.esa_adb.intervals import overlaps

        df = _events_df(
            [
                {
                    "event_id": "id_point",
                    "channel_id": "channel_41",
                    "start_time": "2000-01-01T06:00:00Z",
                    "end_time": "2000-01-01T06:00:00Z",
                    "category": "Anomaly",
                }
            ]
        )
        (event,) = group_events(df, channels=["channel_41"], timeline=_FULL_TIMELINE)
        covering = [
            (pd.Timestamp("2000-01-01T05:00:00Z"), pd.Timestamp("2000-01-01T07:00:00Z"))
        ]
        missing = [
            (pd.Timestamp("2000-01-01T07:00:00Z"), pd.Timestamp("2000-01-01T08:00:00Z"))
        ]
        assert overlaps(event.intervals[0], covering) is True
        assert overlaps(event.intervals[0], missing) is False

    def test_drops_channels_outside_scope(self) -> None:
        df = _events_df(
            [
                {
                    "event_id": "id_1",
                    "channel_id": "channel_21",
                    "start_time": "2000-01-01T00:00:00Z",
                    "end_time": "2000-01-01T01:00:00Z",
                    "category": "Anomaly",
                },
            ]
        )
        events = group_events(df, channels=["channel_41"], timeline=_FULL_TIMELINE)
        assert events == []

    def test_drops_event_entirely_outside_timeline(self) -> None:
        df = _events_df(
            [
                {
                    "event_id": "id_1",
                    "channel_id": "channel_41",
                    "start_time": "1999-01-01T00:00:00Z",
                    "end_time": "1999-01-01T01:00:00Z",
                    "category": "Anomaly",
                },
            ]
        )
        events = group_events(df, channels=["channel_41"], timeline=_FULL_TIMELINE)
        assert events == []

    def test_clips_event_partially_outside_timeline(self) -> None:
        timeline = [(pd.Timestamp("2000-01-01T00:30:00Z"), pd.Timestamp("2000-01-02T00:00:00Z"))]
        df = _events_df(
            [
                {
                    "event_id": "id_1",
                    "channel_id": "channel_41",
                    "start_time": "2000-01-01T00:00:00Z",
                    "end_time": "2000-01-01T01:00:00Z",
                    "category": "Anomaly",
                },
            ]
        )
        events = group_events(df, channels=["channel_41"], timeline=timeline)
        assert len(events) == 1
        assert events[0].intervals[0][0] == pd.Timestamp("2000-01-01T00:30:00Z")
        assert events[0].intervals[0][1] == pd.Timestamp("2000-01-01T01:00:00Z")

    def test_channels_field_restricted_to_scope(self) -> None:
        df = _events_df(
            [
                {
                    "event_id": "id_1",
                    "channel_id": "channel_41",
                    "start_time": "2000-01-01T00:00:00Z",
                    "end_time": "2000-01-01T01:00:00Z",
                    "category": "Anomaly",
                },
                {
                    "event_id": "id_1",
                    "channel_id": "channel_21",
                    "start_time": "2000-01-01T00:00:00Z",
                    "end_time": "2000-01-01T01:00:00Z",
                    "category": "Anomaly",
                },
            ]
        )
        events = group_events(df, channels=["channel_41", "channel_42"], timeline=_FULL_TIMELINE)
        assert len(events) == 1
        assert events[0].channels == frozenset({"channel_41"})

    def test_category_preserved(self) -> None:
        df = _events_df(
            [
                {
                    "event_id": "id_1",
                    "channel_id": "channel_41",
                    "start_time": "2000-01-01T00:00:00Z",
                    "end_time": "2000-01-01T01:00:00Z",
                    "category": "Communication Gap",
                },
            ]
        )
        events = group_events(df, channels=["channel_41"], timeline=_FULL_TIMELINE)
        assert events[0].category == "Communication Gap"

    def test_sorted_by_event_id(self) -> None:
        df = _events_df(
            [
                {
                    "event_id": "id_2",
                    "channel_id": "channel_41",
                    "start_time": "2000-01-01T00:00:00Z",
                    "end_time": "2000-01-01T01:00:00Z",
                    "category": "Anomaly",
                },
                {
                    "event_id": "id_1",
                    "channel_id": "channel_41",
                    "start_time": "2000-01-01T02:00:00Z",
                    "end_time": "2000-01-01T03:00:00Z",
                    "category": "Anomaly",
                },
            ]
        )
        events = group_events(df, channels=["channel_41"], timeline=_FULL_TIMELINE)
        assert [e.event_id for e in events] == ["id_1", "id_2"]
