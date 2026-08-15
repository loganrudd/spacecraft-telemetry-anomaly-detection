"""Tests for esa_adb.timeline — segment-gap-aware observed timeline."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from spacecraft_telemetry.core.config import Settings, load_settings
from spacecraft_telemetry.esa_adb.timeline import (
    channel_timeline,
    channel_timeline_from_metadata,
    mission_timeline,
)
from spacecraft_telemetry.model.dataset import load_series_metadata

_MISSION = "ESA-Mission1-Test"
_FREQ_S = 90

_SERIES_SCHEMA = pa.schema(
    [
        pa.field("telemetry_timestamp", pa.timestamp("us", tz="UTC")),
        pa.field("value_normalized", pa.float32()),
        pa.field("segment_id", pa.int32()),
        pa.field("is_anomaly", pa.bool_()),
    ]
)


def _write_series(processed_dir: Path, mission: str, channel: str, seg_sizes: list[int]) -> None:
    base = datetime(2000, 1, 1, tzinfo=UTC)
    timestamps = []
    seg_ids = []
    t = 0
    for seg_id, size in enumerate(seg_sizes):
        for _ in range(size):
            timestamps.append(
                pa.scalar(base.timestamp() + t * _FREQ_S, type=pa.timestamp("s", tz="UTC")).cast(
                    pa.timestamp("us", tz="UTC")
                )
            )
            seg_ids.append(seg_id)
            t += 1
    n = len(timestamps)
    table = pa.table(
        {
            "telemetry_timestamp": pa.array(timestamps, type=pa.timestamp("us", tz="UTC")),
            "value_normalized": pa.array([0.0] * n, type=pa.float32()),
            "segment_id": pa.array(seg_ids, type=pa.int32()),
            "is_anomaly": pa.array([False] * n, type=pa.bool_()),
        },
        schema=_SERIES_SCHEMA,
    )
    partition_dir = (
        processed_dir / mission / "test" / f"mission_id={mission}" / f"channel_id={channel}"
    )
    partition_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, partition_dir / "part.parquet")


def _settings(processed_dir: Path) -> Settings:
    base_settings = load_settings("test")
    return base_settings.model_copy(
        update={
            "preprocess": base_settings.preprocess.model_copy(
                update={"processed_data_dir": str(processed_dir)}
            )
        }
    )


class TestChannelTimeline:
    def test_single_segment_is_one_interval(self, tmp_path: Path) -> None:
        processed_dir = tmp_path / "processed"
        _write_series(processed_dir, _MISSION, "channel_41", seg_sizes=[10])
        result = channel_timeline(_settings(processed_dir), _MISSION, "channel_41")
        assert len(result) == 1

    def test_two_segments_are_two_disjoint_intervals(self, tmp_path: Path) -> None:
        """A segment gap (LOS) must NOT be bridged into one interval."""
        processed_dir = tmp_path / "processed"
        _write_series(processed_dir, _MISSION, "channel_41", seg_sizes=[5, 5])
        result = channel_timeline(_settings(processed_dir), _MISSION, "channel_41")
        assert len(result) == 2
        # First segment ends strictly before the second starts.
        assert result[0][1] < result[1][0]

    def test_gap_duration_excluded_from_intervals(self, tmp_path: Path) -> None:
        """The interval boundaries must not span the gap between segments."""
        processed_dir = tmp_path / "processed"
        _write_series(processed_dir, _MISSION, "channel_41", seg_sizes=[3, 3])
        result = channel_timeline(_settings(processed_dir), _MISSION, "channel_41")
        seg0_start, seg0_end = result[0]
        seg1_start, seg1_end = result[1]
        # 3 rows at 90s cadence -> 2*90=180s span per segment.
        assert (seg0_end - seg0_start).total_seconds() == 2 * _FREQ_S
        assert (seg1_end - seg1_start).total_seconds() == 2 * _FREQ_S


class TestMissionTimeline:
    def test_unions_across_channels(self, tmp_path: Path) -> None:
        processed_dir = tmp_path / "processed"
        _write_series(processed_dir, _MISSION, "channel_41", seg_sizes=[10])
        _write_series(processed_dir, _MISSION, "channel_42", seg_sizes=[10])
        result = mission_timeline(_settings(processed_dir), _MISSION, ["channel_41", "channel_42"])
        # Both channels share the same timestamps here, so the union collapses to 1.
        assert len(result) == 1

    def test_metadata_by_channel_matches_disk_read(self, tmp_path: Path) -> None:
        """Preloaded metadata must reproduce the disk-reading result exactly.

        Regression for docs/plans/019 P2/P3: build_report preloads each
        channel's (segment_ids, is_anomaly, timestamps) once and reuses them
        here instead of re-reading the partition — must be a pure refactor.
        """
        processed_dir = tmp_path / "processed"
        _write_series(processed_dir, _MISSION, "channel_41", seg_sizes=[5, 5])
        _write_series(processed_dir, _MISSION, "channel_42", seg_sizes=[3, 3])
        settings = _settings(processed_dir)
        channels = ["channel_41", "channel_42"]

        expected = mission_timeline(settings, _MISSION, channels)

        metadata_by_channel = {
            ch: load_series_metadata(processed_dir, _MISSION, ch, "test") for ch in channels
        }
        actual = mission_timeline(
            settings, _MISSION, channels, metadata_by_channel=metadata_by_channel
        )

        assert actual == expected


class TestChannelTimelineFromMetadata:
    def test_matches_channel_timeline(self, tmp_path: Path) -> None:
        processed_dir = tmp_path / "processed"
        _write_series(processed_dir, _MISSION, "channel_41", seg_sizes=[5, 5])
        expected = channel_timeline(_settings(processed_dir), _MISSION, "channel_41")

        segment_ids, _, timestamps = load_series_metadata(
            processed_dir, _MISSION, "channel_41", "test"
        )
        actual = channel_timeline_from_metadata(segment_ids, timestamps)

        assert actual == expected
