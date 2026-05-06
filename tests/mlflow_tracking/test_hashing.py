"""Unit tests for mlflow_tracking/hashing.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from spacecraft_telemetry.mlflow_tracking.hashing import training_data_hash

_MISSION = "ESA-Mission1"
_CHANNEL = "channel_1"


def _make_partition(base: Path, mission: str, channel: str) -> Path:
    """Create a fake train Parquet partition directory with some files."""
    part = base / mission / "train" / f"mission_id={mission}" / f"channel_id={channel}"
    part.mkdir(parents=True)
    (part / "part-00000.parquet").write_bytes(b"fake-parquet-data-a")
    (part / "part-00001.parquet").write_bytes(b"fake-parquet-data-b")
    return part


class TestTrainingDataHash:
    def test_hash_is_stable(self, tmp_path: Path) -> None:
        _make_partition(tmp_path, _MISSION, _CHANNEL)
        h1 = training_data_hash(tmp_path, _MISSION, _CHANNEL)
        h2 = training_data_hash(tmp_path, _MISSION, _CHANNEL)
        assert h1 == h2

    def test_hash_is_hex_string(self, tmp_path: Path) -> None:
        _make_partition(tmp_path, _MISSION, _CHANNEL)
        h = training_data_hash(tmp_path, _MISSION, _CHANNEL)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_hash_differs_when_file_size_changes(self, tmp_path: Path) -> None:
        part = _make_partition(tmp_path, _MISSION, _CHANNEL)
        h1 = training_data_hash(tmp_path, _MISSION, _CHANNEL)

        # Append a byte to change the file size.
        (part / "part-00000.parquet").write_bytes(b"fake-parquet-data-a-EXTRA")
        h2 = training_data_hash(tmp_path, _MISSION, _CHANNEL)
        assert h1 != h2

    def test_hash_differs_when_file_added(self, tmp_path: Path) -> None:
        part = _make_partition(tmp_path, _MISSION, _CHANNEL)
        h1 = training_data_hash(tmp_path, _MISSION, _CHANNEL)

        (part / "part-00002.parquet").write_bytes(b"new-file")
        h2 = training_data_hash(tmp_path, _MISSION, _CHANNEL)
        assert h1 != h2

    def test_missing_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Train partition directory not found"):
            training_data_hash(tmp_path, _MISSION, "nonexistent_channel")

    def test_different_channels_produce_different_hashes(self, tmp_path: Path) -> None:
        part_a = (
            tmp_path / _MISSION / "train"
            / f"mission_id={_MISSION}" / "channel_id=channel_1"
        )
        part_b = (
            tmp_path / _MISSION / "train"
            / f"mission_id={_MISSION}" / "channel_id=channel_2"
        )
        part_a.mkdir(parents=True)
        part_b.mkdir(parents=True)
        (part_a / "part-0.parquet").write_bytes(b"data-a")
        (part_b / "part-0.parquet").write_bytes(b"data-b-different")

        assert training_data_hash(tmp_path, _MISSION, "channel_1") != training_data_hash(
            tmp_path, _MISSION, "channel_2"
        )
