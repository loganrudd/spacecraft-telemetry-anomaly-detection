"""Unit tests for mlflow_tracking/hashing.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from spacecraft_telemetry.mlflow_tracking.hashing import (
    group_partition_hash,
    partition_hash,
    training_data_hash,
)

_MISSION = "ESA-Mission1"
_CHANNEL = "channel_1"


def _make_partition(base: Path, mission: str, channel: str, split: str = "train") -> Path:
    """Create a fake Parquet partition directory with some files."""
    part = base / mission / split / f"mission_id={mission}" / f"channel_id={channel}"
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


class TestPartitionHash:
    """Tests for the general partition_hash function."""

    def test_hash_is_stable_for_train_split(self, tmp_path: Path) -> None:
        _make_partition(tmp_path, _MISSION, _CHANNEL, split="train")
        h1 = partition_hash(tmp_path, _MISSION, _CHANNEL, "train")
        h2 = partition_hash(tmp_path, _MISSION, _CHANNEL, "train")
        assert h1 == h2

    def test_hash_is_stable_for_test_split(self, tmp_path: Path) -> None:
        _make_partition(tmp_path, _MISSION, _CHANNEL, split="test")
        h1 = partition_hash(tmp_path, _MISSION, _CHANNEL, "test")
        h2 = partition_hash(tmp_path, _MISSION, _CHANNEL, "test")
        assert h1 == h2

    def test_train_and_test_splits_produce_different_hashes(self, tmp_path: Path) -> None:
        # Same channel name but different split content → different hashes.
        train_part = _make_partition(tmp_path, _MISSION, _CHANNEL, split="train")
        _make_partition(tmp_path, _MISSION, _CHANNEL, split="test")
        (train_part / "extra.parquet").write_bytes(b"train-only-file")
        h_train = partition_hash(tmp_path, _MISSION, _CHANNEL, "train")
        h_test = partition_hash(tmp_path, _MISSION, _CHANNEL, "test")
        assert h_train != h_test

    def test_training_data_hash_delegates_to_partition_hash(self, tmp_path: Path) -> None:
        """training_data_hash must return the same value as partition_hash(..., 'train')."""
        _make_partition(tmp_path, _MISSION, _CHANNEL, split="train")
        assert training_data_hash(tmp_path, _MISSION, _CHANNEL) == partition_hash(
            tmp_path, _MISSION, _CHANNEL, "train"
        )

    def test_missing_directory_raises_with_split_name(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Test partition directory not found"):
            partition_hash(tmp_path, _MISSION, "nonexistent_channel", "test")

    def test_hash_is_64_char_hex(self, tmp_path: Path) -> None:
        _make_partition(tmp_path, _MISSION, _CHANNEL, split="test")
        h = partition_hash(tmp_path, _MISSION, _CHANNEL, "test")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


class TestGroupPartitionHash:
    """Tests for the multivariate (docs/plans/021) group_partition_hash."""

    def test_single_channel_group_does_not_reproduce_partition_hash(
        self, tmp_path: Path
    ) -> None:
        """The group hash embeds the channel name in its payload, so it must
        differ from partition_hash even for a length-1 group — this is what
        keeps partition_hash's own return value (and every existing
        production training_data_hash tag) untouched by this addition."""
        _make_partition(tmp_path, _MISSION, _CHANNEL, split="train")
        single = partition_hash(tmp_path, _MISSION, _CHANNEL, "train")
        group = group_partition_hash(tmp_path, _MISSION, [_CHANNEL], "train")
        assert single != group

    def test_hash_is_stable(self, tmp_path: Path) -> None:
        _make_partition(tmp_path, _MISSION, "channel_1")
        _make_partition(tmp_path, _MISSION, "channel_2")
        channels = ["channel_1", "channel_2"]
        h1 = group_partition_hash(tmp_path, _MISSION, channels, "train")
        h2 = group_partition_hash(tmp_path, _MISSION, channels, "train")
        assert h1 == h2

    def test_hash_independent_of_channel_order(self, tmp_path: Path) -> None:
        """The payload is sorted internally, so passing the group in a
        different order must not change the hash."""
        _make_partition(tmp_path, _MISSION, "channel_1")
        _make_partition(tmp_path, _MISSION, "channel_2")
        h_forward = group_partition_hash(
            tmp_path, _MISSION, ["channel_1", "channel_2"], "train"
        )
        h_reversed = group_partition_hash(
            tmp_path, _MISSION, ["channel_2", "channel_1"], "train"
        )
        assert h_forward == h_reversed

    def test_hash_differs_from_subset(self, tmp_path: Path) -> None:
        _make_partition(tmp_path, _MISSION, "channel_1")
        _make_partition(tmp_path, _MISSION, "channel_2")
        h_one = group_partition_hash(tmp_path, _MISSION, ["channel_1"], "train")
        h_two = group_partition_hash(tmp_path, _MISSION, ["channel_1", "channel_2"], "train")
        assert h_one != h_two

    def test_hash_differs_when_any_member_changes(self, tmp_path: Path) -> None:
        part_a = _make_partition(tmp_path, _MISSION, "channel_1")
        _make_partition(tmp_path, _MISSION, "channel_2")
        channels = ["channel_1", "channel_2"]
        h1 = group_partition_hash(tmp_path, _MISSION, channels, "train")

        (part_a / "extra.parquet").write_bytes(b"new-file")
        h2 = group_partition_hash(tmp_path, _MISSION, channels, "train")
        assert h1 != h2

    def test_missing_member_raises(self, tmp_path: Path) -> None:
        _make_partition(tmp_path, _MISSION, "channel_1")
        with pytest.raises(ValueError, match="Train partition directory not found"):
            group_partition_hash(tmp_path, _MISSION, ["channel_1", "channel_2"], "train")

    def test_hash_is_64_char_hex(self, tmp_path: Path) -> None:
        _make_partition(tmp_path, _MISSION, "channel_1")
        _make_partition(tmp_path, _MISSION, "channel_2")
        h = group_partition_hash(tmp_path, _MISSION, ["channel_1", "channel_2"], "train")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)
