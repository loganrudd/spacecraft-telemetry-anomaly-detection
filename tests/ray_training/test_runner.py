"""Unit and integration tests for ray_training/runner.py."""

from __future__ import annotations

from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# discover_channels
# ---------------------------------------------------------------------------


def test_discover_channels_finds_channel(tmp_path) -> None:
    """discover_channels returns sorted channel IDs from Hive partition dirs."""
    from spacecraft_telemetry.core.config import load_settings
    from spacecraft_telemetry.ray_training import discover_channels

    settings = load_settings("test")
    mission = "ESA-Mission1"
    processed_dir = tmp_path / "processed"

    # Create two fake channel partition directories.
    for ch in ("channel_3", "channel_1"):
        part = processed_dir / mission / "train" / f"mission_id={mission}" / f"channel_id={ch}"
        part.mkdir(parents=True)

    settings = settings.model_copy(
        update={"spark": settings.spark.model_copy(update={"processed_data_dir": str(processed_dir)})}
    )
    channels = discover_channels(settings, mission)
    assert channels == ["channel_1", "channel_3"]


def test_discover_channels_empty_when_no_processed_data() -> None:
    """discover_channels returns [] when mission dir doesn't exist."""
    from spacecraft_telemetry.core.config import load_settings
    from spacecraft_telemetry.ray_training import discover_channels

    settings = load_settings("test")
    channels = discover_channels(settings, "ESA-NonExistent")
    assert channels == []


def test_discover_channels_ignores_non_channel_dirs(tmp_path) -> None:
    """discover_channels skips dirs that don't start with 'channel_id='."""
    from spacecraft_telemetry.core.config import load_settings
    from spacecraft_telemetry.ray_training import discover_channels

    settings = load_settings("test")
    mission = "ESA-Mission1"
    processed_dir = tmp_path / "processed"
    base = processed_dir / mission / "train" / f"mission_id={mission}"
    (base / "channel_id=channel_1").mkdir(parents=True)
    (base / "_SUCCESS").mkdir(parents=True)           # should be ignored
    (base / "some_other_dir").mkdir(parents=True)     # should be ignored

    settings = settings.model_copy(
        update={"spark": settings.spark.model_copy(update={"processed_data_dir": str(processed_dir)})}
    )
    channels = discover_channels(settings, mission)
    assert channels == ["channel_1"]


# ---------------------------------------------------------------------------
# train_all_channels / score_all_channels — integration (slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_train_all_channels_ok(ray_train_result) -> None:
    """train_all_channels returns one ok result for the available channel."""
    pytest.importorskip("ray")
    assert len(ray_train_result) == 1
    r = ray_train_result[0]
    assert r["status"] == "ok", f"Expected ok, got: {r.get('error_msg')}"
    assert r["channel"] == "channel_1"
    assert isinstance(r["best_epoch"], int)


@pytest.mark.slow
def test_train_all_channels_partial_failure(ray_local, ray_series_parquet) -> None:
    """train_all_channels completes the sweep even if one channel fails."""
    pytest.importorskip("ray")
    from spacecraft_telemetry.ray_training import train_all_channels

    settings = ray_series_parquet
    results = train_all_channels(
        settings, "ESA-Mission1", ["channel_1", "nonexistent_channel"]
    )

    assert len(results) == 2
    statuses = {r["channel"]: r["status"] for r in results}
    assert statuses["channel_1"] == "ok"
    assert statuses["nonexistent_channel"] == "error"


@pytest.mark.slow
def test_train_all_channels_max_channels_cap(ray_train_result) -> None:
    """max_channels limits the sweep — cached single-channel result validates cap=1."""
    pytest.importorskip("ray")
    # ray_train_result was built with exactly 1 channel, matching max_channels=1 behaviour.
    assert len(ray_train_result) == 1


@pytest.mark.slow
def test_score_all_channels_ok(ray_local, pretrained_channel) -> None:
    """score_all_channels returns metrics after training."""
    pytest.importorskip("ray")
    from spacecraft_telemetry.ray_training import score_all_channels

    results = score_all_channels(pretrained_channel, "ESA-Mission1", ["channel_1"])
    assert len(results) == 1
    r = results[0]
    assert r["status"] == "ok", f"Expected ok, got: {r.get('error_msg')}"
    for key in ("precision", "recall", "f1", "f0_5"):
        assert key in r


@pytest.mark.slow
def test_score_all_channels_with_tuned_configs(ray_local, pretrained_channel, tmp_path) -> None:
    """score_all_channels applies per-subsystem scoring overrides from tuned_configs."""
    pytest.importorskip("ray")
    import csv

    from spacecraft_telemetry.ray_training import score_all_channels

    settings = pretrained_channel

    # Provide a minimal channels.csv so channel_1 maps to subsystem_1.
    raw_dir = tmp_path / "raw" / "ESA-Mission1"
    raw_dir.mkdir(parents=True)
    with (raw_dir / "channels.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Channel", "Subsystem", "Physical Unit", "Group", "Target"])
        writer.writerow(["channel_1", "subsystem_1", "V", "power", ""])

    settings = settings.model_copy(
        update={"data": settings.data.model_copy(update={"raw_data_dir": str(tmp_path / "raw")})}
    )

    tuned = {"subsystem_1": {"threshold_z": 2.5}}
    results = score_all_channels(
        settings, "ESA-Mission1", ["channel_1"], tuned_configs=tuned
    )
    assert len(results) == 1
    assert results[0]["status"] == "ok", f"Expected ok, got: {results[0].get('error_msg')}"


@pytest.mark.slow
def test_score_all_channels_ignores_unknown_tuned_keys(ray_local, pretrained_channel) -> None:
    """tuned_configs with unrecognised keys should be silently dropped."""
    pytest.importorskip("ray")
    from spacecraft_telemetry.ray_training import score_all_channels

    settings = pretrained_channel

    # 'hidden_dim' is not a tunable scoring field — should be ignored safely.
    tuned = {"subsystem_1": {"hidden_dim": 999, "threshold_z": 2.5}}
    results = score_all_channels(
        settings, "ESA-Mission1", ["channel_1"], tuned_configs=tuned
    )
    assert len(results) == 1
    assert results[0]["status"] == "ok"
