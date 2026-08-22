"""Unit and integration tests for ray_fanout/runner.py."""

from __future__ import annotations

from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# discover_channels
# ---------------------------------------------------------------------------


def test_discover_channels_finds_channel(tmp_path) -> None:
    """discover_channels returns sorted channel IDs from Hive partition dirs."""
    from spacecraft_telemetry.core.config import load_settings
    from spacecraft_telemetry.ray_fanout import discover_channels

    settings = load_settings("test")
    mission = "ESA-Mission1"
    processed_dir = tmp_path / "processed"

    # Create two fake channel partition directories.
    for ch in ("channel_3", "channel_1"):
        part = processed_dir / mission / "train" / f"mission_id={mission}" / f"channel_id={ch}"
        part.mkdir(parents=True)

    settings = settings.model_copy(
        update={
            "preprocess": settings.preprocess.model_copy(
                update={"processed_data_dir": str(processed_dir)}
            )
        }
    )
    channels = discover_channels(settings, mission)
    assert channels == ["channel_1", "channel_3"]


def test_discover_channels_empty_when_no_processed_data() -> None:
    """discover_channels returns [] when mission dir doesn't exist."""
    from spacecraft_telemetry.core.config import load_settings
    from spacecraft_telemetry.ray_fanout import discover_channels

    settings = load_settings("test")
    channels = discover_channels(settings, "ESA-NonExistent")
    assert channels == []


def test_discover_channels_ignores_non_channel_dirs(tmp_path) -> None:
    """discover_channels skips dirs that don't start with 'channel_id='."""
    from spacecraft_telemetry.core.config import load_settings
    from spacecraft_telemetry.ray_fanout import discover_channels

    settings = load_settings("test")
    mission = "ESA-Mission1"
    processed_dir = tmp_path / "processed"
    base = processed_dir / mission / "train" / f"mission_id={mission}"
    (base / "channel_id=channel_1").mkdir(parents=True)
    (base / "_SUCCESS").mkdir(parents=True)           # should be ignored
    (base / "some_other_dir").mkdir(parents=True)     # should be ignored

    settings = settings.model_copy(
        update={
            "preprocess": settings.preprocess.model_copy(
                update={"processed_data_dir": str(processed_dir)}
            )
        }
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
    from spacecraft_telemetry.ray_fanout import train_all_channels

    settings = ray_series_parquet
    results = train_all_channels(
        settings, "ESA-Mission1", ["channel_1", "nonexistent_channel"]
    )

    assert len(results) == 2
    statuses = {r["channel"]: r["status"] for r in results}
    assert statuses["channel_1"] == "ok"
    assert statuses["nonexistent_channel"] == "error"


@pytest.mark.slow
def test_train_all_channels_max_channels_cap(ray_local, ray_series_parquet) -> None:
    """max_channels=1 with two candidates caps the sweep to the first channel."""
    pytest.importorskip("ray")
    from spacecraft_telemetry.ray_fanout import train_all_channels

    results = train_all_channels(
        ray_series_parquet,
        "ESA-Mission1",
        ["channel_1", "nonexistent_channel"],
        max_channels=1,
    )
    assert len(results) == 1
    assert results[0]["channel"] == "channel_1"


@pytest.mark.slow
def test_score_all_channels_ok(ray_local, pretrained_channel) -> None:
    """score_all_channels returns metrics after training."""
    pytest.importorskip("ray")
    from spacecraft_telemetry.ray_fanout import score_all_channels

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
    import json

    from spacecraft_telemetry.ray_fanout import score_all_channels

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

    # Verify the override was actually written to the MLflow artifact (A1: no filesystem writes).
    from spacecraft_telemetry.mlflow_tracking.conventions import experiment_name
    from spacecraft_telemetry.model.io import download_artifact_bytes, find_latest_run_for_channel

    tracking_uri = settings.mlflow.tracking_uri
    scoring_exp = experiment_name(settings.model.model_type, "scoring", "ESA-Mission1")
    run = find_latest_run_for_channel(scoring_exp, "channel_1", tracking_uri)
    assert run is not None, "No scoring run found in MLflow"
    cfg_bytes = download_artifact_bytes(
        run.info.run_id, "threshold_config.json", tracking_uri, use_cache=False
    )
    saved = json.loads(cfg_bytes.decode())
    assert saved["z"] == pytest.approx(2.5), (
        f"Expected z=2.5 in threshold_config.json, got: {saved}"
    )


@pytest.mark.slow
def test_score_all_channels_ignores_unknown_tuned_keys(ray_local, pretrained_channel) -> None:
    """tuned_configs with unrecognised keys should be silently dropped."""
    pytest.importorskip("ray")
    from spacecraft_telemetry.ray_fanout import score_all_channels

    settings = pretrained_channel

    # 'hidden_dim' is not a tunable scoring field — should be ignored safely.
    tuned = {"subsystem_1": {"hidden_dim": 999, "threshold_z": 2.5}}
    results = score_all_channels(
        settings, "ESA-Mission1", ["channel_1"], tuned_configs=tuned
    )
    assert len(results) == 1
    assert results[0]["status"] == "ok"


@pytest.mark.slow
def test_score_all_channels_tuned_configs_requires_subsystem_map(
    ray_local, pretrained_channel, tmp_path
) -> None:
    """Passing tuned_configs without a subsystem map should fail fast."""
    pytest.importorskip("ray")
    from spacecraft_telemetry.ray_fanout import score_all_channels

    settings = pretrained_channel.model_copy(
        update={
            "data": pretrained_channel.data.model_copy(
                update={"raw_data_dir": tmp_path / "raw"}
            ),
            "preprocess": pretrained_channel.preprocess.model_copy(
                update={"processed_data_dir": tmp_path / "processed"}
            ),
        }
    )
    tuned = {"subsystem_1": {"threshold_z": 2.5}}

    with pytest.raises(ValueError, match="no channel->subsystem map"):
        score_all_channels(settings, "ESA-Mission1", ["channel_1"], tuned_configs=tuned)


# ---------------------------------------------------------------------------
# _with_abs_paths — unit tests
# ---------------------------------------------------------------------------


def test_with_abs_paths_resolves_all_paths(tmp_path) -> None:
    """_with_abs_paths converts all three path fields to absolute paths."""
    from spacecraft_telemetry.core.config import load_settings
    from spacecraft_telemetry.ray_fanout.runner import _with_abs_paths

    settings = load_settings("test")
    # Override with known relative paths so the assertion is meaningful.
    rel = Path("some/relative/path")
    settings = settings.model_copy(
        update={
            "preprocess": settings.preprocess.model_copy(update={"processed_data_dir": rel}),
            "model": settings.model.model_copy(update={"artifacts_dir": rel}),
            "data": settings.data.model_copy(update={"raw_data_dir": rel}),
        }
    )

    result = _with_abs_paths(settings)

    assert result.preprocess.processed_data_dir.is_absolute(), (
        "processed_data_dir should be absolute"
    )
    assert result.model.artifacts_dir.is_absolute(), "artifacts_dir should be absolute"
    assert result.data.raw_data_dir.is_absolute(), "raw_data_dir should be absolute"


# ---------------------------------------------------------------------------
# Multivariate fan-out (docs/plans/021-multivariate-telemanom.md)
# ---------------------------------------------------------------------------


def test_group_channels_by_subsystem_groups_and_orders(tmp_path) -> None:
    """Channels group by subsystem; each group's order matches the input list."""
    import json

    from spacecraft_telemetry.core.config import load_settings
    from spacecraft_telemetry.ray_fanout.runner import _group_channels_by_subsystem

    mission = "ESA-Mission1"
    processed_dir = tmp_path / "processed"
    metadata_dir = processed_dir / mission / "metadata"
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "channel_subsystems.json").write_text(
        json.dumps({
            "channel_42": "subsystem_1", "channel_41": "subsystem_1", "channel_9": "subsystem_2",
        })
    )
    settings = load_settings("test").model_copy(
        update={"preprocess": load_settings("test").preprocess.model_copy(
            update={"processed_data_dir": str(processed_dir)}
        )}
    )

    groups = _group_channels_by_subsystem(
        settings, mission, ["channel_41", "channel_9", "channel_42"]
    )

    assert groups == {
        "subsystem_1": ["channel_41", "channel_42"],
        "subsystem_2": ["channel_9"],
    }


def test_group_channels_by_subsystem_drops_unmapped(tmp_path) -> None:
    """A channel with no subsystem entry is dropped, not silently grouped."""
    import json

    from spacecraft_telemetry.core.config import load_settings
    from spacecraft_telemetry.ray_fanout.runner import _group_channels_by_subsystem

    mission = "ESA-Mission1"
    processed_dir = tmp_path / "processed"
    metadata_dir = processed_dir / mission / "metadata"
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "channel_subsystems.json").write_text(
        json.dumps({"channel_41": "subsystem_1"})
    )
    settings = load_settings("test").model_copy(
        update={"preprocess": load_settings("test").preprocess.model_copy(
            update={"processed_data_dir": str(processed_dir)}
        )}
    )

    groups = _group_channels_by_subsystem(settings, mission, ["channel_41", "channel_unmapped"])

    assert groups == {"subsystem_1": ["channel_41"]}


def test_group_channels_prefers_measured_group_map(tmp_path) -> None:
    """A measured channel_groups.json overrides the subsystem map.

    docs/plans/023: joint modelling needs a shared timestamp grid, and the
    natural ESA group crosses subsystem boundaries — so the grouping key is a
    measured group id, with the subsystem name as the fallback.
    """
    import json

    from spacecraft_telemetry.core.config import load_settings
    from spacecraft_telemetry.ray_fanout.runner import _group_channels_by_subsystem

    mission = "ESA-Mission1"
    processed_dir = tmp_path / "processed"
    metadata_dir = processed_dir / mission / "metadata"
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "channel_subsystems.json").write_text(
        json.dumps({
            "channel_41": "subsystem_5", "channel_47": "subsystem_6", "channel_9": "subsystem_2",
        })
    )
    (metadata_dir / "channel_groups.json").write_text(
        json.dumps({"channel_41": "group_01", "channel_47": "group_01"})
    )
    settings = load_settings("test").model_copy(
        update={"preprocess": load_settings("test").preprocess.model_copy(
            update={"processed_data_dir": str(processed_dir)}
        )}
    )

    groups = _group_channels_by_subsystem(
        settings, mission, ["channel_41", "channel_9", "channel_47"]
    )

    # channel_41 and channel_47 join across subsystems; channel_9 is absent from
    # the group map (a singleton family) and drops out of the multivariate sweep.
    assert groups == {"group_01": ["channel_41", "channel_47"]}


@pytest.mark.slow
def test_train_all_subsystems_ok(ray_local, ray_series_parquet_multichannel) -> None:
    """train_all_subsystems trains one joint model per subsystem group."""
    pytest.importorskip("ray")
    from spacecraft_telemetry.ray_fanout.runner import train_all_subsystems

    settings = ray_series_parquet_multichannel
    results = train_all_subsystems(settings, "ESA-Mission1", ["channel_41", "channel_42"])

    assert len(results) == 1
    r = results[0]
    assert r["status"] == "ok", f"Expected ok, got: {r.get('error_msg')}"
    assert r["channel"] == "subsystem_1"
    assert isinstance(r["best_epoch"], int)


@pytest.mark.slow
def test_train_all_subsystems_registers_joint_model(
    ray_local, ray_series_parquet_multichannel
) -> None:
    """The registered model is keyed by subsystem, and its saved n_channels == 2."""
    pytest.importorskip("ray")
    from spacecraft_telemetry.mlflow_tracking.conventions import registered_model_name
    from spacecraft_telemetry.ray_fanout.runner import train_all_subsystems

    settings = ray_series_parquet_multichannel
    train_all_subsystems(settings, "ESA-Mission1", ["channel_41", "channel_42"])

    import mlflow

    mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
    client = mlflow.tracking.MlflowClient()
    model_name = registered_model_name(settings.model.model_type, "ESA-Mission1", "subsystem_1")
    versions = list(client.search_model_versions(f"name='{model_name}'"))
    assert len(versions) >= 1, f"no registered versions found for {model_name!r}"
    mv = versions[0]
    assert mv.tags.get("subsystem_id") == "subsystem_1"
    assert mv.tags.get("channels") == "channel_41,channel_42"
    assert "channel_id" not in mv.tags


@pytest.mark.slow
def test_score_all_subsystems_ok(ray_local, ray_series_parquet_multichannel) -> None:
    """score_all_subsystems returns per-channel metrics nested under the subsystem."""
    pytest.importorskip("ray")
    from spacecraft_telemetry.ray_fanout.runner import score_all_subsystems, train_all_subsystems

    settings = ray_series_parquet_multichannel
    train_all_subsystems(settings, "ESA-Mission1", ["channel_41", "channel_42"])
    results = score_all_subsystems(settings, "ESA-Mission1", ["channel_41", "channel_42"])

    assert len(results) == 1
    r = results[0]
    assert r["status"] == "ok", f"Expected ok, got: {r.get('error_msg')}"
    assert r["channel"] == "subsystem_1"
    for ch in ("channel_41", "channel_42"):
        assert ch in r, f"expected per-channel metrics dict under key {ch!r} in {r.keys()}"
        assert "precision" in r[ch]
        assert "f0_5" in r[ch]


@pytest.mark.slow
def test_score_all_subsystems_applies_tuned_configs(
    ray_local, ray_series_parquet_multichannel
) -> None:
    """tuned_configs keyed by subsystem apply directly to the group's settings."""
    pytest.importorskip("ray")
    import json

    import mlflow

    from spacecraft_telemetry.mlflow_tracking.conventions import experiment_name
    from spacecraft_telemetry.model.io import download_artifact_bytes
    from spacecraft_telemetry.ray_fanout.runner import score_all_subsystems, train_all_subsystems

    settings = ray_series_parquet_multichannel
    train_all_subsystems(settings, "ESA-Mission1", ["channel_41", "channel_42"])

    tuned = {"subsystem_1": {"threshold_z": 2.5}}
    results = score_all_subsystems(
        settings, "ESA-Mission1", ["channel_41", "channel_42"], tuned_configs=tuned
    )
    assert results[0]["status"] == "ok", f"Expected ok, got: {results[0].get('error_msg')}"

    # A multivariate run has no channel_id tag (see model/scoring.py), so —
    # unlike the univariate test above — look it up by the subsystem tag
    # instead of find_latest_run_for_channel.
    tracking_uri = settings.mlflow.tracking_uri
    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    scoring_exp = experiment_name(settings.model.model_type, "scoring", "ESA-Mission1")
    exp = client.get_experiment_by_name(scoring_exp)
    assert exp is not None
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string="tags.subsystem = 'subsystem_1'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    assert runs, "No scoring run found in MLflow for subsystem_1"
    cfg_bytes = download_artifact_bytes(
        runs[0].info.run_id, "threshold_config.json", tracking_uri, use_cache=False
    )
    saved = json.loads(cfg_bytes.decode())
    assert saved["z"] == pytest.approx(2.5)


def test_min_error_value_is_a_tunable_scoring_field() -> None:
    """min_error_value must survive the tuned_configs whitelist.

    It is filtered by _TUNABLE_SCORING_FIELDS before being applied to Settings.
    If it were absent, a tuned config carrying an absolute error floor would be
    silently dropped and the tuned scoring run would quietly score WITHOUT the
    floor while reporting itself as tuned — invalidating any comparison against
    ESA-ADB's Telemanom-ESA-Pruned. Guard the whitelist membership explicitly.

    Its sibling prune_min_decrease must stay OUT: that one is Hundman §3.3,
    retrospective and not reproducible in the streaming path.
    """
    from spacecraft_telemetry.ray_fanout.runner import _TUNABLE_SCORING_FIELDS

    assert "min_error_value" in _TUNABLE_SCORING_FIELDS
    assert "prune_min_decrease" not in _TUNABLE_SCORING_FIELDS


class TestTunedMeta:
    """The `_meta` provenance reader, shared by both score fan-outs.

    Extracted from two copies (docs/reviews/021 item B4) — the subsystem one
    carried a comment pointing at the channel one, which acknowledged the
    duplication rather than fixing it. Timed for extraction because plan 022.2b
    adds provenance fields, which is now a one-place edit.
    """

    def test_reads_both_fields(self) -> None:
        from spacecraft_telemetry.ray_fanout.runner import _tuned_meta

        assert _tuned_meta({"_meta": {"run_id": "abc123", "source": "ray tune"}}) == (
            "abc123",
            "ray tune",
        )

    def test_grid_config_has_source_but_no_run_id(self) -> None:
        """scripts/threshold_ceiling.py writes provenance with NO run_id —
        there is no Ray Tune run to point at, and fabricating one would corrupt
        the tuned_from_run lineage tag."""
        from spacecraft_telemetry.ray_fanout.runner import _tuned_meta

        run_id, source = _tuned_meta(
            {"_meta": {"source": "scripts/threshold_ceiling.py exhaustive grid"}}
        )
        assert run_id is None
        assert source == "scripts/threshold_ceiling.py exhaustive grid"

    @pytest.mark.parametrize(
        "entry",
        [None, {}, {"threshold_z": 3.0}, {"_meta": None}, {"_meta": "not-a-dict"}],
        ids=["none", "empty", "no-meta", "null-meta", "malformed-meta"],
    )
    def test_missing_or_malformed_meta_yields_no_provenance(self, entry: object) -> None:
        """Absent provenance must degrade to (None, None) rather than raise —
        an untuned channel is an ordinary case, not an error."""
        from spacecraft_telemetry.ray_fanout.runner import _tuned_meta

        assert _tuned_meta(entry) == (None, None)  # type: ignore[arg-type]

    def test_values_are_coerced_to_str(self) -> None:
        """score_channel writes these straight into MLflow tags, which are
        strings — a numeric run_id from hand-edited JSON must not leak through
        as an int."""
        from spacecraft_telemetry.ray_fanout.runner import _tuned_meta

        assert _tuned_meta({"_meta": {"run_id": 12345}}) == ("12345", None)
