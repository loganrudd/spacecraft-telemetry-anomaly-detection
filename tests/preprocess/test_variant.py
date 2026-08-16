"""Variant-axis integration tests for the preprocessing pipeline (Plan 020).

Uses the `settings` / `pipeline_input_dir` fixtures from conftest.py. These
cover the three properties the plan calls out as load-bearing:

  * Variant isolation — preprocessing two variants of the same mission does
    not clobber each other's train/test output (the destructive clear must
    be scoped to {mission}/{variant}/{split} only).
  * mission_id correctness — the Hive partition column carries the real
    mission under every variant; only the *path* changes.
  * No input duplication — raw/sample reads are never variant-scoped; one
    copy of raw input data serves every variant.

Null-default equivalence (variant=None reproduces today's paths byte for
byte) is covered by every other test in this directory, none of which sets
`variant` — plus tests/core/test_paths.py for output_path() directly.
"""

from __future__ import annotations

import json
from pathlib import Path

from spacecraft_telemetry.core.config import Settings
from spacecraft_telemetry.preprocess.pipeline import run_preprocessing

_MISSION = "ESA-Mission1"


class TestVariantIsolation:
    def test_two_variants_both_produce_output(self, settings: Settings) -> None:
        variant_a = settings.model_copy(
            update={"variant": "arm-a"}, deep=True
        )
        variant_b = settings.model_copy(
            update={"variant": "arm-b"}, deep=True
        )
        run_preprocessing(variant_a, _MISSION, parallel=False)
        run_preprocessing(variant_b, _MISSION, parallel=False)

        out = Path(settings.preprocess.processed_data_dir)
        assert (out / _MISSION / "arm-a" / "train").is_dir()
        assert (out / _MISSION / "arm-b" / "train").is_dir()

    def test_rerunning_one_variant_does_not_clear_the_other(self, settings: Settings) -> None:
        variant_a = settings.model_copy(update={"variant": "arm-a"}, deep=True)
        variant_b = settings.model_copy(update={"variant": "arm-b"}, deep=True)
        run_preprocessing(variant_a, _MISSION, parallel=False)
        run_preprocessing(variant_b, _MISSION, parallel=False)

        # Re-run arm-a only. The destructive clear in run_preprocessing must
        # only rm() {mission}/arm-a/{split}, never touch {mission}/arm-b/{split}.
        run_preprocessing(variant_a, _MISSION, parallel=False)

        out = Path(settings.preprocess.processed_data_dir)
        assert (out / _MISSION / "arm-b" / "train").is_dir()
        assert any((out / _MISSION / "arm-b" / "train").rglob("*.parquet"))

    def test_variant_none_and_variant_set_coexist(self, settings: Settings) -> None:
        # A base (variant=None) run plus a variant run for the same mission
        # must not collide — this is what kills the raw-data duplication
        # (plan 019's ESA-Mission1-ADB* pseudo-missions).
        run_preprocessing(settings, _MISSION, parallel=False)
        variant_a = settings.model_copy(update={"variant": "arm-a"}, deep=True)
        run_preprocessing(variant_a, _MISSION, parallel=False)

        out = Path(settings.preprocess.processed_data_dir)
        assert (out / _MISSION / "train").is_dir()          # base, unchanged layout
        assert (out / _MISSION / "arm-a" / "train").is_dir()  # variant, separate namespace

    def test_normalization_params_are_per_variant(self, settings: Settings) -> None:
        variant_a = settings.model_copy(update={"variant": "arm-a"}, deep=True)
        run_preprocessing(variant_a, _MISSION, parallel=False)

        out = Path(settings.preprocess.processed_data_dir)
        assert (out / _MISSION / "arm-a" / "normalization_params.json").exists()
        assert not (out / _MISSION / "normalization_params.json").exists()


class TestMissionIdCorrectness:
    def test_hive_partition_key_carries_real_mission_under_variant(
        self, settings: Settings
    ) -> None:
        # write_series() drops mission_id/channel_id as data columns — they
        # are Hive-partition-encoded in the directory name only (see
        # .claude/rules/preprocess.md). The correctness property the plan
        # cares about is that this partition key is the REAL mission
        # ("ESA-Mission1"), never a pseudo-mission like "ESA-Mission1-arm-a" —
        # variant is a path segment inserted *before* the Hive layout, not
        # baked into it.
        variant = settings.model_copy(update={"variant": "arm-a"}, deep=True)
        run_preprocessing(variant, _MISSION, parallel=False)

        out = Path(settings.preprocess.processed_data_dir)
        partition_dir = (
            out / _MISSION / "arm-a" / "train"
            / f"mission_id={_MISSION}" / "channel_id=channel_1"
        )
        assert partition_dir.is_dir()
        # No pseudo-mission directory was created alongside it.
        assert not (out / f"{_MISSION}-arm-a").exists()

    def test_partition_read_back_via_dataset_api(self, settings: Settings) -> None:
        # The public read API (model.dataset.load_series_parquet) must resolve
        # the same variant-scoped partition the pipeline wrote.
        from spacecraft_telemetry.model.dataset import load_series_parquet

        variant = settings.model_copy(update={"variant": "arm-a"}, deep=True)
        run_preprocessing(variant, _MISSION, parallel=False)

        values, _, _, _ = load_series_parquet(
            variant.preprocess.processed_data_dir, _MISSION, "channel_1", "train",
            variant=variant.variant,
        )
        assert len(values) == 80


class TestNoInputDuplication:
    def test_raw_read_path_is_unaffected_by_variant(
        self, settings: Settings, pipeline_input_dir: Path
    ) -> None:
        # Both variants must read from the SAME raw channel file — one copy
        # of raw input data serves every variant of a mission.
        variant_a = settings.model_copy(update={"variant": "arm-a"}, deep=True)
        variant_b = settings.model_copy(update={"variant": "arm-b"}, deep=True)

        summary_a = run_preprocessing(variant_a, _MISSION, parallel=False)
        summary_b = run_preprocessing(variant_b, _MISSION, parallel=False)

        # Same raw input -> same row counts, regardless of variant.
        assert summary_a["rows_in"] == summary_b["rows_in"]
        # The only raw channel file on disk lives at the mission-only path —
        # no variant-scoped copy was created.
        raw_channels_dir = pipeline_input_dir / _MISSION / "channels"
        assert list(raw_channels_dir.glob("*.parquet")) == [
            raw_channels_dir / "channel_1.parquet"
        ]
        assert not (pipeline_input_dir / _MISSION / "arm-a").exists()
        assert not (pipeline_input_dir / _MISSION / "arm-b").exists()


class TestTunedConfigsArtifactPath:
    """artifacts_dir / mission / [variant /] tuned_configs.json (cli.py, tune.py)."""

    def test_write_tuned_configs_respects_variant_path(self, tmp_path: Path) -> None:
        from spacecraft_telemetry.core.paths import output_path
        from spacecraft_telemetry.ray_fanout.tune import write_tuned_configs

        base_out = output_path(tmp_path, _MISSION, None, "tuned_configs.json")
        variant_out = output_path(tmp_path, _MISSION, "arm-a", "tuned_configs.json")

        write_tuned_configs({"subsystem_1": {"threshold_z": 3.0}}, base_out)
        write_tuned_configs({"subsystem_1": {"threshold_z": 4.0}}, variant_out)

        assert json.loads(base_out.read_text())["subsystem_1"]["threshold_z"] == 3.0
        assert json.loads(variant_out.read_text())["subsystem_1"]["threshold_z"] == 4.0
