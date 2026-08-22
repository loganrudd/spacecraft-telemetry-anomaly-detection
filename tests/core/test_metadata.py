"""Tests for core.metadata.load_channel_subsystem_map.

core/metadata.py had no test file until this one — every appearance of
load_channel_subsystem_map in tests/test_cli.py mocks it out. The one thing
worth pinning here is the `variant` element of the `@lru_cache` key on
`_load_cached`: under Ray fan-out, each worker process caches its own copy of
the subsystem map, and if the cache key ever dropped `variant`, a worker that
had already served one variant's map would silently serve it again for a
different variant in the same process — flowing into wrong tuned-config
overrides (ray_fanout/runner.py) and silently wrong scoring thresholds. See
docs/reviews/020-experiment-variant-axis.md item T2.
"""

from __future__ import annotations

import json
from pathlib import Path

from spacecraft_telemetry.core.config import Settings
from spacecraft_telemetry.core.metadata import (
    _load_cached,
    load_channel_group_map,
    load_channel_subsystem_map,
)


def _settings(tmp_path: Path, variant: str | None = None) -> Settings:
    settings = Settings(variant=variant)
    settings.preprocess.processed_data_dir = str(tmp_path / "processed")
    settings.data.sample_data_dir = str(tmp_path / "sample")
    settings.data.raw_data_dir = str(tmp_path / "raw")
    return settings


def _write_subsystem_map(
    tmp_path: Path, mission: str, variant: str | None, mapping: dict[str, str]
) -> Path:
    base = tmp_path / "processed" / mission
    if variant:
        base = base / variant
    metadata_dir = base / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    path = metadata_dir / "channel_subsystems.json"
    path.write_text(json.dumps(mapping))
    return path


class TestVariantCacheIsolation:
    """Pins the variant element of _load_cached's key, not just the outcome —
    an outcome-only assertion could be satisfied by a change that disables
    caching entirely, which would silently undo the bound this cache exists
    to provide (one file read per process, not per call)."""

    def test_base_and_variant_maps_do_not_contaminate_each_other(
        self, tmp_path: Path
    ) -> None:
        mission = "ESA-Mission1"
        _write_subsystem_map(tmp_path, mission, None, {"channel_1": "power"})
        _write_subsystem_map(tmp_path, mission, "arm-a", {"channel_1": "thermal"})

        base_settings = _settings(tmp_path, variant=None)
        variant_settings = _settings(tmp_path, variant="arm-a")

        base_map = load_channel_subsystem_map(base_settings, mission)
        variant_map = load_channel_subsystem_map(variant_settings, mission)

        assert base_map == {"channel_1": "power"}
        assert variant_map == {"channel_1": "thermal"}

    def test_second_variant_call_is_a_cache_miss_not_a_hit(self, tmp_path: Path) -> None:
        """Pins the cache KEY. If `variant` were ever dropped from the
        _load_cached signature, this call would (wrongly) hit the cache
        entry populated by the first call instead of reading its own file."""
        mission = "ESA-Mission1"
        _write_subsystem_map(tmp_path, mission, None, {"channel_1": "power"})
        _write_subsystem_map(tmp_path, mission, "arm-a", {"channel_1": "thermal"})

        base_settings = _settings(tmp_path, variant=None)
        variant_settings = _settings(tmp_path, variant="arm-a")

        _load_cached.cache_clear()
        load_channel_subsystem_map(base_settings, mission)
        info_after_first = _load_cached.cache_info()
        assert info_after_first.misses == 1

        load_channel_subsystem_map(variant_settings, mission)
        info_after_second = _load_cached.cache_info()
        assert info_after_second.misses == 2
        assert info_after_second.hits == info_after_first.hits


class TestInjectedFallbackUnderVariant:
    """The `_injected` nominal-parent fallback (metadata.py:65-75) must still
    resolve the variant segment when unwinding to the nominal processed dir —
    formalized by docs/reviews/020-experiment-variant-axis.md item A3, which
    otherwise leaves `_injected` combined with variant unimplemented."""

    def test_injected_dir_falls_back_to_nominal_variant_map(self, tmp_path: Path) -> None:
        mission = "ISS"
        _write_subsystem_map(tmp_path, mission, "arm-a", {"S1000003": "thermal"})

        settings = _settings(tmp_path, variant="arm-a")
        settings.preprocess.processed_data_dir = str(
            Path(settings.preprocess.processed_data_dir) / "_injected"
        )

        result = load_channel_subsystem_map(settings, mission)

        assert result == {"S1000003": "thermal"}


class TestCsvFallback:
    def test_falls_back_to_channels_csv_when_no_processed_map(self, tmp_path: Path) -> None:
        mission = "ESA-Mission1"
        settings = _settings(tmp_path, variant="arm-a")
        sample_dir = tmp_path / "sample" / mission
        sample_dir.mkdir(parents=True)
        (sample_dir / "channels.csv").write_text(
            "Channel,Subsystem\nchannel_1,power\n"
        )

        result = load_channel_subsystem_map(settings, mission)

        assert result == {"channel_1": "power"}

    def test_returns_empty_dict_when_nothing_exists(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path, variant="arm-a")
        assert load_channel_subsystem_map(settings, "ESA-Mission1") == {}


# ---------------------------------------------------------------------------
# load_channel_group_map — docs/plans/023 stage .4
# ---------------------------------------------------------------------------


def _write_group_map(
    tmp_path: Path, mission: str, variant: str | None, mapping: dict[str, str]
) -> Path:
    base = tmp_path / "processed" / mission
    if variant:
        base = base / variant
    metadata_dir = base / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    path = metadata_dir / "channel_groups.json"
    path.write_text(json.dumps(mapping))
    return path


class TestLoadChannelGroupMap:
    def test_falls_back_to_subsystem_map_when_absent(self, tmp_path: Path) -> None:
        # Every mission without a measured grouping — ISS, and every ESA tree
        # predating plan 023 — must keep grouping exactly as it does today.
        mission = "ESA-Mission1"
        _write_subsystem_map(tmp_path, mission, None, {"channel_1": "power"})
        assert load_channel_group_map(_settings(tmp_path), mission) == {"channel_1": "power"}

    def test_group_map_wins_when_present(self, tmp_path: Path) -> None:
        # A measured group crosses subsystem boundaries — that is the point.
        mission = "ESA-Mission1"
        _write_subsystem_map(
            tmp_path, mission, None, {"channel_41": "subsystem_5", "channel_47": "subsystem_6"}
        )
        _write_group_map(
            tmp_path, mission, None, {"channel_41": "group_01", "channel_47": "group_01"}
        )
        assert load_channel_group_map(_settings(tmp_path), mission) == {
            "channel_41": "group_01",
            "channel_47": "group_01",
        }

    def test_empty_group_map_falls_back(self, tmp_path: Path) -> None:
        mission = "ESA-Mission1"
        _write_subsystem_map(tmp_path, mission, None, {"channel_1": "power"})
        _write_group_map(tmp_path, mission, None, {})
        assert load_channel_group_map(_settings(tmp_path), mission) == {"channel_1": "power"}

    def test_returns_empty_when_neither_source_exists(self, tmp_path: Path) -> None:
        assert load_channel_group_map(_settings(tmp_path), "ESA-Mission1") == {}

    def test_group_map_is_variant_scoped(self, tmp_path: Path) -> None:
        # The grid variant has its own grouping; the native tree must not see it.
        mission = "ESA-Mission1"
        _write_subsystem_map(tmp_path, mission, None, {"channel_70": "subsystem_3"})
        _write_subsystem_map(tmp_path, mission, "grid-30s", {"channel_70": "subsystem_3"})
        _write_group_map(tmp_path, mission, "grid-30s", {"channel_70": "group_01"})

        assert load_channel_group_map(_settings(tmp_path), mission) == {
            "channel_70": "subsystem_3"
        }
        assert load_channel_group_map(_settings(tmp_path, variant="grid-30s"), mission) == {
            "channel_70": "group_01"
        }
