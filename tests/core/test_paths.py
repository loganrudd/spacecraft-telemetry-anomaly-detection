"""Tests for core.paths — output_path() null-default equivalence (Plan 020).

variant=None must reproduce today's paths byte for byte; this is the entire
risk story for the experiment-variant axis (docs/plans/020-experiment-variant-axis.md).
"""

from __future__ import annotations

from upath import UPath

from spacecraft_telemetry.core.paths import output_path, to_upath


class TestOutputPathNullDefault:
    """variant=None must be string-identical to the pre-variant layout."""

    def test_no_parts(self) -> None:
        assert str(output_path("data/processed", "ESA-Mission1", None)) == (
            str(to_upath("data/processed") / "ESA-Mission1")
        )

    def test_single_part(self) -> None:
        assert str(output_path("data/processed", "ESA-Mission1", None, "train")) == (
            str(to_upath("data/processed") / "ESA-Mission1" / "train")
        )

    def test_multiple_parts_hive_partition(self) -> None:
        got = output_path(
            "data/processed", "ESA-Mission1", None,
            "train", "mission_id=ESA-Mission1", "channel_id=channel_1",
        )
        want = (
            to_upath("data/processed") / "ESA-Mission1"
            / "train" / "mission_id=ESA-Mission1" / "channel_id=channel_1"
        )
        assert str(got) == str(want)

    def test_gcs_root(self) -> None:
        got = output_path("gs://my-bucket-processed-data", "ISS", None, "train")
        want = to_upath("gs://my-bucket-processed-data") / "ISS" / "train"
        assert str(got) == str(want)

    def test_artifacts_dir_tuned_configs(self) -> None:
        got = output_path("models", "ESA-Mission1", None, "tuned_configs.json")
        want = to_upath("models") / "ESA-Mission1" / "tuned_configs.json"
        assert str(got) == str(want)

    def test_empty_string_variant_also_short_circuits(self) -> None:
        # Settings coerces "" to None before this is ever called, but the
        # helper itself treats any falsy variant identically to None.
        assert str(output_path("data/processed", "ESA-Mission1", "")) == (
            str(output_path("data/processed", "ESA-Mission1", None))
        )


class TestOutputPathWithVariant:
    def test_variant_inserted_between_mission_and_parts(self) -> None:
        got = output_path("data/processed", "ESA-Mission1", "adb-24m", "train")
        want = to_upath("data/processed") / "ESA-Mission1" / "adb-24m" / "train"
        assert str(got) == str(want)

    def test_variant_hive_partition_layout(self) -> None:
        got = output_path(
            "data/processed", "ESA-Mission1", "adb-24m",
            "train", "mission_id=ESA-Mission1", "channel_id=channel_41",
        )
        want = (
            to_upath("data/processed") / "ESA-Mission1" / "adb-24m"
            / "train" / "mission_id=ESA-Mission1" / "channel_id=channel_41"
        )
        assert str(got) == str(want)

    def test_different_variants_produce_disjoint_paths(self) -> None:
        a = output_path("data/processed", "ESA-Mission1", "adb-24m", "train")
        b = output_path("data/processed", "ESA-Mission1", "adb-84m", "train")
        assert str(a) != str(b)

    def test_variant_none_and_variant_set_are_disjoint(self) -> None:
        base = output_path("data/processed", "ESA-Mission1", None, "train")
        variant = output_path("data/processed", "ESA-Mission1", "adb-24m", "train")
        assert str(base) != str(variant)
        assert not str(variant).startswith(str(base) + "/mission_id")

    def test_returns_upath(self) -> None:
        assert isinstance(output_path("data/processed", "ESA-Mission1", "adb-24m"), UPath)
