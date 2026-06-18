"""Tests for ingest.iss_channels — registry integrity checks."""

from __future__ import annotations

from spacecraft_telemetry.ingest.iss_channels import (
    CONTEXT_ITEMS,
    ISS_CHANNELS,
    SUBSYSTEMS,
    VALIDATION_CHANNELS,
    ChannelMeta,
    subscription_items,
)

_EXPECTED_SUBSYSTEMS = {"power", "solar_array", "thermal", "attitude"}
_EXPECTED_COUNT = 26


def test_channel_count() -> None:
    assert len(ISS_CHANNELS) == _EXPECTED_COUNT


def test_subsystems_are_valid() -> None:
    for pui, meta in ISS_CHANNELS.items():
        assert meta.subsystem in _EXPECTED_SUBSYSTEMS, (
            f"{pui} has unknown subsystem {meta.subsystem!r}"
        )


def test_subsystems_constant_matches_channels() -> None:
    assert SUBSYSTEMS == _EXPECTED_SUBSYSTEMS


def test_all_channels_have_nonempty_fields() -> None:
    for pui, meta in ISS_CHANNELS.items():
        assert pui.strip(), "PUI must not be empty"
        assert meta.description.strip(), f"{pui}: description must not be empty"
        assert meta.unit.strip(), f"{pui}: unit must not be empty"
        assert meta.subsystem.strip(), f"{pui}: subsystem must not be empty"


def test_validation_channels_subset_of_iss_channels() -> None:
    for pui in VALIDATION_CHANNELS:
        assert pui in ISS_CHANNELS, f"Validation channel {pui!r} not in ISS_CHANNELS"


def test_validation_channel_count() -> None:
    assert len(VALIDATION_CHANNELS) == 6


def test_validation_channels_unique() -> None:
    assert len(VALIDATION_CHANNELS) == len(set(VALIDATION_CHANNELS))


def test_context_items_disjoint_from_channels() -> None:
    overlap = set(CONTEXT_ITEMS) & set(ISS_CHANNELS)
    assert not overlap, f"Context items overlap with telemetry channels: {overlap}"


def test_context_items_nonempty() -> None:
    assert "TIME_000001" in CONTEXT_ITEMS
    assert "USLAB000086" in CONTEXT_ITEMS


def test_sarj_and_soc_excluded() -> None:
    excluded = {"S0000003", "S0000004"}  # SARJ wrap channels
    assert not (excluded & set(ISS_CHANNELS))


def test_subscription_items_validation_includes_context() -> None:
    items = subscription_items("validation")
    for ctx in CONTEXT_ITEMS:
        assert ctx in items


def test_subscription_items_validation_includes_validation_channels() -> None:
    items = subscription_items("validation")
    for pui in VALIDATION_CHANNELS:
        assert pui in items


def test_subscription_items_validation_excludes_non_validation_telemetry() -> None:
    items = set(subscription_items("validation"))
    non_validation = set(ISS_CHANNELS) - set(VALIDATION_CHANNELS)
    overlap = items & non_validation
    assert not overlap, f"Validation set leaked non-validation channels: {overlap}"


def test_subscription_items_all_includes_all_channels() -> None:
    items = subscription_items("all")
    for pui in ISS_CHANNELS:
        assert pui in items
    for ctx in CONTEXT_ITEMS:
        assert ctx in items


def test_subscription_items_are_sorted() -> None:
    for channel_set in ("validation", "all"):
        items = subscription_items(channel_set)  # type: ignore[arg-type]
        assert items == sorted(items), f"{channel_set}: subscription_items not sorted"


def test_subscription_items_no_duplicates() -> None:
    for channel_set in ("validation", "all"):
        items = subscription_items(channel_set)  # type: ignore[arg-type]
        assert len(items) == len(set(items)), f"{channel_set}: duplicate items"


def test_channel_meta_is_namedtuple() -> None:
    meta = ISS_CHANNELS["S1000003"]
    assert isinstance(meta, ChannelMeta)
    assert meta.subsystem == "thermal"


def test_subsystem_counts() -> None:
    from collections import Counter
    counts = Counter(m.subsystem for m in ISS_CHANNELS.values())
    assert counts["power"] == 8
    assert counts["solar_array"] == 8
    assert counts["thermal"] == 4
    assert counts["attitude"] == 6
