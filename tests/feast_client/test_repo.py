"""Tests for feast_client.repo — schema and entity builders."""

from __future__ import annotations

import pytest

feast = pytest.importorskip("feast")

from feast.types import Float32, PrimitiveFeastType  # noqa: E402

from spacecraft_telemetry.feast_client.repo import (  # noqa: E402
    build_entities,
    build_schema_from_definitions,
    feast_dtype_for,
)
from spacecraft_telemetry.features.definitions import FEATURE_DEFINITIONS, get_feature_names  # noqa: E402


class TestBuildSchema:
    def test_schema_matches_registry(self) -> None:
        schema = build_schema_from_definitions(FEATURE_DEFINITIONS)
        schema_names = [f.name for f in schema]
        assert schema_names == get_feature_names()

    def test_all_fields_are_float32(self) -> None:
        schema = build_schema_from_definitions(FEATURE_DEFINITIONS)
        for field in schema:
            assert field.dtype == Float32, f"{field.name} expected Float32, got {field.dtype}"

    def test_schema_length_matches_feature_count(self) -> None:
        schema = build_schema_from_definitions(FEATURE_DEFINITIONS)
        assert len(schema) == len(FEATURE_DEFINITIONS)


class TestFeastDtype:
    def test_float32_maps_to_float32(self) -> None:
        assert feast_dtype_for("float32") == Float32

    def test_float64_maps_to_float32(self) -> None:
        # Feast 0.47 has no Float64; normalized telemetry fits in Float32.
        assert feast_dtype_for("float64") == Float32

    def test_unknown_dtype_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="complex64"):
            feast_dtype_for("complex64")

    def test_empty_string_raises_key_error(self) -> None:
        with pytest.raises(KeyError):
            feast_dtype_for("")


class TestBuildEntities:
    def test_entity_join_keys(self) -> None:
        channel, mission = build_entities()
        # Feast Entity exposes the first join key as `.join_key` (singular)
        assert channel.join_key == "channel_id"
        assert mission.join_key == "mission_id"

    def test_entity_names(self) -> None:
        channel, mission = build_entities()
        assert channel.name == "channel"
        assert mission.name == "mission"

    def test_returns_two_entities(self) -> None:
        result = build_entities()
        assert len(result) == 2
