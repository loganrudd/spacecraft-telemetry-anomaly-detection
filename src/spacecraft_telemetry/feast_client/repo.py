"""Feast adapter layer — builds Feast objects from FEATURE_DEFINITIONS.

This module is the only place in the codebase that constructs Feast
Entity/FeatureView objects programmatically.  It is imported by:

  - feature_repo/registry.py  (Feast CLI entry-point, module-level)
  - tests/feast_client/        (hermetic fixtures via build_feature_view)
  - feast_client/store.py      (lazy-imported inside apply_definitions)
"""

from __future__ import annotations

from datetime import timedelta
from typing import Union

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Bool, ComplexFeastType, Float32, Int64, PrimitiveFeastType, String
from feast.value_type import ValueType

from spacecraft_telemetry.features.definitions import FEATURE_DEFINITIONS, FeatureDefinition

# Union type accepted by Feast's Field.dtype parameter.
_FeastType = Union[ComplexFeastType, PrimitiveFeastType]

# Map dtype strings (from FeatureDefinition.dtype) → Feast types.
# "float64" maps to Float32 — Feast 0.47 online store serialisation is cleanest
# with Float32; normalized telemetry values in the [-5, 5] range don't need it.
_DTYPE_MAP: dict[str, _FeastType] = {
    "float32": Float32,
    "float64": Float32,
    "int64": Int64,
    "string": String,
    "bool": Bool,
}


def feast_dtype_for(dtype_str: str) -> _FeastType:
    """Return the Feast dtype for a dtype string.

    Raises:
        KeyError: if dtype_str is not in the supported set.
    """
    try:
        return _DTYPE_MAP[dtype_str]
    except KeyError:
        raise KeyError(
            f"No Feast dtype mapping for {dtype_str!r}. Known: {sorted(_DTYPE_MAP)}"
        )


def build_schema_from_definitions(defs: list[FeatureDefinition]) -> list[Field]:
    """Convert a list of FeatureDefinitions to Feast Field objects."""
    return [Field(name=fd.name, dtype=feast_dtype_for(fd.dtype)) for fd in defs]


def build_entities() -> tuple[Entity, Entity]:
    """Return the (channel, mission) Entity pair used in all feature views."""
    channel = Entity(
        name="channel",
        join_keys=["channel_id"],
        value_type=ValueType.STRING,
        description="ESA telemetry channel id (e.g. channel_1)",
    )
    mission = Entity(
        name="mission",
        join_keys=["mission_id"],
        value_type=ValueType.STRING,
        description="ESA mission identifier",
    )
    return channel, mission


def build_feature_view(
    *,
    source_path: str,
    view_name: str,
    ttl_days: int,
) -> FeatureView:
    """Build a FeatureView pointing at source_path.

    Constructs its own Entity and FileSource objects so tests can pass a
    tmp_path-based source_path without touching real config.

    Args:
        source_path: Absolute or relative path to the Hive-partitioned Parquet
                     directory (mission_id=.../channel_id=.../part.parquet).
        view_name:   Feast feature view name.
        ttl_days:    Feature TTL in days.
    """
    channel, mission = build_entities()
    source = FileSource(
        name=f"{view_name}_source",
        path=source_path,
        timestamp_field="telemetry_timestamp",
    )
    return FeatureView(
        name=view_name,
        entities=[channel, mission],
        schema=build_schema_from_definitions(FEATURE_DEFINITIONS),
        source=source,
        ttl=timedelta(days=ttl_days),
        online=True,
        tags={"dataset": "esa-anomaly", "channels_anonymized": "true"},
    )
