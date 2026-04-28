"""Feast feature registry — loaded by the Feast CLI and feast_client.store.

This file is the adapter between the framework-agnostic FEATURE_DEFINITIONS
registry and Feast's Entity/FeatureView objects.  Module-level execution is
intentional: Feast resolves registry contents at import time.

The Feast CLI discovers objects in this file by scanning the feature_repo/
directory.  feast_client/store.py lazy-imports this module inside
apply_definitions() so config errors surface with a clear traceback.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureView, FileSource
from feast.value_type import ValueType

from spacecraft_telemetry.core.config import load_settings
from spacecraft_telemetry.features.definitions import FEATURE_DEFINITIONS
from spacecraft_telemetry.feast_client.repo import build_schema_from_definitions

_settings = load_settings()

# Resolve source_path to absolute so Feast doesn't mis-resolve it relative
# to repo_path (feature_repo/).  __file__ is feature_repo/registry.py, so
# its parent's parent is the repo root.
_repo_root = Path(__file__).parent.parent
_source_path = str((_repo_root / _settings.feast.source_path).resolve())

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

telemetry_source = FileSource(
    name="telemetry_features_source",
    path=_source_path,
    timestamp_field="telemetry_timestamp",
)

telemetry_features = FeatureView(
    name=_settings.feast.feature_view_name,
    entities=[channel, mission],
    schema=build_schema_from_definitions(FEATURE_DEFINITIONS),
    source=telemetry_source,
    ttl=timedelta(days=_settings.feast.ttl_days),
    online=True,
    tags={"dataset": "esa-anomaly", "channels_anonymized": "true"},
)
