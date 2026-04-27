"""Feast feature store client — Phase 3.

Public API:
  repo.py   — build_schema_from_definitions, build_entities, build_feature_view, feast_dtype_for
  store.py  — create_feature_store, apply_definitions, materialize, teardown
  client.py — get_historical_features, get_online_features_for_channel
"""

from spacecraft_telemetry.feast_client.client import (
    get_historical_features,
    get_online_features_for_channel,
)
from spacecraft_telemetry.feast_client.repo import (
    build_entities,
    build_feature_view,
    build_schema_from_definitions,
    feast_dtype_for,
)
from spacecraft_telemetry.feast_client.store import (
    apply_definitions,
    create_feature_store,
    materialize,
    teardown,
)

__all__ = [
    "apply_definitions",
    "build_entities",
    "build_feature_view",
    "build_schema_from_definitions",
    "create_feature_store",
    "feast_dtype_for",
    "get_historical_features",
    "get_online_features_for_channel",
    "materialize",
    "teardown",
]
