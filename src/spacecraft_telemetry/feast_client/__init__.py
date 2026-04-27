"""Feast feature store client — Phase 3.

Public API grows as modules are added:
  repo.py   — build_schema_from_definitions, build_entities, build_feature_view, feast_dtype_for
  store.py  — create_feature_store, apply_definitions, materialize, teardown  (Step 5)
  client.py — get_historical_features, get_online_features_for_channel         (Step 6)
"""

from spacecraft_telemetry.feast_client.repo import (
    build_entities,
    build_feature_view,
    build_schema_from_definitions,
    feast_dtype_for,
)

__all__ = [
    "build_entities",
    "build_feature_view",
    "build_schema_from_definitions",
    "feast_dtype_for",
]
