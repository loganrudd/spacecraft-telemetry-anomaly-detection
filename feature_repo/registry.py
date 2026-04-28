"""Feast feature registry — thin shim for the Feast CLI.

This file is the entry point for `feast apply` / `feast materialize` run
directly from the feature_repo/ directory.  All construction logic lives in
feast_client.repo so production code and this shim share one implementation.

The Feast CLI scans this module for Entity and FeatureView objects at import
time, so module-level execution is expected and intentional here.

Do NOT import this module from application code — use
feast_client.store.apply_definitions() instead.
"""

from __future__ import annotations

from pathlib import Path

from spacecraft_telemetry.core.config import load_settings
from spacecraft_telemetry.feast_client.repo import build_entities, build_feature_view

_settings = load_settings()

# Resolve source_path to absolute so Feast doesn't mis-resolve it relative
# to repo_path (feature_repo/).  __file__ is feature_repo/registry.py, so
# its parent's parent is the repo root.
_repo_root = Path(__file__).parent.parent
_source_path = str((_repo_root / _settings.feast.source_path).resolve())

channel, mission = build_entities()
telemetry_features = build_feature_view(
    source_path=_source_path,
    view_name=_settings.feast.feature_view_name,
    ttl_days=_settings.feast.ttl_days,
)
# Expose the FileSource as a module-level name so `feast apply` can register it.
telemetry_source = telemetry_features.batch_source
