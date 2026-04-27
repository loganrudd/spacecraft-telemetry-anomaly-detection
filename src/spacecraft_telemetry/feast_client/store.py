"""Feast store lifecycle — apply, materialize, and teardown.

Usage:
    from spacecraft_telemetry.core.config import load_settings
    from spacecraft_telemetry.feast_client.store import (
        apply_definitions,
        create_feature_store,
        materialize,
        teardown,
    )

    settings = load_settings()
    store = create_feature_store(settings)
    apply_definitions(store)
    materialize(store, end_date=datetime.now(timezone.utc))
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from feast import FeatureStore

from spacecraft_telemetry.core.config import Settings
from spacecraft_telemetry.core.logging import get_logger

log = get_logger(__name__)


def create_feature_store(settings: Settings) -> FeatureStore:
    """Instantiate a FeatureStore pointed at the repo_path in settings.

    The returned store is not yet applied — call apply_definitions() to
    register entities and feature views to the registry.

    Args:
        settings: Populated Settings; feast.repo_path is used to locate
                  feature_store.yaml.
    """
    repo_path = Path(str(settings.feast.repo_path))
    return FeatureStore(repo_path=str(repo_path))


def apply_definitions(store: FeatureStore) -> dict[str, int]:
    """Register entities and feature views to the Feast registry.

    Lazy-imports feature_repo.registry so any config error surfaces with a
    clear traceback at call time, not at module import.

    Args:
        store: FeatureStore instance from create_feature_store().

    Returns:
        Dict with keys "n_entities" and "n_feature_views" (counts of registered objects).
    """
    from feature_repo.registry import channel, mission, telemetry_features, telemetry_source

    project = store.project
    log.info("feast.apply.start", project=project)

    store.apply([channel, mission, telemetry_source, telemetry_features])

    n_entities = len(store.list_entities())
    n_feature_views = len(store.list_feature_views())

    log.info(
        "feast.apply.end",
        project=project,
        n_entities=n_entities,
        n_feature_views=n_feature_views,
    )
    return {"n_entities": n_entities, "n_feature_views": n_feature_views}


def materialize(
    store: FeatureStore,
    end_date: datetime,
    start_date: datetime | None = None,
) -> None:
    """Materialize features from the offline store to the online store.

    Uses incremental materialization when start_date is None (default), which
    picks up from the last materialized timestamp.  Pass start_date for an
    explicit backfill window.

    Args:
        store:      FeatureStore instance; registry must already be applied.
        end_date:   Materialize features up to this timestamp (UTC).
        start_date: If provided, materialize [start_date, end_date].
                    If None, incremental from last materialized timestamp.
    """
    fv_name = store.list_feature_views()[0].name if store.list_feature_views() else "unknown"
    log.info(
        "feast.materialize.start",
        fv_name=fv_name,
        start=start_date.isoformat() if start_date else "incremental",
        end=end_date.isoformat(),
    )

    if start_date is not None:
        store.materialize(start_date=start_date, end_date=end_date)
    else:
        store.materialize_incremental(end_date=end_date)

    log.info("feast.materialize.end", fv_name=fv_name, end=end_date.isoformat())


def teardown(store: FeatureStore) -> None:
    """Delete the registry and online store databases.

    Use for local dev cleanup only — wipes all materialized data.

    Args:
        store: FeatureStore instance to tear down.
    """
    repo_path = Path(store.repo_path)
    registry_path = repo_path / "data" / "registry.db"
    online_path = repo_path / "data" / "online_store.db"

    log.info(
        "feast.teardown",
        registry=str(registry_path),
        online_store=str(online_path),
    )
    store.teardown()
