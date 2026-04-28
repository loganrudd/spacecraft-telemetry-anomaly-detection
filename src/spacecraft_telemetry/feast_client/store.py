"""Feast store lifecycle — apply, materialize, and teardown.

Usage:
    from spacecraft_telemetry.core.config import load_settings
    from spacecraft_telemetry.feast_client.store import (
        apply_definitions,
        create_feature_store,
        ensure_applied,
        materialize,
    )

    settings = load_settings()
    store = create_feature_store(settings)
    apply_definitions(store, settings)
    materialize(store, end_date=datetime.now(UTC))
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


def apply_definitions(store: FeatureStore, settings: Settings) -> dict[str, int]:
    """Register entities and feature views to the Feast registry.

    Builds Feast objects directly from feast_client.repo builders — no import
    of feature_repo.registry, no sys.path mutation.  source_path is resolved
    to absolute from the repo root so Feast doesn't mis-resolve it relative
    to repo_path.

    Args:
        store:    FeatureStore instance from create_feature_store().
        settings: Settings containing feast config (source_path, view_name,
                  ttl_days).

    Returns:
        Dict with keys "entities" and "feature_views" (counts of registered objects).
    """
    from spacecraft_telemetry.feast_client.repo import build_entities, build_feature_view

    repo_root = Path(store.repo_path).resolve().parent
    source_path = str((repo_root / settings.feast.source_path).resolve())

    channel, mission = build_entities()
    fv = build_feature_view(
        source_path=source_path,
        view_name=settings.feast.feature_view_name,
        ttl_days=settings.feast.ttl_days,
    )

    project = store.project
    log.info("feast.apply.start", project=project, source_path=source_path)

    store.apply([channel, mission, fv.source, fv])

    n_entities = len(store.list_entities())
    n_feature_views = len(store.list_feature_views())

    log.info(
        "feast.apply.end",
        project=project,
        n_entities=n_entities,
        n_feature_views=n_feature_views,
    )
    return {"entities": n_entities, "feature_views": n_feature_views}


def ensure_applied(store: FeatureStore, settings: Settings) -> None:
    """Apply definitions only if the registry does not yet exist.

    Used by materialize and retrieve so that a freshly-created store is
    auto-registered without redundant apply calls on every subsequent run.

    Args:
        store:    FeatureStore instance from create_feature_store().
        settings: Settings passed through to apply_definitions if needed.
    """
    registry_path = Path(store.repo_path) / "data" / "registry.db"
    if registry_path.exists():
        log.debug("feast.apply.skip", reason="registry exists", path=str(registry_path))
    else:
        log.info("feast.apply.auto", reason="registry not found, applying now")
        apply_definitions(store, settings)


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
    store.teardown()  # type: ignore[no-untyped-call]
