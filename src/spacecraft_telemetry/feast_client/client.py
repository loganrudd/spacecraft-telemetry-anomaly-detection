"""Feast retrieval helpers — offline (historical) and online lookups.

Phase 4 (training) uses get_historical_features() with a real entity_df
built from telemetry Parquet timestamps.

Phase 9 (serving) uses get_online_features_for_channel() for sub-ms
latest-value lookups from the materialized SQLite online store.

entity_df contract for get_historical_features():

    channel_id        str
    mission_id        str
    event_timestamp   datetime64[ns, UTC]   ← must be timezone-aware
"""

from __future__ import annotations

import pandas as pd
from feast import FeatureStore

from spacecraft_telemetry.features.definitions import get_feature_names


def _all_feature_refs(view_name: str = "telemetry_features") -> list[str]:
    """Return all feature references in "{view_name}:{feature_name}" format."""
    return [f"{view_name}:{name}" for name in get_feature_names()]


def get_historical_features(
    store: FeatureStore,
    entity_df: pd.DataFrame,
    features: list[str] | None = None,
) -> pd.DataFrame:
    """Retrieve historical features via point-in-time join.

    Wraps store.get_historical_features(...).to_df(). Phase 4 builds entity_df
    from real telemetry timestamps in the offline Parquet — not a synthesized
    regular grid — so irregular ESA sampling is handled correctly.

    Args:
        store:     FeatureStore with the registry already applied.
        entity_df: DataFrame with columns [channel_id, mission_id,
                   event_timestamp (datetime64[ns, UTC])].
        features:  Feature refs in "{view}:{name}" format.  If None, all
                   features from FEATURE_DEFINITIONS are returned.

    Returns:
        DataFrame with the original entity_df columns plus one column per
        feature.  Rows where the buffer hasn't warmed up will contain NaN.
    """
    if features is None:
        features = _all_feature_refs()
    retrieval_job = store.get_historical_features(entity_df=entity_df, features=features)
    return retrieval_job.to_df()


def get_online_features_for_channel(
    store: FeatureStore,
    channel_id: str,
    mission_id: str,
    features: list[str] | None = None,
) -> dict[str, float | None]:
    """Retrieve the latest materialized feature values for one channel.

    Performs a single-row online lookup keyed on (channel_id, mission_id).
    The online store must be populated via materialize() before calling this.

    Args:
        store:      FeatureStore with registry applied and online store populated.
        channel_id: Channel entity key (e.g. "channel_1").
        mission_id: Mission entity key (e.g. "ESA-Mission1").
        features:   Feature refs in "{view}:{name}" format.  If None, all
                    features from FEATURE_DEFINITIONS are returned.

    Returns:
        Dict mapping feature name → float value for the latest materialized row.
        Values may be None if the feature was never materialized for this entity.
    """
    if features is None:
        features = _all_feature_refs()

    response = store.get_online_features(
        features=features,
        entity_rows=[{"channel_id": channel_id, "mission_id": mission_id}],
    )
    # to_dict() returns {"view_name__feature_name": [value]} — take the first (only)
    # element and strip the "{view_name}__" prefix so callers get bare feature names.
    view_name = features[0].split(":")[0] if features else "telemetry_features"
    prefix = f"{view_name}__"
    raw = response.to_dict()
    return {k.removeprefix(prefix): (values[0] if values else None) for k, values in raw.items()}
