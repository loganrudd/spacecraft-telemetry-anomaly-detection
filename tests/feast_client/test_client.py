"""Tests for feast_client.client — historical and online retrieval helpers."""

from __future__ import annotations

import pandas as pd
import pytest

feast = pytest.importorskip("feast")

from feast import FeatureStore  # noqa: E402

from spacecraft_telemetry.feast_client.client import (  # noqa: E402
    get_historical_features,
    get_online_features_for_channel,
)
from spacecraft_telemetry.features.definitions import get_feature_names  # noqa: E402
from tests.feast_client.conftest import (  # noqa: E402
    _BASE_TS,
    _CHANNEL,
    _INTERVAL_S,
    _MISSION,
    _N_ROWS,
)


class TestGetHistoricalFeatures:
    def test_schema_includes_all_features(self, materialized_store: FeatureStore) -> None:
        entity_df = pd.DataFrame(
            {
                "channel_id": [_CHANNEL],
                "mission_id": [_MISSION],
                "event_timestamp": pd.array(
                    [_BASE_TS + pd.Timedelta(seconds=_INTERVAL_S * 50)],
                    dtype="datetime64[us, UTC]",
                ),
            }
        )

        result = get_historical_features(materialized_store, entity_df)

        feature_names = set(get_feature_names())
        assert feature_names <= set(result.columns), (
            f"Missing features: {feature_names - set(result.columns)}"
        )

    def test_entity_columns_preserved(self, materialized_store: FeatureStore) -> None:
        entity_df = pd.DataFrame(
            {
                "channel_id": [_CHANNEL],
                "mission_id": [_MISSION],
                "event_timestamp": pd.array(
                    [_BASE_TS + pd.Timedelta(seconds=_INTERVAL_S * 10)],
                    dtype="datetime64[us, UTC]",
                ),
            }
        )

        result = get_historical_features(materialized_store, entity_df)

        assert "channel_id" in result.columns
        assert "mission_id" in result.columns

    def test_values_match_synthetic_data(self, materialized_store: FeatureStore) -> None:
        # Row 50: all feature values == 50 * 0.01 == 0.50
        t_50 = _BASE_TS + pd.Timedelta(seconds=_INTERVAL_S * 50)
        entity_df = pd.DataFrame(
            {
                "channel_id": [_CHANNEL],
                "mission_id": [_MISSION],
                "event_timestamp": pd.array([t_50], dtype="datetime64[us, UTC]"),
            }
        )

        result = get_historical_features(materialized_store, entity_df)

        assert result["rolling_mean_10"].iloc[0] == pytest.approx(50 * 0.01, abs=1e-4)

    def test_handles_query_before_data_window(self, materialized_store: FeatureStore) -> None:
        # Timestamp before any synthetic data — Feast's FileOfflineStore returns an
        # empty DataFrame (no feature rows ≤ entity timestamp). Must not raise.
        pre_data = pd.Timestamp("1999-12-31", tz="UTC")
        entity_df = pd.DataFrame(
            {
                "channel_id": [_CHANNEL],
                "mission_id": [_MISSION],
                "event_timestamp": pd.array([pre_data], dtype="datetime64[us, UTC]"),
            }
        )

        result = get_historical_features(materialized_store, entity_df)

        # Empty result is correct — no feature row has telemetry_timestamp ≤ 1999-12-31.
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_handles_warmup_window_returns_nan(self, materialized_store: FeatureStore) -> None:
        # Row 5 is within the rolling_mean_10 warmup window (rows 0-8 are NaN
        # in the synthetic fixture because window=10 needs 9 prior rows).
        row_5_ts = _BASE_TS + pd.Timedelta(seconds=_INTERVAL_S * 5)
        entity_df = pd.DataFrame(
            {
                "channel_id": [_CHANNEL],
                "mission_id": [_MISSION],
                "event_timestamp": pd.array([row_5_ts], dtype="datetime64[us, UTC]"),
            }
        )

        result = get_historical_features(materialized_store, entity_df)

        assert len(result) == 1, "Expected exactly one row back"
        assert pd.isna(result["rolling_mean_10"].iloc[0]), (
            "rolling_mean_10 should be NaN within the warmup window (rows 0-8)"
        )

    def test_feature_subset_selection(self, materialized_store: FeatureStore) -> None:
        t_20 = _BASE_TS + pd.Timedelta(seconds=_INTERVAL_S * 20)
        entity_df = pd.DataFrame(
            {
                "channel_id": [_CHANNEL],
                "mission_id": [_MISSION],
                "event_timestamp": pd.array([t_20], dtype="datetime64[us, UTC]"),
            }
        )
        requested = ["telemetry_features:rolling_mean_10"]

        result = get_historical_features(materialized_store, entity_df, features=requested)

        assert "rolling_mean_10" in result.columns
        # Columns not requested should be absent
        assert "rolling_mean_50" not in result.columns


class TestGetOnlineFeatures:
    def test_returns_latest_row_values(self, materialized_store: FeatureStore) -> None:
        # Latest row is index (_N_ROWS - 1) == 199; value == 199 * 0.01 == 1.99
        result = get_online_features_for_channel(materialized_store, _CHANNEL, _MISSION)

        # Keys are bare feature names after prefix stripping.
        assert "rolling_mean_10" in result, f"Keys: {list(result)[:5]}"
        assert result["rolling_mean_10"] == pytest.approx((_N_ROWS - 1) * 0.01, abs=1e-4)

    def test_all_features_present(self, materialized_store: FeatureStore) -> None:
        result = get_online_features_for_channel(materialized_store, _CHANNEL, _MISSION)

        for name in get_feature_names():
            assert name in result, f"Feature {name!r} missing from online response"
