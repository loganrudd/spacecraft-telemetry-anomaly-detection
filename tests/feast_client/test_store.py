"""Tests for feast_client.store — apply + materialize lifecycle.

Uses the hermetic tmp_feast_repo fixture (feature_store.yaml + synthetic Parquet)
so the real config and real data are never touched.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

feast = pytest.importorskip("feast")

from feast import FeatureStore  # noqa: E402

from spacecraft_telemetry.feast_client.repo import build_entities, build_feature_view  # noqa: E402
from tests.feast_client.conftest import _BASE_TS, _INTERVAL_S, _N_ROWS  # noqa: E402


def _make_store_with_objects(tmp_feast_repo: Path):
    """Helper: create a FeatureStore + hermetic objects, return (store, channel, mission, fv)."""
    source_dir = tmp_feast_repo.parent / "source"
    channel, mission = build_entities()
    fv = build_feature_view(
        source_path=str(source_dir),
        view_name="telemetry_features",
        ttl_days=99999,
    )
    store = FeatureStore(repo_path=str(tmp_feast_repo))
    return store, channel, mission, fv


class TestApply:
    def test_apply_creates_registry(self, tmp_feast_repo: Path) -> None:
        store, channel, mission, fv = _make_store_with_objects(tmp_feast_repo)

        store.apply([channel, mission, fv.source, fv])

        registry_path = tmp_feast_repo / "data" / "registry.db"
        assert registry_path.exists(), "registry.db should be created after apply"
        assert len(store.list_feature_views()) == 1

    def test_apply_registers_correct_feature_view_name(self, tmp_feast_repo: Path) -> None:
        store, channel, mission, fv = _make_store_with_objects(tmp_feast_repo)

        store.apply([channel, mission, fv.source, fv])

        fv_names = [v.name for v in store.list_feature_views()]
        assert "telemetry_features" in fv_names

    def test_apply_registers_both_entities(self, tmp_feast_repo: Path) -> None:
        store, channel, mission, fv = _make_store_with_objects(tmp_feast_repo)

        store.apply([channel, mission, fv.source, fv])

        entity_names = {e.name for e in store.list_entities()}
        assert {"channel", "mission"} == entity_names

    def test_apply_idempotent(self, tmp_feast_repo: Path) -> None:
        store, channel, mission, fv = _make_store_with_objects(tmp_feast_repo)

        # First apply
        store.apply([channel, mission, fv.source, fv])
        # Second apply must not raise
        store.apply([channel, mission, fv.source, fv])

        assert len(store.list_feature_views()) == 1


class TestMaterialize:
    def test_materialize_populates_online_store(self, tmp_feast_repo: Path) -> None:
        store, channel, mission, fv = _make_store_with_objects(tmp_feast_repo)
        store.apply([channel, mission, fv.source, fv])

        end_dt = (_BASE_TS + pd.Timedelta(seconds=_INTERVAL_S * _N_ROWS)).to_pydatetime()
        store.materialize_incremental(end_date=end_dt)

        online_path = tmp_feast_repo / "data" / "online_store.db"
        assert online_path.exists()
        assert online_path.stat().st_size > 0, (
            "online_store.db should be non-empty after materialize"
        )

    def test_materialize_incremental_called_twice_does_not_raise(
        self, tmp_feast_repo: Path
    ) -> None:
        store, channel, mission, fv = _make_store_with_objects(tmp_feast_repo)
        store.apply([channel, mission, fv.source, fv])

        end_dt = (_BASE_TS + pd.Timedelta(seconds=_INTERVAL_S * _N_ROWS)).to_pydatetime()
        store.materialize_incremental(end_date=end_dt)
        store.materialize_incremental(end_date=end_dt)  # second call must not raise
