"""Fixtures for Feast feature store tests.

All Feast tests are skipped automatically if the [tracking] extra is not
installed (mirrors the JDK-skip pattern in tests/spark/conftest.py).

Fixture hierarchy
-----------------
tmp_feast_repo       — hermetic feature_repo/ dir + synthetic Parquet source
materialized_store   — FeatureStore with registry applied + online store populated
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

feast = pytest.importorskip("feast")  # skip entire module if feast not installed

import yaml  # noqa: E402 — must come after the importorskip guard

from spacecraft_telemetry.feast_client.repo import build_entities, build_feature_view  # noqa: E402
from spacecraft_telemetry.features.definitions import FEATURE_DEFINITIONS  # noqa: E402

# ---------------------------------------------------------------------------
# Constants shared by fixtures and tests
# ---------------------------------------------------------------------------

_N_ROWS = 200
_MISSION = "ESA-Mission1"
_CHANNEL = "channel_1"
_BASE_TS = pd.Timestamp("2000-01-01", tz="UTC")
_INTERVAL_S = 90  # seconds between rows


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synthetic_df() -> pd.DataFrame:
    """200-row synthetic feature Parquet matching the FEATURE_DEFINITIONS schema.

    Row i has all feature values equal to i * 0.01, making point-in-time
    lookups deterministic:  rolling_mean_10 at row 50  == 50 * 0.01 == 0.5.
    """
    timestamps = [_BASE_TS + pd.Timedelta(seconds=_INTERVAL_S * i) for i in range(_N_ROWS)]
    # Cast to microsecond-precision UTC — matches the format written by Phase 2 Spark pipeline.
    ts_array = pd.array(timestamps, dtype="datetime64[us, UTC]")

    data: dict[str, Any] = {
        "telemetry_timestamp": ts_array,
        "channel_id": [_CHANNEL] * _N_ROWS,
        "mission_id": [_MISSION] * _N_ROWS,
        "value_normalized": [float(i) * 0.01 for i in range(_N_ROWS)],
    }
    for fd in FEATURE_DEFINITIONS:
        data[fd.name] = [float(i) * 0.01 for i in range(_N_ROWS)]

    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_feast_repo(tmp_path: Path) -> Generator[Path, None, None]:
    """Build a hermetic Feast repo under tmp_path.

    Layout created:
        {tmp_path}/source/mission_id=ESA-Mission1/channel_id=channel_1/part.parquet
        {tmp_path}/feature_repo/feature_store.yaml   (absolute paths → tmp_path)
        {tmp_path}/feature_repo/data/                (registry + online db written here)

    Does NOT import feature_repo.registry — all objects are built directly
    via feast_client.repo builders so the real config is never touched.
    """
    # 1. Synthetic Parquet under Hive-partitioned directory tree.
    parquet_dir = (
        tmp_path / "source" / f"mission_id={_MISSION}" / f"channel_id={_CHANNEL}"
    )
    parquet_dir.mkdir(parents=True)
    _make_synthetic_df().to_parquet(parquet_dir / "part.parquet", index=False)

    # 2. feature_store.yaml with absolute paths so the store is self-contained.
    feature_repo_dir = tmp_path / "feature_repo"
    data_dir = feature_repo_dir / "data"
    data_dir.mkdir(parents=True)

    config: dict[str, Any] = {
        "project": "spacecraft_telemetry",
        "provider": "local",
        "registry": str(data_dir / "registry.db"),
        "offline_store": {"type": "file"},
        "online_store": {"type": "sqlite", "path": str(data_dir / "online_store.db")},
        "entity_key_serialization_version": 2,
    }
    (feature_repo_dir / "feature_store.yaml").write_text(yaml.dump(config))

    yield feature_repo_dir


@pytest.fixture()
def materialized_store(tmp_feast_repo: Path) -> Generator[Any, None, None]:
    """FeatureStore with registry applied and online store fully materialized.

    Builds a hermetic FeatureView pointing at the synthetic Parquet source
    created by tmp_feast_repo.  Uses ttl_days=99999 so the 2000-era fixture
    data is never considered stale during online retrieval.
    """
    from feast import FeatureStore

    source_dir = tmp_feast_repo.parent / "source"
    channel, mission = build_entities()
    fv = build_feature_view(
        source_path=str(source_dir),
        view_name="telemetry_features",
        ttl_days=99999,
    )

    store = FeatureStore(repo_path=str(tmp_feast_repo))
    store.apply([channel, mission, fv.source, fv])

    # Materialize just past the last synthetic data point.
    end_dt = (_BASE_TS + pd.Timedelta(seconds=_INTERVAL_S * _N_ROWS)).to_pydatetime()
    store.materialize_incremental(end_date=end_dt)

    yield store
