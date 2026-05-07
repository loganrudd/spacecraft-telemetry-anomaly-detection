"""Tests for evidently_monitoring/reference.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from spacecraft_telemetry.core.config import MonitoringConfig, Settings, SparkConfig
from spacecraft_telemetry.evidently_monitoring.reference import (
    MONITORING_FEATURE_COLS,
    build_reference_profile,
    compute_feature_dataframe,
    load_reference_profile,
    reference_profile_path,
    save_reference_profile,
)

# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------

_MISSION = "ESA-Mission1"
_CHANNEL = "channel_1"

# PyArrow schema that mirrors what Spark writes.
# mission_id / channel_id are Hive partition columns (encoded in directory
# names) and are therefore NOT present as data columns in the Parquet files.
_SERIES_FILE_SCHEMA = pa.schema(
    [
        pa.field("telemetry_timestamp", pa.timestamp("us", tz="UTC")),
        pa.field("value_normalized", pa.float32()),
        pa.field("segment_id", pa.int32()),
        pa.field("is_anomaly", pa.bool_()),
    ]
)


def _make_series_df(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """Synthetic SERIES_SCHEMA-like DataFrame — no Parquet on disk needed."""
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2020-01-01", periods=n, freq="1s", tz="UTC")
    return pd.DataFrame(
        {
            "telemetry_timestamp": timestamps,
            "value_normalized": rng.standard_normal(n).astype(np.float32),
            "channel_id": _CHANNEL,
            "mission_id": _MISSION,
            "segment_id": np.zeros(n, dtype=np.int32),
            "is_anomaly": np.zeros(n, dtype=bool),
        }
    )


def _write_train_parquet(
    base: Path,
    mission: str,
    channel: str,
    n: int = 300,
    seed: int = 0,
) -> None:
    """Write a minimal series Parquet at the Hive-partitioned train path."""
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2020-01-01", periods=n, freq="1s", tz="UTC")
    table = pa.table(
        {
            "telemetry_timestamp": pa.array(timestamps.astype("datetime64[us, UTC]")),
            "value_normalized": pa.array(rng.standard_normal(n).astype("float32")),
            "segment_id": pa.array(np.zeros(n, dtype=np.int32)),
            "is_anomaly": pa.array(np.zeros(n, dtype=bool)),
        },
        schema=_SERIES_FILE_SCHEMA,
    )
    partition_dir = (
        base / mission / "train"
        / f"mission_id={mission}"
        / f"channel_id={channel}"
    )
    partition_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, partition_dir / "part.parquet")


# ---------------------------------------------------------------------------
# MONITORING_FEATURE_COLS
# ---------------------------------------------------------------------------


class TestMonitoringFeatureCols:
    def test_value_normalized_is_first(self) -> None:
        assert MONITORING_FEATURE_COLS[0] == "value_normalized"

    def test_contains_14_columns(self) -> None:
        # value_normalized + 4 stats x 3 windows + rate_of_change = 14
        assert len(MONITORING_FEATURE_COLS) == 14

    def test_contains_rate_of_change(self) -> None:
        assert "rate_of_change" in MONITORING_FEATURE_COLS

    def test_contains_expected_rolling_columns(self) -> None:
        for stat in ("mean", "std", "min", "max"):
            for w in (10, 50, 100):
                assert f"rolling_{stat}_{w}" in MONITORING_FEATURE_COLS


# ---------------------------------------------------------------------------
# compute_feature_dataframe
# ---------------------------------------------------------------------------


class TestComputeFeatureDataframe:
    def test_output_columns_match_monitoring_feature_cols(self) -> None:
        result = compute_feature_dataframe(_make_series_df(300), Settings())
        assert list(result.columns) == MONITORING_FEATURE_COLS

    def test_no_nan_rows_in_output(self) -> None:
        result = compute_feature_dataframe(_make_series_df(300), Settings())
        assert not result.isnull().any().any()

    def test_warmup_rows_are_dropped(self) -> None:
        """First max(window)-1 rows are NaN due to rolling; they must be dropped."""
        n = 300
        result = compute_feature_dataframe(_make_series_df(n), Settings())
        # window=100 needs 99 warmup rows; rate_of_change needs 1 → 99 dropped total
        assert len(result) <= n - 99
        assert len(result) > 0

    def test_does_not_mutate_input(self) -> None:
        df = _make_series_df(300)
        original_cols = list(df.columns)
        compute_feature_dataframe(df, Settings())
        assert list(df.columns) == original_cols

    def test_index_is_zero_based_sequential(self) -> None:
        result = compute_feature_dataframe(_make_series_df(300), Settings())
        assert list(result.index) == list(range(len(result)))

    def test_rate_of_change_without_timestamp(self) -> None:
        """Falls back to simple value diff when telemetry_timestamp is absent."""
        df = _make_series_df(300).drop(columns=["telemetry_timestamp"])
        result = compute_feature_dataframe(df, Settings())
        assert "rate_of_change" in result.columns
        # Diff NaN (first row) should be dropped along with rolling warmup
        assert not result["rate_of_change"].isnull().any()

    def test_value_normalized_preserved(self) -> None:
        """Output value_normalized values must be a subset of the input values."""
        df = _make_series_df(300)
        result = compute_feature_dataframe(df, Settings())
        input_vals = set(np.round(df["value_normalized"].astype(float), 6))
        output_vals = set(np.round(result["value_normalized"].astype(float), 6))
        assert output_vals.issubset(input_vals)

    def test_rolling_mean_10_correct_spot_check(self) -> None:
        """rolling_mean_10 must equal pd.rolling(10).mean() for the retained rows.

        window=100 warmup drops rows 0-98; rate_of_change drops row 0 too (already
        covered).  So result rows map 1-to-1 to original rows 99..299 (201 rows).
        """
        df = _make_series_df(300)
        result = compute_feature_dataframe(df, Settings())
        expected = df["value_normalized"].rolling(10).mean()
        # result corresponds to original rows 99..299 (first 99 warmup dropped)
        np.testing.assert_allclose(
            result["rolling_mean_10"].values,
            expected.iloc[99:].values,
            rtol=1e-5,
        )


# ---------------------------------------------------------------------------
# save / load round-trip
# ---------------------------------------------------------------------------


class TestSaveLoadRoundTrip:
    def test_save_and_load_preserves_data(self, tmp_path: Path) -> None:
        feature_df = compute_feature_dataframe(_make_series_df(300), Settings())
        path = tmp_path / "profiles" / _CHANNEL / "reference.parquet"
        save_reference_profile(feature_df, path)
        loaded = load_reference_profile(path)
        pd.testing.assert_frame_equal(feature_df.reset_index(drop=True), loaded)

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        feature_df = compute_feature_dataframe(_make_series_df(300), Settings())
        path = tmp_path / "a" / "b" / "c" / "reference.parquet"
        save_reference_profile(feature_df, path)
        assert path.exists()

    def test_load_missing_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Reference profile not found"):
            load_reference_profile(tmp_path / "missing.parquet")

    def test_loaded_columns_match_monitoring_feature_cols(self, tmp_path: Path) -> None:
        feature_df = compute_feature_dataframe(_make_series_df(300), Settings())
        path = tmp_path / "ref.parquet"
        save_reference_profile(feature_df, path)
        loaded = load_reference_profile(path)
        assert list(loaded.columns) == MONITORING_FEATURE_COLS


# ---------------------------------------------------------------------------
# reference_profile_path
# ---------------------------------------------------------------------------


class TestReferenceProfilePath:
    def test_path_structure(self) -> None:
        settings = Settings(
            monitoring=MonitoringConfig(
                reference_profiles_dir=Path("monitoring/reference_profiles")
            )
        )
        path = reference_profile_path(settings, _MISSION, _CHANNEL)
        assert path == Path(
            f"monitoring/reference_profiles/{_MISSION}/{_CHANNEL}/reference.parquet"
        )

    def test_custom_base_dir(self) -> None:
        settings = Settings(
            monitoring=MonitoringConfig(reference_profiles_dir=Path("/tmp/profiles"))
        )
        path = reference_profile_path(settings, _MISSION, _CHANNEL)
        assert path == Path(f"/tmp/profiles/{_MISSION}/{_CHANNEL}/reference.parquet")

    def test_filename_is_always_reference_parquet(self) -> None:
        path = reference_profile_path(Settings(), _MISSION, _CHANNEL)
        assert path.name == "reference.parquet"


# ---------------------------------------------------------------------------
# build_reference_profile
# ---------------------------------------------------------------------------


class TestBuildReferenceProfile:
    def test_returns_monitoring_cols(self, tmp_path: Path) -> None:
        _write_train_parquet(tmp_path, _MISSION, _CHANNEL, n=300)
        settings = Settings(spark=SparkConfig(processed_data_dir=tmp_path))
        df = build_reference_profile(settings, _MISSION, _CHANNEL)
        assert list(df.columns) == MONITORING_FEATURE_COLS

    def test_no_nans_in_output(self, tmp_path: Path) -> None:
        _write_train_parquet(tmp_path, _MISSION, _CHANNEL, n=300)
        settings = Settings(spark=SparkConfig(processed_data_dir=tmp_path))
        df = build_reference_profile(settings, _MISSION, _CHANNEL)
        assert not df.isnull().any().any()

    def test_caps_at_reference_sample_rows(self, tmp_path: Path) -> None:
        _write_train_parquet(tmp_path, _MISSION, _CHANNEL, n=300)
        settings = Settings(
            spark=SparkConfig(processed_data_dir=tmp_path),
            monitoring=MonitoringConfig(reference_sample_rows=10),
        )
        df = build_reference_profile(settings, _MISSION, _CHANNEL)
        assert len(df) == 10

    def test_does_not_sample_when_under_cap(self, tmp_path: Path) -> None:
        _write_train_parquet(tmp_path, _MISSION, _CHANNEL, n=300)
        settings = Settings(
            spark=SparkConfig(processed_data_dir=tmp_path),
            monitoring=MonitoringConfig(reference_sample_rows=5000),
        )
        df = build_reference_profile(settings, _MISSION, _CHANNEL)
        # 300 rows - 99 warmup - 1 (rate_of_change diff) = 200 rows kept; all below cap
        assert 0 < len(df) <= 300

    def test_sampling_is_reproducible(self, tmp_path: Path) -> None:
        """Two calls with same settings must return identical sampled rows."""
        _write_train_parquet(tmp_path, _MISSION, _CHANNEL, n=300)
        settings = Settings(
            spark=SparkConfig(processed_data_dir=tmp_path),
            monitoring=MonitoringConfig(reference_sample_rows=50),
        )
        df1 = build_reference_profile(settings, _MISSION, _CHANNEL)
        df2 = build_reference_profile(settings, _MISSION, _CHANNEL)
        pd.testing.assert_frame_equal(df1, df2)

    def test_missing_channel_raises_file_not_found(self, tmp_path: Path) -> None:
        settings = Settings(spark=SparkConfig(processed_data_dir=tmp_path))
        with pytest.raises(FileNotFoundError):
            build_reference_profile(settings, _MISSION, "nonexistent_channel")

    def test_index_is_zero_based_sequential(self, tmp_path: Path) -> None:
        _write_train_parquet(tmp_path, _MISSION, _CHANNEL, n=300)
        settings = Settings(spark=SparkConfig(processed_data_dir=tmp_path))
        df = build_reference_profile(settings, _MISSION, _CHANNEL)
        assert list(df.index) == list(range(len(df)))
