"""Integration tests for spark.pipeline.run_preprocessing.

Each test runs the full pipeline on synthetic data (100 rows, 1 channel) and
validates that output files exist, schemas are correct, and counts make sense.

The pipeline now writes per-timestep series rows (SERIES_SCHEMA) rather than
pre-materialised sliding windows. Window construction is deferred to the
PyTorch DataLoader (model/dataset.py).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Shared fixture: Settings wired to tmp_path
# ---------------------------------------------------------------------------


@pytest.fixture()
def settings(pipeline_input_dir: Path, tmp_path: Path):
    """Settings pointing at the synthetic input + a fresh output directory."""
    from spacecraft_telemetry.core.config import DataConfig, Settings, SparkConfig

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return Settings(
        data=DataConfig(sample_data_dir=pipeline_input_dir),
        spark=SparkConfig(
            processed_data_dir=output_dir,
            train_fraction=0.8,
            feature_windows=[10],
        ),
    )


# ---------------------------------------------------------------------------
# Helper: output root
# ---------------------------------------------------------------------------


def _output(settings) -> Path:
    return Path(str(settings.spark.processed_data_dir)) / "ESA-Mission1"


# ---------------------------------------------------------------------------
# Summary dict
# ---------------------------------------------------------------------------


class TestSummaryDict:
    def test_returns_dict_with_expected_keys(self, spark_session, settings) -> None:
        from spacecraft_telemetry.spark.pipeline import run_preprocessing

        summary = run_preprocessing(spark_session, settings, "ESA-Mission1")
        assert set(summary.keys()) == {
            "channels_processed",
            "rows_in",
            "feature_rows_out",
            "train_rows",
            "test_rows",
        }

    def test_channels_processed_equals_one(self, spark_session, settings) -> None:
        from spacecraft_telemetry.spark.pipeline import run_preprocessing

        summary = run_preprocessing(spark_session, settings, "ESA-Mission1")
        assert summary["channels_processed"] == 1

    def test_rows_in_equals_100(self, spark_session, settings) -> None:
        from spacecraft_telemetry.spark.pipeline import run_preprocessing

        summary = run_preprocessing(spark_session, settings, "ESA-Mission1")
        assert summary["rows_in"] == 100

    def test_train_plus_test_equals_rows_in(self, spark_session, settings) -> None:
        from spacecraft_telemetry.spark.pipeline import run_preprocessing

        summary = run_preprocessing(spark_session, settings, "ESA-Mission1")
        assert summary["train_rows"] + summary["test_rows"] == summary["rows_in"]

    def test_train_rows_positive(self, spark_session, settings) -> None:
        from spacecraft_telemetry.spark.pipeline import run_preprocessing

        summary = run_preprocessing(spark_session, settings, "ESA-Mission1")
        # 100 rows, train_fraction=0.8 → 80 train rows
        assert summary["train_rows"] > 0

    def test_test_rows_equals_20(self, spark_session, settings) -> None:
        from spacecraft_telemetry.spark.pipeline import run_preprocessing

        summary = run_preprocessing(spark_session, settings, "ESA-Mission1")
        # 100 rows, train_fraction=0.8 → last 20 rows go to test
        assert summary["test_rows"] == 20

    def test_feature_rows_out_equals_rows_in(self, spark_session, settings) -> None:
        from spacecraft_telemetry.spark.pipeline import run_preprocessing

        summary = run_preprocessing(spark_session, settings, "ESA-Mission1")
        # Features are 1-to-1 with input rows (one feature row per timestamp)
        assert summary["feature_rows_out"] == summary["rows_in"]


# ---------------------------------------------------------------------------
# Output directory existence
# ---------------------------------------------------------------------------


class TestOutputDirectories:
    def test_features_dir_created(self, spark_session, settings) -> None:
        from spacecraft_telemetry.spark.pipeline import run_preprocessing

        run_preprocessing(spark_session, settings, "ESA-Mission1")
        assert (_output(settings) / "features").exists()

    def test_train_dir_created(self, spark_session, settings) -> None:
        from spacecraft_telemetry.spark.pipeline import run_preprocessing

        run_preprocessing(spark_session, settings, "ESA-Mission1")
        assert (_output(settings) / "train").exists()

    def test_test_dir_created(self, spark_session, settings) -> None:
        from spacecraft_telemetry.spark.pipeline import run_preprocessing

        run_preprocessing(spark_session, settings, "ESA-Mission1")
        assert (_output(settings) / "test").exists()

    def test_partition_dirs_created_for_features(self, spark_session, settings) -> None:
        from spacecraft_telemetry.spark.pipeline import run_preprocessing

        run_preprocessing(spark_session, settings, "ESA-Mission1")
        partition = (
            _output(settings) / "features" / "mission_id=ESA-Mission1" / "channel_id=channel_1"
        )
        assert partition.is_dir()

    def test_partition_dirs_created_for_train(self, spark_session, settings) -> None:
        from spacecraft_telemetry.spark.pipeline import run_preprocessing

        run_preprocessing(spark_session, settings, "ESA-Mission1")
        partition = (
            _output(settings) / "train" / "mission_id=ESA-Mission1" / "channel_id=channel_1"
        )
        assert partition.is_dir()

    def test_partition_dirs_created_for_test(self, spark_session, settings) -> None:
        from spacecraft_telemetry.spark.pipeline import run_preprocessing

        run_preprocessing(spark_session, settings, "ESA-Mission1")
        partition = _output(settings) / "test" / "mission_id=ESA-Mission1" / "channel_id=channel_1"
        assert partition.is_dir()


# ---------------------------------------------------------------------------
# Output schemas
# ---------------------------------------------------------------------------


class TestOutputSchemas:
    def test_features_has_required_columns(self, spark_session, settings) -> None:
        from spacecraft_telemetry.spark.pipeline import run_preprocessing

        run_preprocessing(spark_session, settings, "ESA-Mission1")
        df = spark_session.read.parquet(str(_output(settings) / "features"))
        for col in (
            "telemetry_timestamp",
            "value_normalized",
            "rolling_mean_10",
            "rate_of_change",
        ):
            assert col in df.columns, f"Missing feature column: {col}"

    def test_train_has_is_anomaly_column(self, spark_session, settings) -> None:
        from spacecraft_telemetry.spark.pipeline import run_preprocessing

        run_preprocessing(spark_session, settings, "ESA-Mission1")
        df = spark_session.read.parquet(str(_output(settings) / "train"))
        assert "is_anomaly" in df.columns

    def test_train_has_series_schema_columns(self, spark_session, settings) -> None:
        from spacecraft_telemetry.spark.pipeline import run_preprocessing

        run_preprocessing(spark_session, settings, "ESA-Mission1")
        df = spark_session.read.parquet(str(_output(settings) / "train"))
        assert "value_normalized" in df.columns
        assert "telemetry_timestamp" in df.columns
        assert "segment_id" in df.columns

    def test_test_has_is_anomaly_column(self, spark_session, settings) -> None:
        from spacecraft_telemetry.spark.pipeline import run_preprocessing

        run_preprocessing(spark_session, settings, "ESA-Mission1")
        df = spark_session.read.parquet(str(_output(settings) / "test"))
        assert "is_anomaly" in df.columns

    def test_train_no_anomaly_rows(self, spark_session, settings) -> None:
        """Anomalies are excluded from the training set."""
        from pyspark.sql import functions as F

        from spacecraft_telemetry.spark.pipeline import run_preprocessing

        run_preprocessing(spark_session, settings, "ESA-Mission1")
        df = spark_session.read.parquet(str(_output(settings) / "train"))
        assert df.filter(F.col("is_anomaly")).count() == 0


# ---------------------------------------------------------------------------
# normalization_params.json
# ---------------------------------------------------------------------------


class TestNormalizationParams:
    def test_params_file_created(self, spark_session, settings) -> None:
        from spacecraft_telemetry.spark.pipeline import run_preprocessing

        run_preprocessing(spark_session, settings, "ESA-Mission1")
        assert (_output(settings) / "normalization_params.json").exists()

    def test_params_contains_channel(self, spark_session, settings) -> None:
        from spacecraft_telemetry.spark.pipeline import run_preprocessing

        run_preprocessing(spark_session, settings, "ESA-Mission1")
        params = json.loads((_output(settings) / "normalization_params.json").read_text())
        assert "channel_1" in params

    def test_params_has_mean_and_std(self, spark_session, settings) -> None:
        from spacecraft_telemetry.spark.pipeline import run_preprocessing

        run_preprocessing(spark_session, settings, "ESA-Mission1")
        params = json.loads((_output(settings) / "normalization_params.json").read_text())
        assert "mean" in params["channel_1"]
        assert "std" in params["channel_1"]


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


class TestIdempotency:
    def test_rerun_produces_same_counts(self, spark_session, settings) -> None:
        from spacecraft_telemetry.spark.pipeline import run_preprocessing

        summary1 = run_preprocessing(spark_session, settings, "ESA-Mission1")
        summary2 = run_preprocessing(spark_session, settings, "ESA-Mission1")
        assert summary1 == summary2

    def test_rerun_does_not_duplicate_rows(self, spark_session, settings) -> None:
        from spacecraft_telemetry.spark.pipeline import run_preprocessing

        run_preprocessing(spark_session, settings, "ESA-Mission1")
        run_preprocessing(spark_session, settings, "ESA-Mission1")
        # If output was duplicated, feature_rows_out would double
        df = spark_session.read.parquet(str(_output(settings) / "features"))
        assert df.count() == 100


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_missing_channel_dir_raises(self, spark_session, settings, tmp_path) -> None:
        from spacecraft_telemetry.core.config import DataConfig

        bad_settings = settings.model_copy(
            update={"data": DataConfig(sample_data_dir=tmp_path / "nonexistent")}
        )
        from spacecraft_telemetry.spark.pipeline import run_preprocessing

        with pytest.raises(FileNotFoundError, match="No channel Parquet files"):
            run_preprocessing(spark_session, bad_settings, "ESA-Mission1")

    def test_no_labels_file_runs_successfully(
        self, spark_session, settings, pipeline_input_dir
    ) -> None:
        """Pipeline completes when labels.csv is absent; all windows marked nominal."""
        from pyspark.sql import functions as F

        from spacecraft_telemetry.core.config import DataConfig
        from spacecraft_telemetry.spark.pipeline import run_preprocessing

        # Build an input dir without a labels.csv
        no_labels_dir = pipeline_input_dir.parent / "input_no_labels"
        import shutil

        shutil.copytree(pipeline_input_dir, no_labels_dir)
        labels_file = no_labels_dir / "ESA-Mission1" / "labels.csv"
        if labels_file.exists():
            labels_file.unlink()

        no_labels_settings = settings.model_copy(
            update={"data": DataConfig(sample_data_dir=no_labels_dir)}
        )
        summary = run_preprocessing(spark_session, no_labels_settings, "ESA-Mission1")
        assert summary["channels_processed"] == 1

        # All train windows should be nominal (is_anomaly=False) since no labels
        out = Path(str(no_labels_settings.spark.processed_data_dir)) / "ESA-Mission1"
        train_df = spark_session.read.parquet(str(out / "train"))
        assert train_df.filter(F.col("is_anomaly")).count() == 0
