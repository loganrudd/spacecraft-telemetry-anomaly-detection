"""Integration tests for injection/generate.py.

Tests build a minimal ISS test-split fixture (ISS_SERIES_FILE_SCHEMA parquet),
run generate_injected_dataset, and assert on the output layout and contents.
No I/O to GCS, no torch, no Ray.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from spacecraft_telemetry.injection.generate import (
    _discover_channels,
    generate_injected_dataset,
)
from spacecraft_telemetry.preprocess.schemas import ISS_SERIES_FILE_SCHEMA

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MISSION = "ISS"


def _make_iss_test_parquet(
    tmp_path: Path,
    channel: str,
    n: int = 2000,
    seed: int = 0,
) -> Path:
    """Write a minimal ISS test parquet for one channel."""
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2026-01-01", periods=n, freq="30s", tz="UTC")
    values = rng.standard_normal(n).astype(np.float32)
    segment_ids = np.zeros(n, dtype=np.int32)
    segment_ids[n // 2 :] = 1  # two segments
    is_anomaly = np.zeros(n, dtype=bool)
    is_los = np.zeros(n, dtype=bool)
    is_los[100:120] = True

    df = pd.DataFrame({
        "telemetry_timestamp": timestamps,
        "value_normalized": values,
        "segment_id": segment_ids,
        "is_anomaly": is_anomaly,
        "is_los": is_los,
    })

    partition_dir = (
        tmp_path / MISSION / "test"
        / f"mission_id={MISSION}" / f"channel_id={channel}"
    )
    partition_dir.mkdir(parents=True, exist_ok=True)
    out_path = partition_dir / "part.parquet"

    table = pa.Table.from_pandas(df, schema=ISS_SERIES_FILE_SCHEMA, preserve_index=False)
    pq.write_table(table, str(out_path))
    return out_path


def _make_settings(tmp_path: Path, processed_dir: Path, output_dir: Path):
    from spacecraft_telemetry.core.config import InjectionConfig, PreprocessingConfig, Settings

    return Settings(
        preprocess=PreprocessingConfig(processed_data_dir=str(processed_dir)),
        injection=InjectionConfig(
            seed=42,
            faults_per_channel=4,
            output_dir=str(output_dir),
            profiles_path=str(tmp_path / "profiles.json"),  # absent by default → fallback
        ),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGenerateInjectedDataset:
    def test_output_parquet_exists(self, tmp_path: Path) -> None:
        processed = tmp_path / "processed"
        injected = tmp_path / "injected"
        _make_iss_test_parquet(processed, "S1000003")

        settings = _make_settings(tmp_path, processed, injected)
        generate_injected_dataset(settings, MISSION, ["S1000003"])

        out_file = (
            injected / MISSION / "test"
            / f"mission_id={MISSION}" / "channel_id=S1000003" / "part.parquet"
        )
        assert out_file.exists(), "injected parquet must be written"

    def test_output_schema_matches_iss_series_file_schema(self, tmp_path: Path) -> None:
        processed = tmp_path / "processed"
        injected = tmp_path / "injected"
        _make_iss_test_parquet(processed, "S1000003")

        settings = _make_settings(tmp_path, processed, injected)
        generate_injected_dataset(settings, MISSION, ["S1000003"])

        out_file = (
            injected / MISSION / "test"
            / f"mission_id={MISSION}" / "channel_id=S1000003" / "part.parquet"
        )
        schema = pq.read_schema(str(out_file))
        for field in ISS_SERIES_FILE_SCHEMA:
            assert field.name in schema.names, f"column {field.name!r} missing from output"

    def test_both_anomaly_classes_present(self, tmp_path: Path) -> None:
        processed = tmp_path / "processed"
        injected = tmp_path / "injected"
        _make_iss_test_parquet(processed, "S1000003")

        settings = _make_settings(tmp_path, processed, injected)
        generate_injected_dataset(settings, MISSION, ["S1000003"])

        out_file = (
            injected / MISSION / "test"
            / f"mission_id={MISSION}" / "channel_id=S1000003" / "part.parquet"
        )
        df = pd.read_parquet(str(out_file))
        assert df["is_anomaly"].any(), "at least some rows must be anomaly=True"
        assert (~df["is_anomaly"]).any(), "at least some rows must be anomaly=False"

    def test_no_fault_in_los_region(self, tmp_path: Path) -> None:
        processed = tmp_path / "processed"
        injected = tmp_path / "injected"
        _make_iss_test_parquet(processed, "S1000003")

        settings = _make_settings(tmp_path, processed, injected)
        generate_injected_dataset(settings, MISSION, ["S1000003"])

        out_file = (
            injected / MISSION / "test"
            / f"mission_id={MISSION}" / "channel_id=S1000003" / "part.parquet"
        )
        df = pd.read_parquet(str(out_file))
        los_and_anomaly = df["is_los"] & df["is_anomaly"]
        assert not los_and_anomaly.any(), "no fault should land in LOS region"

    def test_row_count_unchanged(self, tmp_path: Path) -> None:
        processed = tmp_path / "processed"
        injected = tmp_path / "injected"
        _make_iss_test_parquet(processed, "S1000003", n=2000)

        settings = _make_settings(tmp_path, processed, injected)
        generate_injected_dataset(settings, MISSION, ["S1000003"])

        out_file = (
            injected / MISSION / "test"
            / f"mission_id={MISSION}" / "channel_id=S1000003" / "part.parquet"
        )
        df = pd.read_parquet(str(out_file))
        assert len(df) == 2000

    def test_manifest_written(self, tmp_path: Path) -> None:
        processed = tmp_path / "processed"
        injected = tmp_path / "injected"
        _make_iss_test_parquet(processed, "S1000003")

        settings = _make_settings(tmp_path, processed, injected)
        generate_injected_dataset(settings, MISSION, ["S1000003"])

        manifest_path = injected / MISSION / "injection_manifest.json"
        assert manifest_path.exists(), "injection_manifest.json must be written"
        manifest = json.loads(manifest_path.read_text())
        assert "S1000003" in manifest
        assert len(manifest["S1000003"]) > 0, "at least one fault record in manifest"

    def test_manifest_records_have_required_fields(self, tmp_path: Path) -> None:
        processed = tmp_path / "processed"
        injected = tmp_path / "injected"
        _make_iss_test_parquet(processed, "S1000003")

        settings = _make_settings(tmp_path, processed, injected)
        generate_injected_dataset(settings, MISSION, ["S1000003"])

        manifest = json.loads((injected / MISSION / "injection_manifest.json").read_text())
        for rec in manifest["S1000003"]:
            assert "type" in rec
            assert "start" in rec
            assert "end" in rec
            assert "duration" in rec
            assert "magnitude_sigma" in rec

    def test_determinism_under_same_seed(self, tmp_path: Path) -> None:
        processed = tmp_path / "processed"
        _make_iss_test_parquet(processed, "S1000003")

        injected_a = tmp_path / "injected_a"
        injected_b = tmp_path / "injected_b"

        settings_a = _make_settings(tmp_path, processed, injected_a)
        settings_b = _make_settings(tmp_path, processed, injected_b)
        generate_injected_dataset(settings_a, MISSION, ["S1000003"])
        generate_injected_dataset(settings_b, MISSION, ["S1000003"])

        def _load(base: Path) -> pd.DataFrame:
            path = (
                base / MISSION / "test"
                / f"mission_id={MISSION}" / "channel_id=S1000003" / "part.parquet"
            )
            return pd.read_parquet(str(path))[["is_anomaly", "value_normalized"]]

        a = _load(injected_a)
        b = _load(injected_b)
        pd.testing.assert_series_equal(a["is_anomaly"], b["is_anomaly"], check_names=False)
        pd.testing.assert_series_equal(
            a["value_normalized"], b["value_normalized"], check_names=False
        )

    def test_metadata_copied_when_present(self, tmp_path: Path) -> None:
        processed = tmp_path / "processed"
        injected = tmp_path / "injected"
        _make_iss_test_parquet(processed, "S1000003")

        # Write a channel_subsystems.json in the source
        meta_dir = processed / MISSION / "metadata"
        meta_dir.mkdir(parents=True, exist_ok=True)
        subsystem_map = {"S1000003": "thermal"}
        (meta_dir / "channel_subsystems.json").write_text(json.dumps(subsystem_map))

        settings = _make_settings(tmp_path, processed, injected)
        generate_injected_dataset(settings, MISSION, ["S1000003"])

        copied = injected / MISSION / "metadata" / "channel_subsystems.json"
        assert copied.exists(), "channel_subsystems.json must be copied to injected dir"
        assert json.loads(copied.read_text()) == subsystem_map

    def test_missing_channel_skipped_gracefully(self, tmp_path: Path) -> None:
        processed = tmp_path / "processed"
        injected = tmp_path / "injected"
        _make_iss_test_parquet(processed, "S1000003")

        settings = _make_settings(tmp_path, processed, injected)
        manifest = generate_injected_dataset(settings, MISSION, ["S1000003", "NOTACHANNEL"])

        # Only the present channel appears in the manifest
        assert "S1000003" in manifest
        assert "NOTACHANNEL" not in manifest

    def test_multi_channel_each_gets_records(self, tmp_path: Path) -> None:
        processed = tmp_path / "processed"
        injected = tmp_path / "injected"
        _make_iss_test_parquet(processed, "S1000003")
        _make_iss_test_parquet(processed, "P1000003", seed=1)

        settings = _make_settings(tmp_path, processed, injected)
        manifest = generate_injected_dataset(settings, MISSION, ["S1000003", "P1000003"])

        assert "S1000003" in manifest
        assert "P1000003" in manifest
        assert len(manifest["S1000003"]) > 0
        assert len(manifest["P1000003"]) > 0


class TestDiscoverChannels:
    def test_empty_when_no_test_dir(self, tmp_path: Path) -> None:
        result = _discover_channels(str(tmp_path), "ISS")
        assert result == []

    def test_finds_written_channels(self, tmp_path: Path) -> None:
        processed = tmp_path / "processed"
        _make_iss_test_parquet(processed, "S1000003")
        _make_iss_test_parquet(processed, "P1000003", seed=1)
        channels = _discover_channels(str(processed), "ISS")
        assert "S1000003" in channels
        assert "P1000003" in channels


class TestLoadWindowLabelsCompatibility:
    """Prove that the injected dir is a valid input to load_window_labels.

    load_window_labels is the downstream consumer that feeds labels into the
    Ray Tune HPO — if this passes, the HPO pipeline can use the injected dir
    unchanged.
    """

    def test_load_window_labels_finds_positives(self, tmp_path: Path) -> None:
        pytest.importorskip("torch")
        from spacecraft_telemetry.core.config import InjectionConfig, PreprocessingConfig, Settings
        from spacecraft_telemetry.model.dataset import load_window_labels

        processed = tmp_path / "processed"
        injected = tmp_path / "injected"
        _make_iss_test_parquet(processed, "S1000003", n=2000)

        settings = Settings(
            preprocess=PreprocessingConfig(processed_data_dir=str(processed)),
            injection=InjectionConfig(
                seed=42,
                faults_per_channel=4,
                output_dir=str(injected),
                profiles_path=str(tmp_path / "profiles.json"),
            ),
        )
        generate_injected_dataset(settings, MISSION, ["S1000003"])

        # Override processed_data_dir to point at injected dir for downstream check
        injected_settings = Settings(
            preprocess=PreprocessingConfig(processed_data_dir=str(injected)),
        )
        labels = load_window_labels(injected_settings, MISSION, "S1000003")
        assert labels.any(), "load_window_labels must return at least one True label"
        assert not labels.all(), "not all windows should be anomalous"
