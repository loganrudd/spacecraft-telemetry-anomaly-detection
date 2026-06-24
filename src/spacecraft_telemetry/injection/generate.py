"""Injection data-gen: inject faults into the ISS nominal test split (Phase 15).

Reads the nominal ISS test Parquet (value_normalized, segment_id, is_los),
injects faults per-channel using profiles from configs/injection_profiles.json,
writes an injected copy conforming to ISS_SERIES_FILE_SCHEMA to settings.injection.output_dir,
copies channel_subsystems.json so ray score / ray tune find their subsystem map,
and emits an injection_manifest.json capturing fault_records for each channel.

Downstream usage:
    inject run --mission ISS
    ray score --mission ISS --processed-dir data/processed_injected
    ray tune  --mission ISS --processed-dir data/processed_injected
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from spacecraft_telemetry.core.config import Settings
from spacecraft_telemetry.core.logging import get_logger
from spacecraft_telemetry.core.paths import to_upath
from spacecraft_telemetry.injection.faults import ChannelProfile, inject_faults
from spacecraft_telemetry.preprocess.io import write_series
from spacecraft_telemetry.preprocess.schemas import ISS_SERIES_FILE_SCHEMA

log = get_logger(__name__)


def _read_iss_test_series(
    processed_dir: Path | str,
    mission: str,
    channel: str,
) -> pd.DataFrame:
    """Read the test split for one ISS channel, returning all ISS columns.

    Returns a DataFrame with columns:
        telemetry_timestamp  datetime64[us, UTC]
        value_normalized     float32
        segment_id           int32
        is_anomaly           bool
        is_los               bool
    """
    partition_dir = (
        to_upath(processed_dir) / mission / "test"
        / f"mission_id={mission}" / f"channel_id={channel}"
    )
    parquet_files = sorted(partition_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No test parquet for mission={mission!r} channel={channel!r}: {partition_dir}"
        )
    columns = ["telemetry_timestamp", "value_normalized", "segment_id", "is_anomaly", "is_los"]
    tables = [pq.read_table(str(f), columns=columns) for f in parquet_files]
    table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
    table = table.sort_by("telemetry_timestamp")
    return table.to_pandas()


def _load_profiles(profiles_path: str) -> dict[str, Any]:
    """Load per-channel injection profiles from JSON, return {} if absent."""
    path = Path(profiles_path)
    if not path.exists():
        log.warning("injection_profiles_not_found", path=str(path))
        return {}
    with path.open() as f:
        return json.load(f)  # type: ignore[no-any-return]


def _copy_metadata(
    source_processed_dir: Path | str,
    dest_processed_dir: Path | str,
    mission: str,
) -> None:
    """Copy channel_subsystems.json from source to dest if present."""
    src = to_upath(source_processed_dir) / mission / "metadata" / "channel_subsystems.json"
    dst_dir = to_upath(dest_processed_dir) / mission / "metadata"

    if not src.exists():
        log.warning("channel_subsystems_not_found", path=str(src))
        return

    dst_dir.mkdir(parents=True, exist_ok=True)
    (dst_dir / "channel_subsystems.json").write_text(src.read_text())
    log.info("metadata_copied", src=str(src), dst=str(dst_dir / "channel_subsystems.json"))


def generate_injected_dataset(
    settings: Settings,
    mission: str,
    channels: list[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Inject faults into the nominal ISS test split for one mission.

    For each channel:
    1. Reads nominal test series from settings.preprocess.processed_data_dir.
    2. Loads per-channel profile from settings.injection.profiles_path.
    3. Calls inject_faults() with a deterministic per-channel seed.
    4. Writes injected series to settings.injection.output_dir using ISS_SERIES_FILE_SCHEMA.

    Also copies channel_subsystems.json metadata so ray score / ray tune can
    find the subsystem grouping without having the original processed dir set.

    Args:
        settings: Loaded Settings object.
        mission:  Mission ID (typically "ISS").
        channels: Channels to process.  If None, discovers from processed_data_dir.

    Returns:
        manifest dict: {channel_id: [fault_records, ...]}.
        Also writes manifest to {output_dir}/{mission}/injection_manifest.json.
    """
    inj = settings.injection
    cfg = settings.preprocess

    if channels is None:
        channels = _discover_channels(cfg.processed_data_dir, mission)
        log.info("discovered_channels", mission=mission, n=len(channels))

    profiles = _load_profiles(inj.profiles_path)
    output_test_dir = to_upath(inj.output_dir) / mission / "test"
    output_test_dir.mkdir(parents=True, exist_ok=True)

    rng_factory = np.random.default_rng(inj.seed)
    manifest: dict[str, list[dict[str, Any]]] = {}

    for i, channel in enumerate(channels):
        log.info("injecting_channel", channel=channel, idx=i, total=len(channels))

        try:
            df = _read_iss_test_series(cfg.processed_data_dir, mission, channel)
        except FileNotFoundError:
            log.warning("channel_missing_skipped", channel=channel)
            continue

        # Per-channel deterministic seed derived from global seed + channel index.
        channel_seed = int(rng_factory.integers(2**31))
        channel_rng = np.random.default_rng(channel_seed)

        profile_dict = profiles.get(channel, {})
        profile = ChannelProfile.from_dict(profile_dict)

        values = df["value_normalized"].to_numpy(dtype=np.float32)
        segment_ids = df["segment_id"].to_numpy(dtype=np.int32)
        # is_los: use existing column if present; default to all-False if absent
        if "is_los" in df.columns:
            is_los = df["is_los"].to_numpy(dtype=bool)
        else:
            is_los = np.zeros(len(values), dtype=bool)

        injected_values, is_anomaly_mask, fault_records = inject_faults(
            values,
            segment_ids,
            is_los,
            channel_rng,
            inj.faults_per_channel,
            profile,
            min_gap=inj.min_gap_between_faults,
            window_size=settings.model.window_size,
        )

        # Build output DataFrame matching ISS_SERIES_SCHEMA_COLS + partition cols
        out_df = pd.DataFrame({
            "telemetry_timestamp": df["telemetry_timestamp"].values,
            "value_normalized": injected_values,
            "channel_id": channel,
            "mission_id": mission,
            "segment_id": segment_ids,
            "is_anomaly": is_anomaly_mask,
            "is_los": is_los,
        })

        write_series(out_df, output_test_dir, schema=ISS_SERIES_FILE_SCHEMA)
        manifest[channel] = fault_records

        anomaly_count = int(is_anomaly_mask.sum())
        log.info(
            "channel_injected",
            channel=channel,
            n_faults=len(fault_records),
            anomaly_timesteps=anomaly_count,
        )

    # Write injection manifest — to_upath so gs:// output_dir works on Cloud Run.
    # _load_profiles intentionally stays as local Path (profiles are committed config).
    manifest_path = to_upath(inj.output_dir) / mission / "injection_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info("manifest_written", path=str(manifest_path), n_channels=len(manifest))

    # Copy subsystem metadata so downstream ray score / ray tune can group by subsystem
    _copy_metadata(cfg.processed_data_dir, inj.output_dir, mission)

    return manifest


def _discover_channels(processed_data_dir: str, mission: str) -> list[str]:
    """List channels available in the ISS test split."""
    test_root = to_upath(processed_data_dir) / mission / "test"
    if not test_root.exists():
        return []
    dirs = sorted(test_root.glob(f"mission_id={mission}/channel_id=*"))
    return [d.name.removeprefix("channel_id=") for d in dirs]
