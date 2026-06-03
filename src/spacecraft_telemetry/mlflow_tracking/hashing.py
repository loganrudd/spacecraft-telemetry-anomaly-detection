"""Training-data fingerprinting for MLflow run tags.

Produces a cheap, stable hash over the Parquet partition that a model was
trained on.  The hash catches realistic data mutations (re-preprocessing
produces different file sizes / different files) without the cost of hashing
actual bytes.

Design: SHA-256 over a JSON-serialised list of (filename, size_bytes) pairs,
sorted by filename for determinism.  Timestamps are intentionally excluded —
they change on every re-run of identical preprocessing and would make the hash
useless as a data-identity signal.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from spacecraft_telemetry.core.paths import to_upath


def partition_hash(
    processed_data_dir: Path | str,
    mission: str,
    channel: str,
    split: str,
) -> str:
    """Return a hex-digest fingerprint of a channel's Parquet partition.

    Scans the Parquet partition directory for the given mission/channel/split
    and hashes sorted (filename, size_in_bytes) pairs.  Two calls with
    identical directory contents return identical strings.

    Args:
        processed_data_dir: Root of preprocessed output (e.g. "data/processed").
        mission:            Mission ID, e.g. "ESA-Mission1".
        channel:            Channel ID, e.g. "channel_1".
        split:              Partition name, e.g. "train" or "test".

    Returns:
        64-character lowercase hex string (SHA-256).

    Raises:
        ValueError: If the partition directory does not exist.
    """
    # to_upath so gs:// URIs resolve in the cloud (plain pathlib mangles
    # gs:// → gs:/ and reports the partition as missing).
    part_dir = (
        to_upath(processed_data_dir)
        / mission
        / split
        / f"mission_id={mission}"
        / f"channel_id={channel}"
    )
    if not part_dir.exists():
        raise ValueError(
            f"{split.capitalize()} partition directory not found: {part_dir}. "
            "Run the preprocessing pipeline before computing partition_hash."
        )
    entries = sorted(
        (p.name, p.stat().st_size)
        for p in part_dir.iterdir()
        if p.is_file()
    )
    payload = json.dumps(entries, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def training_data_hash(
    processed_data_dir: Path | str,
    mission: str,
    channel: str,
) -> str:
    """Return a hex-digest fingerprint of the channel's train Parquet partition.

    Delegates to :func:`partition_hash` with ``split="train"``.
    Kept for backward compatibility — callers that already import this name
    do not need to change.

    Args:
        processed_data_dir: Root of preprocessed output (e.g. "data/processed").
        mission:            Mission ID, e.g. "ESA-Mission1".
        channel:            Channel ID, e.g. "channel_1".

    Returns:
        64-character lowercase hex string (SHA-256).

    Raises:
        ValueError: If the train partition directory does not exist.
    """
    return partition_hash(processed_data_dir, mission, channel, "train")
