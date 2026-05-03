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


def training_data_hash(
    processed_data_dir: Path | str,
    mission: str,
    channel: str,
) -> str:
    """Return a hex-digest fingerprint of the channel's train Parquet partition.

    Scans the Parquet partition directory for the given mission/channel and
    hashes sorted (filename, size_in_bytes) pairs.  Two calls with identical
    directory contents return identical strings.

    Args:
        processed_data_dir: Root of Spark processed output (e.g. "data/processed").
        mission:            Mission ID, e.g. "ESA-Mission1".
        channel:            Channel ID, e.g. "channel_1".

    Returns:
        64-character lowercase hex string (SHA-256).

    Raises:
        ValueError: If the train partition directory does not exist.
    """
    train_dir = (
        Path(str(processed_data_dir))
        / mission
        / "train"
        / f"mission_id={mission}"
        / f"channel_id={channel}"
    )
    if not train_dir.exists():
        raise ValueError(
            f"Train partition directory not found: {train_dir}. "
            "Run the Spark preprocessing pipeline before computing training_data_hash."
        )
    entries = sorted(
        (p.name, p.stat().st_size)
        for p in train_dir.iterdir()
        if p.is_file()
    )
    payload = json.dumps(entries, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()
