"""Raw-tick Parquet I/O for the ISS Live collector.

Writes in-memory tick buffers to Hive-partitioned Parquet shards:
    {raw_ticks_dir}/ISS/ticks/channel_id={PUI}/{YYYYMMDDTHHMMSS}.parquet

The schema is uniform across both telemetry channels and context items
(TIME_000001, USLAB000086) so all items flow through the same flush path.

Phase 13 reads these shards to resample to the 30 s grid and run the
standard preprocessing pipeline. This module has no dependency on that
pipeline — it is purely append-only I/O.

Usage:
    from spacecraft_telemetry.ingest.collector_io import RAW_TICK_SCHEMA, flush_buffer

    flush_buffer(rows, dest_dir=Path("data/raw"), channel_id="S1000003")
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from upath import UPath

from spacecraft_telemetry.core.logging import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

RAW_TICK_SCHEMA = pa.schema(
    [
        # Canonical timestamp: UTC wall-clock receipt time. The ISSLive feed is
        # a near-real-time MERGE push, so receipt time tracks measurement time
        # within seconds — the right anchor for the 30 s resample grid.
        pa.field("telemetry_timestamp", pa.timestamp("us", tz="UTC"), nullable=False),
        # Parsed from the Lightstreamer "Value" field.
        pa.field("value", pa.float32(), nullable=False),
        # Raw ISSLive "TimeStamp" decimal preserved verbatim (AOS timestamp,
        # epoch undocumented). Nullable: some updates omit it. Kept for possible
        # later decoding; not used as the grid anchor.
        pa.field("aos_timestamp", pa.float64(), nullable=True),
    ]
)

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

_MISSION = "ISS"

# Protocols whose paths are backed by a real local filesystem and therefore need
# parent directories created before a write. Object stores (gs://, s3://, ...)
# have no directories — writing the key creates the implied prefix.
_LOCAL_PROTOCOLS = frozenset({"", "file", "local"})


def _is_local_path(path: UPath) -> bool:
    """True when ``path`` is on a local filesystem (needs explicit mkdir).

    Object-store paths (``gs://``, ``s3://``) must NOT be mkdir'd: gcsfs maps
    ``mkdir`` on the top-level component to ``storage.buckets.create``, which a
    write-scoped collector SA neither has nor should have. The bucket already
    exists and writing the object key creates the prefix implicitly.
    """
    return path.protocol in _LOCAL_PROTOCOLS


def shard_path(dest_dir: Path | UPath | str, channel_id: str, bucket_ts: datetime) -> UPath:
    """Return the UPath for a shard file.

    Args:
        dest_dir: Root raw-data directory (local or ``gs://`` URI).
        channel_id: PUI or context item name (e.g. ``"S1000003"``).
        bucket_ts: Any timestamp within the shard's time bucket (minute
            granularity is used for the filename).

    Returns:
        UPath: ``{dest_dir}/ISS/ticks/channel_id={channel_id}/{YYYYMMDDTHHMMSS}.parquet``
    """
    stamp = bucket_ts.strftime("%Y%m%dT%H%M%S")
    return (
        UPath(str(dest_dir))
        / _MISSION
        / "ticks"
        / f"channel_id={channel_id}"
        / f"{stamp}.parquet"
    )


def flush_buffer(
    rows: list[dict[str, Any]],
    dest_dir: Path | UPath | str,
    channel_id: str,
    bucket_ts: datetime | None = None,
) -> UPath | None:
    """Write a list of tick rows to a Parquet shard and return the path.

    Each row must have keys ``telemetry_timestamp`` (UTC-aware ``datetime``),
    ``value`` (float), and ``aos_timestamp`` (float or ``None``), matching
    ``RAW_TICK_SCHEMA``.

    Args:
        rows: List of row dicts. Empty list is a no-op.
        dest_dir: Root raw-data directory.
        channel_id: PUI or context item name.
        bucket_ts: Timestamp used to derive the shard filename. Defaults
            to ``datetime.now(UTC)`` when not provided.

    Returns:
        The ``UPath`` written, or ``None`` if ``rows`` was empty.
    """
    if not rows:
        return None

    ts = bucket_ts if bucket_ts is not None else datetime.now(UTC)
    path = shard_path(dest_dir, channel_id, ts)

    table = pa.Table.from_pylist(rows, schema=RAW_TICK_SCHEMA)

    # Only create parents on local filesystems. On GCS this would 403 (gcsfs
    # maps it to storage.buckets.create); the object write below creates the
    # prefix on its own.
    if _is_local_path(path):
        path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("wb") as fh:
        pq.write_table(table, fh)

    log.info(
        "collector_io.flushed",
        channel_id=channel_id,
        rows=len(rows),
        path=str(path),
    )
    return path
