"""Shared parquet-writing helper for tests/scripts/ — not a conftest fixture
because callers need full control over timestamps/segment ids per call
(phase offsets, fragmentation), not one fixed shape reused across tests.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def write_channel_metadata(
    processed_dir: Path,
    mission: str,
    channel: str,
    split: str,
    timestamps_s: list[int],
    segment_ids: list[int],
) -> None:
    """Write one channel's metadata columns at explicit timestamps/segment ids.

    Full control over both (unlike an evenly-spaced synthetic writer) so tests
    can construct phase offsets and fragmentation directly.
    """
    assert len(timestamps_s) == len(segment_ids)
    n = len(timestamps_s)
    table = pa.table(
        {
            "telemetry_timestamp": pa.array(
                [
                    pa.scalar(t, type=pa.timestamp("s", tz="UTC")).cast(
                        pa.timestamp("us", tz="UTC")
                    )
                    for t in timestamps_s
                ],
                type=pa.timestamp("us", tz="UTC"),
            ),
            "value_normalized": pa.array([0.0] * n, type=pa.float32()),
            "segment_id": pa.array(segment_ids, type=pa.int32()),
            "is_anomaly": pa.array([False] * n, type=pa.bool_()),
        }
    )
    part_dir = (
        processed_dir / mission / split / f"mission_id={mission}" / f"channel_id={channel}"
    )
    part_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, part_dir / "part.parquet")
