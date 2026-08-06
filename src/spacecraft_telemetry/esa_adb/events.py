"""Event-level ground truth for the ESA-ADB-comparable evaluation.

``labels.csv`` has one row per ``(event_id, channel_id)`` fragment — a single
annotated event can appear as several non-contiguous fragments for the same
channel (e.g. a series of short attitude disturbances from one root cause).
The ESA-ADB paper (§3.2.1) treats all fragments sharing an ``event_id`` as one
event so that fragmentation is not double-counted as multiple anomalies —
``group_events`` implements that collapse.

``anomaly_types.csv`` carries the event-level metadata (category, class,
subclass) that ``labels.csv`` does not: exactly one row per event_id.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from spacecraft_telemetry.core.logging import get_logger
from spacecraft_telemetry.core.paths import to_upath
from spacecraft_telemetry.esa_adb.intervals import intersect, normalize
from spacecraft_telemetry.preprocess.io import read_labels

if TYPE_CHECKING:
    from spacecraft_telemetry.core.config import Settings
    from spacecraft_telemetry.esa_adb.intervals import Interval

log = get_logger(__name__)

_ANOMALY_TYPES_COLUMNS = {
    "ID": "event_id",
    "Category": "category",
    "Class": "class_",
    "Subclass": "subclass",
    "Dimensionality": "dimensionality",
}


@dataclass(frozen=True)
class Event:
    """One grouped ESA-ADB event: fragments merged, channels scoped, category kept.

    ``intervals`` is the union of every fragment's span across every annotated
    channel in the evaluation's channel set, clipped to ``timeline`` and
    normalized (sorted, non-overlapping) — i.e. exactly the "logical sum of
    annotations ... across all target channels" the paper computes event-wise
    metrics against.
    """

    event_id: str
    category: str
    intervals: tuple[Interval, ...]
    channels: frozenset[str]


def load_events(settings: Settings, mission: str) -> pd.DataFrame:
    """Read labels.csv joined to anomaly_types.csv for one mission.

    Returns one row per ``(event_id, channel_id)`` fragment (matching
    labels.csv's grain) with columns:
        event_id, channel_id, start_time, end_time,
        category, class_, subclass, dimensionality

    Raises:
        FileNotFoundError: If either source CSV is missing.
        ValueError: If any label references an event_id absent from
            anomaly_types.csv — ground truth would silently lose rows.
    """
    base = to_upath(settings.data.sample_data_dir) / mission
    labels_path = base / "labels.csv"
    types_path = base / "anomaly_types.csv"

    labels = read_labels(labels_path)  # anomaly_id, channel_id, start_time, end_time
    labels = labels.rename(columns={"anomaly_id": "event_id"})

    types = pd.read_csv(str(types_path))
    types = types.rename(columns=_ANOMALY_TYPES_COLUMNS)[
        ["event_id", "category", "class_", "subclass", "dimensionality"]
    ]

    missing = set(labels["event_id"]) - set(types["event_id"])
    if missing:
        raise ValueError(
            f"{len(missing)} event_id(s) in {labels_path} have no matching row in "
            f"{types_path}: {sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}"
        )

    merged = labels.merge(types, on="event_id", how="left", validate="many_to_one")
    log.info(
        "esa_adb.events.loaded",
        mission=mission,
        n_fragments=len(merged),
        n_events=merged["event_id"].nunique(),
    )
    columns = [
        "event_id",
        "channel_id",
        "start_time",
        "end_time",
        "category",
        "class_",
        "subclass",
        "dimensionality",
    ]
    return merged[columns]


def group_events(
    events_df: pd.DataFrame,
    channels: list[str],
    timeline: list[Interval],
) -> list[Event]:
    """Collapse per-fragment rows into one Event per event_id, scoped to ``channels``.

    Fragments whose channel is outside ``channels`` are dropped before
    grouping (an event annotated on channels 21 and 41 contributes only its
    channel_41 fragments to a 41-46-scoped evaluation). An event whose merged
    interval union does not intersect ``timeline`` at all is dropped entirely
    — it has nothing to evaluate in the window under test.

    Args:
        events_df: Output of load_events() (or an equivalent frame with the
            same columns).
        channels:  Channel IDs in scope (e.g. the six ESA-ADB lightweight
            channels 41-46).
        timeline:  The evaluated time span(s) — normalized list of intervals.
            Event intervals are clipped to this before being kept.

    Returns:
        List of Event, one per event_id that survives scoping, sorted by
        event_id for deterministic output.
    """
    scoped = events_df[events_df["channel_id"].isin(channels)]

    events: list[Event] = []
    for event_id, group in scoped.groupby("event_id", sort=True):
        raw_intervals: list[Interval] = list(
            zip(group["start_time"], group["end_time"], strict=False)
        )
        clipped = intersect(normalize(raw_intervals), timeline)
        if not clipped:
            continue
        events.append(
            Event(
                event_id=str(event_id),
                category=str(group["category"].iloc[0]),
                intervals=tuple(clipped),
                channels=frozenset(group["channel_id"]),
            )
        )
    return events
