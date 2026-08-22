"""Preflight check for a multivariate channel group's viability — before any GPU is touched.

docs/plans/023-channel-time-grid.md, stage 023.1. ``model.dataset._align_multi_channel``
takes a strict timestamp intersection across a group and raises if it's empty — but
that's discovered only after a worker (potentially a GPU one) has already spun up.
This tool answers the same question up front, and reports the numbers that
matter even when the group IS viable: two channels can align perfectly and
still be useless for training if their joint segmentation is shattered (see
"Fragmentation" below).

Reads ONLY the metadata columns (segment_id, is_anomaly, telemetry_timestamp) —
never value_normalized — and reuses model.dataset's exact alignment path
(load_multichannel_series_metadata -> _align_multi_channel), so this tool's
answer is the one training will get, not an approximation of it.

Reported per group:
    per_channel_rows        — row count of each member, read independently
    n_aligned                — rows surviving the intersection across ALL members
    alignment_loss_frac      — 1 - n_aligned / max(per_channel_rows): how much of
                                the LARGEST member's own data is thrown away by
                                joining it to the rest of the group
    n_joint_segments         — number of joint segments (see _joint_segment_ids):
                                a boundary exists wherever ANY member's own gap
                                detection drew one
    max_joint_segment_len    — longest joint segment, in rows
    joint_windows            — valid training windows at the configured
                                window_size / prediction_horizon / forecast_steps
                                (model.dataset.window_span), summed over joint
                                segments

Fragmentation: a group can have near-100% alignment and still yield almost no
usable windows, because ``_align_multi_channel`` cuts a joint segment boundary
wherever ANY single member's independently-computed gap detection cut one.
29 channels can agree on every timestamp and still fracture into 900k+ 1-row
segments if one member's cadence is 240x faster than another's. This is why
``joint_windows``, not ``n_aligned``, is the number that decides whether a
group is trainable.

Usage:
    # Check one proposed group
    python scripts/check_channel_group.py --env cloud --mission ESA-Mission1 \\
        --channels channel_41,channel_42,channel_43,channel_44,channel_45,channel_46

    # Enumerate every channel into families that share a timestamp grid
    # (union-find over pairwise probe-window overlap), then report each one
    python scripts/check_channel_group.py --env cloud --mission ESA-Mission1 \\
        --enumerate-families --out outputs/channel_families.json

Requires: base install only (no torch/ray/mlflow) — this is pure PyArrow + pandas + numpy.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

# Allow running as a script without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spacecraft_telemetry.core.config import Settings, load_settings
from spacecraft_telemetry.core.logging import get_logger
from spacecraft_telemetry.model.dataset import (
    load_multichannel_series_metadata,
    load_series_metadata,
    window_span,
)

log = get_logger(__name__)

# Two channels on the same cadence either overlap almost completely within any
# shared window, or not at all (docs/plans/023: channel_70/71, same 30s
# cadence, 16.1s phase offset, ZERO shared timestamps over 2.9M rows each).
# There is no meaningful middle ground to tune a threshold against, so this is
# a coarse divider rather than a sensitive parameter.
_DEFAULT_OVERLAP_THRESHOLD = 0.5
_DEFAULT_PROBE_DAYS = 7


@dataclass(frozen=True)
class GroupReport:
    channels: list[str]
    per_channel_rows: dict[str, int]
    n_aligned: int
    alignment_loss_frac: float
    n_joint_segments: int
    max_joint_segment_len: int
    joint_windows: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "channels": self.channels,
            "per_channel_rows": self.per_channel_rows,
            "n_aligned": self.n_aligned,
            "alignment_loss_frac": round(self.alignment_loss_frac, 4),
            "n_joint_segments": self.n_joint_segments,
            "max_joint_segment_len": self.max_joint_segment_len,
            "joint_windows": self.joint_windows,
        }


def check_group(
    settings: Settings,
    mission: str,
    channels: list[str],
    split: Literal["train", "test"] = "train",
) -> GroupReport:
    """Report one channel group's multivariate viability.

    Per-channel row counts come from independent load_series_metadata calls
    (cheap, metadata-only); alignment and joint segmentation come from
    load_multichannel_series_metadata, i.e. the SAME join
    (_align_multi_channel) training will run — not a re-derivation of it.
    """
    processed_dir = settings.preprocess.processed_data_dir
    variant = settings.variant

    per_channel_rows: dict[str, int] = {}
    for ch in channels:
        _seg, _anom, ts = load_series_metadata(processed_dir, mission, ch, split, variant=variant)
        per_channel_rows[ch] = len(ts)
    max_rows = max(per_channel_rows.values()) if per_channel_rows else 0

    try:
        joint_segment_ids, _is_anomaly, timestamps = load_multichannel_series_metadata(
            processed_dir, mission, channels, split, variant=variant
        )
    except ValueError as exc:
        if "No overlapping timestamps" not in str(exc):
            raise
        return GroupReport(
            channels=channels,
            per_channel_rows=per_channel_rows,
            n_aligned=0,
            alignment_loss_frac=1.0,
            n_joint_segments=0,
            max_joint_segment_len=0,
            joint_windows=0,
        )

    n_aligned = len(timestamps)
    loss_frac = 1.0 - (n_aligned / max_rows) if max_rows else 0.0

    seg_lengths = np.bincount(joint_segment_ids)
    span = window_span(settings.model)
    joint_windows = int(np.clip(seg_lengths - span + 1, 0, None).sum())

    return GroupReport(
        channels=channels,
        per_channel_rows=per_channel_rows,
        n_aligned=n_aligned,
        alignment_loss_frac=loss_frac,
        n_joint_segments=int(seg_lengths.size),
        max_joint_segment_len=int(seg_lengths.max()),
        joint_windows=joint_windows,
    )


def _probe_timestamps(
    settings: Settings,
    mission: str,
    channel: str,
    split: Literal["train", "test"],
    probe_days: int,
) -> pd.DatetimeIndex:
    """One channel's timestamps restricted to the first ``probe_days`` of its own range.

    Used only to decide family membership cheaply — see enumerate_families.
    """
    _seg, _anom, ts = load_series_metadata(
        settings.preprocess.processed_data_dir, mission, channel, split, variant=settings.variant
    )
    idx = pd.DatetimeIndex(ts)
    if len(idx) == 0:
        return idx
    start = idx.min()
    end = start + pd.Timedelta(days=probe_days)
    return idx[(idx >= start) & (idx < end)]


def enumerate_families(
    settings: Settings,
    mission: str,
    channels: list[str],
    split: Literal["train", "test"] = "train",
    probe_days: int = _DEFAULT_PROBE_DAYS,
    overlap_threshold: float = _DEFAULT_OVERLAP_THRESHOLD,
) -> list[list[str]]:
    """Union-find channels into families that share a timestamp grid.

    Compares each pair on a PROBE window (the first ``probe_days`` of each
    channel's own range) rather than a full-series intersection: two channels
    on the same cadence overlap almost completely within any shared window,
    or not at all (see module docstring), so a probe is representative
    without paying for a full-series join on every one of the O(C^2) pairs.

    Returns families sorted largest-first, each a list of channel_ids.
    """
    probes = {ch: _probe_timestamps(settings, mission, ch, split, probe_days) for ch in channels}

    parent = {ch: ch for ch in channels}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, a in enumerate(channels):
        idx_a = probes[a]
        if len(idx_a) == 0:
            continue
        for b in channels[i + 1 :]:
            idx_b = probes[b]
            if len(idx_b) == 0:
                continue
            smaller = min(len(idx_a), len(idx_b))
            overlap = len(idx_a.intersection(idx_b))
            if overlap / smaller >= overlap_threshold:
                union(a, b)

    groups: dict[str, list[str]] = {}
    for ch in channels:
        groups.setdefault(find(ch), []).append(ch)
    return sorted((sorted(g) for g in groups.values()), key=lambda g: (-len(g), g[0]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check a multivariate channel group's viability before training."
    )
    parser.add_argument("--env", default="local", help="Config env (local, cloud, test).")
    parser.add_argument("--mission", required=True, help="Mission name, e.g. ESA-Mission1.")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument(
        "--channels", help="Comma-separated channel_ids to check as ONE group."
    )
    parser.add_argument(
        "--enumerate-families",
        action="store_true",
        help="Union-find every preprocessed channel into families sharing a "
        "timestamp grid, and report each family, instead of checking one "
        "explicit --channels group.",
    )
    parser.add_argument("--probe-days", type=int, default=_DEFAULT_PROBE_DAYS)
    parser.add_argument("--overlap-threshold", type=float, default=_DEFAULT_OVERLAP_THRESHOLD)
    parser.add_argument("--out", help="Also write the JSON report to this path.")
    args = parser.parse_args()

    settings = load_settings(args.env)

    if args.enumerate_families:
        from spacecraft_telemetry.ray_fanout.runner import discover_channels

        channels = discover_channels(settings, args.mission)
        if not channels:
            raise SystemExit(
                f"No preprocessed channels found for mission={args.mission!r} — "
                "nothing to enumerate."
            )
        families = enumerate_families(
            settings,
            args.mission,
            channels,
            split=args.split,
            probe_days=args.probe_days,
            overlap_threshold=args.overlap_threshold,
        )
        log.info(
            "check_channel_group.families_found",
            mission=args.mission, n_channels=len(channels), n_families=len(families),
        )
        reports = [
            check_group(settings, args.mission, family, split=args.split).as_dict()
            for family in families
        ]
        output: dict[str, Any] = {"mission": args.mission, "families": reports}
    else:
        if not args.channels:
            raise SystemExit("--channels is required unless --enumerate-families is set")
        channels = [c.strip() for c in args.channels.split(",") if c.strip()]
        output = check_group(settings, args.mission, channels, split=args.split).as_dict()

    text = json.dumps(output, indent=2, default=str)
    print(text)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text)


if __name__ == "__main__":
    main()
