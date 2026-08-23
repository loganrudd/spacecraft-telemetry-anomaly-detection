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
from spacecraft_telemetry.core.paths import output_path
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

    ⚠ **A family is a CANDIDATE, never a verdict.** Union-find is transitive and
    intersection is not: this unions A with C whenever A-B and B-C both overlap,
    which says nothing about whether A and C share a single timestamp. Always
    read the ``check_group`` report the CLI prints for each family and group on
    ``joint_windows``, never on family membership alone.

    This is not hypothetical. Run against ESA-Mission1 on the plan-023 30 s grid,
    where every channel shares one grid, a chain of pairwise overlaps merged all
    62 channels into a single "family" whose 62-way intersection is ONE ROW.
    Grouping by the enumeration's output would have trained a model on nothing.
    The same effect makes subsystem_6's 41 channels align to 1 row: a common
    sampling grid cannot create a common calendar RANGE, and the strict
    intersection needs both.
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


def _load_named_groups(path: Path) -> dict[str, list[str]]:
    """Read a {group_name: [channel, ...]} JSON file.

    Deliberately does NOT accept an --enumerate-families report: that file's
    families are union-find candidates, and installing them as a training
    topology unchecked is the exact mistake enumerate_families' docstring
    warns about. Point this at a grouping you have measured.
    """
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict) or not raw or not all(
        isinstance(v, list) and v and all(isinstance(c, str) for c in v)
        for v in raw.values()
    ):
        raise SystemExit(f"{path}: expected a non-empty {{group_name: [channel, ...]}} object")
    seen: dict[str, str] = {}
    for name, chans in raw.items():
        for ch in chans:
            if ch in seen:
                raise SystemExit(
                    f"{path}: {ch!r} appears in both {seen[ch]!r} and {name!r}; "
                    "a channel belongs to exactly one multivariate group"
                )
            seen[ch] = name
    return {str(k): [str(c) for c in v] for k, v in raw.items()}


def write_group_map(
    settings: Settings,
    mission: str,
    families: list[list[str]],
    names: list[str] | None = None,
) -> str:
    """Write {channel_id: group_id} to the mission's processed metadata dir.

    This is the file core.metadata.load_channel_group_map reads to decide the
    multivariate fan-out's groups, so it turns a measured family enumeration
    into the actual training topology (docs/plans/023 stage .4).

    Families of one are omitted deliberately: a single-channel "group" is a
    univariate model, and listing it here would route it through the joint
    path — a 1-in/1-out multivariate model — for no benefit. Those channels
    fall out of the multivariate sweep and stay on train_all_channels.

    Group ids default to ``group_NN`` ordered by the family ordering the caller
    passes (largest first from enumerate_families), zero-padded so the registry
    sorts them in that same order. ``names`` overrides them positionally, for a
    grouping that already carries meaningful names (--groups-file).

    Returns the path written, for the caller's report.
    """
    if names is not None and len(names) != len(families):
        raise ValueError(
            f"names has {len(names)} entries but there are {len(families)} groups"
        )
    keep = [
        (name, family)
        for name, family in zip(
            names or [f"group_{i:02d}" for i in range(1, len(families) + 1)],
            families,
            strict=True,
        )
        if len(family) > 1
    ]
    multi = [family for _name, family in keep]
    mapping = {ch: name for name, family in keep for ch in family}

    metadata_dir = output_path(
        settings.preprocess.processed_data_dir, mission, settings.variant, "metadata"
    )
    if not str(metadata_dir).startswith("gs://"):
        metadata_dir.mkdir(parents=True, exist_ok=True)
    path = metadata_dir / "channel_groups.json"
    path.write_text(json.dumps(mapping, indent=2, sort_keys=True))

    log.info(
        "check_channel_group.group_map_written",
        path=str(path),
        n_groups=len(multi),
        n_channels=len(mapping),
        n_singletons=len(families) - len(multi),
    )
    return str(path)


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
    parser.add_argument(
        "--groups-file",
        help="JSON file mapping group name -> channel list. Reports each group, "
        "and with --write-groups installs exactly this grouping. Use this "
        "whenever the grouping was decided by MEASURED joint window yield "
        "rather than by --enumerate-families, whose union-find is only a "
        "candidate generator (see enumerate_families' warning).",
    )
    parser.add_argument(
        "--write-groups",
        action="store_true",
        help="Write the {channel_id: group_id} map to "
        "{processed}/{mission}/[{variant}/]metadata/channel_groups.json, "
        "which core.metadata.load_channel_group_map reads to decide the "
        "multivariate fan-out's groups. Requires --enumerate-families or "
        "--groups-file. Groups of one are omitted — a single-channel group is "
        "a univariate model, and including it would route it through the joint "
        "path for no reason.",
    )
    args = parser.parse_args()
    if sum(bool(x) for x in (args.channels, args.enumerate_families, args.groups_file)) != 1:
        raise SystemExit(
            "exactly one of --channels, --enumerate-families or --groups-file is required"
        )

    settings = load_settings(args.env)

    if args.groups_file:
        named = _load_named_groups(Path(args.groups_file))
        reports = {
            name: check_group(settings, args.mission, chans, split=args.split).as_dict()
            for name, chans in named.items()
        }
        output: dict[str, Any] = {"mission": args.mission, "groups": reports}
        if args.write_groups:
            output["groups_written_to"] = write_group_map(
                settings, args.mission, list(named.values()), names=list(named)
            )
    elif args.enumerate_families:
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
        family_reports = [
            check_group(settings, args.mission, family, split=args.split).as_dict()
            for family in families
        ]
        output = {"mission": args.mission, "families": family_reports}
        if args.write_groups:
            output["groups_written_to"] = write_group_map(settings, args.mission, families)
    else:
        if args.write_groups:
            raise SystemExit("--write-groups requires --enumerate-families or --groups-file")
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
