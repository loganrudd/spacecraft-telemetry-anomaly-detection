"""Measure whether resampling to a common grid fixes multivariate fragmentation — 023.2.

docs/plans/023-channel-time-grid.md, stage 023.2. ``scripts/check_channel_group.py``
(023.1) measures a group's viability on its NATIVE, unresampled timestamps. This
script measures the SAME group under a candidate resample rate, so the two can be
compared side by side before any preprocessing is touched (stage 023.3 is the only
place the on-disk data changes).

Two independent things a common grid can fix, both measured here:
  1. Phase-drift alignment loss — two channels on "the same" cadence whose exact
     tick times drift apart (docs/plans/023: channel_70/channel_71, same 30s
     cadence, 16.1s phase offset, ZERO shared native timestamps over millions of
     rows). Bucketing to a coarser grid puts both channels' ticks in the same
     bucket regardless of exact phase.
  2. Fragmentation — a joint segment boundary exists wherever ANY member's own
     gap detection drew one (model.dataset._joint_segment_ids), so one
     fast/noisy channel's fine-grained gaps can shatter a group that is
     otherwise 100% aligned. Bucketing coarsens away short native gaps that
     don't survive as a genuine missing bucket.

This script buckets via preprocess.transforms.resample_to_grid(...,
gap_preserving=True) — the SAME bucketing production preprocessing runs
(023.3), so a bucket exists in the output iff at least one native tick floors
into it; nothing is invented for buckets no channel ever sampled. Values are
irrelevant here (only bucket timestamps are used downstream), so a dummy
value column is passed in. Segment ids on the bucketed series are then
re-derived with the SAME detect_gaps() production preprocessing already uses
(unmodified, imported directly) so segment semantics match stage 023.3's
eventual output rather than a bespoke approximation of it.

This stage makes NO preprocessing changes — it is a measurement tool. Its
gate is a per-rate cost table archived to outputs/, not a data change.

Usage:
    # One group, one rate, printed and archived
    python scripts/grid_resample_cost.py --env cloud --mission ESA-Mission1 \\
        --channels channel_41,...,channel_49 --rates 30,90,300 \\
        --out-dir outputs

Requires: base install only (no torch/ray/mlflow) — pure PyArrow + pandas + numpy,
same as check_channel_group.py, which this script imports for the native baseline.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

# Allow running as a script without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
# Sibling script, not an installed package — see check_channel_group.py's own
# "reuse the exact same alignment code" rationale; the native baseline here
# must be the SAME computation 023.1 reports, not a second implementation of it.
sys.path.insert(0, str(Path(__file__).parent))

from check_channel_group import check_group

from spacecraft_telemetry.core.config import Settings, load_settings
from spacecraft_telemetry.core.logging import get_logger
from spacecraft_telemetry.model.dataset import (
    _joint_segment_ids,
    load_series_metadata,
    window_span,
)
from spacecraft_telemetry.preprocess.transforms import detect_gaps, resample_to_grid

log = get_logger(__name__)

_DEFAULT_GAP_MULTIPLIER = 3.0
# A group is "materially" fixed by resampling if the best candidate rate at
# least quintuples the native joint-window count — well above noise, well
# below the 31x plan 023 measured on family 1, so it won't fire on a rounding
# difference but also won't demand reproducing the largest measured win.
_MATERIAL_IMPROVEMENT_RATIO = 5.0


@dataclass(frozen=True)
class RateCost:
    rate_s: int
    per_channel_rows: dict[str, int]
    n_aligned: int
    n_joint_segments: int
    max_joint_segment_len: int
    joint_windows: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "rate_s": self.rate_s,
            "per_channel_rows": self.per_channel_rows,
            "n_aligned": self.n_aligned,
            "n_joint_segments": self.n_joint_segments,
            "max_joint_segment_len": self.max_joint_segment_len,
            "joint_windows": self.joint_windows,
        }


def _segment_bucketed(
    bucket_ts: pd.DatetimeIndex, gap_multiplier: float
) -> np.ndarray[Any, np.dtype[np.int32]]:
    """Re-derive segment ids on a bucketed timestamp series via detect_gaps.

    detect_gaps (preprocess/transforms.py, unmodified) only needs a
    'telemetry_timestamp' column, so this reuses production gap semantics at
    the new grid resolution rather than approximating them.
    """
    if len(bucket_ts) == 0:
        return np.empty(0, dtype=np.int32)
    df = pd.DataFrame({"telemetry_timestamp": bucket_ts})
    segmented = detect_gaps(df, gap_multiplier=gap_multiplier)
    return segmented["segment_id"].to_numpy(dtype=np.int32)  # type: ignore[no-any-return]


# Memo for _native_timestamps. Keyed on everything that selects a partition, so
# two missions/variants/splits never collide. LRU-bounded rather than
# process-lifetime unbounded: both 023 scripts advertise "base install only" —
# laptop use — and each entry holds a full channel's native timestamp array
# (measured ~1.4GB total for ESA-Mission1's 62 channels held at once). A
# --groups-file run over --enumerate-families output can touch every channel
# in a mission, so an unbounded cache scales with the mission, not the
# channels actually needed at once; capping trades a possible re-read (a GCS
# round trip) for a fixed memory ceiling.
_TIMESTAMP_CACHE_MAX_ENTRIES = 16
_TIMESTAMP_CACHE: OrderedDict[tuple[str, str, str, str, str | None], pd.DatetimeIndex] = (
    OrderedDict()
)


def _native_timestamps(
    settings: Settings,
    mission: str,
    channel: str,
    split: Literal["train", "test"],
) -> pd.DatetimeIndex:
    """One channel's native timestamps, memoized for the life of the process.

    build_cost_table evaluates every candidate rate over the same channels, and
    --groups-file evaluates many groups; a 29-channel family re-read once per
    rate is a few hundred GCS round trips for data that cannot have changed
    between them. LRU-bounded at _TIMESTAMP_CACHE_MAX_ENTRIES — see that
    constant's comment.
    """
    processed_dir = str(settings.preprocess.processed_data_dir)
    key = (processed_dir, mission, channel, split, settings.variant)
    if key in _TIMESTAMP_CACHE:
        _TIMESTAMP_CACHE.move_to_end(key)
        return _TIMESTAMP_CACHE[key]

    _seg, _anom, ts = load_series_metadata(
        processed_dir, mission, channel, split, variant=settings.variant
    )
    _TIMESTAMP_CACHE[key] = pd.DatetimeIndex(ts)
    if len(_TIMESTAMP_CACHE) > _TIMESTAMP_CACHE_MAX_ENTRIES:
        _TIMESTAMP_CACHE.popitem(last=False)
    return _TIMESTAMP_CACHE[key]


def measure_resampled(
    settings: Settings,
    mission: str,
    channels: list[str],
    rate_s: int,
    split: Literal["train", "test"] = "train",
    gap_multiplier: float = _DEFAULT_GAP_MULTIPLIER,
) -> RateCost:
    """The resampled counterpart to check_channel_group.check_group.

    Reads the same per-channel metadata (native timestamps only — segment_id
    on disk is native-resolution and not reused here, since bucketing changes
    what counts as a gap), buckets each channel independently, inner-joins the
    bucket grids, and re-derives joint segments/windows exactly as
    check_group does for the native case.
    """
    per_channel_buckets: dict[str, pd.DatetimeIndex] = {}
    per_channel_rows: dict[str, int] = {}
    for ch in channels:
        ts = _native_timestamps(settings, mission, ch, split)
        # Only the resulting bucket timestamps are used downstream — segment
        # ids are re-derived separately via detect_gaps — so the value column
        # resample_to_grid's mean aggregation reads is a dummy.
        ticks_df = pd.DataFrame({"telemetry_timestamp": ts, "value": 0.0})
        resampled = resample_to_grid(
            ticks_df, ch, mission, grid_interval_seconds=rate_s, gap_preserving=True
        )
        buckets = pd.DatetimeIndex(resampled["telemetry_timestamp"])
        per_channel_buckets[ch] = buckets
        per_channel_rows[ch] = len(buckets)

    common = per_channel_buckets[channels[0]]
    for ch in channels[1:]:
        common = common.intersection(per_channel_buckets[ch])
    common = common.sort_values()
    n_aligned = len(common)

    if n_aligned == 0:
        return RateCost(rate_s, per_channel_rows, 0, 0, 0, 0)

    per_channel_seg = np.empty((n_aligned, len(channels)), dtype=np.int32)
    for i, ch in enumerate(channels):
        seg_ids = _segment_bucketed(per_channel_buckets[ch], gap_multiplier)
        pos = per_channel_buckets[ch].get_indexer(common)
        per_channel_seg[:, i] = seg_ids[pos]

    joint_seg = _joint_segment_ids(per_channel_seg)
    seg_lengths = np.bincount(joint_seg)
    span = window_span(settings.model)
    joint_windows = int(np.clip(seg_lengths - span + 1, 0, None).sum())

    return RateCost(
        rate_s=rate_s,
        per_channel_rows=per_channel_rows,
        n_aligned=n_aligned,
        n_joint_segments=int(seg_lengths.size),
        max_joint_segment_len=int(seg_lengths.max()),
        joint_windows=joint_windows,
    )


def build_cost_table(
    settings: Settings,
    mission: str,
    channels: list[str],
    rates_s: list[int],
    split: Literal["train", "test"] = "train",
    gap_multiplier: float = _DEFAULT_GAP_MULTIPLIER,
) -> dict[str, Any]:
    """Native baseline + one resampled measurement per candidate rate, plus a verdict.

    The verdict is a per-group heuristic, not the plan's full mission-wide
    branch decision (that compares across many groups of different sizes and
    is a written judgement call, not a formula) — it flags whether THIS group
    clears _MATERIAL_IMPROVEMENT_RATIO at its best candidate rate, which is
    the input that judgement needs.
    """
    native = check_group(settings, mission, channels, split=split)
    resampled = {rate: measure_resampled(settings, mission, channels, rate, split, gap_multiplier)
                 for rate in rates_s}

    best_rate = max(resampled, key=lambda r: resampled[r].joint_windows)
    best_windows = resampled[best_rate].joint_windows
    native_windows = native.joint_windows
    ratio = (best_windows / native_windows) if native_windows else (
        float("inf") if best_windows else 0.0
    )
    verdict = (
        "resampling materially raises joint window yield"
        if ratio >= _MATERIAL_IMPROVEMENT_RATIO
        else "yield stays collapsed — fragmentation is not fixed by resampling"
    )

    return {
        "mission": mission,
        "channels": channels,
        "native": native.as_dict(),
        "resampled": {str(rate): cost.as_dict() for rate, cost in resampled.items()},
        "best_rate_s": best_rate,
        "improvement_ratio": None if ratio == float("inf") else round(ratio, 2),
        "verdict": verdict,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure grid-resample cost/benefit for a channel group (023.2)."
    )
    parser.add_argument("--env", default="local", help="Config env (local, cloud, test).")
    parser.add_argument("--mission", required=True, help="Mission name, e.g. ESA-Mission1.")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument(
        "--channels", help="Comma-separated channel_ids to check as ONE group."
    )
    parser.add_argument(
        "--groups-file",
        help="JSON file mapping group name -> channel list, measured as SEPARATE "
        "groups in one process (so the per-channel metadata read is shared "
        "across every group and rate). Mutually exclusive with --channels. "
        "Accepts either {name: [channels]} or check_channel_group.py's "
        "--enumerate-families output, whose families are named family_1, ....",
    )
    parser.add_argument(
        "--rates", required=True,
        help="Comma-separated candidate grid intervals in seconds, e.g. 30,90,300.",
    )
    parser.add_argument("--gap-multiplier", type=float, default=_DEFAULT_GAP_MULTIPLIER)
    parser.add_argument(
        "--out-dir", default="outputs",
        help="Directory to write grid_cost_<rate>s.json into (one per rate) plus "
        "a combined grid_cost_summary.json.",
    )
    args = parser.parse_args()
    if bool(args.channels) == bool(args.groups_file):
        raise SystemExit("exactly one of --channels or --groups-file is required")

    settings = load_settings(args.env)
    rates = [int(r.strip()) for r in args.rates.split(",") if r.strip()]

    if args.channels:
        groups = {"group": [c.strip() for c in args.channels.split(",") if c.strip()]}
    else:
        groups = _load_groups(Path(args.groups_file))

    tables: dict[str, dict[str, Any]] = {}
    for name, channels in groups.items():
        table = build_cost_table(
            settings, args.mission, channels, rates, split=args.split,
            gap_multiplier=args.gap_multiplier,
        )
        tables[name] = table
        log.info(
            "grid_resample_cost.measured",
            group=name, mission=args.mission, n_channels=len(channels), rates=rates,
            native_windows=table["native"]["joint_windows"],
            best_rate_s=table["best_rate_s"], verdict=table["verdict"],
        )

    print(json.dumps(tables, indent=2, default=str))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for rate in rates:
        rate_report = {
            "mission": args.mission,
            "groups": {
                name: {
                    "channels": t["channels"],
                    "native": t["native"],
                    "resampled": t["resampled"][str(rate)],
                }
                for name, t in tables.items()
            },
        }
        (out_dir / f"grid_cost_{rate}s.json").write_text(json.dumps(rate_report, indent=2))
    (out_dir / "grid_cost_summary.json").write_text(
        json.dumps({"mission": args.mission, "groups": tables}, indent=2)
    )


def _load_groups(path: Path) -> dict[str, list[str]]:
    """Read a group definition file — see --groups-file.

    check_channel_group.py's --enumerate-families output is accepted directly so
    the two 023 tools chain without a hand-written intermediate file.
    """
    raw = json.loads(path.read_text())
    if isinstance(raw, dict) and "families" in raw:
        return {
            f"family_{i}": fam["channels"] for i, fam in enumerate(raw["families"], start=1)
        }
    if not isinstance(raw, dict) or not all(
        isinstance(v, list) and v for v in raw.values()
    ):
        raise SystemExit(
            f"{path}: expected {{group_name: [channel, ...]}} or an --enumerate-families report"
        )
    return {str(k): [str(c) for c in v] for k, v in raw.items()}


if __name__ == "__main__":
    main()
