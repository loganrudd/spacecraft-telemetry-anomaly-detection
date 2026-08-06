"""Build the ESA-ADB-comparable report: our results next to the paper's own numbers.

Produces one row per (scope, source) combination:
  - "ours (protocol-matched)": Hundman-defaults (untuned) scoring, full test
    window — matches the paper's own protocol, which does not tune
    hyperparameters ("experiments do not aim to extensively tune
    hyperparameters or to find the best algorithm").
  - "ours (tuned)": per-subsystem Ray Tune HPO scoring, restricted to the
    held-out final portion of the test window (the same leakage-free split
    score_channel() itself reports under eval_split="final_portion").
  - "paper — Telemanom-ESA" / "paper — Telemanom-ESA-Pruned": hardcoded
    reference values transcribed from the paper (Table 2 / Supplementary
    Table 9), never computed — see _PAPER_REFERENCE below for citations.

Two scopes, matching the paper's own tables:
  - "all_events":     excludes only Communication Gap        (paper Table 2)
  - "anomalies_only": excludes Communication Gap + Rare Event (paper Supp. Table 9)

Residual caveats that survive even after this report (stated as footnotes in
the output, not silently dropped — see docs/plans/019):
  - Model class: Telemanom-ESA is multivariate-in/multi-out with all channels
    + telecommands as input; ours is univariate 1-in/1-out. Permanent,
    disclosed, not closable without reimplementing their model (out of scope).
  - Split (Stage A only): our train/test split (train_fraction=0.8,
    train_lookback=730D) differs from the paper's chronological 50/50 half.
    Stage B (mission="ESA-Mission1-ADB", configs/esa_adb.yaml) closes this.
  - ADTQC and affiliation-based scores (paper priorities 4-5) are not
    computed — out of scope, see docs/plans/019 Non-goals.
"""

from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING, Any

import pandas as pd

from spacecraft_telemetry.core.logging import get_logger
from spacecraft_telemetry.esa_adb.detections import (
    mission_detection_intervals,
    per_channel_detection_intervals,
)
from spacecraft_telemetry.esa_adb.events import group_events, load_events
from spacecraft_telemetry.esa_adb.intervals import intersect, normalize
from spacecraft_telemetry.esa_adb.metrics import (
    alarming_precision,
    channel_aware,
    corrected_event_wise,
)
from spacecraft_telemetry.esa_adb.timeline import mission_timeline
from spacecraft_telemetry.mlflow_tracking import configure_mlflow
from spacecraft_telemetry.model.dataset import window_target_timestamps

if TYPE_CHECKING:
    from spacecraft_telemetry.core.config import Settings
    from spacecraft_telemetry.esa_adb.events import Event
    from spacecraft_telemetry.esa_adb.intervals import Interval
    from spacecraft_telemetry.esa_adb.offline import OfflineRunMap

log = get_logger(__name__)

LIGHTWEIGHT_CHANNELS = [f"channel_{i}" for i in range(41, 47)]

_SCOPE_EXCLUSIONS: dict[str, frozenset[str]] = {
    "all_events": frozenset({"Communication Gap"}),
    "anomalies_only": frozenset({"Communication Gap", "Rare Event"}),
}

# Transcribed verbatim from arXiv:2406.17826 (Kotowski et al. 2024) — never
# computed. F0.5 values are the paper's own "F0.5" column; alarming_precision
# is the paper's "Alarming precision" row. Both tables cover Mission1's
# lightweight subset of channels 41-46 — the same 6 channels this report uses.
_PAPER_REFERENCE: dict[str, dict[str, dict[str, Any]]] = {
    "all_events": {
        "paper — Telemanom-ESA": {
            "precision": 0.148,
            "recall": 0.894,
            "f0_5": 0.178,
            "alarming_precision": 0.868,
            "source": "Table 2, Mission1 lightweight (ch 41-46), all events",
        },
        "paper — Telemanom-ESA-Pruned": {
            "precision": 0.999,
            "recall": 0.424,
            "f0_5": 0.786,
            "alarming_precision": 0.875,
            "source": "Table 2, Mission1 lightweight (ch 41-46), all events",
        },
    },
    "anomalies_only": {
        "paper — Telemanom-ESA": {
            "precision": 0.074,
            "recall": 0.931,
            "f0_5": 0.090,
            "alarming_precision": 0.818,
            "source": "Supplementary Table 9, Mission1 lightweight (ch 41-46), anomalies only",
        },
        "paper — Telemanom-ESA-Pruned": {
            "precision": 0.999,
            "recall": 0.862,
            "f0_5": 0.968,
            "alarming_precision": 0.862,
            "source": "Supplementary Table 9, Mission1 lightweight (ch 41-46), anomalies only",
        },
    },
}

FOOTNOTES = [
    "Model class differs (permanent, not closable here): Telemanom-ESA is "
    "multivariate-in/multi-out (all channels + telecommands as input, "
    "window=256); ours is univariate 1-in/1-out per Hundman defaults "
    "(window_size from settings.model). See docs/plans/019 Non-goals.",
    "Split differs unless this report was run against mission='ESA-Mission1-ADB' "
    "(Stage B, configs/esa_adb.yaml): our default split is train_fraction=0.8 + "
    "train_lookback=730D; the paper's Mission1 lightweight split is a "
    "chronological 50/50 half (84 months train).",
    "The 'ours (protocol-matched)' row uses Hundman-defaults (untuned) scoring "
    "over the FULL test window, matching the paper's own no-tuning protocol. "
    "The 'ours (tuned)' row uses per-subsystem Ray Tune HPO scoring restricted "
    "to the held-out final portion (tune.hpo_eval_fraction) of the test window.",
    "TNR_t's nominal-time denominator is built from per-channel, per-segment "
    "observed spans (esa_adb.timeline) — LOS/data gaps are excluded from both "
    "the annotated and nominal duration, not counted as nominal time.",
    "ADTQC and affiliation-based scores (paper priorities 4-5) are not "
    "computed — out of scope for this report (docs/plans/019 Non-goals).",
    "Subsystem-aware F-score is not reported: channels 41-46 are a single "
    "subsystem (subsystem_5), matching the paper's own omission for "
    "lightweight subsets.",
]


def _hpo_cutoff(settings: Settings, mission: str, channels: list[str]) -> pd.Timestamp:
    """Latest per-channel HPO-portion cutoff across ``channels``.

    score_channel()'s eval_split="final_portion" is the tail after the first
    tune.hpo_eval_fraction of a channel's own test windows. Different
    channels can have slightly different test-window counts, so the report
    uses the MAX (latest) cutoff across all channels — the conservative
    choice that guarantees every channel's HPO-used portion is excluded from
    the mission-level "tuned" row.

    ``ts`` is a tz-naive numpy datetime64 array (PyArrow's to_numpy() drops
    the tz label from the UTC-typed Parquet column) — localized to UTC here
    for the same reason as esa_adb.detections._flags_to_intervals and
    esa_adb.timeline.channel_timeline.
    """
    fraction = settings.tune.hpo_eval_fraction
    cutoffs: list[pd.Timestamp] = []
    for channel in channels:
        ts = window_target_timestamps(settings, mission, channel)
        if len(ts) == 0:
            continue
        idx = min(int(len(ts) * fraction), len(ts) - 1)
        cutoffs.append(pd.Timestamp(ts[idx]).tz_localize("UTC"))
    if not cutoffs:
        raise ValueError(
            f"No test windows found for any of {channels} in mission {mission!r} — "
            "has preprocessing/training run for this mission yet?"
        )
    return max(cutoffs)


def _score_row(
    *,
    scope: str,
    label: str,
    events: list[Event],
    detections: list[Interval],
    timeline: list[Interval],
    excluded: frozenset[str],
    split: str,
    params: str,
) -> dict[str, Any]:
    cew = corrected_event_wise(events, detections, timeline, excluded_categories=excluded)
    ap = alarming_precision(events, detections, excluded_categories=excluded)
    return {
        "scope": scope,
        "label": label,
        "precision": cew["precision"],
        "recall": cew["recall"],
        "f0_5": cew["f_beta"],
        "alarming_precision": ap["alarming_precision"],
        "n_events": cew["n_events"],
        "n_detections": cew["n_detections"],
        "split": split,
        "params": params,
    }


def build_report(
    settings: Settings,
    mission: str = "ESA-Mission1",
    channels: list[str] | None = None,
    run_map: OfflineRunMap | None = None,
) -> dict[str, Any]:
    """Build the full ESA-ADB-comparable report for ``mission``.

    Requires a baseline (untuned) and a tuned scoring run to already exist for
    every channel in ``channels`` — this function only reads already-logged
    scoring artifacts, it never trains or scores. Raises if any channel is
    missing a matching run rather than silently narrowing the evaluated
    channel set.

    Args:
        run_map: When supplied, resolve runs from this offline map and read
            their arrays from staged paths instead of querying the MLflow
            tracking server (see esa_adb/offline.py). Use when the tracking
            backend is unavailable, or to pin exact run IDs for provenance.

    Returns:
        {"mission", "channels", "rows": [...], "footnotes": [...]} — see
        module docstring for row semantics and scripts/esa_adb_report.py for
        a table-formatted driver.
    """
    channels = channels or LIGHTWEIGHT_CHANNELS
    if run_map is None:
        with suppress(Exception):
            configure_mlflow(settings)

    log.info(
        "esa_adb.report.start",
        mission=mission,
        channels=channels,
        offline=run_map is not None,
    )

    events_df = load_events(settings, mission)
    timeline_full = mission_timeline(settings, mission, channels)
    hpo_cutoff = _hpo_cutoff(settings, mission, channels)
    far_future = pd.Timestamp.max.tz_localize("UTC")
    timeline_tuned = intersect(timeline_full, [(hpo_cutoff, far_future)])

    per_channel_untuned = per_channel_detection_intervals(
        settings, mission, channels, tuned=False, run_map=run_map
    )
    detections_untuned = mission_detection_intervals(
        settings, mission, channels, tuned=False, run_map=run_map
    )

    per_channel_tuned_full = per_channel_detection_intervals(
        settings, mission, channels, tuned=True, run_map=run_map
    )
    per_channel_tuned = {
        ch: intersect(normalize(ivs), timeline_tuned) for ch, ivs in per_channel_tuned_full.items()
    }
    detections_tuned_full = mission_detection_intervals(
        settings, mission, channels, tuned=True, run_map=run_map
    )
    detections_tuned = intersect(normalize(detections_tuned_full), timeline_tuned)

    events_full = group_events(events_df, channels, timeline_full)
    events_tuned = group_events(events_df, channels, timeline_tuned)

    rows: list[dict[str, Any]] = []
    for scope, excluded in _SCOPE_EXCLUSIONS.items():
        row_untuned = _score_row(
            scope=scope,
            label="ours (protocol-matched, untuned)",
            events=events_full,
            detections=detections_untuned,
            timeline=timeline_full,
            excluded=excluded,
            split="full test window",
            params="Hundman defaults",
        )
        row_untuned["channel_aware_f0_5"] = channel_aware(
            events_full, per_channel_untuned, excluded_categories=excluded
        )["f_beta"]
        rows.append(row_untuned)

        row_tuned = _score_row(
            scope=scope,
            label="ours (tuned)",
            events=events_tuned,
            detections=detections_tuned,
            timeline=timeline_tuned,
            excluded=excluded,
            split="held-out final portion (tune.hpo_eval_fraction)",
            params="per-subsystem Ray Tune HPO",
        )
        row_tuned["channel_aware_f0_5"] = channel_aware(
            events_tuned, per_channel_tuned, excluded_categories=excluded
        )["f_beta"]
        rows.append(row_tuned)

        for label, ref in _PAPER_REFERENCE[scope].items():
            rows.append(
                {
                    "scope": scope,
                    "label": label,
                    "precision": ref["precision"],
                    "recall": ref["recall"],
                    "f0_5": ref["f0_5"],
                    "alarming_precision": ref["alarming_precision"],
                    "channel_aware_f0_5": None,
                    "n_events": None,
                    "n_detections": None,
                    "split": "their 50/50 chronological half-split",
                    "params": "their defaults (no tuning)",
                    "source": ref["source"],
                }
            )

    log.info("esa_adb.report.done", mission=mission, n_rows=len(rows))
    return {"mission": mission, "channels": channels, "rows": rows, "footnotes": FOOTNOTES}
