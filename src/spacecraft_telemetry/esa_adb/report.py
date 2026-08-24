"""Build the ESA-ADB-comparable report: our results next to the paper's own numbers.

Produces one row per (scope, source) combination:
  - "ours (protocol-matched)": Hundman-defaults (untuned) scoring, full test
    window — matches the paper's own protocol, which does not tune
    hyperparameters ("experiments do not aim to extensively tune
    hyperparameters or to find the best algorithm").
  - "ours (tuned)": tuned scoring restricted to the held-out final portion
    of the test window (the same leakage-free split score_channel() itself
    reports under eval_split="final_portion"). "Tuned" covers more than one
    provenance (Ray Tune HPO, or scripts/threshold_ceiling.py's exhaustive
    grid) — the row's actual provenance is read from the run's MLflow tags
    (esa_adb.detections.tuned_provenance) and rendered in its "params"
    column and the matching footnote, not asserted (docs/plans/022, stage
    022.2b).
  - "paper — Telemanom-ESA" / "paper — Telemanom-ESA-Pruned": hardcoded
    reference values transcribed from the paper (Table 2 / Supplementary
    Table 9), never computed — see _PAPER_REFERENCE below for citations.

Two scopes, matching the paper's own tables:
  - "all_events":     excludes only Communication Gap        (paper Table 2)
  - "anomalies_only": excludes Communication Gap + Rare Event (paper Supp. Table 9)

Residual caveats that survive even after this report (stated as footnotes in
the output, not silently dropped — see docs/plans/019):
  - Model class: Telemanom-ESA is the same Telemanom LSTM (layers [80, 80])
    run 6-in/6-out over channels 41-46 on the lightweight subset, forecasting
    10 steps ahead; ours is 1-in/1-out, one step. Telecommands are NOT an
    input there (only in the full-set runs). Disclosed; closable in principle
    by widening our input/output dims (out of scope for this plan).
  - Split (Stage A only): our train/test split (train_fraction=0.8,
    train_lookback=730D) differs from the paper's chronological 50/50 half.
    Stage B (mission="ESA-Mission1-ADB", configs/esa_adb.yaml) closes this.
  - ADTQC and affiliation-based scores (paper priorities 4-5) are not
    computed — out of scope, see docs/plans/019 Non-goals.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from spacecraft_telemetry.core.logging import get_logger
from spacecraft_telemetry.esa_adb.detections import (
    load_metadata_matching_runs,
    mission_intervals_from_per_channel,
    per_channel_detection_intervals,
    tuned_provenance,
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
from spacecraft_telemetry.model.dataset import (
    window_target_timestamps,
    window_target_timestamps_from_metadata,
)

if TYPE_CHECKING:
    from spacecraft_telemetry.core.config import Settings
    from spacecraft_telemetry.esa_adb.detections import SeriesMetadata
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
    "Model class differs: Telemanom-ESA is the same Telemanom LSTM (layers "
    "[80, 80], as here) but runs 6-in/6-out on the lightweight subset "
    "(mission1_experiments.py sets input_channels=target_channels=channels "
    "41-46) and forecasts 10 steps ahead (prediction_window_size=10); ours is "
    "1-in/1-out, one step ahead. Telecommands are NOT an input in the "
    "lightweight config -- they enter only the full-set runs (which score "
    "0.008). Window size is NOT a difference: their manifest sets "
    "window_size=250, the same Hundman default we use (256 is DC-VAE's "
    "window). See docs/plans/019 Non-goals.",
    "Split differs unless this report was run against mission='ESA-Mission1-ADB' "
    "(Stage B, configs/esa_adb.yaml): our default split is train_fraction=0.8 + "
    "train_lookback=730D; the paper's Mission1 lightweight split is a "
    "chronological 50/50 half (84 months train).",
    # The row-description footnote (was here, static) is now generated by
    # _tuned_row_footnote() and spliced in at this same position by
    # build_report — it must name each row's ACTUAL scoring provenance
    # (docs/plans/019 P2/P3, docs/plans/022 stage 022.2b), which cannot be a
    # module-level constant.
    "TNR_t's nominal-time denominator is built from per-channel, per-segment "
    "observed spans (esa_adb.timeline) — LOS/data gaps are excluded from both "
    "the annotated and nominal duration, not counted as nominal time.",
    "ADTQC and affiliation-based scores (paper priorities 4-5) are not "
    "computed — out of scope for this report (docs/plans/019 Non-goals).",
    "Subsystem-aware F-score is not reported: channels 41-46 are a single "
    "subsystem (subsystem_5), matching the paper's own omission for "
    "lightweight subsets.",
]


def _tuned_row_description(provenance: str | None, *, offline: bool) -> str:
    """Render what the "ours (tuned)" row's scoring params actually were.

    Used for both the row's "params" column and the row-description
    footnote, so the two can never disagree (docs/plans/022, stage 022.2b).
    ``provenance`` comes from esa_adb.detections.tuned_provenance, which
    reads the run's MLflow tags rather than asserting a fixed claim.
    """
    if offline:
        return (
            "tuned scoring (provenance unavailable in offline mode — a run_map "
            "was supplied, so MLflow was never contacted for tags)"
        )
    if provenance is None:
        return "tuned scoring (provenance unavailable — no tagged tuned run found)"
    return provenance


def _tuned_row_footnote(
    provenance: str | None, *, offline: bool, include_tuned: bool = True
) -> str:
    base = (
        "The 'ours (protocol-matched)' row uses Hundman-defaults (untuned) scoring "
        "over the FULL test window, matching the paper's own no-tuning protocol."
    )
    if not include_tuned:
        return base
    description = _tuned_row_description(provenance, offline=offline)
    return (
        f"{base} The 'ours (tuned)' row uses {description}, restricted to the "
        "held-out final portion (tune.hpo_eval_fraction) of the test window."
    )


def hpo_cutoff(
    settings: Settings,
    mission: str,
    channels: list[str],
    *,
    metadata_by_channel: dict[str, SeriesMetadata] | None = None,
) -> pd.Timestamp:
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

    Args:
        metadata_by_channel: When given, reuse each channel's preloaded
            (segment_ids, is_anomaly, timestamps) instead of re-reading the
            parquet partition (see build_report, docs/plans/019 P2/P3).
    """
    fraction = settings.tune.hpo_eval_fraction
    cutoffs: list[pd.Timestamp] = []
    for channel in channels:
        if metadata_by_channel is not None:
            segment_ids, is_anomaly, timestamps = metadata_by_channel[channel]
            ts = window_target_timestamps_from_metadata(
                settings, segment_ids, is_anomaly, timestamps
            )
        else:
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


# Private alias, kept so nothing breaks mid-migration. This function has two
# consumers OUTSIDE this module (ray_fanout/tune.py and
# scripts/threshold_ceiling.py), which made a leading underscore a false
# statement about its scope — a private name with external consumers is a
# likely source of a future silent break. Rename only: no behaviour changed,
# which is what keeps this inside plan 021's "esa_adb is frozen" rule.
_hpo_cutoff = hpo_cutoff


def tuned_eval_window(
    settings: Settings,
    mission: str,
    channels: list[str],
    timeline_full: list[Interval],
    *,
    metadata_by_channel: dict[str, SeriesMetadata] | None = None,
) -> list[Interval]:
    """The held-out portion of ``timeline_full`` the "tuned" row is scored against.

    This is the leakage boundary: everything before the HPO cutoff (see
    hpo_cutoff) was used to select hyperparameters, so scoring against it
    would contaminate every tuned number this report produces. Single source
    for both build_report() and scripts/diag_tp_duration.py, which must
    reproduce the exact same window to classify the same detections.

    Args:
        metadata_by_channel: See hpo_cutoff — preloaded per-channel arrays
            to avoid re-reading the parquet partition.
    """
    cutoff = hpo_cutoff(settings, mission, channels, metadata_by_channel=metadata_by_channel)
    far_future = pd.Timestamp.max.tz_localize("UTC")
    return intersect(timeline_full, [(cutoff, far_future)])


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
    include_tuned: bool = True,
) -> dict[str, Any]:
    """Build the full ESA-ADB-comparable report for ``mission``.

    Requires a baseline (untuned) scoring run — and, unless ``include_tuned``
    is False, a tuned one — to already exist for every channel in ``channels``.
    This function only reads already-logged scoring artifacts, it never trains
    or scores. Raises if any channel is missing a matching run rather than
    silently narrowing the evaluated channel set (which would quietly flatter
    the OR-aggregated metrics).

    Both rows are normally wanted: the paper reports two Telemanom variants,
    and they line up with ours — Telemanom-ESA (no tuning) against our untuned
    row, and Telemanom-ESA-Pruned (their thresholding/pruning applied) against
    our HPO-tuned row. Reporting only one would compare against half the
    paper's result.

    Args:
        run_map: When supplied, resolve runs from this offline map and read
            their arrays from staged paths instead of querying the MLflow
            tracking server (see esa_adb/offline.py). Use when the tracking
            backend is unavailable, or to pin exact run IDs for provenance.
        include_tuned: Emit the "ours (tuned)" row. Set False only when no HPO
            pass exists for the mission yet (e.g. an interrupted replication) —
            the omission is recorded in the footnotes rather than being silent.

    Returns:
        {"mission", "channels", "rows": [...], "footnotes": [...]} — see
        module docstring for row semantics and scripts/esa_adb_report.py for
        a table-formatted driver.
    """
    channels = channels or LIGHTWEIGHT_CHANNELS
    if run_map is None:
        # This is the branch where MLflow IS required (the report reads scoring
        # runs from it) — a failure here must stay visible rather than vanish
        # silently. A misconfigured tracking URI still fails loudly downstream
        # in find_scoring_run with a clear message, so it's safe to continue;
        # what must not happen is losing the root cause (e.g. an ID-token fetch
        # failure masked by ambient gcloud credentials happening to cover it).
        try:
            configure_mlflow(settings)
        except Exception as exc:
            log.warning("esa_adb.report.configure_mlflow_failed", error=str(exc))

    log.info(
        "esa_adb.report.start",
        mission=mission,
        variant=settings.variant,
        channels=channels,
        offline=run_map is not None,
        include_tuned=include_tuned,
    )

    events_df = load_events(settings, mission)

    # Preloaded once per channel and threaded explicitly through every call
    # below that would otherwise re-read the same parquet partition
    # (mission_timeline, hpo_cutoff, and per_channel_detection_intervals for
    # both the untuned and tuned variants) — 4 reads per channel down to 1.
    # Deliberately NOT an lru_cache: at ~100-channel production scale, caching
    # full load_series_parquet() output (which also includes the large
    # values column this report never uses) would be unusable against
    # CLAUDE.md's local memory ceiling. load_series_metadata() only reads the
    # small columns actually needed. See docs/plans/019 P2/P3.
    # Built per SCORING RUN's index, not blindly per channel: a channel scored
    # inside a multivariate group was windowed over the group's joint
    # (intersected) index, and rebuilding its windows from its own partition
    # would disagree with its saved errors.npy — see
    # detections.load_metadata_matching_runs. Called once per untuned/tuned
    # variant (grouping can legitimately differ between them mid-experiment),
    # sharing one group_cache so groups both variants agree on are still
    # loaded once.
    _group_cache: dict[tuple[str, ...], SeriesMetadata] = {}
    metadata_untuned: dict[str, SeriesMetadata] = load_metadata_matching_runs(
        settings, mission, channels, tuned=False, run_map=run_map, group_cache=_group_cache
    )

    timeline_full = mission_timeline(
        settings, mission, channels, metadata_by_channel=metadata_untuned
    )

    per_channel_untuned = per_channel_detection_intervals(
        settings,
        mission,
        channels,
        tuned=False,
        run_map=run_map,
        metadata_by_channel=metadata_untuned,
    )
    detections_untuned = mission_intervals_from_per_channel(per_channel_untuned)
    events_full = group_events(events_df, channels, timeline_full)

    provenance: str | None = None
    if include_tuned:
        # Read from the runs' MLflow tags rather than asserted — see
        # esa_adb.detections.tuned_provenance (docs/plans/022, stage 022.2b).
        provenance = tuned_provenance(settings, mission, channels, run_map=run_map)

        metadata_tuned: dict[str, SeriesMetadata] = load_metadata_matching_runs(
            settings, mission, channels, tuned=True, run_map=run_map, group_cache=_group_cache
        )

        timeline_tuned = tuned_eval_window(
            settings, mission, channels, timeline_full, metadata_by_channel=metadata_tuned
        )

        per_channel_tuned_full = per_channel_detection_intervals(
            settings,
            mission,
            channels,
            tuned=True,
            run_map=run_map,
            metadata_by_channel=metadata_tuned,
        )
        per_channel_tuned = {
            ch: intersect(normalize(ivs), timeline_tuned)
            for ch, ivs in per_channel_tuned_full.items()
        }
        detections_tuned_full = mission_intervals_from_per_channel(per_channel_tuned_full)
        detections_tuned = intersect(normalize(detections_tuned_full), timeline_tuned)
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

        if include_tuned:
            row_tuned = _score_row(
                scope=scope,
                label="ours (tuned)",
                events=events_tuned,
                detections=detections_tuned,
                timeline=timeline_tuned,
                excluded=excluded,
                split="held-out final portion (tune.hpo_eval_fraction)",
                params=_tuned_row_description(provenance, offline=run_map is not None),
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

    footnotes = [
        *FOOTNOTES[:2],
        _tuned_row_footnote(
            provenance, offline=run_map is not None, include_tuned=include_tuned
        ),
        *FOOTNOTES[2:],
    ]
    if not include_tuned:
        footnotes.append(
            "The 'ours (tuned)' row is OMITTED from this report (include_tuned=False): "
            "no HPO pass exists for this mission. The paper's Telemanom-ESA-Pruned "
            "result (its strongest) therefore has no counterpart row here — the "
            "comparison covers only the untuned Telemanom-ESA variant."
        )

    log.info("esa_adb.report.done", mission=mission, n_rows=len(rows))
    return {"mission": mission, "channels": channels, "rows": rows, "footnotes": footnotes}
