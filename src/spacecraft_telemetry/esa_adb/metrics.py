"""ESA-ADB paper metrics: corrected event-wise F-score, alarming precision,
channel-aware F-score (Kotowski et al. 2024, arXiv:2406.17826 §3.2).

All three operate on already-grouped ``Event`` objects (esa_adb.events) and
detection intervals (esa_adb.detections) — this module has no MLflow /
settings / parquet dependencies, so it is exhaustively unit testable.

Exclusion semantics (paper §3.2.2, quoted verbatim): "detections for excluded
events are ignored when counting true and false positives, and a lack of
detection is not counted as a false negative for them." Concretely:
  - an excluded event contributes no TP and no FN;
  - a detection overlapping ONLY excluded events is dropped from FP entirely
    (neither counted nor left to fall through to a "real" false positive);
  - a detection overlapping at least one *included* event counts toward that
    event's TP as usual, regardless of what else it overlaps.

Two scopes are used by the report (esa_adb.report):
  - "all_events":      exclude {"Communication Gap"}                (paper Table 2)
  - "anomalies_only":  exclude {"Communication Gap", "Rare Event"}   (paper Supp. Table 9)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from spacecraft_telemetry.esa_adb.intervals import (
    intersect,
    normalize,
    overlaps,
    subtract,
    total_duration,
)

if TYPE_CHECKING:
    from collections.abc import Collection

    from spacecraft_telemetry.esa_adb.events import Event
    from spacecraft_telemetry.esa_adb.intervals import Interval


def _split_by_exclusion(
    events: list[Event],
    excluded_categories: Collection[str],
) -> tuple[list[Event], list[Event]]:
    included = [e for e in events if e.category not in excluded_categories]
    excluded = [e for e in events if e.category in excluded_categories]
    return included, excluded


def _match(
    events: list[Event],
    detections: list[Interval],
) -> tuple[list[bool], list[bool]]:
    """Return (event_matched, detection_matched) against ``events``' intervals.

    ``event_matched[i]`` is True iff any interval of ``events[i]`` overlaps
    any detection. ``detection_matched[j]`` is True iff ``detections[j]``
    overlaps any interval of any event in ``events``.
    """
    event_matched = [any(overlaps(iv, detections) for iv in event.intervals) for event in events]
    all_event_intervals: list[Interval] = [iv for event in events for iv in event.intervals]
    detection_matched = [overlaps(d, all_event_intervals) for d in detections]
    return event_matched, detection_matched


def corrected_event_wise(
    events: list[Event],
    detections: list[Interval],
    timeline: list[Interval],
    *,
    excluded_categories: Collection[str],
    beta: float = 0.5,
) -> dict[str, Any]:
    """Corrected event-wise precision/recall/F-beta (paper eq. 1-2).

    ``Prec_corr = (TP_e / (TP_e + FP_e)) * TNR_t``, where ``TNR_t`` is the
    time-level true-negative rate over the *nominal* region — this is what
    stops an all-True detector from scoring raw precision 1.0 (paper's
    motivating example, Supplementary Fig. 11): a detector that covers the
    whole timeline has TNR_t ≈ 0, driving corrected precision to ≈ 0 even
    though every event was "detected".

    Excluded-category events are removed from the evaluated timeline
    entirely (their spans count as neither annotated nor nominal) before
    computing ``TNR_t`` — see the module docstring's Open Question note in
    docs/plans/019 for why: a Communication Gap has no real data behind it,
    so it should not be scored as "correctly quiet" nominal time either.

    Returns:
        Dict with precision, recall, f_beta (all corrected/final values),
        plus raw_precision (uncorrected event precision, for diagnostics),
        tnr, tp_e, fp_e, fn_e, n_events, n_detections.
    """
    included, excluded = _split_by_exclusion(events, excluded_categories)
    excluded_intervals = normalize([iv for e in excluded for iv in e.intervals])
    effective_timeline = subtract(normalize(timeline), excluded_intervals)

    event_matched, detection_matched = _match(included, detections)
    tp_e = sum(event_matched)
    fn_e = len(included) - tp_e

    detection_hits_excluded = [overlaps(d, excluded_intervals) for d in detections]
    fp_e = sum(
        1
        for matched, hits_excluded in zip(detection_matched, detection_hits_excluded, strict=True)
        if not matched and not hits_excluded
    )

    annotated_included = normalize([iv for e in included for iv in e.intervals])
    annotated_effective = intersect(annotated_included, effective_timeline)
    nominal_region = subtract(effective_timeline, annotated_effective)
    n_t = total_duration(nominal_region)
    detections_in_nominal = intersect(normalize(detections), nominal_region)
    tn_t = n_t - total_duration(detections_in_nominal)
    tnr_t = float(tn_t / n_t) if n_t > np.timedelta64(0, "ns") else 0.0

    raw_precision = tp_e / (tp_e + fp_e) if (tp_e + fp_e) > 0 else 0.0
    precision = raw_precision * tnr_t
    recall = tp_e / (tp_e + fn_e) if (tp_e + fn_e) > 0 else 0.0

    beta_sq = beta * beta
    f_beta = (
        (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)
        if (beta_sq * precision + recall) > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f_beta": f_beta,
        "raw_precision": raw_precision,
        "tnr": tnr_t,
        "tp_e": tp_e,
        "fp_e": fp_e,
        "fn_e": fn_e,
        "n_events": len(included),
        "n_detections": len(detections),
    }


def alarming_precision(
    events: list[Event],
    detections: list[Interval],
    *,
    excluded_categories: Collection[str],
) -> dict[str, Any]:
    """Event-wise alarming precision (paper eq. 4).

    ``PrA = TP_e / (TP_e + TP_r)``, where ``TP_r`` counts *redundant*
    detections — extra alarms beyond the first for something already
    detected.

    Redundancy is counted **per ground-truth interval, not per event**,
    matching the reference scorer (``kplabs-pl/ESA-ADB``,
    ``timeeval/metrics/ESA_ADB_metrics.py``), which tallies hits per
    interval and then sums ``count - 1`` over intervals hit more than once.
    The distinction is large here, not academic: an ESA event is a *group*
    of annotation fragments (~18 per event across 3589 fragments / 200
    events), and only fragmentation of the *detections* is genuinely
    redundant alarming. Counting per event instead — i.e.
    ``matched_detections - tp_e`` — charges an event's own fragmentation as
    redundancy, so an event whose 18 fragments are each cleanly hit once
    scores 17 redundant alarms instead of 0, biasing PrA low.

    ``tp_e`` stays event-level: it is the same TP the corrected event-wise
    scorer reports, which was verified identical to the reference.
    """
    included, _ = _split_by_exclusion(events, excluded_categories)
    event_matched, detection_matched = _match(included, detections)
    tp_e = sum(event_matched)

    tp_r = 0
    for event in included:
        for interval in event.intervals:
            n_hits = sum(1 for det in detections if overlaps(det, [interval]))
            if n_hits > 1:
                tp_r += n_hits - 1

    pr_a = tp_e / (tp_e + tp_r) if (tp_e + tp_r) > 0 else 0.0
    return {
        "alarming_precision": pr_a,
        "tp_e": tp_e,
        "tp_r": tp_r,
        "n_matched_detections": sum(detection_matched),
    }


def channel_aware(
    events: list[Event],
    per_channel_detections: dict[str, list[Interval]],
    *,
    excluded_categories: Collection[str],
    beta: float = 0.5,
) -> dict[str, Any]:
    """Channel-aware precision/recall/F-beta (paper eq. 3, per-channel variant).

    For each included event and each channel with detections: TP if the
    channel is annotated for the event AND has an overlapping detection
    within the event's span; FN if annotated but no overlapping detection;
    FP if NOT annotated but has an overlapping detection anyway. Channels
    that are neither annotated nor detected for an event are not counted
    (true negatives — uninteresting for a precision/recall pair).

    Args:
        events:                 Already-scoped Event list (event.channels is
            the annotated-channel set restricted to the evaluation's channels).
        per_channel_detections: {channel_id: detection intervals for that
            channel alone} — NOT the mission-level OR'd union; channel
            identity is exactly what this metric needs to preserve.
    """
    included, _ = _split_by_exclusion(events, excluded_categories)
    tp = fp = fn = 0
    for event in included:
        for channel, det_intervals in per_channel_detections.items():
            annotated = channel in event.channels
            detected = any(overlaps(iv, det_intervals) for iv in event.intervals)
            if annotated and detected:
                tp += 1
            elif annotated and not detected:
                fn += 1
            elif detected:  # not annotated, but detected
                fp += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    beta_sq = beta * beta
    f_beta = (
        (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)
        if (beta_sq * precision + recall) > 0
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f_beta": f_beta,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }
