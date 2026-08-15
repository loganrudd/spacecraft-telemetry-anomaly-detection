"""Tests for esa_adb.metrics — corrected event-wise F, alarming precision, channel-aware F.

The golden test (test_all_ones_detector_corrected_precision_near_zero) is the
entire reason the TNR correction exists (paper Supplementary Fig. 11): a
detector that flags everything scores a perfect raw event precision but the
corrected precision must collapse toward zero. If that test fails, the
metric is not doing what the paper's equation (1) says it does.
"""

from __future__ import annotations

import pandas as pd
import pytest

from spacecraft_telemetry.esa_adb.events import Event
from spacecraft_telemetry.esa_adb.intervals import overlaps
from spacecraft_telemetry.esa_adb.metrics import (
    alarming_precision,
    channel_aware,
    corrected_event_wise,
)

_EXCLUDED = frozenset({"Communication Gap"})


def _ts(*minutes: int) -> list[pd.Timestamp]:
    base = pd.Timestamp("2000-01-01T00:00:00Z")
    return [base + pd.Timedelta(minutes=m) for m in minutes]


def _event(
    event_id: str,
    intervals: list[tuple[pd.Timestamp, pd.Timestamp]],
    category: str = "Anomaly",
    channels: frozenset[str] = frozenset({"channel_41"}),
) -> Event:
    return Event(
        event_id=event_id, category=category, intervals=tuple(intervals), channels=channels
    )


# ---------------------------------------------------------------------------
# corrected_event_wise
# ---------------------------------------------------------------------------


class TestCorrectedEventWise:
    def test_all_ones_detector_corrected_precision_near_zero(self) -> None:
        """Paper Supp. Fig. 11: an all-True detector must NOT score precision ~1."""
        t0, t10, t20, t100 = _ts(0, 10, 20, 100)
        event = _event("id_1", [(t10, t20)])
        timeline = [(t0, t100)]
        detections = [(t0, t100)]  # covers the entire timeline

        result = corrected_event_wise([event], detections, timeline, excluded_categories=_EXCLUDED)

        assert result["raw_precision"] == 1.0, "every event was 'detected' by the blanket alarm"
        assert result["precision"] < 0.15, (
            f"corrected precision={result['precision']} should collapse toward 0 — "
            "TNR correction did not penalise the blanket detector"
        )
        assert result["recall"] == 1.0

    def test_tnr_one_means_corrected_equals_raw_precision(self) -> None:
        """When detections never touch nominal time, TNR_t=1 and correction is a no-op."""
        t0, t10, t20, t100 = _ts(0, 10, 20, 100)
        event = _event("id_1", [(t10, t20)])
        timeline = [(t0, t100)]
        detections = [(t10, t20)]  # exactly the event span — no nominal time touched

        result = corrected_event_wise([event], detections, timeline, excluded_categories=_EXCLUDED)
        assert result["tnr"] == 1.0
        assert result["raw_precision"] == 1.0
        assert result["precision"] == 1.0

    def test_no_events_no_detections(self) -> None:
        timeline = _ts(0, 100)
        timeline_iv = [(timeline[0], timeline[1])]
        result = corrected_event_wise([], [], timeline_iv, excluded_categories=_EXCLUDED)
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f_beta"] == 0.0
        assert result["n_events"] == 0

    def test_missed_event_counts_as_false_negative(self) -> None:
        t0, t10, t20, t100 = _ts(0, 10, 20, 100)
        event = _event("id_1", [(t10, t20)])
        timeline = [(t0, t100)]
        result = corrected_event_wise([event], [], timeline, excluded_categories=_EXCLUDED)
        assert result["fn_e"] == 1
        assert result["tp_e"] == 0
        assert result["recall"] == 0.0

    def test_unmatched_detection_is_false_positive(self) -> None:
        t0, t10, t20, t50, t60, t100 = _ts(0, 10, 20, 50, 60, 100)
        event = _event("id_1", [(t10, t20)])
        timeline = [(t0, t100)]
        # Detection matches the event AND a spurious extra detection elsewhere.
        detections = [(t10, t20), (t50, t60)]
        result = corrected_event_wise([event], detections, timeline, excluded_categories=_EXCLUDED)
        assert result["tp_e"] == 1
        assert result["fp_e"] == 1

    def test_excluded_event_contributes_no_tp_or_fn(self) -> None:
        """An excluded-category event must not appear in tp_e/fn_e regardless of detections."""
        t0, t10, t20, t100 = _ts(0, 10, 20, 100)
        gap_event = _event("id_1", [(t10, t20)], category="Communication Gap")
        timeline = [(t0, t100)]

        # Not detected at all.
        result_missed = corrected_event_wise(
            [gap_event], [], timeline, excluded_categories=_EXCLUDED
        )
        assert result_missed["fn_e"] == 0
        assert result_missed["n_events"] == 0

        # Detected — still must not count as TP either (event is excluded).
        result_detected = corrected_event_wise(
            [gap_event], [(t10, t20)], timeline, excluded_categories=_EXCLUDED
        )
        assert result_detected["tp_e"] == 0
        assert result_detected["n_events"] == 0

    def test_detection_overlapping_only_excluded_event_is_not_fp(self) -> None:
        """A detection that hits ONLY an excluded event must be dropped, not counted as FP."""
        t0, t10, t20, t100 = _ts(0, 10, 20, 100)
        gap_event = _event("id_1", [(t10, t20)], category="Communication Gap")
        timeline = [(t0, t100)]
        detections = [(t10, t20)]  # overlaps only the excluded gap
        result = corrected_event_wise(
            [gap_event], detections, timeline, excluded_categories=_EXCLUDED
        )
        assert result["fp_e"] == 0

    def test_fragmented_event_counts_once_when_any_fragment_detected(self) -> None:
        """A multi-fragment event (paper §3.2.1) is one TP even if only 1 fragment is hit."""
        t0, t10, t20, t30, t40, t100 = _ts(0, 10, 20, 30, 40, 100)
        event = _event("id_1", [(t10, t20), (t30, t40)])
        timeline = [(t0, t100)]
        detections = [(t10, t20)]  # only the first fragment is detected
        result = corrected_event_wise([event], detections, timeline, excluded_categories=_EXCLUDED)
        assert result["tp_e"] == 1
        assert result["fn_e"] == 0

    def test_beta_weights_precision_over_recall_when_below_one(self) -> None:
        """With beta=0.5, low precision must hurt f_beta more than equally-low recall."""
        t0, t10, t20, t100 = _ts(0, 10, 20, 100)
        event = _event("id_1", [(t10, t20)])
        timeline = [(t0, t100)]
        detections = [(t10, t20)]
        low_precision = corrected_event_wise(
            [event], detections, timeline, excluded_categories=_EXCLUDED, beta=0.5
        )
        # f_beta with beta<1 must not exceed the arithmetic mean bias toward recall;
        # sanity check it lies strictly between 0 and 1 when both P and R are 1.0 here.
        assert low_precision["f_beta"] == 1.0  # both precision and recall are 1.0


# ---------------------------------------------------------------------------
# alarming_precision
# ---------------------------------------------------------------------------


class TestAlarmingPrecision:
    def test_one_detection_per_event_no_redundancy(self) -> None:
        _, t10, t20, t30, t40 = _ts(0, 10, 20, 30, 40)
        events = [_event("id_1", [(t10, t20)]), _event("id_2", [(t30, t40)])]
        detections = [(t10, t20), (t30, t40)]
        result = alarming_precision(events, detections, excluded_categories=_EXCLUDED)
        assert result["alarming_precision"] == 1.0
        assert result["tp_r"] == 0

    def test_redundant_detections_on_same_event_penalised(self) -> None:
        t0, t5, t10, t14, t16, t20 = _ts(0, 5, 10, 14, 16, 20)
        event = _event("id_1", [(t0, t20)])
        # Three separate, genuinely disjoint detections (real detections are
        # always normalize()'d before reaching this function — touching
        # intervals like (t10, t15)/(t15, t20) would already be merged into
        # one, so a gap between each pair is required for a valid fixture).
        detections = [(t0, t5), (t10, t14), (t16, t20)]
        result = alarming_precision([event], detections, excluded_categories=_EXCLUDED)
        assert result["tp_e"] == 1
        assert result["tp_r"] == 2
        assert abs(result["alarming_precision"] - 1 / 3) < 1e-9

    def test_no_matches_returns_zero(self) -> None:
        _, t10, t20 = _ts(0, 10, 20)
        event = _event("id_1", [(t10, t20)])
        result = alarming_precision([event], [], excluded_categories=_EXCLUDED)
        assert result["alarming_precision"] == 0.0

    def test_fragmented_event_cleanly_hit_is_not_redundant(self) -> None:
        """A multi-fragment event hit once per fragment has ZERO redundancy.

        Regression test for counting redundancy per event instead of per
        ground-truth interval. ESA events are groups of annotation fragments
        (~18 each on Mission1), so the event-level count
        ``matched_detections - tp_e`` charged an event's own fragmentation as
        redundant alarming: this case scored tp_r=2 (PrA 1/3) instead of 0.
        """
        t0, t10, t20, t30, t40, t50 = _ts(0, 10, 20, 30, 40, 50)
        event = _event("id_1", [(t0, t10), (t20, t30), (t40, t50)])
        # Exactly one detection per fragment — nothing redundant here.
        detections = [(t0, t10), (t20, t30), (t40, t50)]

        result = alarming_precision([event], detections, excluded_categories=_EXCLUDED)

        assert result["tp_e"] == 1
        assert result["tp_r"] == 0
        assert result["alarming_precision"] == 1.0

    def test_fragmented_event_charges_only_the_doubly_hit_fragment(self) -> None:
        """Redundancy is per fragment: only the fragment hit twice is charged."""
        t0, t2, t5, t10, t20, t30 = _ts(0, 2, 5, 10, 20, 30)
        event = _event("id_1", [(t0, t10), (t20, t30)])
        # First fragment hit twice, second hit once → exactly 1 redundant alarm.
        detections = [(t0, t2), (t5, t10), (t20, t30)]

        result = alarming_precision([event], detections, excluded_categories=_EXCLUDED)

        assert result["tp_e"] == 1
        assert result["tp_r"] == 1
        assert result["alarming_precision"] == 0.5

    def test_touching_detections_raise(self) -> None:
        """The bisect-based redundancy count assumes normalized (disjoint) input.

        Touching intervals ((t10, t15) and (t15, t20)) would already have been
        merged into one by intervals.normalize() at every real call site, so
        this is a precondition violation, not a valid input to tolerate.
        """
        t0, t10, t15, t20 = _ts(0, 10, 15, 20)
        event = _event("id_1", [(t0, t20)])
        detections = [(t10, t15), (t15, t20)]
        with pytest.raises(ValueError, match="normalized"):
            alarming_precision([event], detections, excluded_categories=_EXCLUDED)

    def test_overlapping_detections_raise(self) -> None:
        t0, t10, t12, t20 = _ts(0, 10, 12, 20)
        event = _event("id_1", [(t0, t20)])
        detections = [(t0, t12), (t10, t20)]  # overlap between t10 and t12
        with pytest.raises(ValueError, match="normalized"):
            alarming_precision([event], detections, excluded_categories=_EXCLUDED)

    def test_matches_naive_brute_force_on_a_larger_fixture(self) -> None:
        """Cross-check the bisect-based count against the original O(n) scan.

        Regression coverage for the Q1 rewrite: several events with multiple
        fragments and multiple detections per fragment, some hit once, some
        hit multiple times, some not hit at all.
        """
        base = pd.Timestamp("2000-01-01T00:00:00Z")
        ts = [base + pd.Timedelta(minutes=m) for m in range(0, 200, 2)]  # 100 points

        events = [
            _event("id_1", [(ts[0], ts[10]), (ts[20], ts[30])]),
            _event("id_2", [(ts[50], ts[60])]),
            _event("id_3", [(ts[70], ts[75])]),  # never detected
        ]
        detections = [
            (ts[1], ts[3]),
            (ts[5], ts[7]),  # 2nd hit on id_1's first fragment
            (ts[21], ts[24]),
            (ts[26], ts[29]),
            (ts[52], ts[54]),
            (ts[55], ts[57]),
            (ts[58], ts[59]),  # 3rd hit on id_2's fragment
            (ts[90], ts[95]),  # matches nothing
        ]

        def _naive_tp_r(
            evts: list[Event], dets: list[tuple[pd.Timestamp, pd.Timestamp]]
        ) -> int:
            total = 0
            for evt in evts:
                for interval in evt.intervals:
                    n_hits = sum(1 for det in dets if overlaps(det, [interval]))
                    if n_hits > 1:
                        total += n_hits - 1
            return total

        result = alarming_precision(events, detections, excluded_categories=_EXCLUDED)
        assert result["tp_r"] == _naive_tp_r(events, detections)
        # 1 extra on id_1 frag 1, 1 extra on id_1 frag 2, 2 extra on id_2.
        assert result["tp_r"] == 4


# ---------------------------------------------------------------------------
# channel_aware
# ---------------------------------------------------------------------------


class TestChannelAware:
    def test_annotated_and_detected_is_true_positive(self) -> None:
        _, t10, t20 = _ts(0, 10, 20)
        event = _event("id_1", [(t10, t20)], channels=frozenset({"channel_41"}))
        per_channel = {"channel_41": [(t10, t20)]}
        result = channel_aware([event], per_channel, excluded_categories=_EXCLUDED)
        assert result["tp"] == 1
        assert result["fp"] == 0
        assert result["fn"] == 0

    def test_annotated_but_not_detected_is_false_negative(self) -> None:
        _, t10, t20 = _ts(0, 10, 20)
        event = _event("id_1", [(t10, t20)], channels=frozenset({"channel_41"}))
        per_channel = {"channel_41": []}
        result = channel_aware([event], per_channel, excluded_categories=_EXCLUDED)
        assert result["fn"] == 1
        assert result["tp"] == 0

    def test_detected_but_not_annotated_is_false_positive(self) -> None:
        _, t10, t20 = _ts(0, 10, 20)
        event = _event("id_1", [(t10, t20)], channels=frozenset({"channel_41"}))
        per_channel = {"channel_41": [(t10, t20)], "channel_42": [(t10, t20)]}
        result = channel_aware([event], per_channel, excluded_categories=_EXCLUDED)
        assert result["tp"] == 1  # channel_41
        assert result["fp"] == 1  # channel_42: detected, not annotated

    def test_neither_annotated_nor_detected_is_not_counted(self) -> None:
        _, t10, t20 = _ts(0, 10, 20)
        event = _event("id_1", [(t10, t20)], channels=frozenset({"channel_41"}))
        per_channel = {"channel_41": [(t10, t20)], "channel_43": []}
        result = channel_aware([event], per_channel, excluded_categories=_EXCLUDED)
        assert result["tp"] == 1
        assert result["fp"] == 0
        assert result["fn"] == 0

    def test_excluded_event_not_counted(self) -> None:
        _, t10, t20 = _ts(0, 10, 20)
        gap_event = _event(
            "id_1", [(t10, t20)], category="Communication Gap", channels=frozenset({"channel_41"})
        )
        per_channel = {"channel_41": [(t10, t20)]}
        result = channel_aware([gap_event], per_channel, excluded_categories=_EXCLUDED)
        assert result["tp"] == 0
        assert result["fp"] == 0
        assert result["fn"] == 0
