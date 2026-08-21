"""Tests for esa_adb.report.tuned_eval_window / hpo_cutoff — the leakage boundary.

hpo_cutoff separates the HPO-tuned portion of each channel's test window from
the held-out portion the "tuned" report row is scored against. If this is
wrong, every tuned number on this branch is contaminated (see docs/plans/019).

Reuses the series-writing pattern from test_timeline.py / test_report.py.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from spacecraft_telemetry.core.config import Settings, load_settings
from spacecraft_telemetry.esa_adb.intervals import normalize
from spacecraft_telemetry.esa_adb.report import hpo_cutoff, tuned_eval_window
from spacecraft_telemetry.esa_adb.timeline import mission_timeline

_MISSION = "ESA-Mission1-TunedWindowTest"
_FREQ_S = 90
_WINDOW_SIZE = 3
_PREDICTION_HORIZON = 1
_SPAN = _WINDOW_SIZE + _PREDICTION_HORIZON  # 4

_SERIES_SCHEMA = pa.schema(
    [
        pa.field("telemetry_timestamp", pa.timestamp("us", tz="UTC")),
        pa.field("value_normalized", pa.float32()),
        pa.field("segment_id", pa.int32()),
        pa.field("is_anomaly", pa.bool_()),
    ]
)


def _write_channel(processed_dir: Path, channel: str, n_rows: int) -> None:
    base = datetime(2000, 1, 1, tzinfo=UTC)
    timestamps = [
        pa.scalar(base.timestamp() + i * _FREQ_S, type=pa.timestamp("s", tz="UTC")).cast(
            pa.timestamp("us", tz="UTC")
        )
        for i in range(n_rows)
    ]
    table = pa.table(
        {
            "telemetry_timestamp": pa.array(timestamps, type=pa.timestamp("us", tz="UTC")),
            "value_normalized": pa.array([0.0] * n_rows, type=pa.float32()),
            "segment_id": pa.array([0] * n_rows, type=pa.int32()),
            "is_anomaly": pa.array([False] * n_rows, type=pa.bool_()),
        },
        schema=_SERIES_SCHEMA,
    )
    partition_dir = (
        processed_dir / _MISSION / "test" / f"mission_id={_MISSION}" / f"channel_id={channel}"
    )
    partition_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, partition_dir / "part.parquet")


def _settings(processed_dir: Path, hpo_eval_fraction: float = 0.6) -> Settings:
    base_settings = load_settings("test")
    return base_settings.model_copy(
        update={
            "preprocess": base_settings.preprocess.model_copy(
                update={"processed_data_dir": str(processed_dir)}
            ),
            "model": base_settings.model.model_copy(
                update={"window_size": _WINDOW_SIZE, "prediction_horizon": _PREDICTION_HORIZON}
            ),
            # model_copy(update=...) bypasses field validation, which is required
            # to exercise the fraction=1.0 clamp test below (the field_validator
            # rejects fraction=1.0 at normal construction time).
            "tune": base_settings.tune.model_copy(update={"hpo_eval_fraction": hpo_eval_fraction}),
        }
    )


class TestHpoCutoff:
    def test_cutoff_is_max_across_channels_not_min(self, tmp_path: Path) -> None:
        """A min would silently leak the HPO-tuned portion into the tuned row."""
        processed_dir = tmp_path / "processed"
        # channel_a: 20 rows -> M = 20 - 4 + 1 = 17 windows. idx = int(17*0.6) = 10.
        # channel_b: 14 rows -> M = 14 - 4 + 1 = 11 windows. idx = int(11*0.6) = 6.
        _write_channel(processed_dir, "channel_a", n_rows=20)
        _write_channel(processed_dir, "channel_b", n_rows=14)
        settings = _settings(processed_dir)

        cutoff = hpo_cutoff(settings, _MISSION, ["channel_a", "channel_b"])

        base = datetime(2000, 1, 1, tzinfo=UTC)
        # channel_a's cutoff: target timestamp at window index 10 -> row (3+10)=13.
        expected_a = base.timestamp() + 13 * _FREQ_S
        # channel_b's cutoff: target timestamp at window index 6 -> row (3+6)=9.
        expected_b = base.timestamp() + 9 * _FREQ_S
        assert expected_a > expected_b, "test fixture must produce two distinct cutoffs"

        assert cutoff.timestamp() == expected_a, (
            "cutoff must be the MAX across channels (channel_a's, the later one) — "
            "a min would leak channel_a's still-HPO-used windows into the tuned row"
        )

    def test_fraction_one_clamps_to_last_window(self, tmp_path: Path) -> None:
        """min(int(len*fraction), len-1) must not index past the last window."""
        processed_dir = tmp_path / "processed"
        _write_channel(processed_dir, "channel_a", n_rows=20)  # M = 17 windows
        settings = _settings(processed_dir, hpo_eval_fraction=1.0)

        cutoff = hpo_cutoff(settings, _MISSION, ["channel_a"])

        base = datetime(2000, 1, 1, tzinfo=UTC)
        # Without the clamp, int(17*1.0)=17 would index out of bounds (valid
        # indices are 0..16). The clamp must select index 16 (the last window).
        expected_last = base.timestamp() + (3 + 16) * _FREQ_S
        assert cutoff.timestamp() == expected_last

    def test_returned_cutoff_is_tz_aware_utc(self, tmp_path: Path) -> None:
        processed_dir = tmp_path / "processed"
        _write_channel(processed_dir, "channel_a", n_rows=20)
        settings = _settings(processed_dir)

        cutoff = hpo_cutoff(settings, _MISSION, ["channel_a"])

        assert cutoff.tzinfo is not None
        assert str(cutoff.tzinfo) == "UTC"

    def test_raises_when_no_channel_has_test_windows(self, tmp_path: Path) -> None:
        """A channel with fewer rows than window_size+prediction_horizon has 0 windows."""
        processed_dir = tmp_path / "processed"
        _write_channel(processed_dir, "channel_a", n_rows=_SPAN - 1)  # too few rows
        settings = _settings(processed_dir)

        with pytest.raises(ValueError, match="No test windows found"):
            hpo_cutoff(settings, _MISSION, ["channel_a"])


class TestTunedEvalWindow:
    def test_returned_window_is_subset_of_timeline_full(self, tmp_path: Path) -> None:
        processed_dir = tmp_path / "processed"
        _write_channel(processed_dir, "channel_a", n_rows=20)
        settings = _settings(processed_dir)

        timeline_full = mission_timeline(settings, _MISSION, ["channel_a"])
        result = tuned_eval_window(settings, _MISSION, ["channel_a"], timeline_full)

        assert result, "fixture should produce a non-empty tuned window"
        full_norm = normalize(timeline_full)
        for start, end in result:
            assert any(fs <= start and end <= fe for fs, fe in full_norm), (
                f"tuned window interval ({start}, {end}) is not a subset of "
                f"timeline_full {full_norm}"
            )

    def test_excludes_the_hpo_tuned_portion(self, tmp_path: Path) -> None:
        processed_dir = tmp_path / "processed"
        _write_channel(processed_dir, "channel_a", n_rows=20)
        settings = _settings(processed_dir)

        timeline_full = mission_timeline(settings, _MISSION, ["channel_a"])
        cutoff = hpo_cutoff(settings, _MISSION, ["channel_a"])
        result = tuned_eval_window(settings, _MISSION, ["channel_a"], timeline_full)

        assert all(start >= cutoff for start, _ in result), (
            "no interval in the tuned window may start before the HPO cutoff"
        )


def test_private_alias_still_resolves_to_the_public_name() -> None:
    """`_hpo_cutoff` had two consumers OUTSIDE esa_adb (ray_fanout/tune.py and
    scripts/threshold_ceiling.py), which made the leading underscore a false
    statement about its scope. The rename keeps the private name as an alias so
    nothing breaks mid-migration; this pins that they are the same object, not
    two implementations that could drift.
    """
    from spacecraft_telemetry.esa_adb import report

    assert report._hpo_cutoff is report.hpo_cutoff
