"""Tests for ISS-specific transforms in preprocess/transforms.py.

All tests use synthetic in-memory DataFrames.  No I/O or Ray required.
"""

from __future__ import annotations

import pandas as pd

from spacecraft_telemetry.preprocess.transforms import (
    augment_with_los,
    compute_los_mask,
    resample_to_grid,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_raw_ticks(
    n: int = 60,
    start: str = "2026-06-01T00:00:00Z",
    interval_s: float = 2.0,
) -> pd.DataFrame:
    """Return a synthetic raw-tick DataFrame (2-second cadence by default)."""
    base = pd.Timestamp(start, tz="UTC")
    ts = [base + pd.Timedelta(seconds=i * interval_s) for i in range(n)]
    return pd.DataFrame(
        {
            "telemetry_timestamp": pd.array(ts, dtype="datetime64[us, UTC]"),
            "value": pd.array([float(i % 10) for i in range(n)], dtype="float32"),
            "aos_timestamp": [None] * n,
        }
    )


def _make_all_ticks_df(
    channels: list[str],
    n_per_channel: int = 60,
    start: str = "2026-06-01T00:00:00Z",
    interval_s: float = 2.0,
) -> pd.DataFrame:
    """Return a combined all-channel tick DataFrame for LOS detection."""
    frames = []
    base = pd.Timestamp(start, tz="UTC")
    for ch in channels:
        ts = [base + pd.Timedelta(seconds=i * interval_s) for i in range(n_per_channel)]
        mini = pd.DataFrame(
            {
                "telemetry_timestamp": pd.array(ts, dtype="datetime64[us, UTC]"),
                "channel_id": ch,
            }
        )
        frames.append(mini)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# TestResampleToGrid
# ---------------------------------------------------------------------------


class TestResampleToGrid:
    def test_output_columns(self) -> None:
        ticks = _make_raw_ticks(n=120)
        result = resample_to_grid(ticks, "S1000003", "ISS", grid_interval_seconds=30)
        assert list(result.columns) == [
            "telemetry_timestamp",
            "value",
            "channel_id",
            "mission_id",
        ]

    def test_channel_and_mission_columns(self) -> None:
        ticks = _make_raw_ticks(n=60)
        result = resample_to_grid(ticks, "P4000001", "ISS", grid_interval_seconds=30)
        assert (result["channel_id"] == "P4000001").all()
        assert (result["mission_id"] == "ISS").all()

    def test_grid_is_regular(self) -> None:
        ticks = _make_raw_ticks(n=120)
        result = resample_to_grid(ticks, "S1000003", "ISS", grid_interval_seconds=30)
        diffs = result["telemetry_timestamp"].diff().iloc[1:].dt.total_seconds()
        assert (diffs == 30.0).all()

    def test_mean_aggregation_within_bucket(self) -> None:
        # Two ticks in the same 30s bucket: values 0.0 and 10.0 → mean = 5.0
        base = pd.Timestamp("2026-06-01T00:00:00Z")
        ticks = pd.DataFrame(
            {
                "telemetry_timestamp": pd.array(
                    [base, base + pd.Timedelta(seconds=5)],
                    dtype="datetime64[us, UTC]",
                ),
                "value": pd.array([0.0, 10.0], dtype="float32"),
                "aos_timestamp": [None, None],
            }
        )
        result = resample_to_grid(ticks, "S1000003", "ISS", grid_interval_seconds=30)
        assert abs(float(result["value"].iloc[0]) - 5.0) < 1e-4

    def test_ffill_for_sparse_bucket(self) -> None:
        # Tick at t=0s, gap to t=91s, then tick at t=91s.
        # The middle bucket (t=30s) gets no tick → ffill from t=0s.
        base = pd.Timestamp("2026-06-01T00:00:00Z")
        ticks = pd.DataFrame(
            {
                "telemetry_timestamp": pd.array(
                    [base, base + pd.Timedelta(seconds=91)],
                    dtype="datetime64[us, UTC]",
                ),
                "value": pd.array([42.0, 99.0], dtype="float32"),
                "aos_timestamp": [None, None],
            }
        )
        result = resample_to_grid(ticks, "P4000001", "ISS", grid_interval_seconds=30)
        # t=30s bucket is ffill'd to 42.0
        t30 = base + pd.Timedelta(seconds=30)
        row = result[result["telemetry_timestamp"] == t30]
        assert len(row) == 1
        assert abs(float(row["value"].iloc[0]) - 42.0) < 1e-4

    def test_value_is_float32(self) -> None:
        ticks = _make_raw_ticks(n=60)
        result = resample_to_grid(ticks, "S1000003", "ISS", grid_interval_seconds=30)
        assert result["value"].dtype == "float32"

    def test_timestamp_is_utc(self) -> None:
        ticks = _make_raw_ticks(n=60)
        result = resample_to_grid(ticks, "S1000003", "ISS", grid_interval_seconds=30)
        assert str(result["telemetry_timestamp"].dt.tz) == "UTC"

    def test_output_has_fewer_rows_than_input(self) -> None:
        ticks = _make_raw_ticks(n=120, interval_s=2.0)
        result = resample_to_grid(ticks, "S1000003", "ISS", grid_interval_seconds=30)
        assert len(result) < len(ticks)


# ---------------------------------------------------------------------------
# TestComputeLosMask
# ---------------------------------------------------------------------------


class TestComputeLosMask:
    def test_no_los_when_all_buckets_covered(self) -> None:
        # Dense ticks at 2s cadence over 2 minutes → every 30s bucket has ticks.
        all_ticks = _make_all_ticks_df(["S1000003", "P4000001"], n_per_channel=60)
        mask = compute_los_mask(all_ticks, grid_interval_seconds=30)
        assert mask.dtype == bool
        assert not mask.any()

    def test_silent_bucket_marked_los(self) -> None:
        # Build ticks with a 91s gap so the middle 30s bucket has no ticks.
        base = pd.Timestamp("2026-06-01T00:00:00Z")
        before = [base + pd.Timedelta(seconds=i * 2) for i in range(10)]
        after = [base + pd.Timedelta(seconds=91 + i * 2) for i in range(10)]
        all_ts = before + after
        df = pd.DataFrame(
            {
                "telemetry_timestamp": pd.array(all_ts, dtype="datetime64[us, UTC]"),
                "channel_id": "S1000003",
            }
        )
        mask = compute_los_mask(df, grid_interval_seconds=30)
        assert mask.any(), "Expected at least one LOS bucket"

    def test_smear_expands_one_bucket_each_side(self) -> None:
        # Single completely silent 30s bucket in the middle of coverage.
        # After smear: the silent bucket + its two neighbours should all be True.
        base = pd.Timestamp("2026-06-01T00:00:00Z")
        # Ticks only before 30s and after 61s (the 30-60s bucket is empty).
        before = [base + pd.Timedelta(seconds=i * 2) for i in range(10)]  # 0-18s
        after = [base + pd.Timedelta(seconds=61 + i * 2) for i in range(10)]  # 61-79s
        df = pd.DataFrame(
            {
                "telemetry_timestamp": pd.array(before + after, dtype="datetime64[us, UTC]"),
                "channel_id": "S1000003",
            }
        )
        mask = compute_los_mask(df, grid_interval_seconds=30)
        # The silent bucket at 30s and its neighbours (0s and 60s) should all be True.
        assert mask.sum() >= 3

    def test_returns_bool_series(self) -> None:
        all_ticks = _make_all_ticks_df(["S1000003"], n_per_channel=30)
        mask = compute_los_mask(all_ticks, grid_interval_seconds=30)
        assert isinstance(mask, pd.Series)
        assert mask.dtype == bool

    def test_index_is_utc_datetimeindex(self) -> None:
        all_ticks = _make_all_ticks_df(["S1000003"], n_per_channel=30)
        mask = compute_los_mask(all_ticks, grid_interval_seconds=30)
        assert isinstance(mask.index, pd.DatetimeIndex)
        assert str(mask.index.tz) == "UTC"

    def test_empty_input_returns_empty_series(self) -> None:
        df = pd.DataFrame(columns=["telemetry_timestamp", "channel_id"])
        mask = compute_los_mask(df, grid_interval_seconds=30)
        assert isinstance(mask, pd.Series)
        assert len(mask) == 0

    def test_multiple_channels_combined(self) -> None:
        # All channels have ticks → no LOS even though each individual channel
        # has a gap (they cover different buckets).
        base = pd.Timestamp("2026-06-01T00:00:00Z")
        ch1 = [base + pd.Timedelta(seconds=i * 2) for i in range(15)]  # 0-28s
        ch2 = [base + pd.Timedelta(seconds=31 + i * 2) for i in range(15)]  # 31-59s
        df = pd.DataFrame(
            {
                "telemetry_timestamp": pd.array(
                    ch1 + ch2, dtype="datetime64[us, UTC]"
                ),
                "channel_id": ["A"] * 15 + ["B"] * 15,
            }
        )
        mask = compute_los_mask(df, grid_interval_seconds=30)
        # The 0-30s bucket is covered by ch1, the 30-60s bucket by ch2 → no raw LOS.
        # Smear may bleed into adjacent buckets but the core should not be LOS.
        assert not mask.iloc[0]


# ---------------------------------------------------------------------------
# TestAugmentWithLos
# ---------------------------------------------------------------------------


class TestAugmentWithLos:
    def _make_resampled(self, n: int = 5) -> pd.DataFrame:
        base = pd.Timestamp("2026-06-01T00:00:00Z")
        ts = [base + pd.Timedelta(seconds=i * 30) for i in range(n)]
        return pd.DataFrame(
            {
                "telemetry_timestamp": pd.array(ts, dtype="datetime64[us, UTC]"),
                "value": pd.array([float(i) for i in range(n)], dtype="float32"),
                "channel_id": "S1000003",
                "mission_id": "ISS",
            }
        )

    def _make_los_mask(
        self, los_indices: list[int], n: int = 5
    ) -> pd.Series:
        base = pd.Timestamp("2026-06-01T00:00:00Z")
        idx = pd.DatetimeIndex(
            [base + pd.Timedelta(seconds=i * 30) for i in range(n)], tz="UTC"
        )
        values = [i in los_indices for i in range(n)]
        return pd.Series(values, index=idx, dtype=bool, name="is_los")

    def test_is_los_column_added(self) -> None:
        df = self._make_resampled()
        mask = self._make_los_mask([])
        result = augment_with_los(df, mask)
        assert "is_los" in result.columns

    def test_los_timestamps_are_true(self) -> None:
        df = self._make_resampled()
        mask = self._make_los_mask([1, 3])
        result = augment_with_los(df, mask)
        assert result.iloc[1]["is_los"]
        assert result.iloc[3]["is_los"]

    def test_non_los_timestamps_are_false(self) -> None:
        df = self._make_resampled()
        mask = self._make_los_mask([1])
        result = augment_with_los(df, mask)
        assert not result.iloc[0]["is_los"]
        assert not result.iloc[2]["is_los"]

    def test_unmatched_timestamps_default_false(self) -> None:
        # Resampled df has timestamps not in the mask → should be False.
        base = pd.Timestamp("2026-06-01T00:00:00Z")
        df = pd.DataFrame(
            {
                "telemetry_timestamp": pd.array(
                    [base + pd.Timedelta(seconds=i * 30) for i in range(5)],
                    dtype="datetime64[us, UTC]",
                ),
                "value": pd.array([0.0] * 5, dtype="float32"),
                "channel_id": "S1000003",
                "mission_id": "ISS",
            }
        )
        # mask covers only the first 3 timestamps
        idx = pd.DatetimeIndex(
            [base + pd.Timedelta(seconds=i * 30) for i in range(3)], tz="UTC"
        )
        mask = pd.Series([False, True, False], index=idx, dtype=bool, name="is_los")
        result = augment_with_los(df, mask)
        # rows 3 and 4 have no entry in the mask → should be False
        assert not result.iloc[3]["is_los"]
        assert not result.iloc[4]["is_los"]

    def test_is_los_is_bool(self) -> None:
        df = self._make_resampled()
        mask = self._make_los_mask([0])
        result = augment_with_los(df, mask)
        assert result["is_los"].dtype == bool

    def test_row_count_unchanged(self) -> None:
        df = self._make_resampled(n=10)
        mask = self._make_los_mask([2, 5], n=10)
        result = augment_with_los(df, mask)
        assert len(result) == 10
