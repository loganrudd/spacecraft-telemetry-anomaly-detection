"""Tests for preprocess/transforms.py — handle_nulls, detect_gaps, normalize,
temporal_train_test_split, label_timesteps."""

from __future__ import annotations

import numpy as np
import pandas as pd

from spacecraft_telemetry.preprocess.transforms import (
    detect_gaps,
    handle_nulls,
    label_timesteps,
    normalize,
    temporal_train_test_split,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_channel_df(
    n: int = 20,
    freq: str = "90s",
    channel_id: str = "channel_1",
    mission_id: str = "ESA-Mission1",
    start: str = "2000-01-01",
) -> pd.DataFrame:
    """Regular 90-second spaced channel DataFrame."""
    ts = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    return pd.DataFrame(
        {
            "telemetry_timestamp": ts,
            "value": pd.array(
                [float(i % 5) + 1.0 for i in range(n)], dtype="float32"
            ),
            "channel_id": channel_id,
            "mission_id": mission_id,
        }
    )


def _make_labels_df(
    channel_id: str = "channel_1",
    segments: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """Build a labels DataFrame with the standard schema."""
    if segments is None:
        segments = [("2000-01-01T00:15:00Z", "2000-01-01T00:22:30Z")]
    rows = []
    for i, (start, end) in enumerate(segments):
        rows.append(
            {
                "anomaly_id": f"id_{i}",
                "channel_id": channel_id,
                "start_time": pd.Timestamp(start),
                "end_time": pd.Timestamp(end),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# handle_nulls
# ---------------------------------------------------------------------------


class TestHandleNulls:
    def test_no_op_when_no_nulls(self) -> None:
        df = _make_channel_df()
        out = handle_nulls(df)
        assert len(out) == len(df)
        assert out["value"].isna().sum() == 0

    def test_fills_interior_nulls(self) -> None:
        df = _make_channel_df(n=10)
        df.loc[3, "value"] = None
        df.loc[7, "value"] = None
        out = handle_nulls(df)
        assert out["value"].isna().sum() == 0
        assert len(out) == 10

    def test_drops_leading_nulls(self) -> None:
        df = _make_channel_df(n=10)
        df.loc[0, "value"] = None
        df.loc[1, "value"] = None
        out = handle_nulls(df)
        assert len(out) == 8
        assert out["value"].isna().sum() == 0

    def test_filled_value_equals_previous(self) -> None:
        df = _make_channel_df(n=10)
        prev_val = float(df.loc[4, "value"])
        df.loc[5, "value"] = None
        out = handle_nulls(df.sort_values("telemetry_timestamp").reset_index(drop=True))
        assert float(out.loc[5, "value"]) == prev_val

    def test_returns_dataframe(self) -> None:
        df = _make_channel_df()
        out = handle_nulls(df)
        assert isinstance(out, pd.DataFrame)

    def test_empty_df_after_all_nulls_dropped(self) -> None:
        df = _make_channel_df(n=3)
        df["value"] = None
        out = handle_nulls(df)
        assert out.empty


# ---------------------------------------------------------------------------
# detect_gaps
# ---------------------------------------------------------------------------


class TestDetectGaps:
    def test_segment_id_zero_for_uniform_series(self) -> None:
        df = _make_channel_df(n=50)
        out = detect_gaps(df)
        assert (out["segment_id"] == 0).all()

    def test_adds_segment_id_and_is_gap_columns(self) -> None:
        df = _make_channel_df(n=10)
        out = detect_gaps(df)
        assert "segment_id" in out.columns
        assert "is_gap" in out.columns

    def test_segment_id_increments_at_gap(
        self, irregular_channel_pd: pd.DataFrame
    ) -> None:
        # irregular_channel_pd has a 20-minute gap at row 50.
        df = irregular_channel_pd.reset_index().rename(
            columns={"datetime": "telemetry_timestamp", "channel_2": "value"}
        )
        df["channel_id"] = "channel_2"
        df["mission_id"] = "ESA-Mission1"
        out = detect_gaps(df)
        assert out["segment_id"].max() == 1
        assert out["segment_id"].nunique() == 2

    def test_is_gap_false_for_first_row(self) -> None:
        df = _make_channel_df(n=20)
        out = detect_gaps(df)
        assert not out["is_gap"].iloc[0]

    def test_segment_id_dtype_is_int32(self) -> None:
        df = _make_channel_df(n=10)
        out = detect_gaps(df)
        assert out["segment_id"].dtype == np.int32

    def test_gap_multiplier_respected(self) -> None:
        # Gap is exactly 10x the median — should be detected at multiplier=3, not at 15.
        base = pd.Timestamp("2000-01-01", tz="UTC")
        timestamps = [base + pd.Timedelta(seconds=90 * i) for i in range(10)]
        timestamps.append(base + pd.Timedelta(seconds=90 * 10 + 90 * 9))  # ~10x gap
        timestamps += [
            base + pd.Timedelta(seconds=90 * 10 + 90 * 9 + 90 * i) for i in range(1, 5)
        ]
        df = pd.DataFrame(
            {
                "telemetry_timestamp": timestamps,
                "value": pd.array([1.0] * len(timestamps), dtype="float32"),
                "channel_id": "channel_1",
                "mission_id": "ESA-Mission1",
            }
        )
        out3 = detect_gaps(df, gap_multiplier=3.0)
        out15 = detect_gaps(df, gap_multiplier=15.0)
        assert out3["segment_id"].max() == 1
        assert out15["segment_id"].max() == 0


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------


class TestNormalize:
    def test_returns_tuple(self) -> None:
        df = _make_channel_df()
        result = normalize(df)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_value_normalized_column_added(self) -> None:
        df = _make_channel_df()
        out, _ = normalize(df)
        assert "value_normalized" in out.columns

    def test_value_normalized_is_float32(self) -> None:
        df = _make_channel_df()
        out, _ = normalize(df)
        assert out["value_normalized"].dtype == np.float32

    def test_params_contains_mean_and_std(self) -> None:
        df = _make_channel_df()
        _, params = normalize(df)
        assert "channel_1" in params
        assert "mean" in params["channel_1"]
        assert "std" in params["channel_1"]

    def test_normalized_mean_approx_zero(self) -> None:
        df = _make_channel_df(n=100)
        out, _ = normalize(df)
        assert abs(float(out["value_normalized"].mean())) < 1e-5

    def test_normalized_std_approx_one(self) -> None:
        df = _make_channel_df(n=100)
        out, _ = normalize(df)
        assert abs(float(out["value_normalized"].std(ddof=1)) - 1.0) < 1e-4

    def test_constant_channel_normalized_to_zero(self) -> None:
        df = _make_channel_df()
        df["value"] = pd.array([5.0] * len(df), dtype="float32")
        out, params = normalize(df)
        assert (out["value_normalized"] == np.float32(0.0)).all()
        assert params["channel_1"]["std"] == 0.0

    def test_params_std_uses_ddof1(self) -> None:
        df = _make_channel_df(n=10)
        _, params = normalize(df)
        expected_std = float(df["value"].std(ddof=1))
        assert abs(params["channel_1"]["std"] - expected_std) < 1e-6


# ---------------------------------------------------------------------------
# temporal_train_test_split
# ---------------------------------------------------------------------------


class TestTemporalTrainTestSplit:
    def test_returns_two_dataframes(self) -> None:
        df = _make_channel_df(n=100)
        train, test = temporal_train_test_split(df)
        assert isinstance(train, pd.DataFrame)
        assert isinstance(test, pd.DataFrame)

    def test_total_rows_preserved(self) -> None:
        df = _make_channel_df(n=100)
        train, test = temporal_train_test_split(df)
        assert len(train) + len(test) == 100

    def test_train_fraction_80_approx(self) -> None:
        df = _make_channel_df(n=100)
        train, _test = temporal_train_test_split(df, train_fraction=0.8)
        # Allow ±2 rows — cutoff falls on a timestamp boundary.
        assert 78 <= len(train) <= 82

    def test_train_before_test(self) -> None:
        df = _make_channel_df(n=100)
        train, test = temporal_train_test_split(df)
        assert train["telemetry_timestamp"].max() < test["telemetry_timestamp"].min()

    def test_no_overlap_between_splits(self) -> None:
        df = _make_channel_df(n=100)
        train, test = temporal_train_test_split(df)
        train_ts = set(train["telemetry_timestamp"].astype(str))
        test_ts = set(test["telemetry_timestamp"].astype(str))
        assert train_ts.isdisjoint(test_ts)

    def test_train_fraction_100_puts_all_in_train(self) -> None:
        df = _make_channel_df(n=20)
        train, test = temporal_train_test_split(df, train_fraction=1.0)
        assert len(train) == 20
        assert len(test) == 0

    def test_shuffled_input_still_splits_correctly(self) -> None:
        df = _make_channel_df(n=100)
        shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
        train, test = temporal_train_test_split(shuffled)
        assert train["telemetry_timestamp"].max() < test["telemetry_timestamp"].min()


# ---------------------------------------------------------------------------
# label_timesteps
# ---------------------------------------------------------------------------


class TestLabelTimesteps:
    def test_adds_is_anomaly_column(self) -> None:
        df = _make_channel_df(n=20)
        labels = _make_labels_df()
        out = label_timesteps(df, labels)
        assert "is_anomaly" in out.columns

    def test_is_anomaly_dtype_is_bool(self) -> None:
        df = _make_channel_df(n=20)
        labels = _make_labels_df()
        out = label_timesteps(df, labels)
        assert out["is_anomaly"].dtype == bool

    def test_timestamps_inside_interval_are_anomalous(self) -> None:
        df = _make_channel_df(n=20)
        # Row 2 is at 00:03:00, row 4 is at 00:06:00.
        labels = _make_labels_df(
            segments=[("2000-01-01T00:03:00Z", "2000-01-01T00:06:01Z")]
        )
        out = label_timesteps(df, labels)
        row2 = df["telemetry_timestamp"].iloc[2]
        row4 = df["telemetry_timestamp"].iloc[4]
        assert out.loc[out["telemetry_timestamp"] == row2, "is_anomaly"].iloc[0]
        assert out.loc[out["telemetry_timestamp"] == row4, "is_anomaly"].iloc[0]

    def test_end_time_exclusive(self) -> None:
        df = _make_channel_df(n=10)
        # Row 3 is at 00:04:30 exactly. Interval ends at 00:04:30 — should NOT be anomalous.
        row3_ts = df["telemetry_timestamp"].iloc[3].isoformat().replace("+00:00", "Z")
        labels = _make_labels_df(segments=[("2000-01-01T00:03:00Z", row3_ts)])
        out = label_timesteps(df, labels)
        assert not out.loc[3, "is_anomaly"]

    def test_no_labels_for_channel_returns_all_false(self) -> None:
        df = _make_channel_df(n=20)
        labels = _make_labels_df(channel_id="channel_99")  # different channel
        out = label_timesteps(df, labels)
        assert not out["is_anomaly"].any()

    def test_multiple_label_segments(self) -> None:
        df = _make_channel_df(n=30)
        labels = _make_labels_df(
            segments=[
                ("2000-01-01T00:03:00Z", "2000-01-01T00:06:00Z"),
                ("2000-01-01T00:21:00Z", "2000-01-01T00:24:00Z"),
            ]
        )
        out = label_timesteps(df, labels)
        assert out["is_anomaly"].sum() > 0
        # Timestamps well before either segment must be nominal.
        assert not out["is_anomaly"].iloc[0]

    def test_empty_labels_returns_all_false(self) -> None:
        df = _make_channel_df(n=20)
        labels = pd.DataFrame(
            columns=["anomaly_id", "channel_id", "start_time", "end_time"]
        )
        out = label_timesteps(df, labels)
        assert not out["is_anomaly"].any()
