"""Data exploration: scan sample Parquet files and produce structured reports.

Expects the directory layout written by SampleCreator:
    data_dir/{mission}/channels/{channel}.parquet
    data_dir/{mission}/labels.csv

Usage:
    explorer = DataExplorer(Path("data/sample"))
    explorer.print_report("ESA-Mission1")
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
from pydantic import BaseModel
from rich.console import Console
from rich.table import Table

from spacecraft_telemetry.core.logging import get_logger

log = get_logger(__name__)

# Column names considered timestamp columns, in priority order.
_TS_CANDIDATES = ("timestamp", "time", "datetime", "ts", "date")

# Column names considered anomaly-type columns, in priority order.
_TYPE_CANDIDATES = ("anomaly_type", "type", "label")

# Column names considered channel identifier columns, in priority order.
_CHANNEL_CANDIDATES = ("channel", "chan", "channel_id", "name")


# ---------------------------------------------------------------------------
# Report models
# ---------------------------------------------------------------------------


class ChannelSummary(BaseModel):
    """Detailed statistics for a single telemetry channel."""

    channel: str
    n_rows: int
    n_columns: int
    dtypes: dict[str, str]  # column → dtype string
    null_counts: dict[str, int]  # column → count of nulls
    value_stats: dict[str, dict[str, float]]  # numeric column → {min, max, mean, std}
    time_range: tuple[str, str] | None = None  # (earliest, latest) ISO strings


class MissionReport(BaseModel):
    """Summary across all channels for one mission."""

    mission: str
    n_channels: int
    channel_names: list[str]
    total_rows: int
    time_range: tuple[str, str] | None = None  # union of all channel time ranges
    sampling_interval_s: float | None = None  # median interval estimated from first channel


class LabelReport(BaseModel):
    """Summary of anomaly labels for one mission."""

    mission: str
    n_labeled_channels: int
    n_anomaly_segments: int
    anomaly_types: dict[str, int] = {}  # anomaly type label → segment count
    channels_with_labels: list[str] = []


# ---------------------------------------------------------------------------
# DataExplorer
# ---------------------------------------------------------------------------


class DataExplorer:
    """Explores sample Parquet data and produces structured reports.

    Args:
        data_dir: Root directory of the sample data (e.g. Path("data/sample")).
    """

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir

    def mission_report(self, mission: str) -> MissionReport:
        """Summarise all channels for a mission.

        Raises:
            FileNotFoundError: If no Parquet files exist for the mission.
        """
        channel_dir = self.data_dir / mission / "channels"
        parquet_files = sorted(channel_dir.glob("*.parquet"))

        if not parquet_files:
            raise FileNotFoundError(f"No Parquet files found in {channel_dir}")

        channel_names = [p.stem for p in parquet_files]
        total_rows = 0
        earliest: pd.Timestamp | None = None
        latest: pd.Timestamp | None = None
        sampling_interval_s: float | None = None

        for i, path in enumerate(parquet_files):
            df = pd.read_parquet(path)
            total_rows += len(df)
            ts_col = _detect_time_column(df)
            if ts_col:
                ts = pd.to_datetime(df[ts_col])
                col_min, col_max = ts.min(), ts.max()
                earliest = col_min if earliest is None else min(earliest, col_min)
                latest = col_max if latest is None else max(latest, col_max)
                if i == 0:
                    sampling_interval_s = _estimate_interval_s(df, ts_col)

        time_range = (
            (earliest.isoformat(), latest.isoformat())
            if earliest is not None and latest is not None
            else None
        )

        return MissionReport(
            mission=mission,
            n_channels=len(channel_names),
            channel_names=channel_names,
            total_rows=total_rows,
            time_range=time_range,
            sampling_interval_s=sampling_interval_s,
        )

    def channel_summary(self, mission: str, channel: str) -> ChannelSummary:
        """Return detailed statistics for one channel.

        Raises:
            FileNotFoundError: If the channel Parquet file does not exist.
        """
        path = self.data_dir / mission / "channels" / f"channel_{channel}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Channel file not found: {path}")

        df = pd.read_parquet(path)
        dtypes = {col: str(df[col].dtype) for col in df.columns}
        null_counts = {col: int(df[col].isna().sum()) for col in df.columns}

        value_stats: dict[str, dict[str, float]] = {}
        for col in df.select_dtypes(include="number").columns:
            s = df[col].dropna()
            value_stats[col] = {
                "min": float(s.min()) if len(s) else float("nan"),
                "max": float(s.max()) if len(s) else float("nan"),
                "mean": float(s.mean()) if len(s) else float("nan"),
                "std": float(s.std()) if len(s) > 1 else float("nan"),
            }

        ts_col = _detect_time_column(df)
        if ts_col:
            ts = pd.to_datetime(df[ts_col])
            time_range: tuple[str, str] | None = (ts.min().isoformat(), ts.max().isoformat())
        else:
            time_range = None

        return ChannelSummary(
            channel=channel,
            n_rows=len(df),
            n_columns=len(df.columns),
            dtypes=dtypes,
            null_counts=null_counts,
            value_stats=value_stats,
            time_range=time_range,
        )

    def label_report(self, mission: str) -> LabelReport:
        """Summarise anomaly labels for a mission.

        Returns an empty report (zeros) if labels.csv is absent.
        """
        path = self.data_dir / mission / "labels.csv"
        if not path.exists():
            return LabelReport(mission=mission, n_labeled_channels=0, n_anomaly_segments=0)

        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return LabelReport(mission=mission, n_labeled_channels=0, n_anomaly_segments=0)
        if df.empty:
            return LabelReport(mission=mission, n_labeled_channels=0, n_anomaly_segments=0)

        channel_col = _first_match(df.columns, _CHANNEL_CANDIDATES)
        channels_with_labels = (
            sorted(df[channel_col].dropna().unique().tolist()) if channel_col else []
        )

        type_col = _first_match(df.columns, _TYPE_CANDIDATES)
        anomaly_types: dict[str, int] = (
            {str(k): int(v) for k, v in df[type_col].value_counts().items()} if type_col else {}
        )

        return LabelReport(
            mission=mission,
            n_labeled_channels=len(channels_with_labels),
            n_anomaly_segments=len(df),
            anomaly_types=anomaly_types,
            channels_with_labels=channels_with_labels,
        )

    def print_report(self, mission: str, console: Console | None = None) -> None:
        """Print a formatted summary report using rich tables.

        Args:
            mission: Mission name to report on.
            console: Optional rich Console (defaults to stdout). Injected for tests.
        """
        con = console or Console()

        # --- Mission overview ---
        try:
            mr = self.mission_report(mission)
        except FileNotFoundError as exc:
            con.print(f"[red]Mission data not found:[/red] {exc}")
            return

        overview = Table(title=f"Mission overview: {mission}", show_header=True)
        overview.add_column("Property", style="bold cyan", no_wrap=True)
        overview.add_column("Value")
        overview.add_row("Channels", str(mr.n_channels))
        overview.add_row("Total rows", f"{mr.total_rows:,}")
        if mr.time_range:
            overview.add_row("Time range", f"{mr.time_range[0]}  →  {mr.time_range[1]}")
        if mr.sampling_interval_s is not None:
            overview.add_row("Sampling interval", f"{mr.sampling_interval_s:.3f} s")
        overview.add_row("Channel list", ", ".join(mr.channel_names))
        con.print(overview)

        # --- Per-channel stats ---
        for channel in mr.channel_names:
            # channel_summary expects just the numeric id (e.g. "1"), but
            # mission_report returns full parquet stems (e.g. "channel_1").
            channel_id = channel.removeprefix("channel_")
            try:
                ch = self.channel_summary(mission, channel_id)
            except FileNotFoundError:
                con.print(f"[yellow]Skipping {channel} (file missing)[/yellow]")
                continue

            tbl = Table(title=f"Channel: {channel}", show_header=True)
            tbl.add_column("Column", style="bold")
            tbl.add_column("Dtype")
            tbl.add_column("Nulls", justify="right")
            tbl.add_column("Min", justify="right")
            tbl.add_column("Max", justify="right")
            tbl.add_column("Mean", justify="right")
            tbl.add_column("Std", justify="right")

            for col in ch.dtypes:
                stats = ch.value_stats.get(col, {})
                tbl.add_row(
                    col,
                    ch.dtypes[col],
                    str(ch.null_counts.get(col, 0)),
                    f"{stats['min']:.4g}" if stats else "—",
                    f"{stats['max']:.4g}" if stats else "—",
                    f"{stats['mean']:.4g}" if stats else "—",
                    f"{stats['std']:.4g}" if stats else "—",
                )
            con.print(tbl)

        # --- Labels ---
        lr = self.label_report(mission)
        lbl_tbl = Table(title="Anomaly labels", show_header=True)
        lbl_tbl.add_column("Property", style="bold cyan")
        lbl_tbl.add_column("Value")
        lbl_tbl.add_row("Labeled channels", str(lr.n_labeled_channels))
        lbl_tbl.add_row("Anomaly segments", str(lr.n_anomaly_segments))
        for atype, count in sorted(lr.anomaly_types.items()):
            lbl_tbl.add_row(f"  {atype}", str(count))
        con.print(lbl_tbl)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _detect_time_column(df: pd.DataFrame) -> str | None:
    """Return the name of a timestamp column in df, or None.

    Checks in order: datetime64 dtype, then common column names.
    """
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col  # type: ignore[no-any-return]
    return _first_match(df.columns, _TS_CANDIDATES)


def _estimate_interval_s(df: pd.DataFrame, ts_col: str) -> float | None:
    """Estimate the median sampling interval in seconds from a timestamp column."""
    if len(df) < 2:
        return None
    ts = pd.to_datetime(df[ts_col]).sort_values()
    diffs = ts.diff().dropna()
    if diffs.empty:
        return None
    return float(diffs.median().total_seconds())


def _first_match(columns: pd.Index, candidates: tuple[str, ...]) -> str | None:
    """Return the first candidate that appears in columns, or None."""
    for name in candidates:
        if name in columns:
            return name
    return None
