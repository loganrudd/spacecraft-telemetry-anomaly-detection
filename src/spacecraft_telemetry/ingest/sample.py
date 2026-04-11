"""Sample creation: raw pickle → Parquet for local dev.

Selects the first N channels (alphabetical), takes the first M% of rows
(contiguous time slice — preserves temporal order), converts to Parquet,
filters labels.csv, and writes a manifest.json for reproducibility.

Usage:
    creator = SampleCreator(
        raw_dir=Path("data/raw"),
        sample_dir=Path("data/sample"),
        sample_fraction=0.01,
        sample_channels=5,
    )
    manifest = creator.create_sample("ESA-Mission1")
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from pydantic import BaseModel

from spacecraft_telemetry.core.logging import get_logger

log = get_logger(__name__)


class SampleManifest(BaseModel):
    """Records the parameters used to create a sample dataset."""

    mission: str
    created_at: str  # ISO 8601
    sample_fraction: float
    sample_channels: int
    channels: list[str]
    row_counts: dict[str, int]  # channel → rows written to Parquet
    source_dir: str
    sample_dir: str


class SampleCreator:
    """Converts raw pickle channel files to a Parquet sample for local dev.

    Channel selection is deterministic: files in channels/ are sorted
    alphabetically by filename and the first sample_channels are taken.
    Rows are the first N% of each channel (contiguous slice, not random).

    Args:
        raw_dir:          Root of raw data. Mission dirs live here directly
                          (e.g. raw_dir/ESA-Mission1/channels/).
        sample_dir:       Root of sample output (e.g. data/sample/).
        sample_fraction:  Fraction of rows to keep (0 < f ≤ 1, default 0.01).
        sample_channels:  Max channels to include per mission (default 5).
    """

    def __init__(
        self,
        raw_dir: Path,
        sample_dir: Path,
        sample_fraction: float = 0.01,
        sample_channels: int = 5,
    ) -> None:
        self.raw_dir = raw_dir
        self.sample_dir = sample_dir
        self.sample_fraction = sample_fraction
        self.sample_channels = sample_channels

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_sample(self, mission: str) -> SampleManifest:
        """Create a Parquet sample for one mission.

        Writes:
          sample_dir/{mission}/channels/{channel}.parquet  (one per channel)
          sample_dir/{mission}/labels.csv                  (filtered)
          sample_dir/{mission}/manifest.json

        Args:
            mission: Mission directory name under raw_dir.

        Returns:
            SampleManifest describing what was created.
        """
        channels = self._select_channels(mission)
        row_counts: dict[str, int] = {}

        for channel in channels:
            df = self._load_channel(mission, channel)
            sampled = self._take_first_n_rows(df)
            dest = self.sample_dir / mission / "channels" / f"{channel}.parquet"
            dest.parent.mkdir(parents=True, exist_ok=True)
            sampled.to_parquet(dest, index=True, engine="pyarrow")
            row_counts[channel] = len(sampled)
            log.info(
                "channel sampled",
                mission=mission,
                channel=channel,
                sample_rows=len(sampled),
                source_rows=len(df),
            )

        self._write_labels(mission, channels)

        manifest = SampleManifest(
            mission=mission,
            created_at=datetime.now(UTC).isoformat(),
            sample_fraction=self.sample_fraction,
            sample_channels=self.sample_channels,
            channels=channels,
            row_counts=row_counts,
            source_dir=str(self.raw_dir / mission),
            sample_dir=str(self.sample_dir / mission),
        )
        manifest_path = self.sample_dir / mission / "manifest.json"
        manifest_path.write_text(manifest.model_dump_json(indent=2))
        log.info("manifest written", mission=mission, path=str(manifest_path))

        return manifest

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _select_channels(self, mission: str) -> list[str]:
        """Return the first sample_channels channel names, sorted alphabetically."""
        channel_dir = self.raw_dir / mission / "channels"
        if not channel_dir.exists():
            raise FileNotFoundError(f"Channel directory not found: {channel_dir}")

        pkl_files = sorted(
            (
                p
                for p in channel_dir.iterdir()
                if p.suffix == ".pkl" or p.name.endswith(".pkl.zip")
            ),
            key=lambda p: p.name,
        )

        if not pkl_files:
            raise FileNotFoundError(f"No pickle files found in {channel_dir}")

        channel_names = [_channel_name(p) for p in pkl_files]
        selected = channel_names[: self.sample_channels]
        log.info(
            "channels selected",
            mission=mission,
            selected=selected,
            total_available=len(channel_names),
        )
        return selected

    def _load_channel(self, mission: str, channel: str) -> pd.DataFrame:
        """Load a channel DataFrame from pickle (plain or zipped)."""
        channel_dir = self.raw_dir / mission / "channels"
        for filename in (f"{channel}.pkl", f"{channel}.pkl.zip"):
            path = channel_dir / filename
            if path.exists():
                obj = pd.read_pickle(path)
                return obj if isinstance(obj, pd.DataFrame) else pd.DataFrame(obj)
        raise FileNotFoundError(f"No pickle file for channel {channel!r} in {channel_dir}")

    def _take_first_n_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return the first sample_fraction rows as a contiguous slice."""
        n = max(1, int(len(df) * self.sample_fraction))
        return df.iloc[:n]

    def _write_labels(self, mission: str, channels: list[str]) -> None:
        """Filter labels.csv to selected channels and write to sample dir."""
        src = self.raw_dir / mission / "labels.csv"
        dst = self.sample_dir / mission / "labels.csv"
        dst.parent.mkdir(parents=True, exist_ok=True)

        if not src.exists():
            log.warning("labels.csv not found, skipping", mission=mission)
            return

        df = pd.read_csv(src)
        col = _detect_channel_column(df)
        if col:
            df = df[df[col].isin(channels)].reset_index(drop=True)

        df.to_csv(dst, index=False)
        log.info("labels written", mission=mission, rows=len(df))


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _channel_name(path: Path) -> str:
    """Derive a clean channel name from a pickle file path."""
    name = path.name
    for suffix in (".pkl.zip", ".pkl"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _detect_channel_column(df: pd.DataFrame) -> str | None:
    """Return the name of the channel identifier column, or None if unknown."""
    for candidate in ("channel", "chan", "channel_id", "name"):
        if candidate in df.columns:
            return candidate
    return None
