"""Channel suitability profiler — identifies degenerate channels before preprocessing.

Reads raw channel zips directly (no download-sample step required) and emits a
channel_suitability.json manifest that the preprocessing pipeline consults to
skip channels that would produce useless or misleading models:

  empty     — fewer than min_rows numeric values (sensor offline / format error)
  constant  — only one unique value (sensor stuck at a fixed reading)
  flat      — frac_zero_diff >= flat_threshold (sensor changes too rarely to
               produce a useful forecast signal; the LSTM trains to predict
               "same as last time" and the threshold is never exceeded)

Usage (CLI):
    spacecraft-telemetry preprocess profile --mission ESA-Mission2

Usage (Python):
    from spacecraft_telemetry.preprocess.profiler import profile_mission
    manifest = profile_mission(raw_data_dir, mission, flat_threshold=0.99)
"""

from __future__ import annotations

import io
import json
import pickle
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from upath import UPath

from spacecraft_telemetry.core.logging import get_logger
from spacecraft_telemetry.core.paths import to_upath

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChannelProfile:
    channel: str
    n_rows: int
    std: float | None
    nunique: int
    frac_zero_diff: float | None
    status: str          # "ok" | "skip"
    skip_reason: str | None   # "empty" | "constant" | "flat" | None


# ---------------------------------------------------------------------------
# Raw channel loading (mirrors SampleCreator._load_channel in ingest/sample.py)
# ---------------------------------------------------------------------------


def _load_raw_series(raw_mission_dir: Path, channel: str) -> pd.Series:
    """Load the value series from a raw channel zip or pickle file.

    Tries channel.zip, channel.pkl.zip, channel.pkl in order.
    Returns the first numeric column as a float64 Series with NaN dropped.
    """
    channel_dir = raw_mission_dir / "channels"
    for filename in (f"{channel}.zip", f"{channel}.pkl.zip", f"{channel}.pkl"):
        path = channel_dir / filename
        if not path.exists():
            continue
        if path.suffix == ".pkl" and not filename.endswith(".pkl.zip"):
            obj = pd.read_pickle(str(path))
        else:
            with zipfile.ZipFile(str(path)) as zf, zf.open(zf.namelist()[0]) as f:
                obj = pickle.load(io.BytesIO(f.read()))
        df = obj if isinstance(obj, pd.DataFrame) else pd.DataFrame(obj)
        for col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if not s.empty:
                return s.astype("float64")
        return pd.Series(dtype="float64")
    raise FileNotFoundError(f"No channel file for {channel!r} in {channel_dir}")


# ---------------------------------------------------------------------------
# Per-channel profiling
# ---------------------------------------------------------------------------


def profile_channel(
    raw_mission_dir: Path,
    channel: str,
    *,
    flat_threshold: float,
    min_rows: int,
) -> ChannelProfile:
    """Compute suitability stats for one channel and classify it."""
    try:
        series = _load_raw_series(raw_mission_dir, channel)
    except Exception as exc:
        log.warning("profiler.channel.load_error", channel=channel, error=str(exc))
        return ChannelProfile(
            channel=channel, n_rows=0, std=None, nunique=0,
            frac_zero_diff=None, status="skip", skip_reason="empty",
        )

    n = len(series)
    if n < min_rows:
        return ChannelProfile(
            channel=channel, n_rows=n, std=None, nunique=int(series.nunique()),
            frac_zero_diff=None, status="skip", skip_reason="empty",
        )

    nunique = int(series.nunique())
    if nunique <= 1:
        return ChannelProfile(
            channel=channel, n_rows=n, std=float(series.std()),
            nunique=nunique, frac_zero_diff=None,
            status="skip", skip_reason="constant",
        )

    frac_zero_diff = float((series.diff() == 0).mean())
    std = float(series.std())
    if frac_zero_diff >= flat_threshold:
        return ChannelProfile(
            channel=channel, n_rows=n, std=std, nunique=nunique,
            frac_zero_diff=frac_zero_diff, status="skip", skip_reason="flat",
        )

    return ChannelProfile(
        channel=channel, n_rows=n, std=std, nunique=nunique,
        frac_zero_diff=frac_zero_diff, status="ok", skip_reason=None,
    )


# ---------------------------------------------------------------------------
# Mission-level profiling
# ---------------------------------------------------------------------------


def profile_mission(
    raw_data_dir: Path,
    mission: str,
    channels: list[str] | None = None,
    *,
    flat_threshold: float = 0.99,
    min_rows: int = 1000,
) -> dict[str, Any]:
    """Profile all channels for a mission and return a suitability manifest dict.

    Args:
        raw_data_dir:   Root raw-data directory (contains mission subdirs).
        mission:        Mission name, e.g. "ESA-Mission2".
        channels:       Explicit list of channel IDs to profile. When None,
                        discovers all channel files under raw_data_dir/mission/channels/.
        flat_threshold: Fraction of zero-diff steps above which a channel is
                        classified as flat (default 0.99).
        min_rows:       Minimum numeric rows required to be considered non-empty
                        (default 1000).

    Returns:
        Manifest dict suitable for JSON serialisation.
    """
    raw_mission_dir = raw_data_dir / mission
    channel_dir = raw_mission_dir / "channels"

    if channels is None:
        channels = sorted(
            p.stem.removesuffix(".pkl")
            for p in channel_dir.iterdir()
            if p.suffix in (".zip", ".pkl")
        )

    if not channels:
        raise FileNotFoundError(f"No channel files found in {channel_dir}")

    log.info("profiler.mission.start", mission=mission, n_channels=len(channels),
             flat_threshold=flat_threshold, min_rows=min_rows)

    profiles: list[ChannelProfile] = []
    for channel in channels:
        profile = profile_channel(
            raw_mission_dir, channel,
            flat_threshold=flat_threshold, min_rows=min_rows,
        )
        profiles.append(profile)
        log.info(
            "profiler.channel.done",
            channel=channel, status=profile.status, skip_reason=profile.skip_reason,
            n_rows=profile.n_rows, frac_zero_diff=round(profile.frac_zero_diff, 3)
            if profile.frac_zero_diff is not None else None,
        )

    n_ok = sum(1 for p in profiles if p.status == "ok")
    log.info("profiler.mission.end", mission=mission, n_ok=n_ok,
             n_skip=len(profiles) - n_ok)

    return {
        "mission": mission,
        "generated_at": datetime.now(UTC).isoformat(),
        "thresholds": {"flat_threshold": flat_threshold, "min_rows": min_rows},
        "channels": {
            p.channel: {
                "n_rows": p.n_rows,
                "std": p.std,
                "nunique": p.nunique,
                "frac_zero_diff": p.frac_zero_diff,
                "status": p.status,
                "skip_reason": p.skip_reason,
            }
            for p in profiles
        },
    }


# ---------------------------------------------------------------------------
# Manifest I/O helpers (used by pipeline and CLI)
# ---------------------------------------------------------------------------


def suitability_manifest_path(sample_data_dir: str | Path, mission: str) -> UPath:
    """Standard path for the suitability manifest.

    Lives in sample_data_dir (not raw_data_dir) so the same path works both
    locally (data/sample/{mission}/) and in cloud (gs://…/sample/{mission}/).
    The profiler writes here; the pipeline and cloud preprocess read from here.

    Uses to_upath so a gs:// sample_data_dir round-trips correctly — plain
    pathlib.Path collapses "gs://" to "gs:/" and cannot stat/read GCS.
    """
    return to_upath(str(sample_data_dir)) / mission / "channel_suitability.json"


def load_suitability_manifest(manifest_path: str | Path) -> dict[str, str]:
    """Return {channel: status} from a manifest file; empty dict if absent.

    Reads through to_upath so gs:// URIs work in the cloud preprocess RayJob.
    """
    p = to_upath(str(manifest_path))
    if not p.exists():
        return {}
    data = json.loads(p.read_text())
    return {ch: info["status"] for ch, info in data.get("channels", {}).items()}


def filter_channels(
    channels: list[str],
    manifest_path: Path,
) -> tuple[list[str], list[str]]:
    """Split channels into (ok, skipped) using the suitability manifest.

    Channels absent from the manifest pass through as ok — no manifest means
    no filter applied, preserving backward compatibility.
    """
    statuses = load_suitability_manifest(manifest_path)
    if not statuses:
        return channels, []
    ok = [ch for ch in channels if statuses.get(ch, "ok") == "ok"]
    skipped = [ch for ch in channels if statuses.get(ch, "ok") != "ok"]
    return ok, skipped
