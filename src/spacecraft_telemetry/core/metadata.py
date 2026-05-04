"""Lightweight mission metadata helpers — no Ray, no MLflow, no torch.

Placed in `core/` so that `model/training.py`, `model/scoring.py`, and
`ray_training/tune.py` can all import without creating a circular dependency
through `ray_training/runner.py`.

Previously, `load_channel_subsystem_map` lived in `ray_training/runner.py`.
That forced `model/training.py` and `model/scoring.py` to use a lazy
`try/except` import to avoid a layering violation (model/ importing ray_training/).
Moving it here restores clean layering:

    core/metadata  ←  model/training, model/scoring, ray_training/tune, ray_training/runner
"""

from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path

from spacecraft_telemetry.core.config import Settings
from spacecraft_telemetry.core.logging import get_logger

log = get_logger(__name__)


def load_channel_subsystem_map(settings: Settings, mission: str) -> dict[str, str]:
    """Return a mapping of channel_id → subsystem name.

    Lookup order:
    1) Processed metadata file:
       {spark.processed_data_dir}/{mission}/metadata/channel_subsystems.json
    2) Raw metadata CSV fallback:
       {data.raw_data_dir}/{mission}/channels.csv

    The processed metadata path keeps training/scoring metadata colocated with
    processed artifacts. CSV fallback is retained for backward compatibility.

    Returns an empty dict (not an exception) when neither file exists — callers
    treat the subsystem tag as optional/best-effort metadata.

    The mapping is cached per (processed_dir, raw_dir, mission) tuple inside each
    process.  Under Ray fan-out each worker process caches its own copy, so disk
    reads are bounded to at most one read per file per process regardless of how
    many channels that worker handles sequentially.
    """
    processed_dir = str(Path(str(settings.spark.processed_data_dir)).resolve())
    raw_dir = str(Path(str(settings.data.raw_data_dir)).resolve())
    return _load_cached(processed_dir, raw_dir, mission)


@lru_cache(maxsize=32)
def _load_cached(processed_dir: str, raw_dir: str, mission: str) -> dict[str, str]:
    processed_map_path = (
        Path(processed_dir)
        / mission
        / "metadata"
        / "channel_subsystems.json"
    )
    if processed_map_path.exists():
        try:
            loaded = json.loads(processed_map_path.read_text())
        except json.JSONDecodeError:
            log.warning(
                "processed subsystem map is invalid JSON; falling back to channels.csv",
                path=str(processed_map_path),
            )
        else:
            if isinstance(loaded, dict):
                processed_mapping = {
                    str(channel): str(subsystem)
                    for channel, subsystem in loaded.items()
                    if str(channel).strip() and str(subsystem).strip()
                }
                if processed_mapping:
                    return processed_mapping
                log.warning(
                    "processed subsystem map is empty; falling back to channels.csv",
                    path=str(processed_map_path),
                )

    # CSV fallback for compatibility with current preprocessing output.
    csv_path = Path(raw_dir) / mission / "channels.csv"
    if not csv_path.exists():
        log.warning(
            "channels.csv not found; tuned_configs will not be applied",
            path=str(csv_path),
        )
        return {}
    mapping: dict[str, str] = {}
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            channel = row.get("Channel", "").strip()
            subsystem = row.get("Subsystem", "").strip()
            if channel and subsystem:
                mapping[channel] = subsystem
    return mapping
