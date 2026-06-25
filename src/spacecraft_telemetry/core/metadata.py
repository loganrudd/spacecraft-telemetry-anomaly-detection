"""Lightweight mission metadata helpers — no Ray, no MLflow, no torch.

Placed in `core/` so that `model/training.py`, `model/scoring.py`, and
`ray_fanout/tune.py` can all import without creating a circular dependency
through `ray_fanout/runner.py`.

Previously, `load_channel_subsystem_map` lived in `ray_fanout/runner.py`.
That forced `model/training.py` and `model/scoring.py` to use a lazy
`try/except` import to avoid a layering violation (model/ importing ray_fanout/).
Moving it here restores clean layering:

    core/metadata  ←  model/training, model/scoring, ray_fanout/tune, ray_fanout/runner
"""

from __future__ import annotations

import csv
import io
import json
from functools import lru_cache

from spacecraft_telemetry.core.config import Settings
from spacecraft_telemetry.core.logging import get_logger
from spacecraft_telemetry.core.paths import to_upath

log = get_logger(__name__)


def load_channel_subsystem_map(settings: Settings, mission: str) -> dict[str, str]:
    """Return a mapping of channel_id → subsystem name.

    Lookup order:
    1) Processed metadata file:
       {preprocess.processed_data_dir}/{mission}/metadata/channel_subsystems.json
    2) CSV fallback (channels.csv) tried in BOTH the sample and raw dirs — in
       cloud only the sample bucket exists; locally only raw has it.

    The processed metadata path keeps training/scoring metadata colocated with
    processed artifacts. CSV fallback is retained for backward compatibility.

    Returns an empty dict (not an exception) when no source exists — callers
    treat the subsystem tag as optional/best-effort metadata.

    All paths go through to_upath so gs:// URIs resolve in the cloud (plain
    pathlib mangles gs:// → gs:/ and cannot read GCS).

    The mapping is cached per (processed_dir, sample_dir, raw_dir, mission)
    tuple inside each process.  Under Ray fan-out each worker process caches its
    own copy, so reads are bounded to one per file per process.
    """
    return _load_cached(
        str(settings.preprocess.processed_data_dir),
        str(settings.data.sample_data_dir),
        str(settings.data.raw_data_dir),
        mission,
    )


@lru_cache(maxsize=32)
def _load_cached(
    processed_dir: str, sample_dir: str, raw_dir: str, mission: str
) -> dict[str, str]:
    # When running against an injected test split (_injected subdirectory), the
    # channel_subsystems.json lives in the nominal processed bucket, not under
    # _injected. Try the nominal parent as a transparent fallback so that
    # INJECTED=1 tune jobs find the subsystem map without a separate env var.
    candidate_dirs = [processed_dir]
    _INJECTED_SUFFIX = "/_injected"
    if processed_dir.rstrip("/").endswith("/_injected") or "/_injected/" in processed_dir:
        nominal = processed_dir.rstrip("/")
        if nominal.endswith("/_injected"):
            nominal = nominal[: -len(_INJECTED_SUFFIX)]
        candidate_dirs.append(nominal)

    for _dir in candidate_dirs:
        processed_map_path = (
            to_upath(_dir) / mission / "metadata" / "channel_subsystems.json"
        )
        if not processed_map_path.exists():
            continue
        try:
            loaded = json.loads(processed_map_path.read_text())
        except json.JSONDecodeError:
            log.warning(
                "processed subsystem map is invalid JSON; falling back to channels.csv",
                path=str(processed_map_path),
            )
            continue
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

    # CSV fallback. Try sample dir first (the one that exists in cloud), then
    # raw (local dev). Both via to_upath so gs:// resolves.
    for base in (sample_dir, raw_dir):
        csv_path = to_upath(base) / mission / "channels.csv"
        if not csv_path.exists():
            continue
        mapping: dict[str, str] = {}
        reader = csv.DictReader(io.StringIO(csv_path.read_text()))
        for row in reader:
            channel = row.get("Channel", "").strip()
            subsystem = row.get("Subsystem", "").strip()
            if channel and subsystem:
                mapping[channel] = subsystem
        return mapping

    log.warning(
        "channels.csv not found; tuned_configs will not be applied",
        sample_path=str(to_upath(sample_dir) / mission / "channels.csv"),
        raw_path=str(to_upath(raw_dir) / mission / "channels.csv"),
    )
    return {}
