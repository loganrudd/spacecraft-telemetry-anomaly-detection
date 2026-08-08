"""Run the ESA-ADB report without an MLflow tracking server.

``esa_adb.detections`` normally resolves a channel's scoring run by querying
MLflow run tags (``channel_id``, ``tuned_from_run``). Those tags live only in
the tracking backend (Postgres in cloud deployments), so when that server is
down the artifacts in GCS are orphaned: ``errors.npy`` and ``threshold.npy``
carry no channel identity of their own.

The mapping is still recoverable from object storage alone — see
``scripts/stage_adb_offline.sh`` and docs/plans/019 for the derivation:

  1. ``hpo_channel_manifest.json`` (logged by the HPO sweep) records
     ``channel -> scoring_run_id`` for the baseline runs it consumed.
  2. ``threshold_config.json`` ``{window, z}`` separates baseline runs
     (Hundman defaults) from tuned re-scores (per-subsystem HPO values in
     ``tuned_configs.json``).
  3. Where several channels share a window count, tuned runs are matched to
     channels by correlating ``errors.npy`` content against each channel's
     known baseline run (two EWMA smoothings of the same raw error series
     correlate far more strongly than different channels do).

This module consumes the *result* of that recovery: an explicit run map,
which also serves as a provenance record — it pins the exact run IDs a
published number came from, independent of any later re-scoring.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from spacecraft_telemetry.core.logging import get_logger
from spacecraft_telemetry.core.paths import to_upath

if TYPE_CHECKING:
    from pathlib import Path

log = get_logger(__name__)

_VARIANTS = ("baseline", "tuned")


@dataclass(frozen=True)
class RunSpec:
    """Everything needed to reconstruct one scoring run's flags, sans MLflow.

    ``threshold_min_anomaly_len`` and ``min_error_value`` are carried explicitly
    because they are logged *params*, not part of ``threshold_config.json``
    (which holds only window and z) — so they cannot be recovered from the
    staged artifacts themselves.

    ``min_error_value`` defaults to 0.0, which is exactly the behaviour of runs
    logged before the absolute error floor existed.
    """

    run_id: str
    errors_path: str
    threshold_path: str
    threshold_min_anomaly_len: int
    min_error_value: float = 0.0


@dataclass(frozen=True)
class OfflineRunMap:
    """Resolved ``channel -> {baseline, tuned}`` scoring runs for one mission."""

    mission: str
    baseline: dict[str, RunSpec]
    tuned: dict[str, RunSpec]

    def channels(self) -> list[str]:
        return sorted(self.baseline)

    def get(self, channel: str, *, tuned: bool) -> RunSpec:
        table = self.tuned if tuned else self.baseline
        if channel not in table:
            variant = "tuned" if tuned else "baseline"
            raise KeyError(
                f"No {variant} run for channel {channel!r} in the offline run map "
                f"(have: {sorted(table)}). Re-run scripts/stage_adb_offline.sh or "
                "add the channel to the run-map JSON."
            )
        return table[channel]


def load_run_map(path: str | Path) -> OfflineRunMap:
    """Load and validate an offline run map JSON file.

    Expected shape::

        {
          "mission": "ESA-Mission1",
          "runs_dir": "data/adb_offline/runs",
          "channels": {
            "channel_41": {
              "baseline": {"run_id": "39ca…", "threshold_min_anomaly_len": 3},
              "tuned":    {"run_id": "57b8…", "threshold_min_anomaly_len": 7}
            }
          }
        }

    Artifact paths are resolved as ``{runs_dir}/{run_id}/{errors,threshold}.npy``.
    ``runs_dir`` may be a local path or a ``gs://`` URI.

    Raises:
        ValueError: On a missing key, an unknown variant, or an empty channel map —
            a malformed map must fail loudly rather than silently narrow the
            evaluated channel set (which would quietly flatter the OR-aggregated
            metrics).
    """
    raw: dict[str, Any] = json.loads(to_upath(str(path)).read_text())

    for key in ("mission", "runs_dir", "channels"):
        if key not in raw:
            raise ValueError(f"Offline run map {path} is missing required key {key!r}.")

    runs_dir = str(raw["runs_dir"]).rstrip("/")
    channels: dict[str, Any] = raw["channels"]
    if not channels:
        raise ValueError(f"Offline run map {path} has an empty 'channels' map.")

    baseline: dict[str, RunSpec] = {}
    tuned: dict[str, RunSpec] = {}
    for channel, variants in channels.items():
        unknown = set(variants) - set(_VARIANTS)
        if unknown:
            raise ValueError(
                f"Offline run map {path}, channel {channel!r}: unknown variant(s) "
                f"{sorted(unknown)}; expected a subset of {list(_VARIANTS)}."
            )
        for variant, target in (("baseline", baseline), ("tuned", tuned)):
            if variant not in variants:
                continue
            entry = variants[variant]
            for key in ("run_id", "threshold_min_anomaly_len"):
                if key not in entry:
                    raise ValueError(
                        f"Offline run map {path}, channel {channel!r}, variant "
                        f"{variant!r}: missing required key {key!r}."
                    )
            run_id = str(entry["run_id"])
            target[channel] = RunSpec(
                run_id=run_id,
                errors_path=f"{runs_dir}/{run_id}/errors.npy",
                threshold_path=f"{runs_dir}/{run_id}/threshold.npy",
                threshold_min_anomaly_len=int(entry["threshold_min_anomaly_len"]),
                # Optional: absent for runs predating the absolute error floor.
                min_error_value=float(entry.get("min_error_value", 0.0)),
            )

    log.info(
        "esa_adb.offline.run_map_loaded",
        mission=raw["mission"],
        n_baseline=len(baseline),
        n_tuned=len(tuned),
    )
    return OfflineRunMap(mission=str(raw["mission"]), baseline=baseline, tuned=tuned)
