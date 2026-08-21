"""Report each arm's ACHIEVABLE threshold ceiling, to check tuning parity.

docs/plans/021 stage 021.5b. Plan 021's first result credited the multivariate
architecture with +0.061 mean segF0.5 over the univariate arm — but sweeping
both arms' saved error arrays showed the gap collapses to ~+0.020 at their
respective ceilings. Most of the apparent win was the univariate arm's HPO
landing on a worse operating point *inside* a search space that contained a
better one. Comparing two architectures is only meaningful once both are tuned
to comparable quality; this makes that check cheap enough to run every time.

Reads only existing MLflow scoring runs — no GPU, no training, no inference.
Works for univariate runs (channel_id-tagged, errors.npy at the run root) and
multivariate ones (subsystem-tagged, errors/{channel}.npy) via
esa_adb.detections.find_scoring_run_and_artifacts.

Usage:
    # Univariate arm A
    python scripts/threshold_ceiling.py --env cloud --mission ESA-Mission1-ADB

    # Multivariate arm (variant comes from SPACECRAFT_VARIANT)
    SPACECRAFT_VARIANT=adb-84m python scripts/threshold_ceiling.py \\
        --env cloud --mission ESA-Mission1 \\
        --processed-dir gs://PROJECT-processed-data

Requires: .[tracking] (mlflow). configure_mlflow handles Cloud Run ID-token auth.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spacecraft_telemetry.core.config import load_settings
from spacecraft_telemetry.core.logging import get_logger
from spacecraft_telemetry.esa_adb.detections import find_scoring_run_and_artifacts
from spacecraft_telemetry.mlflow_tracking import configure_mlflow, experiment_name
from spacecraft_telemetry.model.dataset import (
    load_series_metadata,
    load_window_labels_from_metadata,
)
from spacecraft_telemetry.model.io import bytes_to_errors, download_artifact_bytes
from spacecraft_telemetry.ray_fanout.threshold_grid import (
    best_point,
    bounds_report,
    sweep_group,
    sweep_group_mission_level,
)

log = get_logger(__name__)

_DEFAULT_CHANNELS = [f"channel_{i}" for i in range(41, 47)]
# Spans the tuned search space (threshold_z 2.5-8.0, min_error_value 0.0-0.6)
# rather than stopping short of it — an optimum on the grid edge only tells you
# the ceiling is a lower bound, which is the answer nobody wants.
_DEFAULT_Z = [2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
_DEFAULT_FLOORS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


def _sweep_mission(
    settings: Any,
    mission: str,
    channels: list[str],
    per_channel: dict[str, tuple[Any, Any]],
    eval_slice: slice,
    *,
    metadata_by_channel: dict[str, Any],
    select_on: str,
    threshold_window: int,
    min_run_length: int,
    z_values: list[float],
    floor_values: list[float],
) -> dict[tuple[float, float], float]:
    """Assemble ESA-ADB ground truth and sweep on the mission-level metric.

    Mirrors ray_fanout.tune.run_hpo_sweep's 021.4b preparation exactly — same
    timeline, same cutoff, same event grouping — so a config chosen here means
    the same thing a tune trial's ``mission_f0_5`` would have meant. Building
    it differently would reintroduce precisely the objective mismatch this
    mode exists to remove.

    The timeline must match ``select_on``: scoring detections from the HPO
    portion against the FULL timeline would count the held-out portion's
    nominal time as un-alarmed, inflating TNR_t and therefore precision.

    ``metadata_by_channel`` carries each channel's preloaded
    (segment_ids, is_anomaly, timestamps) so the three derivations below —
    timeline, HPO cutoff, target timestamps — reuse one read instead of
    issuing three more (docs/reviews/021, item D1).
    """
    import pandas as pd

    from spacecraft_telemetry.esa_adb.events import group_events, load_events
    from spacecraft_telemetry.esa_adb.intervals import intersect, subtract
    from spacecraft_telemetry.esa_adb.report import hpo_cutoff
    from spacecraft_telemetry.esa_adb.timeline import mission_timeline
    from spacecraft_telemetry.model.dataset import window_target_timestamps_from_metadata

    timeline_full = mission_timeline(
        settings, mission, channels, metadata_by_channel=metadata_by_channel
    )
    cutoff = hpo_cutoff(
        settings, mission, channels, metadata_by_channel=metadata_by_channel
    )
    far_past = pd.Timestamp.min.tz_localize("UTC")
    if select_on == "hpo_portion":
        timeline = intersect(timeline_full, [(far_past, cutoff)])
    else:
        # The held-out remainder: the complement of the HPO portion, which is
        # what esa_adb.report.tuned_eval_window computes for the tuned rows.
        timeline = subtract(timeline_full, [(far_past, cutoff)])
    events = group_events(load_events(settings, mission), channels, timeline)
    log.info(
        "threshold_ceiling.mission_prep",
        select_on=select_on, n_events=len(events), n_timeline_spans=len(timeline),
    )

    channel_timestamps = {
        channel: window_target_timestamps_from_metadata(
            settings, *metadata_by_channel[channel]
        )
        for channel in channels
    }
    for channel, stamps in channel_timestamps.items():
        n = len(per_channel[channel][0])
        if len(stamps) != n:
            raise SystemExit(
                f"Channel {channel!r}: {len(stamps)} target timestamps but {n} saved "
                "error windows. They must be index-aligned or detections map to the "
                "wrong instants — re-check window_size/prediction_horizon."
            )

    return sweep_group_mission_level(
        per_channel, channel_timestamps, events, timeline,
        threshold_window=threshold_window,
        min_run_length=min_run_length,
        z_values=z_values,
        floor_values=floor_values,
        eval_slice=eval_slice,
    )


def _parse_floats(raw: str | None, default: list[float]) -> list[float]:
    if not raw:
        return default
    return [float(x) for x in raw.split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env", default="cloud")
    p.add_argument("--mission", default="ESA-Mission1")
    p.add_argument("--channels", default=None, help="Comma-separated. Default: 41-46.")
    p.add_argument("--z-values", default=None, help="Comma-separated threshold_z grid.")
    p.add_argument("--floor-values", default=None, help="Comma-separated min_error_value grid.")
    p.add_argument("--processed-dir", default=None)
    p.add_argument(
        "--sample-dir",
        default=None,
        help="Override settings.data.sample_data_dir (holds labels.csv + "
        "anomaly_types.csv). Required with --objective mission, which needs "
        "ESA-ADB ground truth; unused by --objective per_channel.",
    )
    p.add_argument("--tracking-uri", default=None)
    p.add_argument(
        "--untuned",
        action="store_true",
        help="Sweep the untuned baseline run instead of the tuned one. The saved "
        "smoothed array differs (different error_smoothing_window), so this "
        "measures a different conditional ceiling — not a like-for-like swap.",
    )
    p.add_argument(
        "--select-on",
        choices=("final_portion", "hpo_portion"),
        default="final_portion",
        help="Which slice the grid is scored on. 'final_portion' (default) answers "
        "'how high could this arm reach?' — a CEILING, and not safe to report as a "
        "result because the config was chosen on the same data. 'hpo_portion' "
        "selects on the first tune.hpo_eval_fraction exactly as Ray Tune does, so "
        "the winning config can then be scored on the held-out remainder "
        "leakage-free — use this to produce a config for --tuned-configs.",
    )
    p.add_argument(
        "--emit-tuned-configs",
        default=None,
        metavar="JSON",
        help="Write the winning config as a tuned_configs.json for the given "
        "subsystem (see --subsystem-name), consumable by `ray score --tuned-configs`. "
        "Intended with --select-on hpo_portion; refuses otherwise, since a config "
        "selected on the reported slice would leak.",
    )
    p.add_argument(
        "--subsystem-name",
        default="subsystem_5",
        help="Subsystem key for --emit-tuned-configs. tuned_configs.json is keyed "
        "by subsystem (see ray_fanout/runner.py's schema).",
    )
    p.add_argument(
        "--objective",
        choices=("per_channel", "mission"),
        default="per_channel",
        help="What the grid maximises. 'per_channel' (default) = mean seg_f0_5, "
        "what Ray Tune optimises. 'mission' = mission-level corrected event-wise "
        "F0.5, what scripts/esa_adb_report.py publishes. THEY DIVERGE: a config "
        "optimal on the first drove mission detections 33 -> 112 and F0.5 "
        "0.3759 -> 0.1232 on plan 021's univariate arm. Select on whichever "
        "metric you intend to report.",
    )
    p.add_argument("--out", default=None, metavar="JSON")
    args = p.parse_args()

    if args.emit_tuned_configs and args.select_on != "hpo_portion":
        raise SystemExit(
            "--emit-tuned-configs requires --select-on hpo_portion. Selecting a "
            "config on final_portion and then reporting metrics on that same slice "
            "is leakage: the number would be optimistically biased and not "
            "comparable to arm A's HPO-selected result."
        )

    settings = load_settings(args.env)
    updates: dict[str, Any] = {}
    if args.processed_dir:
        updates["preprocess"] = settings.preprocess.model_copy(
            update={"processed_data_dir": args.processed_dir}
        )
    if args.sample_dir:
        updates["data"] = settings.data.model_copy(
            update={"sample_data_dir": args.sample_dir}
        )
    if args.tracking_uri:
        updates["mlflow"] = settings.mlflow.model_copy(
            update={"tracking_uri": args.tracking_uri}
        )
    if updates:
        settings = settings.model_copy(update=updates)
    configure_mlflow(settings)

    channels = (
        [c.strip() for c in args.channels.split(",") if c.strip()]
        if args.channels
        else _DEFAULT_CHANNELS
    )
    z_values = _parse_floats(args.z_values, _DEFAULT_Z)
    floor_values = _parse_floats(args.floor_values, _DEFAULT_FLOORS)

    exp = experiment_name(settings.model.model_type, "scoring", args.mission, settings.variant)
    tuned = not args.untuned

    per_channel: dict[str, tuple[Any, Any]] = {}
    # One parquet read per channel, reused by every downstream derivation.
    # Previously this script read each channel's test partition FOUR times —
    # load_window_labels here, plus mission_timeline, hpo_cutoff and
    # window_target_timestamps inside _sweep_mission — even though all four
    # derive from the same (segment_ids, is_anomaly, timestamps). The codebase
    # already solved this for esa_adb.report (docs/plans/019 P2/P3); this
    # script, written later, called the re-reading variants.
    metadata_by_channel: dict[str, Any] = {}
    threshold_window: int | None = None
    min_run_length: int | None = None
    # Captured so --emit-tuned-configs can pin it. The grid sweeps a saved
    # SMOOTHED array, so its result is only valid for a scoring run that
    # reproduces that smoothing; scoring recomputes it from scratch and would
    # otherwise fall back to the settings default (30), silently invalidating
    # the chosen (z, floor).
    smoothing_window: int | None = None

    def _load_channel(channel: str) -> dict[str, Any]:
        """Fetch one channel's errors + metadata. Pure I/O, no shared state.

        Runs on a worker thread (see below), so it must not touch the
        threshold_window/min_run_length/smoothing_window accumulators — those
        are reconciled afterwards, in channel order, to keep the "first channel
        wins, the rest must agree" semantics deterministic.
        """
        import mlflow

        run_id, errors_artifact, _threshold_artifact = find_scoring_run_and_artifacts(
            settings, args.mission, exp, channel, tuned=tuned
        )
        client = mlflow.MlflowClient(tracking_uri=settings.mlflow.tracking_uri)
        params = client.get_run(run_id).data.params
        smoothed = bytes_to_errors(
            download_artifact_bytes(run_id, errors_artifact, settings.mlflow.tracking_uri)
        )
        metadata = load_series_metadata(
            settings.preprocess.processed_data_dir, args.mission, channel, "test",
            variant=settings.variant,
        )
        return {
            "channel": channel,
            "run_id": run_id,
            "params": params,
            "smoothed": smoothed,
            "metadata": metadata,
        }

    # This tool is I/O-bound, not compute-bound: ~40-70 s of network per
    # channel at 0.6-3% CPU, against a 4-7 min sweep (docs/reviews/021, D2).
    # PyArrow and the GCS client both release the GIL, so threads recover most
    # of the serial cost. ThreadPoolExecutor.map yields in SUBMISSION order, so
    # `loaded` stays aligned with `channels` — completion order would silently
    # attribute each channel's errors to a different channel.
    with ThreadPoolExecutor(max_workers=min(6, len(channels))) as pool:
        loaded = list(pool.map(_load_channel, channels))

    for entry in loaded:
        channel = entry["channel"]
        params = entry["params"]
        # All channels in one arm share these; capture from the first and
        # verify the rest agree, so a mixed-config arm can't be averaged
        # together silently.
        tw = int(params["threshold_window"])
        mrl = int(params["threshold_min_anomaly_len"])
        esw = int(params["error_smoothing_window"])
        if threshold_window is None:
            threshold_window, min_run_length, smoothing_window = tw, mrl, esw
        elif (tw, mrl, esw) != (threshold_window, min_run_length, smoothing_window):
            raise SystemExit(
                f"Channel {channel!r} was scored with threshold_window={tw}, "
                f"min_run_length={mrl}, error_smoothing_window={esw}, but earlier "
                f"channels used {threshold_window}/{min_run_length}/{smoothing_window}. "
                "Averaging across differing configs would not describe any single "
                "achievable operating point."
            )
        metadata = entry["metadata"]
        metadata_by_channel[channel] = metadata
        labels = load_window_labels_from_metadata(settings, metadata[0], metadata[1])
        per_channel[channel] = (entry["smoothed"], labels)
        log.info(
            "threshold_ceiling.channel_loaded",
            channel=channel, run_id=entry["run_id"], n_windows=len(entry["smoothed"]),
        )

    assert threshold_window is not None and min_run_length is not None
    n_windows = len(next(iter(per_channel.values()))[1])
    n_hpo = int(n_windows * settings.tune.hpo_eval_fraction)
    # final_portion = the held-out remainder score_channel reports on (a ceiling
    # when selected on). hpo_portion = the slice Ray Tune actually optimises
    # against, so a config chosen here can be scored on the remainder without
    # leakage — the same separation the HPO pipeline relies on.
    eval_slice = (
        slice(n_hpo, None) if args.select_on == "final_portion" else slice(None, n_hpo)
    )

    if args.objective == "mission":
        grid = _sweep_mission(
            settings, args.mission, channels, per_channel, eval_slice,
            metadata_by_channel=metadata_by_channel,
            select_on=args.select_on,
            threshold_window=threshold_window,
            min_run_length=min_run_length,
            z_values=z_values,
            floor_values=floor_values,
        )
    else:
        grid = sweep_group(
            per_channel,
            threshold_window=threshold_window,
            min_run_length=min_run_length,
            z_values=z_values,
            floor_values=floor_values,
            eval_slice=eval_slice,
        )
    (best_z, best_floor), best_score = best_point(grid)
    report = bounds_report(grid, z_values, floor_values)

    print(f"\nmission={args.mission} variant={settings.variant} "
          f"{'tuned' if tuned else 'untuned'} channels={len(channels)}")
    print(f"fixed: threshold_window={threshold_window} min_run_length={min_run_length} "
          f"(error_smoothing_window is baked into the saved array and NOT swept)")
    print(f"\n{'z\\floor':>9}" + "".join(f"{f:>8}" for f in floor_values))
    for z in z_values:
        print(f"{z:>9}" + "".join(f"{grid[(z, f)]:8.3f}" for f in floor_values))
    label = "CEILING" if args.select_on == "final_portion" else "BEST-ON-HPO-PORTION"
    metric = (
        "mission-level corrected event-wise F0.5"
        if args.objective == "mission"
        else "mean per-channel segF0.5"
    )
    print(f"\n{label}  {metric} = {best_score:.3f}  at z={best_z}, floor={best_floor}")
    print(f"  (objective={args.objective}, selected on {args.select_on})")
    if report["is_lower_bound"]:
        print("  ⚠  optimum sits on a GRID EDGE — this is a LOWER BOUND. "
              "Widen --z-values / --floor-values before quoting it.")
    else:
        print("  ✓  optimum is interior to the swept grid.")

    if args.emit_tuned_configs:
        # Same schema run_all_sweeps writes and score_all_channels reads
        # (ray_fanout/runner.py). `_meta.run_id` is omitted deliberately: there
        # is no HPO run behind this config, and fabricating one would corrupt
        # the tuned_from_run lineage tag that scoring writes.
        entry = {
            args.subsystem_name: {
                "threshold_z": best_z,
                "min_error_value": best_floor,
                "threshold_window": threshold_window,
                "threshold_min_anomaly_len": min_run_length,
                # MUST be pinned, not omitted. The grid swept a saved SMOOTHED
                # array; scoring recomputes smoothing from scratch and would
                # otherwise use the settings default, producing a different
                # array for which the chosen (z, floor) was never evaluated.
                "error_smoothing_window": smoothing_window,
                # Schema shared with ray_fanout.tune's Ray Tune writer — see
                # write_tuned_configs' docstring (docs/plans/022, stage 022.2).
                "_meta": {
                    "provenance": "exhaustive_grid",
                    "source": (
                        f"scripts/threshold_ceiling.py exhaustive grid ({args.objective})"
                    ),
                    # No HPO run backs a grid-selected config — fabricating an
                    # id would corrupt the tuned_from_run lineage tag.
                    "run_id": None,
                    "objective_name": (
                        "mission_corrected_event_wise_f0_5"
                        if args.objective == "mission"
                        else "mean_per_channel_seg_f0_5"
                    ),
                    "objective_value": best_score,
                    "selected_on": args.select_on,
                    "hpo_eval_fraction": settings.tune.hpo_eval_fraction,
                    "outer_split": "chronological_50_50",
                    "error_smoothing_window": smoothing_window,
                    "threshold_window": threshold_window,
                    "min_run_length": min_run_length,
                    "axes": {"threshold_z": z_values, "min_error_value": floor_values},
                    # This hand-driven CLI has no widening driver yet
                    # (docs/plans/022, stage 022.1) — a manually re-run grid
                    # is not a recorded expansion.
                    "expansions": 0,
                    "interior": not report["is_lower_bound"],
                },
            }
        }
        Path(args.emit_tuned_configs).parent.mkdir(parents=True, exist_ok=True)
        Path(args.emit_tuned_configs).write_text(json.dumps(entry, indent=2))
        print(f"\nWrote tuned_configs → {args.emit_tuned_configs}")
        print(f"  error_smoothing_window pinned to {smoothing_window} (the value the "
              "swept arrays were produced with) — scoring must reproduce it or the "
              "grid result does not apply.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "mission": args.mission,
            "variant": settings.variant,
            "tuned": tuned,
            "objective": args.objective,
            "select_on": args.select_on,
            "channels": channels,
            "threshold_window": threshold_window,
            "min_run_length": min_run_length,
            "z_values": z_values,
            "floor_values": floor_values,
            "grid": {f"{z}|{f}": v for (z, f), v in grid.items()},
            **report,
        }, indent=2))
        print(f"\nWrote → {args.out}")


if __name__ == "__main__":
    main()
