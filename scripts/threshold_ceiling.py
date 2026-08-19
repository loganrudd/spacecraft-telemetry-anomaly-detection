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
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spacecraft_telemetry.core.config import load_settings
from spacecraft_telemetry.core.logging import get_logger
from spacecraft_telemetry.esa_adb.detections import find_scoring_run_and_artifacts
from spacecraft_telemetry.mlflow_tracking import configure_mlflow, experiment_name
from spacecraft_telemetry.model.dataset import load_window_labels
from spacecraft_telemetry.model.io import bytes_to_errors, download_artifact_bytes
from spacecraft_telemetry.ray_fanout.threshold_grid import (
    best_point,
    bounds_report,
    sweep_group,
)

log = get_logger(__name__)

_DEFAULT_CHANNELS = [f"channel_{i}" for i in range(41, 47)]
# Spans the tuned search space (threshold_z 2.5-8.0, min_error_value 0.0-0.6)
# rather than stopping short of it — an optimum on the grid edge only tells you
# the ceiling is a lower bound, which is the answer nobody wants.
_DEFAULT_Z = [2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
_DEFAULT_FLOORS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


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
    threshold_window: int | None = None
    min_run_length: int | None = None
    for channel in channels:
        run_id, errors_artifact, _threshold_artifact = find_scoring_run_and_artifacts(
            settings, args.mission, exp, channel, tuned=tuned
        )
        import mlflow

        client = mlflow.MlflowClient(tracking_uri=settings.mlflow.tracking_uri)
        params = client.get_run(run_id).data.params
        # All channels in one arm share these; capture from the first and
        # verify the rest agree, so a mixed-config arm can't be averaged
        # together silently.
        tw, mrl = int(params["threshold_window"]), int(params["threshold_min_anomaly_len"])
        if threshold_window is None:
            threshold_window, min_run_length = tw, mrl
        elif (tw, mrl) != (threshold_window, min_run_length):
            raise SystemExit(
                f"Channel {channel!r} was scored with threshold_window={tw}, "
                f"min_run_length={mrl}, but earlier channels used "
                f"{threshold_window}/{min_run_length}. Averaging across differing "
                "configs would not describe any single achievable operating point."
            )
        smoothed = bytes_to_errors(
            download_artifact_bytes(run_id, errors_artifact, settings.mlflow.tracking_uri)
        )
        labels = load_window_labels(settings, args.mission, channel)
        per_channel[channel] = (smoothed, labels)
        log.info(
            "threshold_ceiling.channel_loaded",
            channel=channel, run_id=run_id, n_windows=len(smoothed),
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
    print(f"\n{label}  mean segF0.5 = {best_score:.3f}  at z={best_z}, floor={best_floor}")
    print(f"  (selected on {args.select_on})")
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
                "_meta": {
                    "seg_f0_5": best_score,
                    "source": "scripts/threshold_ceiling.py exhaustive grid",
                    "selected_on": args.select_on,
                },
            }
        }
        Path(args.emit_tuned_configs).parent.mkdir(parents=True, exist_ok=True)
        Path(args.emit_tuned_configs).write_text(json.dumps(entry, indent=2))
        print(f"\nWrote tuned_configs → {args.emit_tuned_configs}")
        print("  NOTE: error_smoothing_window is NOT included — it is baked into the "
              "saved errors array this grid swept, so the scoring run must keep the "
              "same value it already used. Overriding it would invalidate the grid.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "mission": args.mission,
            "variant": settings.variant,
            "tuned": tuned,
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
