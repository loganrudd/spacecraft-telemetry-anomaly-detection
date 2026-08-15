"""Ablation: per-channel vs per-subsystem HPO for one subsystem.

Answers a concrete question raised while debugging why S4000001/P6000001
(ISS power subsystem) miss an injected drift anomaly that other channels in
the same subsystem detect: is per-subsystem-pooled HPO ("ray tune", the
current default -- see ray_fanout/tune.py) actually leaving detection on the
table relative to per-channel HPO, and if per-channel is switched on, does it
hurt any channel that the pooled sweep already serves well?

Runs run_hpo_sweep() twice for the given subsystem:

  1. Pooled  -- exactly what ``ray tune --subsystem <name>`` does today: one
     config chosen to maximise the MEAN objective across every channel in
     the subsystem (module docstring in ray_fanout/tune.py).
  2. Per-channel -- run_hpo_sweep() called once per channel with
     channels=[ch], so each channel gets its own optimizer run.

Both configs are then evaluated against each channel's held-out final
portion -- the tail slice ``settings.tune.hpo_eval_fraction`` reserves and
that NO HPO trial (pooled or per-channel) has optimized against -- so the
comparison isn't just re-reporting in-sample HPO scores. Reports seg_f0_5 /
seg_precision / seg_recall / nominal_fp_rate side by side per channel.

This directly tests the concern that per-channel tuning could overfit a
channel's sparse injected-fault labels and generalise worse than the
pooled sweep: if per-channel doesn't beat pooled on the held-out portion,
that overfitting risk materialised for that channel.

No model re-training occurs -- only cached errors.npy scoring passes, same
as run_hpo_sweep() itself. Requires each channel to already have a nominal
AND an injected scoring run in MLflow (`ray score --mission ISS` and
`ray score --mission ISS --injected`).

Usage (requires cloud MLflow + GCS -- see --window-size note below for ISS):
    SPACECRAFT_MLFLOW__TRACKING_URI=... \\
    SPACECRAFT_PREPROCESS__PROCESSED_DATA_DIR=gs://<project>-processed-data \\
    SSL_CERT_FILE=$(uv run python -m certifi) \\
    uv run python scripts/ablation_per_channel_hpo.py \\
        --env cloud --mission ISS --subsystem power --window-size 128

    # Faster/cheaper sweep for a quick look:
    uv run python scripts/ablation_per_channel_hpo.py \\
        --env cloud --mission ISS --subsystem power --window-size 128 --num-samples 15
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any

from spacecraft_telemetry.core.config import load_settings
from spacecraft_telemetry.mlflow_tracking import configure_mlflow
from spacecraft_telemetry.model import scoring
from spacecraft_telemetry.ray_fanout import load_channel_subsystem_map
from spacecraft_telemetry.ray_fanout import tune as tune_mod
from spacecraft_telemetry.ray_fanout.tune import run_hpo_sweep

# Imported as a module (not `from ... import _prepare_channel_data`) so that
# --errors-dir can swap the loader for run_hpo_sweep AND for the held-out
# comparison below through a single assignment. A direct name import would
# bind the original function into this module's globals, and the held-out
# pass would silently keep using MLflow while the sweeps used local files.


def _held_out_eval(
    config: dict[str, Any],
    errors: Any,
    labels: Any,
    fraction: float,
    nominal_errors: Any | None,
) -> dict[str, float]:
    """Score one config against the held-out tail (last 1-fraction of *errors*).

    Mirrors _scoring_trial()'s math exactly (smooth -> threshold -> flag ->
    evaluate_overlap) so results are directly comparable to what the sweep
    itself optimizes, just on data no trial has seen.
    """
    n_hpo = int(len(errors) * fraction)
    held_errors = errors[n_hpo:]
    held_labels = labels[n_hpo:]
    smoothed = scoring.smooth_errors(held_errors, int(config["error_smoothing_window"]))
    thresh = scoring.dynamic_threshold(
        smoothed, int(config["threshold_window"]), float(config["threshold_z"])
    )
    flags = scoring.flag_anomalies(smoothed, thresh, int(config["threshold_min_anomaly_len"]))
    seg = scoring.evaluate_overlap(held_labels, flags)

    fp_rate = 0.0
    if nominal_errors is not None:
        nom_smoothed = scoring.smooth_errors(
            nominal_errors, int(config["error_smoothing_window"])
        )
        nom_thresh = scoring.dynamic_threshold(
            nom_smoothed, int(config["threshold_window"]), float(config["threshold_z"])
        )
        nom_flags = scoring.flag_anomalies(
            nom_smoothed, nom_thresh, int(config["threshold_min_anomaly_len"])
        )
        fp_rate = float(nom_flags.mean()) if len(nom_flags) else 0.0

    return {
        "seg_f0_5": seg["seg_f0_5"],
        "seg_precision": seg["seg_precision"],
        "seg_recall": seg["seg_recall"],
        "nominal_fp_rate": fp_rate,
    }


def _install_local_errors_source(errors_dir: str) -> None:
    """Source errors.npy from a local directory instead of an MLflow server.

    The cloud MLflow *service* is gone, but its Postgres backend and GCS
    artifacts survive, so errors.npy can be fetched out-of-band into
    ``{errors_dir}/{channel}.npy`` (see scripts/ in the scratchpad, or any
    equivalent copy). This swaps only the two data-loading seams in tune.py;
    the sweep itself — HyperOptSearch, SEARCH_SPACE, the baseline guard,
    _scoring_trial — runs completely unmodified, so the ablation still measures
    the production HPO path.

    Nominal-baseline errors are returned empty: these ESA scoring runs predate
    the Phase 15 ``data_source`` tag, so the FP penalty was never active for
    them in the cloud either. Returning {} reproduces that faithfully rather
    than inventing a baseline, and reduces the objective to pure seg_f0_5 —
    identically for both the pooled and per-channel arms.
    """
    import numpy as np

    from spacecraft_telemetry.model.dataset import load_window_labels

    root = Path(errors_dir)

    def _local_prepare(settings: Any, mission: str, channels: list[str]) -> Any:
        prepared: dict[str, tuple[Any, Any]] = {}
        run_ids: dict[str, str | None] = {}
        missing: list[str] = []
        for ch in channels:
            path = root / f"{ch}.npy"
            if not path.exists():
                missing.append(ch)
                continue
            errors = np.load(path)
            labels = load_window_labels(settings, mission, ch)
            if labels.shape != errors.shape:
                raise ValueError(
                    f"{ch}: labels{labels.shape} != errors{errors.shape}. The local "
                    "test split must be the same one that produced errors.npy "
                    "(same window_size / train_fraction / gap_multiplier)."
                )
            prepared[ch] = (errors, labels)
            run_ids[ch] = None
        if not prepared:
            raise ValueError(f"No errors.npy found under {root} for {channels}")
        if missing:
            print(f"  [warn] no errors.npy for: {missing}")

        # Same HPO-portion slice _prepare_channel_data applies; the held-out
        # tail is reserved for final eval.
        fraction = settings.tune.hpo_eval_fraction
        for ch in list(prepared):
            errors, labels = prepared[ch]
            n_hpo = int(len(errors) * fraction)
            prepared[ch] = (errors[:n_hpo], labels[:n_hpo])
        return prepared, run_ids

    def _local_nominal(settings: Any, mission: str, channels: list[str]) -> dict[str, Any]:
        return {}

    tune_mod._prepare_channel_data = _local_prepare  # type: ignore[assignment]
    tune_mod._load_nominal_errors = _local_nominal   # type: ignore[assignment]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--env", default="cloud")
    ap.add_argument(
        "--errors-dir", default=None,
        help="Directory of {channel}.npy error arrays. When set, errors are read "
        "from here instead of MLflow — lets the whole ablation run locally with "
        "no tracking server and no re-training.",
    )
    ap.add_argument("--mission", required=True)
    ap.add_argument("--subsystem", required=True)
    ap.add_argument(
        "--num-samples", type=int, default=None,
        help="override tune.num_samples for a faster/cheaper ablation run",
    )
    ap.add_argument(
        "--ray-address", default="local",
        help="Ray address for _ray_session (default: 'local' -- start a fresh "
        "local cluster on this machine). configs/cloud.yaml sets ray.address="
        "'auto', which is correct for cloud_tune.sh (a RayJob running inside "
        "GKE, attaching to the cluster's own head) but wrong for this script: "
        "we want cloud DATA (GCS/MLflow) with LOCAL compute, so this overrides "
        "it. Pass --ray-address auto only if you're actually running this "
        "inside a cluster with a reachable head.",
    )
    ap.add_argument(
        "--window-size", type=int, default=None,
        help="Override model.window_size (LSTM sliding-window length) -- must "
        "match whatever window_size the channels were actually SCORED with, "
        "or load_window_labels() and the cached errors.npy will disagree in "
        "length. ISS was scored at window_size=128 (see .claude/rules/iss.md; "
        "cloud_train.sh/cloud_score.sh/cloud_tune.sh all set "
        "SPACECRAFT_MODEL__WINDOW_SIZE=128 for it) vs the ESA default of 250 "
        "-- pass --window-size 128 for ISS.",
    )
    args = ap.parse_args()

    settings = load_settings(args.env)
    settings = settings.model_copy(
        update={"ray": settings.ray.model_copy(update={"address": args.ray_address})}
    )
    if args.num_samples is not None:
        settings = settings.model_copy(
            update={"tune": settings.tune.model_copy(
                update={"num_samples": args.num_samples}
            )}
        )
    if args.window_size is not None:
        settings = settings.model_copy(
            update={"model": settings.model.model_copy(
                update={"window_size": args.window_size}
            )}
        )
    configure_mlflow(settings)
    if args.errors_dir:
        _install_local_errors_source(args.errors_dir)
        print(f"[local mode] errors.npy from {args.errors_dir}; "
              "nominal FP penalty disabled (no data_source-tagged runs)\n")

    subsystem_map = load_channel_subsystem_map(settings, args.mission)
    channels = sorted(ch for ch, sub in subsystem_map.items() if sub == args.subsystem)
    if not channels:
        raise SystemExit(f"No channels found for subsystem {args.subsystem!r}")

    # A channel with no cached errors was never scored; the per-channel arm
    # would abort on it. Drop it from BOTH arms so the pooled and per-channel
    # sweeps see an identical channel set (an asymmetric set would make the
    # comparison meaningless).
    if args.errors_dir:
        root = Path(args.errors_dir)
        have = [c for c in channels if (root / f"{c}.npy").exists()]
        skipped = [c for c in channels if c not in have]
        if skipped:
            print(f"[local mode] no errors.npy, excluded from both arms: {skipped}")
        channels = have
        if not channels:
            raise SystemExit(f"No channels with errors.npy under {root}")
    print(f"mission={args.mission} subsystem={args.subsystem} "
          f"window_size={settings.model.window_size} "
          f"num_samples={settings.tune.num_samples} channels={channels}\n")

    from spacecraft_telemetry.cli import _ray_session

    with _ray_session(settings):
        print(f"--- pooled sweep ({len(channels)} channels together, "
              f"the current 'ray tune' default) ---")
        pooled = run_hpo_sweep(args.subsystem, channels, settings, args.mission)
        print(f"  config: {pooled['config']}")
        print(f"  hpo-portion seg_f0_5={pooled['seg_f0_5']:.3f} "
              f"nominal_fp_rate={pooled['nominal_fp_rate']:.3f}\n")

        print("--- per-channel sweeps ---")
        per_channel_config: dict[str, dict[str, Any]] = {}
        for ch in channels:
            r = run_hpo_sweep(f"{args.subsystem}__{ch}", [ch], settings, args.mission)
            per_channel_config[ch] = r["config"]
            print(f"  {ch}: {r['config']}  "
                  f"hpo-portion seg_f0_5={r['seg_f0_5']:.3f} "
                  f"nominal_fp_rate={r['nominal_fp_rate']:.3f}")
        print()

    # Held-out comparison: reload each channel's FULL (unsliced) errors/labels
    # by passing hpo_eval_fraction=1.0 to _prepare_channel_data (its internal
    # slice becomes a no-op), then slice the held-out tail ourselves using the
    # REAL fraction -- the same split run_hpo_sweep's baseline guard reserves,
    # so neither the pooled nor any per-channel sweep has optimized against it.
    fraction = settings.tune.hpo_eval_fraction
    full_settings = settings.model_copy(
        update={"tune": settings.tune.model_copy(update={"hpo_eval_fraction": 1.0})}
    )
    with warnings.catch_warnings():
        # _prepare_channel_data warns when its OWN held-out slice (empty, since
        # fraction=1.0 here) has no labels -- expected and irrelevant; we do
        # the real held-out slicing ourselves right below.
        warnings.simplefilter("ignore", UserWarning)
        full_data, _ = tune_mod._prepare_channel_data(full_settings, args.mission, channels)
    nominal = tune_mod._load_nominal_errors(settings, args.mission, channels)

    print(f"--- held-out comparison (last {1 - fraction:.0%} of test data, "
          f"unseen by any HPO trial) ---")
    for ch in channels:
        errors, labels = full_data[ch]
        nom = nominal.get(ch)
        pooled_m = _held_out_eval(pooled["config"], errors, labels, fraction, nom)
        per_ch_m = _held_out_eval(per_channel_config[ch], errors, labels, fraction, nom)

        print(f"{ch}")
        print(f"  {'':<10} {'seg_f0_5':>9} {'seg_prec':>9} {'seg_recall':>10} {'fp_rate':>8}")
        print(f"  {'pooled':<10} {pooled_m['seg_f0_5']:>9.3f} "
              f"{pooled_m['seg_precision']:>9.3f} {pooled_m['seg_recall']:>10.3f} "
              f"{pooled_m['nominal_fp_rate']:>8.3f}")
        print(f"  {'per-chan':<10} {per_ch_m['seg_f0_5']:>9.3f} "
              f"{per_ch_m['seg_precision']:>9.3f} {per_ch_m['seg_recall']:>10.3f} "
              f"{per_ch_m['nominal_fp_rate']:>8.3f}")
        delta = per_ch_m["seg_f0_5"] - pooled_m["seg_f0_5"]
        fp_delta = per_ch_m["nominal_fp_rate"] - pooled_m["nominal_fp_rate"]
        if delta > 0.01:
            verdict = "IMPROVED"
        elif delta < -0.01:
            verdict = "WORSE (overfit risk materialised)"
        else:
            verdict = "~same"
        print(f"  delta seg_f0_5={delta:+.3f}  delta fp_rate={fp_delta:+.3f}  [{verdict}]\n")


if __name__ == "__main__":
    main()
