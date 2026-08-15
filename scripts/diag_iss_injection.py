"""Diagnostic: inject a drift fault offline and trace why some channels miss it.

Reproduces the live ISS demo's "Inject Fault (drift)" exactly through the real
serving engine (ChannelInferenceEngine + EventBroadcaster.apply_fault), for the
6 stationary demo channels, and reports per channel:

  * the tuned scoring params actually in force (so you can SEE whether all
    channels in a subsystem share one (z, K, span, window) -- the per-subsystem
    HPO question), and
  * the per-tick residual / smoothed-error / threshold trace across the
    injection window, plus whether is_anomaly_predicted ever fired, and
  * a z-sweep: the lowest threshold_z at which THIS channel would have fired.

The z-sweep is the discriminator between the two candidate causes of a miss:

  (A) HPO-compromise threshold -- the residual bump DOES clear a lower z than
      the subsystem's chosen z. Per-channel tuning (a la Hundman) would rescue
      it: the channel is being out-voted by the subsystem-mean objective.

  (B) Forecaster tracks the drift -- the smoothed error barely rises during the
      plateau (a one-step LSTM predicts ~= the drifted level), so NO z clears
      it. Per-channel HPO would not help; the point-forecaster is structurally
      blind to a slow sustained drift, and the honest signal is the drift panel.

Run against the same cloud env as the live ISS service, e.g.:

    SPACECRAFT_API__MISSION=ISS \\
    SPACECRAFT_MODEL__WINDOW_SIZE=128 \\
    SPACECRAFT_MLFLOW__TRACKING_URI=... MLFLOW_TRACKING_TOKEN=... \\
    SPACECRAFT_PREPROCESS__PROCESSED_DATA_DIR=gs://<project>-processed-data \\
    SSL_CERT_FILE=$(python -m certifi) \\
    uv run python scripts/diag_iss_injection.py

Optional flags: --channels a,b,c  --magnitude 5.0  --duration 20  --lead-in 40
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from spacecraft_telemetry.api.broadcast import EventBroadcaster
from spacecraft_telemetry.api.inference import ChannelInferenceEngine
from spacecraft_telemetry.core.config import load_settings
from spacecraft_telemetry.mlflow_tracking import configure_mlflow
from spacecraft_telemetry.mlflow_tracking.conventions import registered_model_name
from spacecraft_telemetry.model import scoring
from spacecraft_telemetry.model.dataset import load_series_parquet
from spacecraft_telemetry.model.device import resolve_device
from spacecraft_telemetry.model.io import load_model_for_scoring, load_scoring_params

# The 6 stationary demo channels (see .claude/rules/iss.md demo tiering).
_DEMO_CHANNELS = [
    "P4000001", "S4000001", "P6000001", "S6000001",  # power PV voltages
    "S1000003", "P1000003",                          # thermal loops
]

_Z_SWEEP = [3.0, 2.5, 2.0, 1.5, 1.25, 1.0, 0.75, 0.5]


def _run_channel(
    ch: str,
    settings: object,
    device: object,
    magnitude: float,
    duration: int,
    lead_in: int,
) -> None:
    s = settings  # local alias
    mission = s.api.mission  # type: ignore[attr-defined]
    tracking_uri = s.mlflow.tracking_uri  # type: ignore[attr-defined]
    processed_dir = s.preprocess.processed_data_dir  # type: ignore[attr-defined]

    values, _seg, _anom, ts = load_series_parquet(processed_dir, mission, ch, "test")

    name = registered_model_name("telemanom", mission, ch)
    model, window_size = load_model_for_scoring(name, device, tracking_uri)
    model.eval()
    params = load_scoring_params(channel=ch, mission=mission, tracking_uri=tracking_uri)

    engine = ChannelInferenceEngine(
        mission=mission, channel=ch, model=model,
        window_size=window_size, params=params, device=device,
    )

    # Prime exactly like startup (window + threshold ring buffer warm) from the
    # head of the test series, then run the remainder tick-by-tick.
    warm = window_size + params.threshold_window
    if len(values) < warm + lead_in + duration + 10:
        print(f"=== {ch} ===  SKIPPED: only {len(values)} test rows, "
              f"need >= {warm + lead_in + duration + 10}\n")
        return
    engine.prime_with_scoring([float(x) for x in values[:warm]])

    # Injection starts `lead_in` ticks after priming; a fresh broadcaster gives
    # byte-identical drift math to the live pump / replay loop.
    tail_v = values[warm:]
    # load_series_parquet returns numpy.datetime64; TelemetryEvent (pydantic)
    # requires a real datetime -- same conversion broadcast.py's run_shared_loop
    # applies per-tick (pd.Timestamp(...).to_pydatetime()), done upfront here.
    tail_ts = [pd.Timestamp(t).to_pydatetime() for t in ts[warm:]]
    inject_start = lead_in
    broadcaster = EventBroadcaster()
    broadcaster.request_injection("drift", frozenset(), magnitude, duration)

    n = min(len(tail_v), lead_in + duration + 20)
    residual = np.full(n, np.nan)
    smoothed = np.full(n, np.nan)
    thresh = np.full(n, np.nan)
    injected_flag = np.zeros(n, dtype=bool)
    pred_flag = np.zeros(n, dtype=bool)

    for i in range(n):
        raw_val = float(tail_v[i])
        if i == inject_start:
            pass  # injection already pending; begin_tick activates it below
        if inject_start <= i < inject_start + duration:
            broadcaster.begin_tick()
            val, was_inj = broadcaster.apply_fault(ch, raw_val)
            injected_flag[i] = was_inj
        else:
            val = raw_val
        ev = engine.step(val, tail_ts[i], injected_flag[i])
        if ev.residual is not None:
            residual[i] = ev.residual
        if ev.smoothed_error is not None:
            smoothed[i] = ev.smoothed_error
        if ev.threshold is not None:
            thresh[i] = ev.threshold
        pred_flag[i] = ev.is_anomaly_predicted
        if inject_start <= i < inject_start + duration:
            broadcaster.end_tick()

    fired = bool(pred_flag[inject_start:inject_start + duration + 10].any())

    # z-sweep on the captured smoothed errors: lowest z at which a run of >= K
    # crossings appears inside the injection+decay window. Answers "would a
    # lower (per-channel) z have caught it?".
    lo, hi = inject_start, min(n, inject_start + duration + 10)
    min_z_fire: float | None = None
    for z in _Z_SWEEP:
        th = scoring.dynamic_threshold(smoothed, params.threshold_window, z)
        fl = scoring.flag_anomalies(smoothed, th, params.threshold_min_anomaly_len)
        if fl[lo:hi].any():
            min_z_fire = z

    seg = smoothed[lo:hi]
    peak = float(np.nanmax(seg)) if np.isfinite(seg).any() else float("nan")
    th_seg = thresh[lo:hi]
    th_at_peak = (
        float(th_seg[int(np.nanargmax(seg))]) if np.isfinite(seg).any() else float("nan")
    )

    print(f"=== {ch} ===")
    print(f"  params: z={params.threshold_z} K={params.threshold_min_anomaly_len} "
          f"ewma_span={params.error_smoothing_window} "
          f"threshold_window={params.threshold_window} window_size={window_size}")
    print(f"  injected {magnitude}sigma drift over {duration} ticks at tick {inject_start}")
    print(f"  is_anomaly_predicted fired: {fired}")
    print(f"  peak smoothed err in window = {peak:.4f}  vs threshold@peak = {th_at_peak:.4f}")
    print(f"  lowest z that would fire (this channel) = "
          f"{min_z_fire if min_z_fire is not None else 'never in sweep'}")
    print("  per-tick trace (I=injected, *=predicted anomaly):")
    for i in range(max(0, inject_start - 2), hi):
        r = "nan" if np.isnan(residual[i]) else f"{residual[i]:+.4f}"
        sm = "nan" if np.isnan(smoothed[i]) else f"{smoothed[i]:.4f}"
        th = "nan" if np.isnan(thresh[i]) else f"{thresh[i]:.4f}"
        tags = ("I" if injected_flag[i] else " ") + ("*" if pred_flag[i] else " ")
        print(f"    t{i:>4} {tags} resid={r:>9} smoothed={sm:>8} thresh={th:>8}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--env", default="cloud")
    ap.add_argument("--channels", default=",".join(_DEMO_CHANNELS),
                    help="comma-separated channel IDs")
    ap.add_argument("--magnitude", type=float, default=5.0, help="drift sigma")
    ap.add_argument("--duration", type=int, default=20, help="fault length in ticks")
    ap.add_argument("--lead-in", type=int, default=40,
                    help="nominal ticks after priming before injection")
    args = ap.parse_args()

    s = load_settings(args.env)
    configure_mlflow(s)
    device = resolve_device(s.model.device)
    print(f"mission={s.api.mission} device={device} tracking_uri={s.mlflow.tracking_uri}")
    print(f"injection: {args.magnitude}sigma drift / {args.duration} ticks / "
          f"lead-in {args.lead_in}\n")

    for ch in [c.strip() for c in args.channels.split(",") if c.strip()]:
        try:
            _run_channel(ch, s, device, args.magnitude, args.duration, args.lead_in)
        except Exception as exc:  # one bad channel shouldn't abort the sweep
            print(f"=== {ch} ===  ERROR: {exc}\n")


if __name__ == "__main__":
    main()
