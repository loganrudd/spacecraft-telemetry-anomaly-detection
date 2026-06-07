"""Definitive: run the REAL online engine with TUNED params on a warmed slice.

Loads the model + tuned scoring params from MLflow (via the gcloud proxy on
localhost:5001), then replays a slice with a LARGE warmup so the anomaly lands
well past the detector's warm point (window_size + threshold_window). This is
exactly what the deployed service would do after the replay_warmup_rows fix.
Reports whether is_anomaly_predicted fires within the labeled anomaly span.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from spacecraft_telemetry.api.inference import ChannelInferenceEngine
from spacecraft_telemetry.api.replay import _anomaly_slice
from spacecraft_telemetry.core.config import load_settings
from spacecraft_telemetry.model.dataset import load_series_parquet
from spacecraft_telemetry.model.device import resolve_device
from spacecraft_telemetry.model.io import load_model_for_scoring, load_scoring_params
from spacecraft_telemetry.mlflow_tracking.conventions import registered_model_name

URI = "http://localhost:5001"
CHANNELS = sys.argv[1:] or ["channel_44"]
WARMUP = 1200  # > window_size(250) + tuned threshold_window(351) = 601, big margin
MAX_ROWS = 3500


def main() -> None:
    import mlflow
    mlflow.set_tracking_uri(URI)
    mlflow.set_registry_uri(URI)
    s = load_settings("cloud")
    mission = s.api.mission
    device = resolve_device(s.model.device)
    print(f"warmup_rows={WARMUP} max_rows={MAX_ROWS} device={device}\n")

    for ch in CHANNELS:
        values, _seg, anom, ts = load_series_parquet(
            s.preprocess.processed_data_dir, mission, ch, "test"
        )
        sl = _anomaly_slice(anom, WARMUP, MAX_ROWS)
        v, a, t = values[sl], anom[sl], ts[sl]

        name = registered_model_name("telemanom", mission, ch)
        model, ws = load_model_for_scoring(name, device, URI)
        model.eval()
        params = load_scoring_params(channel=ch, mission=mission, tracking_uri=URI)
        engine = ChannelInferenceEngine(
            mission=mission, channel=ch, model=model,
            window_size=ws, params=params, device=device,
        )
        engine.reset()

        warm_at = ws + params.threshold_window
        anom_rows = np.where(a)[0]
        first_anom = int(anom_rows[0]) if anom_rows.size else -1

        pred = np.zeros(len(v), dtype=bool)
        sm = np.full(len(v), np.nan)
        th = np.full(len(v), np.nan)
        for i, val in enumerate(v):
            ev = engine.step(float(val), pd.Timestamp(t[i]).to_pydatetime(), bool(a[i]))
            pred[i] = ev.is_anomaly_predicted
            if ev.smoothed_error is not None:
                sm[i] = ev.smoothed_error
            if ev.threshold is not None:
                th[i] = ev.threshold

        fired_in_span = int(pred[anom_rows].sum()) if anom_rows.size else 0
        peak = float(np.nanmax(sm[anom_rows])) if anom_rows.size else float("nan")
        th_at_anom = th[anom_rows]
        th_med = float(np.nanmedian(th_at_anom)) if anom_rows.size else float("nan")
        ratio = peak / th_med if th_med and not np.isnan(th_med) else float("nan")

        print(f"=== {ch} (z={params.threshold_z:.2f} tw={params.threshold_window} "
              f"span={params.error_smoothing_window} K={params.threshold_min_anomaly_len}) ===")
        print(f"  detector warm at row {warm_at}; anomaly at row {first_anom} "
              f"-> {'WARM ✓' if first_anom > warm_at else 'STILL IN WARMUP ✗'}")
        print(f"  total predicted ticks: {int(pred.sum())}")
        print(f"  fired WITHIN anomaly span: {fired_in_span}  "
              f"{'<-- FIRES ✓' if fired_in_span else '<-- still silent ✗'}")
        print(f"  anomaly smoothed peak={peak:.4f}  threshold@anomaly(median)={th_med:.4f}  "
              f"ratio={ratio:.3f}  (need >= 1.0 for K={params.threshold_min_anomaly_len} consecutive)")

        # Sweep z and K to find what WOULD fire (re-replay engine per combo).
        import dataclasses
        print("  param sweep (fired_in_span / total_pred):")
        for z in [4.84, 4.0, 3.0, 2.5, 2.0]:
            line = f"    z={z:<4}"
            for K in [7, 3, 1]:
                p2 = dataclasses.replace(params, threshold_z=z, threshold_min_anomaly_len=K)
                eng = ChannelInferenceEngine(
                    mission=mission, channel=ch, model=model,
                    window_size=ws, params=p2, device=device,
                )
                eng.reset()
                pr = np.zeros(len(v), dtype=bool)
                for i, val in enumerate(v):
                    e2 = eng.step(float(val), pd.Timestamp(t[i]).to_pydatetime(), bool(a[i]))
                    pr[i] = e2.is_anomaly_predicted
                fis = int(pr[anom_rows].sum()) if anom_rows.size else 0
                tot = int(pr.sum())
                tag = "FIRE" if fis else "----"
                line += f"   K={K}:{tag}({fis}/{tot})"
            print(line)
        print()


if __name__ == "__main__":
    main()
