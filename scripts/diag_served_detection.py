"""Diagnostic: replay the exact served slice through the real serving engine.

Replicates api/broadcast.py serving for a handful of channels and reports,
around each labeled anomaly row, what the ChannelInferenceEngine actually
produces (smoothed_error vs threshold, is_anomaly_predicted). This tells us
whether the miss is a warmup-boundary problem, a threshold-sensitivity
problem, or something else — without guessing.
"""

from __future__ import annotations

import numpy as np

from spacecraft_telemetry.api.inference import ChannelInferenceEngine
from spacecraft_telemetry.api.replay import _anomaly_slice
from spacecraft_telemetry.core.config import load_settings
from spacecraft_telemetry.model.dataset import load_series_parquet
from spacecraft_telemetry.model.device import resolve_device
from spacecraft_telemetry.model.io import load_model_for_scoring, load_scoring_params
from spacecraft_telemetry.mlflow_tracking.conventions import registered_model_name
from spacecraft_telemetry.mlflow_tracking import configure_mlflow

CHANNELS = ["channel_41", "channel_43", "channel_44", "channel_45"]


def main() -> None:
    s = load_settings("cloud")
    configure_mlflow(s)
    mission = s.api.mission
    device = resolve_device(s.model.device)
    warmup, max_rows = s.api.replay_warmup_rows, s.api.replay_max_rows
    print(f"mission={mission} device={device} warmup_rows={warmup} max_rows={max_rows}")
    print(f"tracking_uri={s.mlflow.tracking_uri}\n")

    for ch in CHANNELS:
        values, _seg, anom, ts = load_series_parquet(
            s.preprocess.processed_data_dir, mission, ch, "test"
        )
        sl = _anomaly_slice(anom, warmup, max_rows)
        v, a = values[sl], anom[sl]

        name = registered_model_name("telemanom", mission, ch)
        model, window_size = load_model_for_scoring(name, device, s.mlflow.tracking_uri)
        model.eval()
        params = load_scoring_params(
            channel=ch, mission=mission, tracking_uri=s.mlflow.tracking_uri
        )
        engine = ChannelInferenceEngine(
            mission=mission, channel=ch, model=model,
            window_size=window_size, params=params, device=device,
        )
        engine.reset()

        warm_at = window_size + params.threshold_window  # first finite threshold
        anom_rows = np.where(a)[0]
        first_anom = int(anom_rows[0]) if anom_rows.size else -1

        pred_flags = np.zeros(len(v), dtype=bool)
        smoothed = np.full(len(v), np.nan)
        thresh = np.full(len(v), np.nan)
        for i, val in enumerate(v):
            ev = engine.step(float(val), ts[sl][i], bool(a[i]))
            pred_flags[i] = ev.is_anomaly_predicted
            if ev.smoothed_error is not None:
                smoothed[i] = ev.smoothed_error
            if ev.threshold is not None:
                thresh[i] = ev.threshold

        n_pred = int(pred_flags.sum())
        # How often did smoothed exceed threshold at the labeled anomaly rows?
        crossed_at_anom = int(
            np.sum(
                (smoothed[anom_rows] > thresh[anom_rows])
                & ~np.isnan(thresh[anom_rows])
            )
        ) if anom_rows.size else 0

        print(f"=== {ch} ===")
        print(f"  window_size={window_size} threshold_window={params.threshold_window} "
              f"z={params.threshold_z} K={params.threshold_min_anomaly_len} "
              f"ewma_span={params.error_smoothing_window}")
        print(f"  detector warm at row {warm_at}; first labeled anomaly at row {first_anom}")
        print(f"  labeled anomaly rows in slice: {anom_rows.size}; "
              f"predicted-anomaly rows: {n_pred}; "
              f"raw smoothed>thresh at anomaly rows: {crossed_at_anom}")
        # Sample the detector state right around the first anomaly run.
        if first_anom >= 0:
            lo, hi = max(0, first_anom - 2), min(len(v), first_anom + 12)
            print(f"  around first anomaly [rows {lo}..{hi}):")
            for i in range(lo, hi):
                sm = "nan" if np.isnan(smoothed[i]) else f"{smoothed[i]:.5f}"
                th = "nan" if np.isnan(thresh[i]) else f"{thresh[i]:.5f}"
                print(f"    row {i:4d} label={int(a[i])} smoothed={sm:>9} "
                      f"thresh={th:>9} pred={int(pred_flags[i])}")
        print()


if __name__ == "__main__":
    main()
