"""Capture the live per-tick residuals, then run the BATCH scoring pipeline.

The online engine returns threshold=None until window_size+threshold_window
ticks, so an anomaly sitting at slice-row 500 (inside that dead-zone) can never
fire. This script captures the exact (value, prediction) the live model
produces over one pass, then runs the warm batch pipeline (scoring.py) on the
residuals. If the batch detector fires at the labeled rows, the data IS
detectable and the only fix needed is more warmup lead-in (replay_warmup_rows).
"""

from __future__ import annotations

import json
import sys

import httpx
import numpy as np

from spacecraft_telemetry.core.config import load_settings
from spacecraft_telemetry.model import scoring

API = "https://api-pb5fb25noa-uc.a.run.app"
CH = sys.argv[1] if len(sys.argv) > 1 else "channel_41"


def capture_one_pass(ch: str) -> list[dict]:
    url = f"{API}/api/stream/telemetry?channels={ch}&speed=50"
    rows: list[dict] = []
    capturing = False
    last_ts = None
    with httpx.stream("GET", url, timeout=httpx.Timeout(30.0, read=None)) as r:
        r.raise_for_status()
        buf = ""
        for chunk in r.iter_text():
            buf += chunk
            while "\n\n" in buf:
                frame, buf = buf.split("\n\n", 1)
                dl = next((ln[5:].strip() for ln in frame.splitlines()
                           if ln.startswith("data:")), None)
                if not dl:
                    continue
                ev = json.loads(dl)
                ts = ev["timestamp"]
                restart = last_ts is not None and ts < last_ts
                last_ts = ts
                if restart:
                    if capturing:
                        return rows
                    capturing = True
                    rows = []
                if capturing:
                    rows.append(ev)


def main() -> None:
    s = load_settings("cloud")
    cfg = s.model
    print(f"capturing one pass of {CH} from live API ...")
    rows = capture_one_pass(CH)
    n = len(rows)
    values = np.array([e["value_normalized"] for e in rows], dtype=np.float64)
    # Use the model's own predictions; residual where present, else NaN->skip.
    preds = np.array(
        [e["prediction"] if e["prediction"] is not None else np.nan for e in rows],
        dtype=np.float64,
    )
    labels = np.array([bool(e["is_anomaly"]) for e in rows])

    # Residuals start once predictions exist (after window warmup).
    valid = ~np.isnan(preds)
    first_valid = int(np.argmax(valid))
    v = values[valid]
    p = preds[valid]
    lab = labels[valid]
    errors = p - v  # same orientation as scoring.py (preds - targets)

    anom_idx = np.where(lab)[0]
    smoothed = scoring.smooth_errors(errors, cfg.error_smoothing_window)
    print(f"\n{CH}: pass_len={n} first_pred_at_row={first_valid} "
          f"valid_ticks={len(v)} labeled_anomaly_ticks={len(anom_idx)}")
    print(f"params: ewma_span={cfg.error_smoothing_window} "
          f"threshold_window={cfg.threshold_window} min_len={cfg.threshold_min_anomaly_len}")

    if not anom_idx.size:
        return
    lo, hi = int(anom_idx[0]), int(anom_idx[-1]) + 1

    # Sweep z: find the largest z (most conservative) that still fires within
    # the labeled span. Lower z = more sensitive. Search floor is 1.5.
    print(f"\n  z      thresh@peak  fires_in_span  seg_recall  total_pred")
    for z in [3.0, 2.5, 2.0, 1.5, 1.25, 1.0, 0.75]:
        th = scoring.dynamic_threshold(smoothed, cfg.threshold_window, z)
        fl = scoring.flag_anomalies(smoothed, th, cfg.threshold_min_anomaly_len)
        fires = int(fl[lo:hi].sum())
        rec = scoring.evaluate_overlap(lab, fl)["seg_recall"]
        th_at_peak = th[lo:hi][np.argmax(smoothed[lo:hi])]
        mark = "  <-- FIRES" if fires > 0 else ""
        print(f"  {z:<5} {th_at_peak:>10.4f}  {fires:>13} {rec:>11.3f} "
              f"{int(fl.sum()):>11}{mark}")
    peak = smoothed[lo:hi].max()
    print(f"  anomaly peak smoothed error = {peak:.4f}")


if __name__ == "__main__":
    main()
