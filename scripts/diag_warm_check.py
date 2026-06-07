"""Would channel_44 fire IF warmup were fixed, using the LIVE tuned params?

The live SSE exposes the tuned-param smoothed_error every tick and the tuned
threshold in the warm region. The anomaly sits in the warmup dead-zone (threshold
gated to None), but its smoothed errors are real. If we moved the anomaly past
warmup, the threshold there would equal the tuned threshold seen in nearby warm
NOMINAL rows. So: does the anomaly's smoothed-error peak exceed that nominal
threshold for >= K consecutive ticks? That is the fixable-or-not question.
"""

from __future__ import annotations

import json
import sys

import httpx
import numpy as np

API = "https://api-pb5fb25noa-uc.a.run.app"
CH = sys.argv[1] if len(sys.argv) > 1 else "channel_44"


def capture_pass(ch: str) -> list[dict]:
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
    print(f"capturing {CH} ...")
    rows = capture_pass(CH)
    n = len(rows)
    sm = np.array([e["smoothed_error"] if e["smoothed_error"] is not None else np.nan
                   for e in rows])
    th = np.array([e["threshold"] if e["threshold"] is not None else np.nan
                   for e in rows])
    lab = np.array([bool(e["is_anomaly"]) for e in rows])
    pred = np.array([bool(e["is_anomaly_predicted"]) for e in rows])

    anom_idx = np.where(lab)[0]
    warm_idx = np.where(~np.isnan(th))[0]
    print(f"\n{CH}: pass={n} ticks; predicted_total={int(pred.sum())}")
    print(f"anomaly rows: {anom_idx[0]}..{anom_idx[-1]} (count {len(anom_idx)})")
    print(f"first warm (threshold present) row: {warm_idx[0]}")

    # Tuned threshold level in the warm NOMINAL region (exclude any anomaly rows).
    nominal_warm = [i for i in warm_idx if not lab[i]]
    th_nom = th[nominal_warm]
    sm_nom = sm[nominal_warm]
    print(f"\ntuned threshold over warm NOMINAL rows: "
          f"min={np.nanmin(th_nom):.4f} median={np.nanmedian(th_nom):.4f} "
          f"max={np.nanmax(th_nom):.4f}")
    print(f"tuned smoothed over warm NOMINAL rows:  "
          f"median={np.nanmedian(sm_nom):.4f} p99={np.nanpercentile(sm_nom,99):.4f} "
          f"max={np.nanmax(sm_nom):.4f}")

    # Anomaly smoothed errors (these are real even though threshold is gated).
    sm_anom = sm[anom_idx]
    print(f"\nanomaly smoothed errors: max={np.nanmax(sm_anom):.4f} "
          f"mean={np.nanmean(sm_anom):.4f}")

    # Simulate: if the anomaly were past warmup, threshold ~= nominal median.
    # Count consecutive anomaly ticks exceeding that level.
    th_ref_median = float(np.nanmedian(th_nom))
    th_ref_min = float(np.nanmin(th_nom))
    for label, ref in [("median nominal thr", th_ref_median),
                       ("min nominal thr", th_ref_min)]:
        over = sm_anom > ref
        # longest consecutive run
        best = run = 0
        for x in over:
            run = run + 1 if x else 0
            best = max(best, run)
        print(f"  vs {label}={ref:.4f}: anomaly ticks over={int(over.sum())}/"
              f"{len(over)}, longest_consecutive_run={best}")

    print("\nVERDICT: if longest_consecutive_run >= K (tuned min_anomaly_len, "
          "usually 1-3), the anomaly WOULD fire once warmup is fixed.")


if __name__ == "__main__":
    main()
