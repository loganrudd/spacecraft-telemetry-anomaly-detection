"""Count labeled vs predicted anomalies per channel from the LIVE SSE stream.

Connects to the deployed API's telemetry stream (the real production serving
path: shared broadcast loop + ChannelInferenceEngine), parses events, and
tallies per channel: events seen, label-true ticks, predicted-true ticks,
and how many predicted ticks were non-warmup (threshold present).
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict

import httpx

API = "https://api-pb5fb25noa-uc.a.run.app"
CHANNELS = ["channel_41", "channel_43", "channel_44", "channel_45"]
RUN_SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 else 240.0


def main() -> None:
    url = f"{API}/api/stream/telemetry?channels={','.join(CHANNELS)}&speed=50"
    seen = defaultdict(int)
    label_true = defaultdict(int)
    pred_true = defaultdict(int)
    thresh_present = defaultdict(int)
    first_ts: dict[str, str] = {}
    last_ts: dict[str, str] = {}
    wraps = defaultdict(int)
    # At labeled-anomaly ticks: best smoothed/threshold ratio (how close to firing)
    best_ratio_at_anom = defaultdict(float)
    # Overall best ratio anywhere (does it ever get near firing at all?)
    best_ratio_any = defaultdict(float)

    print(f"connecting {url}\nrunning {RUN_SECONDS}s ...")
    deadline = time.time() + RUN_SECONDS
    with httpx.stream("GET", url, timeout=httpx.Timeout(30.0, read=None)) as r:
        r.raise_for_status()
        buf = ""
        for chunk in r.iter_text():
            buf += chunk
            while "\n\n" in buf:
                frame, buf = buf.split("\n\n", 1)
                data_line = next(
                    (ln[5:].strip() for ln in frame.splitlines() if ln.startswith("data:")),
                    None,
                )
                if not data_line:
                    continue
                ev = json.loads(data_line)
                ch = ev["channel"]
                seen[ch] += 1
                if ev.get("is_anomaly"):
                    label_true[ch] += 1
                if ev.get("is_anomaly_predicted"):
                    pred_true[ch] += 1
                th = ev.get("threshold")
                sm = ev.get("smoothed_error")
                if th is not None:
                    thresh_present[ch] += 1
                    if th > 0 and sm is not None:
                        ratio = sm / th
                        best_ratio_any[ch] = max(best_ratio_any[ch], ratio)
                        if ev.get("is_anomaly"):
                            best_ratio_at_anom[ch] = max(
                                best_ratio_at_anom[ch], ratio
                            )
                ts = ev["timestamp"]
                if ch not in first_ts:
                    first_ts[ch] = ts
                if ch in last_ts and ts < last_ts[ch]:
                    wraps[ch] += 1  # timestamp went backwards => loop restarted
                last_ts[ch] = ts
            if time.time() > deadline:
                break

    print("\n=== results ===")
    for ch in CHANNELS:
        print(f"{ch}: seen={seen[ch]:5d} label_true={label_true[ch]:4d} "
              f"pred_true={pred_true[ch]:4d} thresh_present={thresh_present[ch]:5d} "
              f"loop_restarts={wraps[ch]}")
        print(f"        best smoothed/threshold ratio  AT anomaly ticks: "
              f"{best_ratio_at_anom[ch]:.3f}   anywhere: {best_ratio_any[ch]:.3f}")
    print("\nratio >= 1.0 => would fire. A ratio at-anomaly of e.g. 0.9 means a "
          "small z reduction would catch it; 0.3 means the model predicts the "
          "anomaly too well for error-threshold detection to ever fire.")


if __name__ == "__main__":
    main()
