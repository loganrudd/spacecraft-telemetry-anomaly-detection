"""Scan all loaded channels for anomaly detections in the live SSE stream.

Connects once to the full-mission stream (speed=50), captures one complete
replay pass, and reports per channel: labeled ticks, predicted ticks, and
seg_recall from the stream output. Channels with predicted > 0 are candidates
for demo. No model or MLflow auth required — reads the deployed API directly.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

import httpx

from spacecraft_telemetry.model.scoring import _find_sequences

API = "https://api-pb5fb25noa-uc.a.run.app"


def main() -> None:
    # First get the channel list from health.
    h = httpx.get(f"{API}/health", timeout=10).json()
    all_channels: list[str] = sorted(h.get("channels_loaded", []))
    if not all_channels:
        print("no channels loaded"); return
    print(f"scanning {len(all_channels)} channels (speed=50x)...\n")

    channels_param = ",".join(all_channels)
    url = f"{API}/api/stream/telemetry?channels={channels_param}&speed=50"

    ev_buf: dict[str, list[dict]] = defaultdict(list)
    last_ts: dict[str, str] = {}
    pass_done: set[str] = set()

    with httpx.stream("GET", url, timeout=httpx.Timeout(30.0, read=None)) as r:
        r.raise_for_status()
        buf = ""
        for chunk in r.iter_text():
            buf += chunk
            while "\n\n" in buf:
                frame, buf = buf.split("\n\n", 1)
                dl = next((ln[5:] for ln in frame.splitlines()
                           if ln.startswith("data:")), None)
                if not dl:
                    continue
                ev = json.loads(dl)
                ch = ev["channel"]
                if ch in pass_done:
                    continue
                ts = ev["timestamp"]
                if ch in last_ts and ts < last_ts[ch]:
                    # loop restarted → first pass complete for this channel
                    pass_done.add(ch)
                else:
                    ev_buf[ch].append(ev)
                last_ts[ch] = ts
            if pass_done == set(all_channels):
                break

    # Summarise
    rows: list[tuple] = []
    for ch in all_channels:
        evs = ev_buf[ch]
        n = len(evs)
        label_true = sum(1 for e in evs if e.get("is_anomaly"))
        pred_true = sum(1 for e in evs if e.get("is_anomaly_predicted"))
        lab_arr = [bool(e.get("is_anomaly")) for e in evs]
        pred_arr = [bool(e.get("is_anomaly_predicted")) for e in evs]

        import numpy as np
        lab = np.array(lab_arr)
        pred = np.array(pred_arr)
        true_seqs = _find_sequences(lab)
        pred_seqs = _find_sequences(pred)

        def _overlaps_any(s, e, cands):
            return any(cs < e and ce > s for cs, ce in cands)

        n_true = len(true_seqs)
        tp_r = sum(1 for s, e in true_seqs if _overlaps_any(s, e, pred_seqs))
        seg_recall = tp_r / n_true if n_true > 0 else 0.0
        tp_p = sum(1 for s, e in pred_seqs if _overlaps_any(s, e, true_seqs))
        seg_prec = tp_p / len(pred_seqs) if pred_seqs else 0.0
        rows.append((ch, n, label_true, pred_true, n_true, len(pred_seqs),
                     seg_recall, seg_prec))

    print(f"{'channel':>12} {'ticks':>6} {'label':>6} {'pred':>5} "
          f"{'true_seqs':>9} {'pred_seqs':>9} {'seg_rec':>8} {'seg_prec':>9}")
    print("-" * 80)
    for ch, n, lt, pt, ts, ps, sr, sp in sorted(rows, key=lambda r: -r[6]):
        mark = " *** FIRES ***" if pt > 0 else ""
        print(f"{ch:>12} {n:>6} {lt:>6} {pt:>5} {ts:>9} {ps:>9} "
              f"{sr:>8.3f} {sp:>9.3f}{mark}")

    hits = [(ch, sr, sp) for ch, *_, ts, ps, sr, sp in rows if sr > 0]
    print(f"\n{len(hits)} channel(s) with seg_recall > 0:")
    for ch, sr, sp in hits:
        print(f"  {ch}  seg_recall={sr:.3f}  seg_prec={sp:.3f}")


if __name__ == "__main__":
    main()
