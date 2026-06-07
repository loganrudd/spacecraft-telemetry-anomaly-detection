"""Dump an ordered per-tick trace for ONE channel from the live SSE stream.

Captures one full replay pass (between loop restarts) and prints the detector
state at every labeled-anomaly tick plus surrounding context, so we can see
exactly why the detector does/doesn't fire.
"""

from __future__ import annotations

import json
import sys

import httpx

API = "https://api-pb5fb25noa-uc.a.run.app"
CH = sys.argv[1] if len(sys.argv) > 1 else "channel_41"


def main() -> None:
    url = f"{API}/api/stream/telemetry?channels={CH}&speed=50"
    print(f"connecting {url}")
    rows: list[dict] = []  # one full pass: list of event dicts in order
    capturing = False
    last_ts = None

    with httpx.stream("GET", url, timeout=httpx.Timeout(30.0, read=None)) as r:
        r.raise_for_status()
        buf = ""
        for chunk in r.iter_text():
            buf += chunk
            while "\n\n" in buf:
                frame, buf = buf.split("\n\n", 1)
                dl = next(
                    (ln[5:].strip() for ln in frame.splitlines()
                     if ln.startswith("data:")),
                    None,
                )
                if not dl:
                    continue
                ev = json.loads(dl)
                ts = ev["timestamp"]
                # Detect loop restart: timestamp goes backwards.
                restart = last_ts is not None and ts < last_ts
                last_ts = ts
                if restart:
                    if capturing:
                        # finished one full pass
                        _report(rows)
                        return
                    capturing = True
                    rows = []
                if capturing:
                    rows.append(ev)


def _report(rows: list[dict]) -> None:
    n = len(rows)
    anom_idx = [i for i, e in enumerate(rows) if e.get("is_anomaly")]
    pred_idx = [i for i, e in enumerate(rows) if e.get("is_anomaly_predicted")]
    warm_idx = [i for i, e in enumerate(rows) if e.get("threshold") is not None]
    print(f"\n=== {CH}: one full pass, {n} ticks ===")
    print(f"labeled-anomaly ticks: {len(anom_idx)} "
          f"(rows {anom_idx[:3]}..{anom_idx[-3:] if anom_idx else []})")
    print(f"predicted ticks: {len(pred_idx)}")
    print(f"threshold-present (warm) ticks: {len(warm_idx)}; "
          f"first warm at row {warm_idx[0] if warm_idx else None}")

    if not anom_idx:
        return
    lo = max(0, anom_idx[0] - 5)
    hi = min(n, anom_idx[-1] + 6)
    print(f"\ntrace rows {lo}..{hi} (around labeled anomalies):")
    print(f"{'row':>5} {'label':>5} {'pred':>4} {'value':>10} "
          f"{'prediction':>11} {'resid':>10} {'smoothed':>10} {'threshold':>10} {'ratio':>7}")
    for i in range(lo, hi):
        e = rows[i]
        v = e.get("value_normalized")
        p = e.get("prediction")
        res = e.get("residual")
        sm = e.get("smoothed_error")
        th = e.get("threshold")
        ratio = (sm / th) if (sm is not None and th not in (None, 0)) else None

        def f(x, w=10):
            return f"{x:>{w}.4f}" if isinstance(x, (int, float)) else f"{'—':>{w}}"

        print(f"{i:>5} {int(bool(e.get('is_anomaly'))):>5} "
              f"{int(bool(e.get('is_anomaly_predicted'))):>4} "
              f"{f(v)} {f(p,11)} {f(res)} {f(sm)} {f(th)} "
              f"{f(ratio,7) if ratio is not None else '      —'}")


if __name__ == "__main__":
    main()
