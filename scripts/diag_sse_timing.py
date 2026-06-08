"""Measure SSE event inter-arrival timing from the deployed telemetry stream.

Goal: confirm whether Cloud Run's front end delivers events smoothly or in
periodic clumps (which the dashboard renders as a cyclic "skip").

Run this against the DEPLOYED API, not localhost — the whole point is to
measure the GFE/network delivery path. Because it runs from a plain HTTP
client with zero browser/rendering involved, any clumping seen here is purely
network-side, isolating it from the React/recharts layer.

Method: each telemetry channel emits exactly one event per replay tick, so on
a healthy stream every per-channel inter-arrival gap clusters near the tick
interval (1 / speed seconds; ~50 ms at the default 20x). Clumped delivery
collapses gaps into many ~0 ms (events dumped together in one chunk) plus
periodic large gaps between clumps — a bimodal distribution is the tell.

Usage:
    python scripts/diag_sse_timing.py [run_seconds] [channels_csv]

    run_seconds   how long to sample (default 30)
    channels_csv  comma-separated channels (default: all loaded — matches the
                  dashboard, which streams every served channel)
"""

from __future__ import annotations

import json
import os
import sys
import time

import httpx

# Override with SSE_API=http://127.0.0.1:8000 to A/B the same probe locally.
API = os.environ.get("SSE_API", "https://api-pb5fb25noa-uc.a.run.app")
RUN_SECONDS = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
CHANNELS = sys.argv[2] if len(sys.argv) > 2 else None

# A per-channel gap larger than this is well above the ~50 ms tick and flags a
# clump boundary (events held by the proxy, then released in a burst).
ABNORMAL_MS = 120.0


def _pct(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    i = min(len(sorted_vals) - 1, int(p / 100 * len(sorted_vals)))
    return sorted_vals[i]


def main() -> None:
    query = "" if not CHANNELS else f"?channels={CHANNELS}"
    url = f"{API}/api/stream/telemetry{query}"
    print(f"connecting {url}")
    print(f"running {RUN_SECONDS:.0f}s at the server default speed ...\n")

    last: dict[str, float] = {}      # channel -> last arrival (perf_counter)
    gaps_ms: list[float] = []        # all per-channel inter-arrival gaps
    big: list[tuple[float, str, float]] = []  # (t_since_start, channel, gap_ms)
    window: list[float] = []         # gaps in the current 2 s report window

    start = time.perf_counter()
    deadline = start + RUN_SECONDS
    next_report = start + 2.0

    with httpx.stream("GET", url, timeout=httpx.Timeout(30.0, read=None)) as r:
        r.raise_for_status()
        buf = ""
        for chunk in r.iter_text():
            now = time.perf_counter()
            if now >= deadline:
                break
            buf += chunk
            while "\n\n" in buf:
                frame, buf = buf.split("\n\n", 1)
                data = next(
                    (ln[5:].strip() for ln in frame.splitlines()
                     if ln.startswith("data:")),
                    None,
                )
                if not data:
                    continue
                ev = json.loads(data)
                ch = ev["channel"]
                arr = time.perf_counter()
                if ch in last:
                    gap = (arr - last[ch]) * 1000.0
                    gaps_ms.append(gap)
                    window.append(gap)
                    if gap > ABNORMAL_MS:
                        big.append((arr - start, ch, gap))
                last[ch] = arr

            if now >= next_report:
                w = sorted(window)
                window.clear()
                if w:
                    n_abn = sum(1 for x in w if x > ABNORMAL_MS)
                    print(
                        f"[t={now - start:4.0f}s] n={len(w):4d}  "
                        f"median={_pct(w, 50):6.1f}ms  p95={_pct(w, 95):6.1f}ms  "
                        f"max={w[-1]:6.1f}ms  abnormal(>{ABNORMAL_MS:.0f}ms)={n_abn}"
                    )
                next_report = now + 2.0

    s = sorted(gaps_ms)
    print(f"\n=== summary ({RUN_SECONDS:.0f}s) ===")
    print(f"channels: {len(last)}   gap samples: {len(s)}")
    if s:
        print(
            f"per-channel inter-arrival gap (ms): "
            f"median={_pct(s, 50):.1f}  p95={_pct(s, 95):.1f}  "
            f"p99={_pct(s, 99):.1f}  max={s[-1]:.1f}"
        )
    print(f"abnormal gaps (>{ABNORMAL_MS:.0f}ms): {len(big)}")
    if len(big) >= 2:
        times = [t for t, _, _ in big]
        cadence = (times[-1] - times[0]) / (len(times) - 1)
        sizes = [g for _, _, g in big]
        print(
            f"  abnormal-gap cadence: ~1 every {cadence:.2f}s   "
            f"(gap sizes {min(sizes):.0f}-{max(sizes):.0f}ms)"
        )
        print("  -> periodic clumping: consistent with GFE/proxy buffering")
    elif s and _pct(s, 95) < 1.5 * _pct(s, 50):
        print("  -> smooth at the network layer (no clumping seen here)")


if __name__ == "__main__":
    main()
