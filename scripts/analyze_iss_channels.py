"""Rank ISS raw-tick channels by suitability for forecasting / the demo.

Reads the raw tick shards written by the Phase 12 collector (local or gs://)
and reports, per telemetry channel:

  - coverage: rows, time span, median tick cadence, largest gap
  - dynamics: raw value range, std
  - flatness: flat_frac — fraction of 30 s-grid steps with no change (the
    profiler's frac_zero_diff signal; note it is inflated by forward-fill
    during slow-cadence / LOS stretches, so it is reported but not the verdict)
  - orbital structure: autocorrelation of the 30 s-gridded series at the
    ~92 min orbital lag — high positive value = clean, learnable orbital cycle

Then prints a ranked verdict (STRONG / OK / WEAK / DEAD) to guide channel
selection, keyed on orbital_ac (a parked sensor cannot sustain orbital-lag
autocorrelation, so this is the cleanest single suitability signal).

Usage:
    SSL_CERT_FILE=$(uv run python -m certifi) \
      uv run python scripts/analyze_iss_channels.py \
      --base gs://spacecraft-telemetry-ads-raw-data
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from upath import UPath

from spacecraft_telemetry.ingest.iss_channels import CONTEXT_ITEMS, ISS_CHANNELS

ORBIT_SECONDS = 92 * 60          # ISS orbital period ≈ 92 min
GRID_SECONDS = 30               # candidate resample grid (Phase 13 contract)
ORBIT_LAG = ORBIT_SECONDS // GRID_SECONDS  # autocorr lag in grid buckets


def _read_channel(base: UPath, channel: str) -> pd.DataFrame:
    """Read all shards for one channel into a sorted DataFrame."""
    chan_dir = base / "ISS" / "ticks" / f"channel_id={channel}"
    tables = []
    for shard in sorted(chan_dir.glob("*.parquet")):
        with shard.open("rb") as fh:
            tables.append(pq.read_table(fh, columns=["telemetry_timestamp", "value"]))
    if not tables:
        return pd.DataFrame(columns=["telemetry_timestamp", "value"])
    df = pa.concat_tables(tables).to_pandas()
    df = df.sort_values("telemetry_timestamp").reset_index(drop=True)
    df["telemetry_timestamp"] = pd.to_datetime(df["telemetry_timestamp"], utc=True)
    return df


def _gridded(df: pd.DataFrame) -> pd.Series:
    """30 s-gridded, forward-filled series (the Phase 13 resample contract)."""
    s = df.set_index("telemetry_timestamp")["value"]
    return s.resample(f"{GRID_SECONDS}s").mean().ffill()


def _analyze(df: pd.DataFrame) -> dict[str, float]:
    ts = df["telemetry_timestamp"]
    v = df["value"].astype("float64")
    dt = ts.diff().dt.total_seconds().dropna()
    grid = _gridded(df)

    # Orbital structure: autocorrelation at the ~92 min lag.
    if len(grid) <= ORBIT_LAG + 2 or grid.std() == 0:
        orbital_ac = float("nan")
    else:
        orbital_ac = float(grid.autocorr(lag=ORBIT_LAG))

    # Flatness: fraction of 30 s-grid steps with no change. This is exactly the
    # profiler's frac_zero_diff signal — offset/unit-invariant, always in [0, 1],
    # and well-defined even for quantized sensors (where a diff-based SNR breaks).
    # A parked sensor ffills to long constant runs → ~1.0; a dynamic channel → ~0.
    grid_diff = grid.diff().dropna()
    flat_frac = float((grid_diff == 0).mean()) if len(grid_diff) else float("nan")

    return {
        "rows": len(df),
        "span_h": (ts.iloc[-1] - ts.iloc[0]).total_seconds() / 3600,
        "median_dt": float(dt.median()) if len(dt) else float("nan"),
        "max_gap": float(dt.max()) if len(dt) else float("nan"),
        "vmin": float(v.min()),
        "vmax": float(v.max()),
        "std": float(v.std(ddof=1)),
        "range": float(v.max() - v.min()),
        "flat_frac": flat_frac,
        "orbital_ac": orbital_ac,
    }


def _verdict(m: dict[str, float]) -> str:
    """Heuristic label for demo/forecasting suitability.

    orbital_ac is the primary axis: a genuinely flat/parked sensor cannot
    sustain high autocorrelation at the orbital lag, while flat_frac is
    confounded by forward-fill during slow-cadence / LOS stretches.
    """
    if m["rows"] < 50:
        return "DEAD"          # never published live data (e.g. SA current PUIs)
    ac = m["orbital_ac"]
    if np.isnan(ac) or ac < 0.55:
        return "WEAK"          # no real orbital cycle (CMG, TRRJ, q3)
    if ac > 0.80:
        return "STRONG"        # clean orbital cycle a forecaster can learn
    return "OK"                # real but partial structure (power V, slow channels)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Raw-data root (gs:// or local path)")
    args = ap.parse_args()

    base = UPath(args.base)
    rows: list[dict[str, Any]] = []
    total = len(ISS_CHANNELS)
    for i, (channel, meta) in enumerate(ISS_CHANNELS.items(), start=1):
        t0 = time.monotonic()
        try:
            df = _read_channel(base, channel)
        except Exception as exc:
            print(f"  !! {channel}: read failed ({exc})")
            continue
        # Progress to stderr so it streams even while stdout is buffered.
        print(
            f"[{i}/{total}] {channel}: {len(df)} rows in {time.monotonic()-t0:.1f}s",
            file=sys.stderr,
            flush=True,
        )
        if df.empty:
            print(f"  !! {channel}: no data")
            continue
        m = _analyze(df)
        row: dict[str, Any] = {
            **m,
            "channel": channel,
            "subsystem": meta.subsystem,
            "verdict": _verdict(m),
        }
        rows.append(row)

    if not rows:
        print("\nNo channels could be read — check --base path and credentials.")
        return

    res = pd.DataFrame(rows)
    order = {"STRONG": 0, "OK": 1, "WEAK": 2, "DEAD": 3}
    res = res.sort_values(
        by=["verdict", "orbital_ac"],
        key=lambda c: c.map(order) if c.name == "verdict" else c,
        ascending=[True, False],
    )

    hdr = (
        f"{'channel':12s} {'subsys':11s} {'verdict':7s} {'rows':>7s} {'med_dt':>6s} "
        f"{'flat_frac':>9s} {'orbit_ac':>8s} {'range':>16s}"
    )
    print("\n" + hdr)
    print("-" * len(hdr))
    for _, r in res.iterrows():
        print(
            f"{r['channel']:12s} {r['subsystem']:11s} {r['verdict']:7s} "
            f"{int(r['rows']):7d} {r['median_dt']:6.1f} {r['flat_frac']:9.3f} "
            f"{r['orbital_ac']:8.3f} [{r['vmin']:.4g}, {r['vmax']:.4g}]"
        )

    print("\nVerdict counts:", res["verdict"].value_counts().to_dict())
    print("By subsystem:")
    for sub, g in res.groupby("subsystem"):
        print(f"  {sub:11s}: {g['verdict'].value_counts().to_dict()}")
    print(f"\n(context items not scored: {CONTEXT_ITEMS})")


if __name__ == "__main__":
    main()
