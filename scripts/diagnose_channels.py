"""Diagnose why scored channels have F0.5 = 0 by bucketing them against ground truth.

Joins the anomaly labels (ground truth, model-independent) with the per-channel
MLflow scoring metrics and sorts every deployed channel into one of:

  no_anomalies          channel never appears in labels.csv → nothing to detect,
                        ever. F0.5 = 0 is correct, not a failure. Drop from serving
                        expectations.
  outside_eval_window   has labeled anomalies, but none landed in the scored slice
                        (scoring uses eval_split="final_portion" = last 40%). F0.5 = 0
                        is structural — tuning cannot fix it. Re-score full_test to
                        judge the model.
  not_scored            has anomalies but no scoring run found (skipped / untrained).
  detected              anomalies in the eval window AND seg_f0_5 > 0 → working.
  missed                anomalies in the eval window but seg_f0_5 = 0 → the real
                        failures worth debugging (or dropping).

Usage:
    # All discovered channels, cloud backend:
    python scripts/diagnose_channels.py --env cloud --mission ESA-Mission1

    # Restrict to the channels you actually deployed (local or gs:// file):
    python scripts/diagnose_channels.py --env cloud --mission ESA-Mission1 \\
        --channels-from gs://my-project-processed-data/ESA-Mission1/channels.txt

    # Only consider tuned runs, and write the per-channel table to CSV:
    python scripts/diagnose_channels.py --env cloud --mission ESA-Mission1 \\
        --eval-split final_portion --out channel_diagnosis.csv

Requires: .[tracking,gcp] (mlflow, gcsfs). Reads labels from
settings.data.sample_data_dir and metrics from settings.mlflow.tracking_uri;
configure_mlflow handles the Cloud Run GCP ID-token auth automatically.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a script without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spacecraft_telemetry.core.config import Settings, load_settings
from spacecraft_telemetry.core.logging import get_logger
from spacecraft_telemetry.core.paths import to_upath
from spacecraft_telemetry.mlflow_tracking import configure_mlflow, experiment_name
from spacecraft_telemetry.preprocess.io import read_labels
from spacecraft_telemetry.ray_training import discover_channels

log = get_logger(__name__)

# Bucket order = severity for the operator: actionable failures last.
_BUCKETS = [
    "detected",
    "no_anomalies",
    "outside_eval_window",
    "not_scored",
    "missed",
]


def _read_channels_from_file(path: str) -> list[str]:
    """Read channel IDs from a local or GCS text file, one per line."""
    import fsspec

    with fsspec.open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]  # type: ignore[union-attr]


def _labels_per_channel(settings: Settings, mission: str) -> dict[str, int]:
    """Return {channel_id: n_anomaly_segments} from labels.csv; {} if absent.

    This is ground truth across the *whole* series — independent of any model
    or eval split. A channel missing from the result has zero labeled anomalies.
    """
    labels_path = to_upath(settings.data.sample_data_dir) / mission / "labels.csv"
    if not labels_path.exists():
        # Fail loudly — a missing labels.csv silently buckets EVERY channel as
        # "no_anomalies", which looks like a catastrophic result but is really a
        # misconfiguration. The usual cause: running --env cloud locally without
        # pointing sample_data_dir at GCS (it defaults to a local path).
        raise SystemExit(
            f"labels.csv not found at {labels_path}.\n"
            "If reading cloud data locally, export the GCS location first, e.g.:\n"
            "  SPACECRAFT_DATA__SAMPLE_DATA_DIR=gs://<PROJECT_ID>-sample-data"
        )
    df = read_labels(labels_path)
    counts = df.groupby("channel_id").size()
    return {str(ch): int(n) for ch, n in counts.items()}


def _scoring_metrics(mission: str, eval_split: str | None) -> dict[str, dict[str, float]]:
    """Return {channel_id: {n_true_seqs, seg_f0_5, f0_5, n_true_positive_labels}}.

    Pulls the scoring experiment's runs and keeps the most recent run per channel
    (so tuned re-scores supersede the baseline). Optionally filters to one
    eval_split tag. Returns {} when the experiment has no runs yet.
    """
    import mlflow
    import pandas as pd

    exp = experiment_name("telemanom", "scoring", mission)
    try:
        runs = mlflow.search_runs(
            experiment_names=[exp],
            order_by=["attributes.start_time DESC"],  # newest first → first seen wins
        )
    except Exception as exc:  # surface the cause, don't crash the report
        log.warning("diagnose.search_runs_failed", experiment=exp, error=str(exc))
        return {}

    if isinstance(runs, list):
        rows = [
            {
                "tags.channel_id": run.data.tags.get("channel_id"),
                "tags.mlflow.runName": run.data.tags.get("mlflow.runName"),
                "tags.eval_split": run.data.tags.get("eval_split"),
                "metrics.n_true_seqs": run.data.metrics.get("n_true_seqs", float("nan")),
                "metrics.seg_f0_5": run.data.metrics.get("seg_f0_5", float("nan")),
                "metrics.f0_5": run.data.metrics.get("f0_5", float("nan")),
                "metrics.n_true_positive_labels": run.data.metrics.get(
                    "n_true_positive_labels", float("nan")
                ),
            }
            for run in runs
        ]
        runs_df = pd.DataFrame(rows)
    else:
        runs_df = runs

    if runs_df is None or runs_df.empty:
        log.warning("diagnose.no_scoring_runs", experiment=exp)
        return {}

    if eval_split is not None and "tags.eval_split" in runs_df.columns:
        runs_df = runs_df[runs_df["tags.eval_split"] == eval_split]

    def _isna(v: object) -> bool:
        # A missing tag comes back as float NaN, which is truthy — so a plain
        # `a or b` fallback never fires and yields the string 'nan'. Test explicitly.
        return v is None or (isinstance(v, float) and v != v)

    out: dict[str, dict[str, float]] = {}
    for _, row in runs_df.iterrows():
        ch = row.get("tags.channel_id")
        if _isna(ch):
            ch = row.get("tags.mlflow.runName")
        if _isna(ch) or str(ch) in out:  # newest-first → keep the first (latest) seen
            continue
        out[str(ch)] = {
            "n_true_seqs": row.get("metrics.n_true_seqs", float("nan")),
            "seg_f0_5": row.get("metrics.seg_f0_5", float("nan")),
            "f0_5": row.get("metrics.f0_5", float("nan")),
            "n_true_positive_labels": row.get("metrics.n_true_positive_labels", float("nan")),
        }
    return out


def _bucket(label_count: int, metrics: dict[str, float] | None) -> str:
    """Classify one channel from its label count and (optional) scoring metrics."""
    if label_count == 0:
        return "no_anomalies"
    if metrics is None:
        return "not_scored"
    n_true = metrics.get("n_true_seqs")
    if n_true is None or n_true != n_true:  # NaN → metric absent on the run
        return "not_scored"
    if n_true == 0:
        return "outside_eval_window"
    seg = metrics.get("seg_f0_5")
    return "detected" if (seg is not None and seg == seg and seg > 0) else "missed"


def main() -> None:
    parser = argparse.ArgumentParser(description="Bucket channels by anomaly detectability.")
    parser.add_argument("--env", default="local", help="Config env (local, cloud, test).")
    parser.add_argument("--mission", required=True, help="Mission name, e.g. ESA-Mission1.")
    parser.add_argument(
        "--channels-from",
        default=None,
        metavar="PATH",
        help="Text file (local or gs://) of channel IDs to diagnose. "
        "Defaults to discover_channels() over the processed-data dir.",
    )
    parser.add_argument(
        "--eval-split",
        default=None,
        help="Only consider scoring runs with this eval_split tag "
        "(e.g. final_portion). Default: latest run per channel regardless.",
    )
    parser.add_argument(
        "--out", default=None, metavar="CSV", help="Write the per-channel table to this CSV."
    )
    args = parser.parse_args()

    import mlflow
    import pandas as pd
    from rich.console import Console
    from rich.table import Table

    settings = load_settings(args.env)
    # configure_mlflow first: sets the tracking URI and installs the GCP ID token
    # for the Cloud Run backend before any search_runs call goes out.
    configure_mlflow(settings)

    # Echo the resolved sources — both the labels dir and the tracking URI default
    # to LOCAL paths under --env cloud unless overridden, which silently produces
    # bogus buckets (all no_anomalies, or all not_scored). Make them visible, and
    # fail loudly if a cloud run is pointed at a local MLflow store.
    tracking_uri = mlflow.get_tracking_uri()
    log.info(
        "diagnose.sources",
        env=args.env,
        tracking_uri=tracking_uri,
        labels_dir=str(settings.data.sample_data_dir),
    )
    if args.env != "local" and tracking_uri.startswith(("sqlite:", "file:")):
        raise SystemExit(
            f"--env {args.env} but MLflow tracking_uri resolved to a LOCAL store "
            f"({tracking_uri}); cloud scoring runs won't be found (you'd get all "
            "'not_scored'). Point it at the Cloud Run MLflow, e.g.:\n"
            "  SPACECRAFT_MLFLOW__TRACKING_URI=$(gcloud run services describe mlflow "
            "--region $REGION --project $PROJECT_ID --format='value(status.url)')"
        )

    if args.channels_from is not None:
        channels = _read_channels_from_file(args.channels_from)
    else:
        channels = discover_channels(settings, args.mission)
    if not channels:
        raise SystemExit(f"No channels found for {args.mission} (pass --channels-from?).")

    labels = _labels_per_channel(settings, args.mission)
    metrics = _scoring_metrics(args.mission, args.eval_split)
    log.info(
        "diagnose.loaded",
        mission=args.mission,
        n_channels=len(channels),
        n_labeled_channels=len(labels),
        n_scored_channels=len(metrics),
    )

    rows: list[dict[str, object]] = []
    for ch in sorted(channels):
        m = metrics.get(ch)
        bucket = _bucket(labels.get(ch, 0), m)
        rows.append({
            "channel": ch,
            "labels_total": labels.get(ch, 0),
            "n_true_seqs": (m or {}).get("n_true_seqs"),
            "seg_f0_5": (m or {}).get("seg_f0_5"),
            "bucket": bucket,
        })
    df = pd.DataFrame(rows)

    con = Console()

    # Summary: counts per bucket.
    summary = Table(title=f"Channel diagnosis: {args.mission} ({len(channels)} channels)")
    summary.add_column("bucket")
    summary.add_column("count", justify="right")
    summary.add_column("meaning")
    meanings = {
        "detected": "anomalies in eval window, seg_f0_5 > 0 — working",
        "no_anomalies": "no labels anywhere — nothing to detect (drop)",
        "outside_eval_window": "labeled, but none in scored 40% — tuning can't help",
        "not_scored": "labeled, but no scoring run found",
        "missed": "anomalies in eval window, seg_f0_5 = 0 — DEBUG these",
    }
    counts = df["bucket"].value_counts().to_dict()
    for b in _BUCKETS:
        summary.add_row(b, str(counts.get(b, 0)), meanings[b])
    con.print(summary)

    # Detail: the two actionable buckets in full; others collapsed to channel lists.
    detail = Table(title="Actionable channels")
    for c in ("channel", "labels_total", "n_true_seqs", "seg_f0_5", "bucket"):
        detail.add_column(c)
    for _, r in df[df["bucket"].isin(["missed", "not_scored"])].sort_values("channel").iterrows():
        detail.add_row(
            str(r["channel"]),
            str(r["labels_total"]),
            "" if pd.isna(r["n_true_seqs"]) else f"{r['n_true_seqs']:.0f}",
            "" if pd.isna(r["seg_f0_5"]) else f"{r['seg_f0_5']:.3f}",
            str(r["bucket"]),
        )
    if detail.row_count:
        con.print(detail)
    else:
        con.print("[green]No 'missed' or 'not_scored' channels — every F0.5=0 is benign.[/green]")

    if args.out:
        df.to_csv(args.out, index=False)
        con.print(f"Wrote per-channel table → {args.out}")


if __name__ == "__main__":
    main()
