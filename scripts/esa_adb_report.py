"""Print the ESA-ADB-comparable report (Plan 019): our results next to the
paper's own Telemanom-ESA / Telemanom-ESA-Pruned numbers, on the same 6
channels (41-46), under the same metric (corrected event-wise F0.5).

Requires a baseline (untuned) AND a tuned scoring run to already exist in
MLflow for every requested channel — this script only reads existing scoring
artifacts, it never trains or scores. If a channel is missing either run,
esa_adb.detections.find_scoring_run raises with the exact tags it searched
for rather than silently narrowing the evaluated channel set.

Usage:
    # Default: ESA-Mission1, channels 41-46, local backend.
    python scripts/esa_adb_report.py --env local

    # Cloud backend, Stage B split-matched mission, write JSON alongside stdout:
    python scripts/esa_adb_report.py --env cloud --mission ESA-Mission1-ADB \\
        --out outputs/esa_adb_report.json

Requires: .[tracking] (mlflow) + rich. Reads labels/anomaly_types from
settings.data.sample_data_dir and scoring runs from settings.mlflow.tracking_uri;
configure_mlflow handles the Cloud Run GCP ID-token auth automatically.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Allow running as a script without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spacecraft_telemetry.core.config import load_settings
from spacecraft_telemetry.core.logging import get_logger
from spacecraft_telemetry.esa_adb.report import LIGHTWEIGHT_CHANNELS, build_report

log = get_logger(__name__)

_COLUMNS = (
    "label",
    "precision",
    "recall",
    "f0_5",
    "alarming_precision",
    "channel_aware_f0_5",
    "n_events",
    "n_detections",
    "split",
    "params",
)


def _fmt(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ESA-ADB-comparable evaluation report (Plan 019)."
    )
    parser.add_argument("--env", default="local", help="Config env (local, cloud, test).")
    parser.add_argument("--mission", default="ESA-Mission1", help="Mission name.")
    parser.add_argument(
        "--channels",
        default=None,
        help="Comma-separated channel IDs. Default: the ESA-ADB lightweight "
        f"subset ({', '.join(LIGHTWEIGHT_CHANNELS)}).",
    )
    parser.add_argument(
        "--out", default=None, metavar="JSON", help="Write the full report to this JSON file."
    )
    parser.add_argument(
        "--run-map",
        default=None,
        metavar="JSON",
        help="Offline run map (channel -> baseline/tuned run IDs + staged artifact "
        "paths). When given, the MLflow tracking server is never contacted — runs "
        "are read from the staged paths instead. See scripts/stage_adb_offline.sh.",
    )
    parser.add_argument(
        "--processed-dir",
        default=None,
        metavar="PATH",
        help="Override settings.preprocess.processed_data_dir (e.g. a locally "
        "staged copy of the processed test partitions).",
    )
    parser.add_argument(
        "--sample-dir",
        default=None,
        metavar="PATH",
        help="Override settings.data.sample_data_dir (holds labels.csv + "
        "anomaly_types.csv, the ground truth).",
    )
    args = parser.parse_args()

    from rich.console import Console
    from rich.table import Table

    from spacecraft_telemetry.esa_adb.offline import load_run_map
    from spacecraft_telemetry.mlflow_tracking import configure_mlflow

    settings = load_settings(args.env)
    if args.processed_dir:
        settings = settings.model_copy(
            update={
                "preprocess": settings.preprocess.model_copy(
                    update={"processed_data_dir": args.processed_dir}
                )
            }
        )
    if args.sample_dir:
        settings = settings.model_copy(
            update={"data": settings.data.model_copy(update={"sample_data_dir": args.sample_dir})}
        )

    run_map = load_run_map(args.run_map) if args.run_map else None
    if run_map is None:
        # configure_mlflow first: sets the tracking URI and installs the GCP ID
        # token for the Cloud Run backend before any search_runs call goes out.
        # Skipped entirely in offline mode — there is no server to reach.
        configure_mlflow(settings)

    channels = args.channels.split(",") if args.channels else None
    log.info(
        "esa_adb_report.start",
        env=args.env,
        mission=args.mission,
        channels=channels or LIGHTWEIGHT_CHANNELS,
        offline=run_map is not None,
    )
    report: dict[str, Any] = build_report(settings, args.mission, channels, run_map=run_map)

    con = Console()
    for scope in ("all_events", "anomalies_only"):
        table = Table(title=f"{report['mission']} — scope={scope}")
        for col in _COLUMNS:
            table.add_column(col)
        for row in report["rows"]:
            if row["scope"] != scope:
                continue
            table.add_row(*(_fmt(row[c]) for c in _COLUMNS))
        con.print(table)

    con.print("\n[bold]Footnotes[/bold]")
    for note in report["footnotes"]:
        con.print(f"  • {note}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, default=str))
        con.print(f"\nWrote full report → {out_path}")


if __name__ == "__main__":
    main()
