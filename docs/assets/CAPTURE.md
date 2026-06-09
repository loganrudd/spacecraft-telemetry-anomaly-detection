# README Visual Assets — Capture Checklist

This folder holds the screenshots and demo recording embedded in the top-level
[`README.md`](../../README.md). The embeds are already wired up — each one lights up the
moment a file with the matching name lands here.

> ⚠️ **Do not push / merge `phase11` until every file below exists.** Until then the
> README shows broken-image icons. Capture all assets first, drop them in this folder,
> then commit. Run `ls docs/assets/` and confirm every filename in the table is present.

All paths are relative to the repo root. Use the recommended widths so the README layout
stays consistent; PNGs should be trimmed to the relevant UI (no full-desktop screenshots).

| Filename | README placement | What it must show | How to capture | Width |
|---|---|---|---|---|
| `dashboard.gif` | Hero, top of README | The live dashboard mid-stream: telemetry charts scrolling with at least one **anomaly band** lighting up (red true / yellow predicted), plus the subsystem overview. ~8–15s loop. | Open the [live demo](https://api-pb5fb25noa-uc.a.run.app) (or `make api-serve` + `make frontend-dev`), pick a subsystem with a labeled anomaly, screen-record the window, export to GIF (e.g. Kap / Gifski / `ffmpeg`). Keep it under ~8 MB. | 900px |
| `mlflow-experiments.png` | Screenshots gallery | The MLflow experiments list showing the separate **training / scoring / HPO** experiments with run counts. | `make mlflow-ui` → Experiments view. Crop to the experiment list + a few runs. | 800px |
| `mlflow-registry.png` | Screenshots gallery | The MLflow **Model Registry** with `telemanom-*` registered models and the **`@champion`** alias visible on a version. | MLflow UI → Models. Crop to one model's versions showing the champion alias. | 800px |
| `ray-tune-sweep.png` | Screenshots gallery | A **Ray Tune** sweep for one subsystem — trials table or parallel-coordinates over the scoring params (error-smoothing window, threshold z, min anomaly length), ideally showing ASHA early-stopping. | Ray dashboard during/after `make ray-tune`, or the MLflow nested-run view of the sweep. | 800px |
| `evidently-drift-report.png` | Screenshots gallery | An **Evidently** batch drift report (the HTML deliverable) — the data-drift summary table / per-feature drift section. | Open the HTML report logged as an MLflow artifact by `drift batch-mission`, screenshot the drift summary. | 800px |
| `architecture.png` | _Optional_ | A rendered architecture diagram. **Usually skip** — the README already embeds a Mermaid diagram that renders natively on GitHub. Only add if you want a styled export. | Export the Mermaid diagram (e.g. mermaid.live) if desired. | 900px |

## Notes

- The live **drift panel** in the dashboard is disabled by default (`DRIFT_DISABLED = true`
  in `frontend/src/App.tsx`) for short demo windows, so the gallery uses the Evidently
  **batch** HTML report instead of a live-panel screenshot. If you do enable the panel and
  capture it, you can add `drift-panel.png` and embed it next to the GIF.
- Prefer light-mode UI for screenshots — it reads better against GitHub's default README
  background in both themes.
- If you rename a file, update the matching embed in `README.md`.
