# Spacecraft Telemetry Anomaly Detection System

End-to-end MLOps platform for detecting anomalies in real spacecraft telemetry, built on
the [ESA Anomaly Dataset](https://zenodo.org/records/12528696) — 31 GB of sensor data
from 3 ESA missions, ~225 telemetry channels.

The model ([Telemanom LSTM, Hundman et al. 2018](https://arxiv.org/abs/1802.04431)) is
intentionally off-the-shelf: one small LSTM per channel trains on nominal data and flags
when sensor readings diverge from its predictions. The engineering emphasis is the platform
wrapping it — pandas + PyArrow preprocessing with Ray Core fan-out across hundreds of
channels, Ray Tune for per-subsystem HPO, MLflow for experiment tracking and model registry,
Evidently for drift monitoring, FastAPI with SSE for real-time stream replay, and Cloud Run
for serving.

Built as a portfolio project targeting ML Platform Engineer / ML Infrastructure roles.

**Live demo:** https://api-pb5fb25noa-uc.a.run.app

**Deployment guide:** [docs/deployment.md](docs/deployment.md)

## Status

**Phase 10 complete — GCP deployment, ESA-Mission1, live demo.**

Completed:
- Phase 1: repo scaffold + ingestion
- Phase 2: Pandas + Ray Core preprocessing
- Phase 3: Telemanom model drop-in
- Phase 4: Ray parallel training + scoring
- Phase 5: Ray Tune scoring-parameter HPO
- Phase 6: MLflow experiment tracking + model registry
- Phase 7: Evidently batch drift detection + MLflow artifact logging
- Phase 8: FastAPI serving layer — SSE telemetry stream replay + `/health` endpoint
- Phase 9: React dashboard — live telemetry charts + anomaly & real-time drift alerts
- Phase 10: GCP deployment — Cloud Run serving, GKE/Ray training, Terraform IaC, keyless CI/CD

## What Works Today

- End-to-end local workflow on sampled ESA data
- Preprocessing to partitioned Parquet outputs (pandas + Ray Core fan-out per channel)
- Per-channel Telemanom training + scoring artifacts, tracked in MLflow
- Ray fan-out training and scoring across channels
- Ray Tune HPO over scoring parameters per subsystem
- MLflow experiment tracking (training, scoring, HPO experiments per mission)
- MLflow model registry with `telemanom-{mission}-{channel}` naming convention
- `mlflow promote` CLI sets the `@champion` alias on a model version (MLflow 3.x)
- Evidently batch drift detection (14 features: `value_normalized` + rolling stats)
- Drift reports (HTML) logged as MLflow artifacts; per-feature metrics logged per run
- `drift batch` CLI: build reference profile → run report → log to MLflow
- `drift batch-mission` CLI: run all channels for a mission, print summary table
- **FastAPI serving layer** (`make serve`):
  - `GET /health` — model version, uptime, loaded channels, MLflow URI
  - `GET /api/stream/telemetry` — SSE stream replaying preprocessed Parquet with
    real-time LSTM inference; per-tick anomaly scores and predicted/labeled anomaly flags
  - `?speed=N` replay speed multiplier; `?channels=ch1,ch2` channel filter
  - Structured logging with correlation IDs on every request
- **React dashboard** (`make frontend-dev`):
  - Vite + TypeScript + React 18, served at `http://localhost:5173`
  - Live per-channel time-series charts (Recharts) — value + prediction overlay
  - Ground-truth anomaly bands (red) and model-predicted bands (yellow) on each chart
  - Rising-edge anomaly alert panel with TP/FP indicator
  - Channel picker with performance warning above 5 simultaneous charts
  - Mission-control dark theme; single SSE connection multiplexed across channels
  - Real-time drift panel: per-channel Evidently KS drift scores + subsystem gauge
    (automatically hidden when no reference profiles are available)
- Fast test, lint, and typecheck workflows

## Quick Start

Prerequisites:
- Python 3.12
- [uv](https://docs.astral.sh/uv/)

```bash
# 1) Install dependencies
make setup

# 2) Download and sample an entire ESA mission's data
make download-sample MISSION=ESA-Mission1

# you can also specify a specific subsystem and/or channel you want to sample/preprocess/train/score/tune
make download-sample MISSION=ESA-Mission1 SUBSYSTEM=subsystem_6

# 3) Preprocess data (pandas + Ray Core)
make preprocess MISSION=ESA-Mission1 CHANNEL=channel_22

# 4) Run test suite (fast + slow mix per project config)
make test
```

## Demo Workflow

Full Phase 6 lifecycle (train → baseline score → HPO tune → tuned score → promote):

```bash
# 1) Train all discovered channels (logged to MLflow training experiment)
make ray-train MISSION=ESA-Mission1

# 2) Score with Hundman defaults on full test set (baseline)
make ray-score MISSION=ESA-Mission1

# 3) Tune scoring parameters per subsystem via Ray Tune
#    HPO evaluates on the first 60% of each channel's test windows.
make ray-tune MISSION=ESA-Mission1

# 4) Re-score using tuned params, evaluated on the held-out final 40%
#    (avoids leakage between HPO search and reported metrics)
#    TUNED=1 auto-derives models/ESA-Mission1/tuned_configs.json (mirrors cloud-score).
#    Use TUNED_CONFIGS=<path> instead to point at a non-canonical configs file.
make ray-score MISSION=ESA-Mission1 TUNED=1

# 5) Inspect experiments and registered models
make mlflow-server                      # opens at http://localhost:5001

# 6) Promote a model — sets the @champion alias, required before serving
make mlflow-promote MISSION=ESA-Mission1 CHANNEL=channel_22          # local MLflow
make mlflow-promote MISSION=ESA-Mission1 CHANNEL=channel_22 ENV=cloud # cloud MLflow
# Promote all channels in a mission (or a subsystem):
make mlflow-promote MISSION=ESA-Mission1 ENV=cloud
make mlflow-promote MISSION=ESA-Mission1 SUBSYSTEM=subsystem_6 ENV=cloud

# 7) Run drift monitoring for a single channel
#    Builds reference profile from train split, compares to test split,
#    logs HTML report + per-feature metrics to telemanom-monitoring-ESA-Mission1 experiment.
uv run spacecraft-telemetry drift batch --mission ESA-Mission1 --channel channel_22

# 8) Run drift monitoring for all discovered channels in a mission
uv run spacecraft-telemetry drift batch-mission --mission ESA-Mission1

# 9) Start the FastAPI serving layer (separate terminal)
#    Only loads channels with a @champion alias — promote before starting.
make serve
# or with overrides:
uv run spacecraft-telemetry --env local api serve --subsystem subsystem_6 --reload

# Health check
curl -s http://127.0.0.1:8000/health | jq
# → 200, {"status": "ok", "channels_loaded": [...], "uptime_seconds": ..., ...}

# SSE stream — observe events with channel values, anomaly scores, and flags
curl -sN "http://127.0.0.1:8000/api/stream/telemetry?speed=50" 2>/dev/null | head -60

# Filter to a single channel
curl -sN "http://127.0.0.1:8000/api/stream/telemetry?speed=100&channels=channel_1" 2>/dev/null | head -30

# Look for a labeled anomaly event
curl -N "http://127.0.0.1:8000/api/stream/telemetry?speed=200" \
    | grep -m1 '"is_anomaly":true'

# 10) Drift stream — per-channel KS drift scores (requires reference profiles built in step 8)
curl -N 'http://127.0.0.1:8000/api/stream/drift?channels=channel_1'

# 12) Run the React dashboard (second separate terminal)
#     Requires: make serve running in another terminal
#     First time only:
make frontend-install
#     Start dev server:
make frontend-dev
# Open http://localhost:5173 in a browser.
# Select a channel from the left panel — live telemetry charts, anomaly alerts,
# and the real-time drift panel appear on the right.
```

**Temporal split rationale:** The test set is split at 60% / 40%. HPO trials search
over the first 60% (`hpo_portion`) to find optimal threshold parameters. Final
reported metrics use the last 40% (`final_portion`) — data the HPO search never saw.
This prevents the tuning process from inflating held-out F0.5 scores.

Expected key artifacts:
- `models/ESA-Mission1/tuned_configs.json` — per-subsystem scoring params + HPO lineage
- `mlflow.db` — local SQLite MLflow tracking store (experiments, runs, registered models)

Cloud lifecycle on GKE mirrors local exactly — baseline score first, then tune, then tuned score:

```bash
# 1) Train
make cloud-train MISSION=ESA-Mission1

# 2) Baseline score — Hundman defaults, full test set (no TUNED flag)
make cloud-score MISSION=ESA-Mission1

# 3) Tune — writes tuned_configs.json to GCS
make cloud-tune MISSION=ESA-Mission1

# 4) Tuned score — HPO params, held-out final 40% only
make cloud-score MISSION=ESA-Mission1 TUNED=1
```

`TUNED=1` fetches `tuned_configs.json` from GCS and passes it to the scorer; omitting it
(or `TUNED=0`) always runs a clean Hundman-defaults baseline regardless of what's in the bucket.

## Repository Map

Top-level directories:
- `src/spacecraft_telemetry/`: application modules (ingest, preprocess, model, ray, api)
- `frontend/`: React dashboard (Vite + TypeScript, `npm run dev` / `make frontend-dev`)
- `tests/`: unit and integration tests mirroring source structure
- `configs/`: environment YAML configs (`local`, `test`, `cloud`)
- `data/`: raw, sampled, and processed telemetry
- `models/`: per-mission/channel model and scoring artifacts
- `docs/`: plans, reviews, architecture notes

## Architecture Overview

```text
ESA Parquet (Zenodo/GCS)
  -> Download + Sample
  -> Pandas + Ray Core preprocessing (per-channel fan-out)
  -> Telemanom LSTM (per-channel)
  -> Ray parallel train/score
  -> Ray Tune scoring HPO
  -> MLflow experiment tracking + model registry
  -> Evidently drift monitoring (batch, per-channel, HTML reports in MLflow)
  -> FastAPI + SSE serving  [Phase 8 - complete]
       GET /health
       GET /api/stream/telemetry  (SSE, real-time LSTM inference)
       GET /api/stream/drift      (SSE, rolling Evidently KS drift per channel)
  -> React dashboard (Vite/TS, Recharts)  [Phase 9 - complete]
       Live telemetry charts + anomaly bands
       Rising-edge anomaly alert panel
  -> Real-time drift panel  [Phase 9.5 - complete]
       Per-channel KS drift scores (value_normalized feature)
       Subsystem-level gauge: % channels drifting, alert at ≥30%
  -> GCP deployment  [Phase 10 - complete]
       Cloud Run (API + MLflow), GKE Autopilot (Ray training), GCS
       Terraform IaC, GitHub Actions CI/CD, Workload Identity Federation
       Billing kill-switch, cost guardrails
```

## Deployment

Full instructions in [docs/deployment.md](docs/deployment.md). Summary:

```bash
# Provision infrastructure
cd infra && terraform init && terraform apply

# Run training pipeline (after cloud-up)
make cloud-preprocess MISSION=ESA-Mission1
make cloud-train      MISSION=ESA-Mission1
make cloud-score      MISSION=ESA-Mission1          # baseline
make cloud-tune       MISSION=ESA-Mission1
make cloud-score      MISSION=ESA-Mission1 TUNED=1  # tuned, held-out 40%

# Seed drift reference profiles + promote + serve
make seed-reference-profiles MISSION=ESA-Mission1
make mlflow-promote   MISSION=ESA-Mission1 ENV=cloud
make cloud-deploy
```

Architecture decisions: [docs/architecture/phase-10-gcp.md](docs/architecture/phase-10-gcp.md)

## Known Limitations

**Channel toggle restarts replay from row 0.**  
The SSE endpoint is stateless: each request opens a new replay from the
beginning. Adding or removing a channel tears down the current stream and
reopens it with the updated channel list, resetting all charts. This is
intentional for the demo — it keeps the server simple — but means you cannot
add a channel mid-session without losing the existing view. A future design
could key the stream by a replay cursor and support live channel subscription
changes without restart.

**Replay speed does not persist across channel changes.**  
The speed selector is wired to the same `useEffect` that controls the stream.
Changing channels restarts at the currently selected speed, which is the
expected behaviour.

---

## Roadmap

| Phase | Description | Status |
|---|---|---|
| 1 | Repo scaffold + data ingestion | Complete |
| 2 | Preprocessing pipeline | Complete |
| 3 | Telemanom model drop-in | Complete |
| 4 | Ray parallel training | Complete |
| 5 | Ray Tune HPO | Complete |
| 6 | MLflow integration | Complete |
| 7 | Evidently monitoring | Complete |
| 8 | FastAPI serving layer | Complete |
| 9 | React dashboard | Complete |
| 10 | GCP deployment | Complete |
| 11 | Documentation + polish | In Progress |

## Links

- [Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding](https://arxiv.org/abs/1802.04431)
  (Hundman et al., 2018): original Telemanom method this project uses as the model baseline.
- [ESA-ADB: The European Space Agency Anomaly Detection Benchmark](https://arxiv.org/abs/2406.17826)
  (Kotowski et al., 2024): benchmark paper describing the ESA-ADB evaluation context.
