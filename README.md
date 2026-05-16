# Spacecraft Telemetry Anomaly Detection System

End-to-end MLOps platform for detecting anomalies in real spacecraft telemetry, built on
the [ESA Anomaly Dataset](https://zenodo.org/records/12528696) — 31 GB of sensor data
from 3 ESA missions, ~225 telemetry channels.

The model ([Telemanom LSTM, Hundman et al. 2018](https://arxiv.org/abs/1802.04431)) is
intentionally off-the-shelf: one small LSTM per channel trains on nominal data and flags
when sensor readings diverge from its predictions. The engineering emphasis is the platform
wrapping it — Spark preprocessing, Ray Core for parallelizing
hundreds of training jobs, Ray Tune for per-subsystem HPO, MLflow for experiment tracking
and model registry, Evidently for drift monitoring, FastAPI with SSE for real-time stream
replay, and Cloud Run for serving.

Built as a portfolio project targeting ML Platform Engineer / ML Infrastructure roles.

## Status

**Current phase: 8 of 11 complete (FastAPI serving layer).**

Completed:
- Phase 1: repo scaffold + ingestion
- Phase 2: PySpark preprocessing
- Phase 3: Telemanom model drop-in
- Phase 4: Ray parallel training + scoring
- Phase 5: Ray Tune scoring-parameter HPO
- Phase 6: MLflow experiment tracking + model registry
- Phase 7: Evidently batch drift detection + MLflow artifact logging
- Phase 8: FastAPI serving layer — SSE telemetry stream replay + `/health` endpoint

## What Works Today

- End-to-end local workflow on sampled ESA data
- Spark preprocessing to partitioned Parquet outputs
- Per-channel Telemanom training + scoring artifacts, tracked in MLflow
- Ray fan-out training and scoring across channels
- Ray Tune HPO over scoring parameters per subsystem
- MLflow experiment tracking (training, scoring, HPO experiments per mission)
- MLflow model registry with `telemanom-{mission}-{channel}` naming convention
- `mlflow promote` CLI for Staging → Production stage transitions
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
- Fast test, lint, and typecheck workflows

## Quick Start

Prerequisites:
- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- JDK 21 (required for PySpark locally)

```bash
# 1) Install dependencies
make setup

# 2) (Optional) set Java for Spark
export JAVA_HOME=$(brew --prefix openjdk@21)

# 3) Download sampled ESA data
make download-sample MISSION=ESA-Mission1

# 4) Preprocess data with Spark
make spark-preprocess MISSION=ESA-Mission1

# 5) Run test suite (fast + slow mix per project config)
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
make ray-score MISSION=ESA-Mission1 TUNED_CONFIGS=models/ESA-Mission1/tuned_configs.json

# 5) Inspect experiments and registered models
make mlflow-server                      # opens at http://localhost:5001

# 6) Promote a model to Production
make mlflow-promote MISSION=ESA-Mission1 CHANNEL=channel_1 STAGE=Production

# 7) Run drift monitoring for a single channel
#    Builds reference profile from train split, compares to test split,
#    logs HTML report + per-feature metrics to telemanom-monitoring-ESA-Mission1 experiment.
uv run spacecraft-telemetry drift batch --mission ESA-Mission1 --channel channel_1

# 8) Run drift monitoring for all discovered channels in a mission
uv run spacecraft-telemetry drift batch-mission --mission ESA-Mission1

# 9) Start the FastAPI serving layer (separate terminal)
#    Loads all scored models for the configured subsystem (default: subsystem_6)
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
    | grep -m1 '"is_anomaly_true": true'
```

**Temporal split rationale:** The test set is split at 60% / 40%. HPO trials search
over the first 60% (`hpo_portion`) to find optimal threshold parameters. Final
reported metrics use the last 40% (`final_portion`) — data the HPO search never saw.
This prevents the tuning process from inflating held-out F0.5 scores.

Expected key artifacts:
- `models/ESA-Mission1/tuned_configs.json` — per-subsystem scoring params + HPO lineage
- `mlflow.db` — local SQLite MLflow tracking store (experiments, runs, registered models)

## Repository Map

Top-level directories:
- `src/spacecraft_telemetry/`: application modules (ingest, spark, model, ray, api)
- `tests/`: unit and integration tests mirroring source structure
- `configs/`: environment YAML configs (`local`, `test`, `cloud`)
- `data/`: raw, sampled, and processed telemetry
- `models/`: per-mission/channel model and scoring artifacts
- `docs/`: plans, reviews, architecture notes

## Architecture Overview

```text
ESA Parquet (Zenodo/GCS)
  -> Download + Sample
  -> Spark preprocessing
  -> Telemanom LSTM (per-channel)
  -> Ray parallel train/score
  -> Ray Tune scoring HPO
  -> MLflow experiment tracking + model registry
  -> Evidently drift monitoring (batch, per-channel, HTML reports in MLflow)
  -> FastAPI + SSE serving  [Phase 8 - complete]
       GET /health
       GET /api/stream/telemetry  (SSE, real-time LSTM inference)
  -> React dashboard (next)
  -> GCP deployment
```

## Roadmap

| Phase | Description | Status |
|---|---|---|
| 1 | Repo scaffold + data ingestion | Complete |
| 2 | PySpark preprocessing pipeline | Complete |
| 3 | Telemanom model drop-in | Complete |
| 4 | Ray parallel training | Complete |
| 5 | Ray Tune HPO | Complete |
| 6 | MLflow integration | Complete |
| 7 | Evidently monitoring | Complete |
| 8 | FastAPI serving layer | Complete |
| 9 | React dashboard | In Progress |
| 10 | GCP deployment | Planned |
| 11 | Documentation + polish | Planned |

## Links

- [Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding](https://arxiv.org/abs/1802.04431)
  (Hundman et al., 2018): original Telemanom method this project uses as the model baseline.
- [ESA-ADB: The European Space Agency Anomaly Detection Benchmark](https://arxiv.org/abs/2406.17826)
  (Kotowski et al., 2024): benchmark paper describing the ESA-ADB evaluation context.
