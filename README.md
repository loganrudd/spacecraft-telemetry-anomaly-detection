# Spacecraft Telemetry Anomaly Detection System

Production-style MLOps pipeline for spacecraft telemetry anomaly detection using the
[ESA Anomaly Dataset](https://zenodo.org/records/12528696).

The project emphasizes platform engineering over novel model research: reliable data
pipelines, feature store integration, parallel training, tuning, monitoring, and serving.

## Status

**Current phase: 6 of 12 complete (Ray Tune HPO).**

Completed:
- Phase 1: repo scaffold + ingestion
- Phase 2: PySpark preprocessing
- Phase 3: Feast feature store integration
- Phase 4: Telemanom model drop-in
- Phase 5: Ray parallel training + scoring
- Phase 6: Ray Tune scoring-parameter HPO

## What Works Today

- End-to-end local workflow on sampled ESA data
- Spark preprocessing to partitioned Parquet outputs
- Feast apply/materialize and historical/online retrieval helpers
- Per-channel Telemanom training + scoring artifacts
- Ray fan-out training and scoring across channels
- Ray Tune HPO over scoring parameters per subsystem
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

Minimal Ray workflow (train -> tune -> score):

```bash
# Train all discovered channels
make ray-train MISSION=ESA-Mission1

# Tune scoring params per subsystem
make ray-tune MISSION=ESA-Mission1

# Score channels using tuned params
uv run spacecraft-telemetry ray score \
  --mission ESA-Mission1 \
  --tuned-configs models/ESA-Mission1/tuned_configs.json
```

Expected key artifact after tuning:
- `models/ESA-Mission1/tuned_configs.json`

## Repository Map

Top-level directories:
- `src/spacecraft_telemetry/`: application modules (ingest, spark, feast, model, ray, api)
- `tests/`: unit and integration tests mirroring source structure
- `configs/`: environment YAML configs (`local`, `test`, `cloud`)
- `data/`: raw, sampled, and processed telemetry
- `models/`: per-mission/channel model and scoring artifacts
- `feature_repo/`: Feast repository and local store config
- `docs/`: plans, reviews, architecture notes

## Architecture Overview

```text
ESA Parquet (Zenodo/GCS)
  -> Download + Sample
  -> Spark preprocessing
  -> Feast feature store
  -> Telemanom LSTM (per-channel)
  -> Ray parallel train/score
  -> Ray Tune scoring HPO
  -> MLflow tracking (next)
  -> Evidently monitoring (next)
  -> FastAPI + SSE serving (next)
  -> React dashboard (next)
  -> GCP deployment
```

## Roadmap

| Phase | Description | Status |
|---|---|---|
| 1 | Repo scaffold + data ingestion | Complete |
| 2 | PySpark preprocessing pipeline | Complete |
| 3 | Feast feature store integration | Complete |
| 4 | Telemanom model drop-in | Complete |
| 5 | Ray parallel training | Complete |
| 6 | Ray Tune HPO | Complete |
| 7 | MLflow integration | Planned |
| 8 | Evidently monitoring | Planned |
| 9 | FastAPI serving layer | Planned |
| 10 | React dashboard | Planned |
| 11 | GCP deployment | Planned |
| 12 | Documentation + polish | Planned |

## Links

- [Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding](https://arxiv.org/abs/1802.04431)
  (Hundman et al., 2018): original Telemanom method this project uses as the model baseline.
- [ESA-ADB: The European Space Agency Anomaly Detection Benchmark](https://arxiv.org/abs/2406.17826)
  (Kotowski et al., 2024): benchmark paper describing the ESA-ADB evaluation context.
