# Spacecraft Telemetry Anomaly Detection System

End-to-end ML infrastructure for real-time spacecraft telemetry anomaly detection

**Status: Phase 1 of 12 — Repo Scaffold + Data Ingestion**

## Tech Stack

- Python 3.12, uv
- PySpark 4.1 (preprocessing)
- Ray 2.x Core + Tune (parallel training, HPO)
- PyTorch (Telemanom LSTM)
- MLflow 3.x (experiment tracking, model registry)
- Feast (feature store)
- Evidently (drift monitoring)
- FastAPI + SSE (serving)
- React + Recharts (dashboard)
- GCP: Cloud Run, Dataproc, GCS

## What Works Today

```bash
# Install dependencies
make setup

# Download ESA Anomaly Dataset sample (~1% of data)
make download-sample

# Explore dataset schema, quality, and channel distribution
make explore

# Run tests
make test

# Lint
make lint
```

## Architecture

```
ESA Parquet (Zenodo/GCS)
  → Download + Sample (Phase 1)  ← you are here
  → PySpark preprocessing        (Phase 2)
  → Feast feature store          (Phase 3)
  → Telemanom LSTM training      (Phase 4)
  → Ray parallel training        (Phase 5)
  → Ray Tune HPO                 (Phase 6)
  → MLflow tracking              (Phase 7)
  → Evidently monitoring         (Phase 8)
  → FastAPI + SSE serving        (Phase 9)
  → React dashboard              (Phase 10)
  → GCP deployment               (Phase 11)
  → Documentation + polish       (Phase 12)
```

## Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Repo scaffold + data ingestion | In Progress |
| 2 | PySpark preprocessing pipeline | Planned |
| 3 | Feast feature store integration | Planned |
| 4 | Telemanom model drop-in | Planned |
| 5 | Ray parallel training | Planned |
| 6 | Ray Tune HPO | Planned |
| 7 | MLflow integration | Planned |
| 8 | Evidently monitoring | Planned |
| 9 | FastAPI serving layer | Planned |
| 10 | React dashboard | Planned |
| 11 | GCP deployment | Planned |
| 12 | Documentation + polish | Planned |

## Dataset

[ESA Anomaly Dataset](https://zenodo.org/records/12528696) — real spacecraft telemetry from 3 ESA missions (~31GB, 176 channels).
