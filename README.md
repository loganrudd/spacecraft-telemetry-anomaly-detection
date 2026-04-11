# Spacecraft Telemetry Anomaly Detection System

End-to-end ML infrastructure for real-time spacecraft telemetry anomaly detection using the [ESA Anomaly Dataset](https://zenodo.org/records/12528696).

**Status: Phase 1 of 12 complete — Repo Scaffold + Data Ingestion**

## Tech Stack

- Python 3.12, uv
- PySpark 4.1 (preprocessing — Phase 2)
- Ray 2.x Core + Tune (parallel training, HPO — Phases 5–6)
- PyTorch (Telemanom LSTM — Phase 4)
- MLflow 3.x (experiment tracking, model registry — Phase 7)
- Feast (feature store — Phase 3)
- Evidently (drift monitoring — Phase 8)
- FastAPI + SSE (serving — Phase 9)
- React + Recharts (dashboard — Phase 10)
- GCP: Cloud Run, Dataproc, GCS (Phase 11)

## What Works Today

```bash
# Install all dependencies
make setup

# Run tests (110 tests, all passing)
make test

# Lint
make lint

# Download ESA Anomaly Dataset sample from Zenodo (~1% of one mission)
# Requires internet access — downloads ~300MB, saves ~3MB Parquet sample
make download-sample MISSION=ESA-Mission1

# Explore the sample: schema, row counts, time range, anomaly label summary
make explore MISSION=ESA-Mission1

# Or call the CLI directly
uv run spacecraft-telemetry --help
uv run spacecraft-telemetry download --mission ESA-Mission1 --sample --sample-fraction 0.01
uv run spacecraft-telemetry explore --mission ESA-Mission1
uv run spacecraft-telemetry explore --mission ESA-Mission1 --channel A-1
```

## Architecture

```
ESA Parquet (Zenodo/GCS)
  → Download + Sample (Phase 1)  ← complete
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

## Phase 1 Components

| Module | Description |
|--------|-------------|
| `core/config.py` | Pydantic-settings `Settings` with YAML + env var layering |
| `core/logging.py` | structlog setup (console / JSON), `get_logger()` |
| `ingest/download.py` | `ZenodoDownloader`: file list, streaming download, MD5 verify, 429 backoff |
| `ingest/sample.py` | `SampleCreator`: pickle → Parquet, deterministic channel selection, manifest |
| `ingest/explore.py` | `DataExplorer`: `MissionReport`, `ChannelSummary`, `LabelReport`, rich tables |
| `cli.py` | Click CLI: `download`, `explore`, `version` subcommands |

## Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Repo scaffold + data ingestion | ✅ Complete |
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

[ESA Anomaly Dataset](https://zenodo.org/records/12528696) — real spacecraft telemetry from 3 ESA missions (~31GB total, 176 channels). Pre-labeled anomaly segments used for evaluation only; the Telemanom LSTM trains on nominal data.

Local dev uses a 1% sample (~5 channels per mission) stored as Parquet in `data/sample/`.
