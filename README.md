# Spacecraft Telemetry Anomaly Detection System

End-to-end ML infrastructure for real-time spacecraft telemetry anomaly detection using the [ESA Anomaly Dataset](https://zenodo.org/records/12528696).

**Status: Phase 2 of 12 complete — PySpark Preprocessing Pipeline**

## Tech Stack

- Python 3.12, uv
- PySpark 4.1 (preprocessing — Phase 2 ✅)
- Ray 2.x Core + Tune (parallel training, HPO — Phases 5–6)
- PyTorch (Telemanom LSTM — Phase 4)
- MLflow 3.x (experiment tracking, model registry — Phase 7)
- Feast (feature store — Phase 3)
- Evidently (drift monitoring — Phase 8)
- FastAPI + SSE (serving — Phase 9)
- React + Recharts (dashboard — Phase 10)
- GCP: Cloud Run, Dataproc, GCS (Phase 11)

## Prerequisites

- Python 3.12
- [uv](https://docs.astral.sh/uv/) — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- JDK 21 (required for PySpark — JDK 22+ breaks Hadoop file I/O)
  ```bash
  brew install openjdk@21
  export JAVA_HOME=$(brew --prefix openjdk@21)
  ```

## What Works Today

```bash
# Install all dependencies
make setup

# Run all 266 tests (Spark tests need JDK 21 — skip gracefully if absent)
make test

# Lint
make lint

# Download ESA Anomaly Dataset sample from Zenodo (~1% of one mission)
# Requires internet access — downloads ~300MB, saves ~3MB Parquet sample
make download-sample MISSION=ESA-Mission1

# Explore the sample: schema, row counts, time range, anomaly label summary
make explore MISSION=ESA-Mission1

# Run the Spark preprocessing pipeline on sample data
# Outputs to data/processed/ESA-Mission1/{features,train,test}/
make spark-preprocess MISSION=ESA-Mission1

# Run only the Spark tests
make spark-test

# Or call the CLI directly
uv run spacecraft-telemetry --help
uv run spacecraft-telemetry download --mission ESA-Mission1 --sample
uv run spacecraft-telemetry explore --mission ESA-Mission1
uv run spacecraft-telemetry spark preprocess --mission ESA-Mission1
```

## Architecture

```
ESA Parquet (Zenodo/GCS)
  → Download + Sample (Phase 1)        ← complete
  → PySpark preprocessing (Phase 2)    ← complete
  → Feast feature store (Phase 3)
  → Telemanom LSTM training (Phase 4)
  → Ray parallel training (Phase 5)
  → Ray Tune HPO (Phase 6)
  → MLflow tracking (Phase 7)
  → Evidently monitoring (Phase 8)
  → FastAPI + SSE serving (Phase 9)
  → React dashboard (Phase 10)
  → GCP deployment (Phase 11)
  → Documentation + polish (Phase 12)
```

## Phase 1 Components

| Module | Description |
|--------|-------------|
| `core/config.py` | Pydantic-settings `Settings` with YAML + env var layering |
| `core/logging.py` | structlog setup (console / JSON), `get_logger()` |
| `ingest/download.py` | `ZenodoDownloader`: streaming download, MD5 verify, 429 backoff |
| `ingest/sample.py` | `SampleCreator`: pickle → Parquet, deterministic channel selection, manifest |
| `ingest/explore.py` | `DataExplorer`: `MissionReport`, `ChannelSummary`, `LabelReport`, rich tables |
| `cli.py` | Click CLI: `download`, `explore`, `version` subcommands |

## Phase 2 Components

| Module | Description |
|--------|-------------|
| `spark/transforms.py` | `handle_nulls`, `detect_gaps`, `normalize`, `add_rolling_features`, `create_windows`, `temporal_train_test_split`, `join_anomaly_labels`, `exclude_anomalies_from_train` |
| `spark/io.py` | `read_channel`, `read_labels`, `write_features`, `write_windows` — partitioned Parquet I/O |
| `spark/pipeline.py` | `run_preprocessing` — full preprocessing orchestration (read → clean → normalize → features + windows) |
| `spark/session.py` | `create_spark_session` factory (local mode for dev, configurable for Dataproc) |
| `spark/schemas.py` | Explicit `StructType` schemas for all pipeline stages |
| `features/definitions.py` | `FEATURE_DEFINITIONS` registry shared between Spark (training) and inference (serving) to prevent train-serve skew |

### Spark Pipeline Output

For each channel, `make spark-preprocess` writes to `data/processed/{mission}/`:

| Directory | Contents |
|-----------|----------|
| `features/` | One row per timestamp: `value_normalized`, rolling mean/std/min/max at windows [10, 50, 100] samples, `rate_of_change`. Partitioned by `mission_id` + `channel_id`. For Feast (Phase 3). |
| `train/` | Sliding-window sequences (default 250 samples each) for LSTM training. Anomaly-labeled windows excluded. |
| `test/` | Sliding-window sequences for LSTM evaluation. Includes labeled anomaly windows for scoring. |
| `normalization_params.json` | Per-channel mean + std for applying the same z-score transform at inference time. |

## Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Repo scaffold + data ingestion | ✅ Complete |
| 2 | PySpark preprocessing pipeline | ✅ Complete |
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
