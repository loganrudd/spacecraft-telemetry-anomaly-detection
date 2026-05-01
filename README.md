# Spacecraft Telemetry Anomaly Detection System

End-to-end ML infrastructure for real-time spacecraft telemetry anomaly detection using the [ESA Anomaly Dataset](https://zenodo.org/records/12528696).

**Status: Phase 5 of 12 complete — Ray Parallel Training**

## Tech Stack

- Python 3.12, uv
- PySpark 4.1 (preprocessing — Phase 2 ✅)
- Feast 0.47 (feature store — Phase 3 ✅)
- PyTorch (Telemanom LSTM — Phase 4 ✅)
- Ray 2.x Core + Tune (parallel training, HPO — Phases 5–6)
- MLflow 3.x (experiment tracking, model registry — Phase 7)
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

# Run all tests (Spark tests need JDK 21 — skip gracefully if absent)
make test

# Lint
make lint

# Remove build artifacts and Python caches (safe, instant)
make clean

# Remove Spark output (re-run spark-preprocess to rebuild)
make clean-processed

# Remove trained model artifacts (re-run model-train to rebuild)
make clean-models

# Wipe Feast local registry and online store (re-run feast-apply + feast-materialize)
make clean-feast

# Remove downloaded raw + sample data (re-run download-sample)
make clean-data

# Remove everything
make clean-all

# Download ESA Mission 1/2/3 Anomaly Dataset
# ~4GB .zip file if not already downloaded, 
# then samples the mission's data according to configs/
make download-sample MISSION=ESA-Mission1

# Explore the sample: schema, row counts, time range, anomaly label summary
make explore MISSION=ESA-Mission1

# Run the Spark preprocessing pipeline on sample data
# Outputs to data/processed/ESA-Mission1/{features,train,test}/
make spark-preprocess MISSION=ESA-Mission1

# Register Feast feature view definitions to the local registry
make feast-apply MISSION=ESA-Mission1

# Materialize offline features into the SQLite online store
make feast-materialize MISSION=ESA-Mission1

# Run only the Feast tests
make feast-test

# Run only the Spark tests
make spark-test

# Train Telemanom LSTM on a single channel (requires spark-preprocess first)
make model-evaluate MISSION=ESA-Mission1 CHANNEL=channel_1

# Train all preprocessed channels in parallel with Ray
make ray-train-all MISSION=ESA-Mission1

# Score all trained channels in parallel with Ray
make ray-score-all MISSION=ESA-Mission1

# Smoke test: train exactly 1 channel via Ray (fast sanity check)
make ray-train-smoke MISSION=ESA-Mission1

# Run Ray training unit tests (excludes slow integration tests)
make ray-test

# Ray three-step workflow (Phase 5 → Phase 6 → Phase 5b)
# 1. Train all channels
uv run spacecraft-telemetry ray train-all --mission ESA-Mission1
# 2. (Phase 6) Run HPO → produces tuned_configs.json
# 3. Score all channels, optionally applying HPO output
uv run spacecraft-telemetry ray score-all --mission ESA-Mission1
uv run spacecraft-telemetry ray score-all --mission ESA-Mission1 --tuned-configs tuned_configs.json

# Or call other CLI commands directly
uv run spacecraft-telemetry --help
uv run spacecraft-telemetry download --mission ESA-Mission1 --sample
uv run spacecraft-telemetry explore --mission ESA-Mission1
uv run spacecraft-telemetry spark preprocess --mission ESA-Mission1
uv run spacecraft-telemetry feast apply --mission ESA-Mission1
uv run spacecraft-telemetry feast materialize --mission ESA-Mission1
uv run spacecraft-telemetry feast retrieve --channel channel_1 --mission ESA-Mission1
uv run spacecraft-telemetry model train --mission ESA-Mission1 --channel channel_1
uv run spacecraft-telemetry model score --mission ESA-Mission1 --channel channel_1
```

## Architecture

```
ESA Parquet (Zenodo/GCS)
  → Download + Sample (Phase 1)        ← complete
  → PySpark preprocessing (Phase 2)    ← complete
  → Feast feature store (Phase 3)      ← complete
  → Telemanom LSTM training (Phase 4)  ← complete
  → Ray parallel training (Phase 5)    ← complete
  → Ray Tune HPO (Phase 6)
  → MLflow tracking (Phase 7)
  → Evidently monitoring (Phase 8)
  → FastAPI + SSE serving (Phase 9)
  → React dashboard (Phase 10)
  → GCP deployment (Phase 11)
  → Documentation + polish (Phase 12)

```

## Phase 5 Components

| Module | Description |
|--------|-------------|
| `ray_training/tasks.py` | `make_train_task(num_gpus)` / `make_score_task(num_gpus)` — factory pattern wraps Phase 4 functions as `@ray.remote` tasks. GPU fraction set at call time (0.0 locally, 0.25 on cloud T4). Catches all exceptions; returns `status="error"` with traceback so partial failures don't abort the sweep. |
| `ray_training/runner.py` | `discover_channels` scans Hive partition dirs for available channels. `train_all_channels` / `score_all_channels` fan out tasks via Ray Core; one `ray.put(settings)` per unique config variant. `score_all_channels` reads `channels.csv` to apply per-subsystem tuned configs from Phase 6. |

### Ray Workflow

```
Phase 5a  →  ray train-all   fan out train_channel; write model.pt + errors.npy per channel
Phase 6   →  Ray Tune HPO    one Tune experiment per subsystem; output: tuned_configs.json
Phase 5b  →  ray score-all   fan out score_channel; optionally apply tuned_configs.json
```

### Ray CLI

```bash
# Train all channels in parallel (discovers channels automatically from processed data)
make ray-train-all MISSION=ESA-Mission1

# Score all trained channels (Hundman defaults)
make ray-score-all MISSION=ESA-Mission1

# Score with Phase 6 HPO output
uv run spacecraft-telemetry ray score-all --mission ESA-Mission1 --tuned-configs tuned_configs.json

# Limit to N channels for local testing
uv run spacecraft-telemetry ray train-all --mission ESA-Mission1 --max-channels 4
uv run spacecraft-telemetry ray train-all --mission ESA-Mission1 --channels channel_1,channel_2
```

## Phase 4 Components

| Module | Description |
|--------|-------------|
| `model/architecture.py` | `TelemanomLSTM`: 2-layer LSTM (hidden=80, dropout=0.3) → linear head. `build_model(cfg)` factory. |
| `model/dataset.py` | Reads per-timestep series Parquet via PyArrow (Plan 002.5 — no pre-materialized windows). `_build_window_index` constructs LSTM windows on-the-fly. `make_dataloaders`, `make_test_dataloader`. |
| `model/training.py` | `train_channel(settings, mission, channel)` — Adam + MSE, early stopping, per-call seed. Returns `TrainingResult`. |
| `model/scoring.py` | Torch-free at module level. EWMA error smoothing, causal rolling threshold, run-length anomaly flagging, precision/recall/F1/F0.5. `score_channel` persists artifacts. |
| `model/io.py` | `_write_bytes`/`_read_bytes` indirection point (Phase 5 will widen to `gs://`). `save_model`/`load_model` via `BytesIO`. `artifact_paths` for consistent directory layout. |
| `model/device.py` | `resolve_device(setting)` — auto-selects CUDA→MPS→CPU; raises on unavailable explicit backend. |

### Model Artifact Layout

```
models/
  {mission}/
    {channel}/
      model.pt                 # LSTM state dict (BytesIO serialized)
      model_config.json        # Architecture config (load_model ignores current Settings)
      normalization_params.json
      errors.npy               # Smoothed prediction errors for the test split
      threshold.npy            # Rolling threshold series
      threshold_config.json    # Threshold params: {window, z}
      metrics.json             # precision, recall, f1, f0_5, support counts
      train_log.json           # Per-epoch train/val loss
```

### Model Commands

```bash
# Full end-to-end: train + score on one channel
make model-evaluate MISSION=ESA-Mission1 CHANNEL=channel_1

# Train only (saves model.pt + train_log.json)
make model-train MISSION=ESA-Mission1 CHANNEL=channel_1

# Score only (requires trained model)
make model-score MISSION=ESA-Mission1 CHANNEL=channel_1

# Fast model tests (excludes @pytest.mark.slow training loop tests)
make model-test
```

## Phase 3 Components

| Module | Description |
|--------|-------------|
| `feast_client/repo.py` | `build_schema_from_definitions`, `build_entities`, `build_feature_view` — Feast object builders driven by `FEATURE_DEFINITIONS` |
| `feast_client/store.py` | `create_feature_store`, `apply_definitions`, `materialize`, `teardown` — store lifecycle |
| `feast_client/client.py` | `get_historical_features(store, entity_df)`, `get_online_features_for_channel(store, channel_id, mission_id)` — retrieval helpers for Phase 4 (training) and Phase 9 (serving) |
| `feature_repo/registry.py` | Feast CLI adapter — `Entity` + `FeatureView` objects discovered by `feast apply` |
| `feature_repo/feature_store.yaml` | Local store config: `FileOfflineStore` (reads Phase 2 Parquet directly) + `SqliteOnlineStore` |

### feature_repo/ Layout

```
feature_repo/
  feature_store.yaml        # Feast project config (committed)
  registry.py               # Entities + FeatureView definitions (committed)
  data/
    registry.db             # Written by feast apply (gitignored)
    online_store.db         # Written by feast materialize (gitignored)
```

### Phase 4 Contract

Phase 4 (Telemanom training) retrieves features via:

```python
from spacecraft_telemetry.feast_client.client import get_historical_features

# entity_df columns: channel_id (str), mission_id (str),
#                    event_timestamp (datetime64[us, UTC])
df = get_historical_features(store, entity_df)
```

Phase 9 (FastAPI serving) retrieves latest materialized values via:

```python
from spacecraft_telemetry.feast_client.client import get_online_features_for_channel

features = get_online_features_for_channel(store, channel_id="channel_1", mission_id="ESA-Mission1")
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
| `spark/transforms.py` | `handle_nulls`, `detect_gaps`, `normalize`, `add_rolling_features`, `label_timesteps`, `temporal_train_test_split` |
| `spark/io.py` | `read_channel`, `read_labels`, `write_features`, `write_series` — partitioned Parquet I/O |
| `spark/pipeline.py` | `run_preprocessing` — full preprocessing orchestration (read → clean → normalize → features + windows) |
| `spark/session.py` | `create_spark_session` factory (local mode for dev, configurable for Dataproc) |
| `spark/schemas.py` | Explicit `StructType` schemas for all pipeline stages |
| `features/definitions.py` | `FEATURE_DEFINITIONS` registry shared between Spark (training) and inference (serving) to prevent train-serve skew |

### Spark Pipeline Output

For each channel, `make spark-preprocess` writes to `data/processed/{mission}/`:

| Directory | Contents |
|-----------|----------|
| `features/` | One row per timestamp: `value_normalized`, rolling mean/std/min/max at windows [10, 50, 100] samples, `rate_of_change`. Partitioned by `mission_id` + `channel_id`. For Feast (Phase 3). |
| `train/` | Per-timestep series rows (`value_normalized`, `segment_id`, `is_anomaly`). Windows built on-the-fly by the DataLoader at training time (Plan 002.5). |
| `test/` | Per-timestep series rows for LSTM evaluation. Includes anomalous timesteps for scoring (not excluded at Spark time). |
| `normalization_params.json` | Per-channel mean + std for applying the same z-score transform at inference time. |

## Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Repo scaffold + data ingestion | ✅ Complete |
| 2 | PySpark preprocessing pipeline | ✅ Complete |
| 3 | Feast feature store integration | ✅ Complete |
| 4 | Telemanom model drop-in | ✅ Complete |
| 5 | Ray parallel training | ✅ Complete |
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
