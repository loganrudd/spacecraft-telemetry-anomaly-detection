# GCP Deployment Guide

End-to-end instructions for deploying the Spacecraft Telemetry Anomaly Detection system
to Google Cloud Platform — from a fresh GCP project to a live React dashboard streaming
real-time anomaly detection on ESA-Mission1 telemetry.

## Prerequisites

| Tool | Version | Notes |
|------|---------|-------|
| gcloud CLI | latest | `gcloud auth login` + `gcloud auth application-default login` |
| Terraform | ≥ 1.7 | |
| kubectl | ≥ 1.28 | |
| helm | ≥ 3.14 | |
| gh (GitHub CLI) | ≥ 2.0 | for viewing Actions status |

GCP requirements:
- Billing account with budget headroom (~$20 for a full training run; ~$3/mo idle floor)
- L4 GPU quota in us-central1 (≥ 4 L4 for training; request via quota increase if needed)
- Owner or Editor role on the project

## Infrastructure Provisioning

```bash
# 1. Create infra/terraform.tfvars from the example
cp infra/terraform.tfvars.example infra/terraform.tfvars
# Fill in: project_id, billing_account, and optionally region

# 2. Initialise and apply (takes ~15 min on first run)
unset GOOGLE_APPLICATION_CREDENTIALS   # use user ADC, not the dev SA key
cd infra
terraform init
terraform apply
```

Key resources created:
- Cloud Run: `api` (public) + `mlflow` (IAM-authenticated)
- Cloud SQL: PostgreSQL `mlflow-pg` instance
- GKE Autopilot cluster: `ray-cluster`
- GCS buckets: `{project}-raw-data`, `{project}-sample-data`, `{project}-processed-data`, `{project}-artifacts`
- Artifact Registry repo: `spacecraft-telemetry`
- GitHub WIF identity pool + `sa-deployer` service account

Note the outputs — you'll need them later:
```bash
terraform output api_url          # live dashboard URL
terraform output mlflow_url       # MLflow tracking server URL
terraform output wif_provider     # for GitHub Actions secrets
terraform output deployer_sa_email
```

### GitHub Actions Secrets

In your GitHub repo → Settings → Secrets and Variables → Actions:

| Secret | Value |
|--------|-------|
| `GCP_WIF_PROVIDER` | `terraform output -raw wif_provider` |
| `GCP_DEPLOYER_SA` | `terraform output -raw deployer_sa_email` |

| Variable | Value |
|----------|-------|
| `GCP_PROJECT_ID` | your GCP project ID |

## First Deploy

Push to `main` — GitHub Actions (`.github/workflows/deploy.yml`) builds and deploys both
the API and MLflow images:

```bash
git push origin main
gh run watch   # watch deploy progress
```

The API boots immediately. Until models are trained and promoted, `/health` returns:
```json
{"status": "degraded", "channels_loaded": [], "channels_total": 0}
```
This is expected — `channels_total: 0` means no champion models are in the registry yet.

## Training Pipeline

The full batch pipeline runs on a KubeRay cluster. Start it with `make cloud-up` (provisions
the KubeRay operator namespace into the GKE cluster), run your jobs, then `make cloud-down`
to stop billing.

```bash
export PROJECT_ID=spacecraft-telemetry-ads
export REGION=us-central1
export MISSION=ESA-Mission1

# Bring up the Ray operator
make cloud-up

# 1. Preprocess all channels (pandas + Ray Core fan-out)
make cloud-preprocess MISSION=$MISSION

# 2. Train one LSTM per channel (Ray Core, T4 GPU nodes)
make cloud-train MISSION=$MISSION

# 3. Baseline score — Hundman defaults, full test set
make cloud-score MISSION=$MISSION

# 4. HPO tune — per-subsystem threshold search via Ray Tune
make cloud-tune MISSION=$MISSION

# 5. Tuned score — held-out final 40%, HPO params applied
make cloud-score MISSION=$MISSION TUNED=1

# Tear down Ray operator (stops GPU billing)
make cloud-down
```

Each `make cloud-*` target submits a RayJob to GKE and tails logs until completion.
Wall-clock times for ESA-Mission1 (~62 channels):
- Preprocessing: ~20 min
- Training: ~45 min (8× L4 packing, 0.125 GPU/task)
- Baseline score: ~10 min
- Tune: ~30 min (50 trials, 2 concurrent subsystem sweeps)
- Tuned score: ~10 min

## Seed Reference Profiles

Build and upload Evidently reference profiles to GCS (used by the live drift panel):

```bash
make seed-reference-profiles MISSION=$MISSION
```

This reads train-split Parquet from GCS and uploads one `reference.parquet` per channel to
`gs://{project}-artifacts/reference_profiles/`. Run after `cloud-preprocess` completes.

## Promote Models

Set the `@champion` alias on trained models so the API can discover and load them:

```bash
# Authenticate to cloud MLflow
export SPACECRAFT_MLFLOW__TRACKING_URI=$(gcloud run services describe mlflow \
  --region $REGION --project $PROJECT_ID --format='value(status.url)')

# Promote all channels with registered scoring runs
make mlflow-promote MISSION=$MISSION ENV=cloud

# Or a single channel
make mlflow-promote MISSION=$MISSION CHANNEL=channel_22 ENV=cloud

# Or a subsystem
make mlflow-promote MISSION=$MISSION SUBSYSTEM=subsystem_6 ENV=cloud
```

`GOOGLE_APPLICATION_CREDENTIALS` must point at a service account key with
`roles/run.invoker` on the MLflow Cloud Run service (the `spacecraft-dev` SA is
pre-granted via `mlflow_admin_invokers` in `terraform.tfvars`).

## Reload the API

The API loads champion models at startup. After promoting new models, redeploy to pick
them up:

```bash
make cloud-deploy
```

This triggers `gcloud run deploy api --image <current-image>` which creates a new revision
with the same image but forces a fresh cold start that re-runs the lifespan model loader.

## Verification

```bash
API_URL=$(gcloud run services describe api --region $REGION --project $PROJECT_ID \
  --format='value(status.url)')

# Health — should show channels_loaded for promoted channels
curl -s "$API_URL/health" | python3 -m json.tool

# SSE telemetry stream (Ctrl-C to stop)
curl -sN "$API_URL/api/stream/telemetry?speed=10"

# React dashboard
open "$API_URL"
```

A healthy deployment returns:
```json
{
  "status": "ok",
  "mission": "ESA-Mission1",
  "channels_loaded": ["channel_22", "channel_23", ...],
  "channels_ready": 62
}
```

## Updating Models

To retrain after data or code changes, run the full pipeline again:

```bash
make cloud-up
make cloud-preprocess MISSION=$MISSION   # if data changed
make cloud-train MISSION=$MISSION
make cloud-score MISSION=$MISSION        # baseline
make cloud-tune MISSION=$MISSION
make cloud-score MISSION=$MISSION TUNED=1
make cloud-down
make seed-reference-profiles MISSION=$MISSION
make mlflow-promote MISSION=$MISSION ENV=cloud
make cloud-deploy
```

**Important:** `cloud-score` (baseline) must run before `cloud-tune`. A stale
`tuned_configs.json` from a previous cycle is **not** applied automatically — the baseline
pass never reads it (`TUNED` flag unset). After `cloud-tune` completes, run
`cloud-score TUNED=1` to evaluate on the held-out 40% with HPO params.

## MLflow Database Migrations

The deploy pipeline handles schema migrations automatically. When a new MLflow image is
deployed, `deploy-mlflow` in `.github/workflows/deploy.yml` first runs the
`mlflow-db-upgrade` Cloud Run Job to apply any pending schema migrations before the new
service revision starts. This prevents the "out-of-date schema" startup failure.

If you need to run the migration manually:
```bash
gcloud run jobs update mlflow-db-upgrade \
  --image <new-image-uri> --region $REGION --project $PROJECT_ID
gcloud run jobs execute mlflow-db-upgrade \
  --region $REGION --project $PROJECT_ID --wait
```

## ISS Collector (Phase 12)

The ISS Live telemetry collector runs as a long-lived Docker container on an always-on,
non-preemptible `e2-small` GCE VM. It must be running for ~9–10 days before Phase 13
(preprocessing + training) can start.

### Build and push the collector image

CI builds and pushes `collector:latest` automatically when collector-related
files change on `main` (`.github/workflows/build-collector-image.yml`). You can
also trigger a manual build via `workflow_dispatch` before launching the VM.

To build locally (e.g. for a quick test before merging), use `--platform
linux/amd64` — the e2-small COS VM is amd64; an M1 build without this flag
produces arm64 and crashes with "exec format error".

```bash
export PROJECT_ID=your-project-id
export REGION=us-central1

docker build --platform linux/amd64 \
  -f deploy/collector/Dockerfile \
  -t ${REGION}-docker.pkg.dev/${PROJECT_ID}/spacecraft-telemetry/collector:latest .

docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/spacecraft-telemetry/collector:latest
```

**VM rollout is intentionally manual.** Restarting the VM mid-collection would
punch a gap in the 9–10 day data window. After merging changes that affect the
collector, wait for the CI build to finish, then restart the VM:
```bash
gcloud compute instances reset iss-collector \
  --zone=${REGION}-a --project=${PROJECT_ID}
```
The startup-script pulls `:latest` and relaunches the container automatically.

### Deploy the VM

```bash
# Provision the service account, IAM bindings, and GCE instance.
# Run terraform apply in the infra/ directory (or target just the collector resources):
terraform -chdir=infra apply \
  -target=google_service_account.collector \
  -target=google_storage_bucket_iam_member.collector_raw_writer \
  -target=google_project_iam_member.collector_ar_reader \
  -target=google_project_iam_member.collector_log_writer \
  -target=google_compute_instance.collector \
  -var="project_id=${PROJECT_ID}" \
  -var="billing_account=${BILLING_ACCOUNT}"
```

The VM startup script automatically pulls and runs the collector container.

### Verify collection is running

```bash
# SSH into the VM (may take 60s for startup-script to complete):
gcloud compute ssh iss-collector --zone=${REGION}-a --project=${PROJECT_ID}

# Inside the VM — check container logs:
sudo docker logs collector -f
# You should see: "collector.starting" and then "collector.subscribed" events.
# Every 5 minutes: "collector_io.flushed" for each active channel.

# Verify data is landing in GCS:
gsutil ls gs://${PROJECT_ID}-raw-data/ISS/ticks/
```

### 1-hour local dry-run (before deploying)

Validate the feed format and per-channel cadence locally before committing the VM:

```bash
spacecraft-telemetry collect --duration 3600
# After 1 hour:
python - <<'EOF'
import pyarrow.parquet as pq, glob
for p in glob.glob("data/raw/ISS/ticks/channel_id=S1000003/*.parquet"):
    t = pq.read_table(p, partitioning=None)
    print(f"{p}: {len(t)} rows, {t.schema}")
EOF
```

Check the per-channel median tick interval in the logs. If most channels update faster
than ~30 s, update `grid_interval_seconds: 30` and `window_size: 256` in
`configs/local.yaml` and `configs/cloud.yaml` before starting Phase 13.

### Teardown

```bash
terraform -chdir=infra destroy \
  -target=google_compute_instance.collector \
  -var="project_id=${PROJECT_ID}" \
  -var="billing_account=${BILLING_ACCOUNT}"
# The raw-data bucket and its contents are preserved — raw ticks are not auto-deleted.
```

**Cost:** e2-small ~$12/mo → ≈$4–5 for the 10-day collection window. Under the $50 alert
threshold. Stop the VM once Phase 13 preprocessing is complete.

## ISS Preprocessing (Phase 13)

After ~9–10 days of collection, run the ISS preprocessing pipeline on the banked ticks:

```bash
export PROJECT_ID=spacecraft-telemetry-ads
export REGION=us-central1

make cloud-up

# Preprocess all 18 ISS channels (reads gs://{project}-raw-data/ISS/ticks/)
make cloud-preprocess MISSION=ISS

# Or just the 6 Phase-12 validation channels (one+ per subsystem) for the demo:
make cloud-preprocess MISSION=ISS \
  CHANNELS=S1000003,P1000003,P4000007,S4000007,P4000001,USLAB000018

make cloud-down
```

`SPACECRAFT_COLLECT__RAW_TICKS_DIR` is wired in `deploy/ray/cluster_preprocess.yaml` (same
manifest used for ESA — the env var is ignored by the ESA path).  Output lands in
`gs://{project}-processed-data/ISS/{train,test}/…/part.parquet` alongside
`normalization_params.json`.

> **ISS prerequisite — raw-data read for the Ray SA.** ESA preprocessing reads the *sample*
> bucket, but ISS reads ticks from `gs://{project}-raw-data/ISS/ticks/`, so the Ray service
> account needs `objectViewer` on the raw-data bucket (`ray_raw_viewer` / `ray_wif_raw_viewer`
> in `infra/iam.tf`). Provisioned by `make cloud-up`. Without it the RayJob fails with
> `storage.objects.list denied on …-raw-data`.

Wall-clock: ~10 min (18 channels, each a 30 s-grid ~14k-row Parquet).

Verify the output before starting Phase 14 training:
```bash
gsutil ls "gs://${PROJECT_ID}-processed-data/ISS/train/mission_id=ISS/"
gsutil cat "gs://${PROJECT_ID}-processed-data/ISS/normalization_params.json" \
  | python3 -m json.tool | head -20
```

## ISS Training + Injection-Driven HPO (Phases 14–16)

ISS has no labeled anomalies, so detection is evaluated by **fault injection**: inject known
faults into the nominal test split to manufacture `is_anomaly` ground truth, then run the same
Ray Tune F0.5 HPO against them. The demo uses the 6 Phase-12 validation channels (drop
`CHANNELS` to run all 18).

```bash
export PROJECT_ID=spacecraft-telemetry-ads
export REGION=us-central1
export MLFLOW_URL=$(gcloud run services describe mlflow --region $REGION --format='value(status.url)')
CH=S1000003,P1000003,P4000007,S4000007,P4000001,USLAB000018

make cloud-up

# 1. Train one telemanom-ISS-{channel} LSTM per channel
make cloud-train  MISSION=ISS CHANNELS=$CH

# 2. Inject faults into the nominal test split → gs://{project}-processed-data/_injected
#    Runs locally (single-process, reads/writes GCS) — no GKE job, it never touches the models.
make cloud-inject MISSION=ISS CHANNELS=$CH

# 3. Baseline score on the injected data (errors.npy + Hundman-default metrics)
make cloud-score  MISSION=ISS INJECTED=1 CHANNELS=$CH EVAL_SPLIT=full_test

# 4. Tune scoring params on the injected hpo_portion → artifacts/ISS/tuned_configs.json
make cloud-tune   MISSION=ISS INJECTED=1 CHANNELS=$CH

# 5. Tuned re-score on the held-out final_portion
make cloud-score  MISSION=ISS INJECTED=1 CHANNELS=$CH TUNED=1

make cloud-down

# 6. Promote champions (the 6 channels span 3 subsystems)
make mlflow-promote MISSION=ISS SUBSYSTEM=thermal     ENV=cloud
make mlflow-promote MISSION=ISS SUBSYSTEM=solar_array ENV=cloud
make mlflow-promote MISSION=ISS SUBSYSTEM=power       ENV=cloud
make mlflow-promote MISSION=ISS SUBSYSTEM=attitude    ENV=cloud
```

`INJECTED=1` points `cloud-score`/`cloud-tune` at `gs://{project}-processed-data/_injected` and
selects channels explicitly (the injected dir has no `channels.txt`). The ESA path is
unchanged — omit `INJECTED`/`CHANNELS` for the nominal flow. Injection is logically a
*post-preprocessing* step (it depends on the test split, not the models), so it runs as a cheap
local step rather than a RayJob.

> **ISS prerequisite — mlflow pin.** The Ray and API images pin `mlflow==3.13.0`
> (`deploy/ray/Dockerfile`, `deploy/api/Dockerfile`). mlflow 3.14 made `pt2` the default
> pytorch serialization format, which requires an `input_example` and breaks model logging with
> *"If `serialization_format` is set to 'pt2', then input_example is required"*; 3.13 keeps the
> pickle default. The MLflow server image is pinned to the same version. The Ray image is built
> by `.github/workflows/build-ray-image.yml`, which triggers on `main` **and** `iss_ext`.

### Serving ISS as a selectable mission

The serving layer is mission-parameterized: `settings.api.mission` drives model loading
(`telemanom-ISS-{channel}@champion`), the replay path, and the dashboard. ESA and ISS run as
separate processes (and, in Phase 18, separate Cloud Run services from one image). The
dashboard mission switcher (`available_missions` on `/health`) navigates between them, and the
**Inject Fault** button (`POST /api/inject`) drives a live spike/drift/flatline on the shared
replay loop so every viewer sees the same anomaly — the primary way to surface anomalies for
ISS, which has no pre-labeled segments.

To validate ISS serving locally against the cloud champions (before a dedicated Cloud Run
service exists), point a local API at the cloud MLflow + processed bucket:

```bash
SPACECRAFT_MLFLOW__TRACKING_URI=$(gcloud run services describe mlflow --region $REGION --format='value(status.url)') \
MLFLOW_TRACKING_TOKEN=$(gcloud auth print-identity-token) \
SPACECRAFT_PREPROCESS__PROCESSED_DATA_DIR=gs://${PROJECT_ID}-processed-data \
SSL_CERT_FILE=$(uv run python -m certifi) \
make serve MISSION=ISS PORT=8001
```

Run `make serve` (ESA, `:8000`) in another terminal so the mission switcher has both. The
identity token expires after ~1h.

## Cost Guardrails

**Budget alerts** are configured in Terraform at $50, $100, and $150. Alerts email
`notification_email` (set in `terraform.tfvars`).

**Billing kill-switch** (`deploy/billing_kill/`): a Cloud Function triggered by the
Pub/Sub billing alert topic. If spend exceeds the $150 threshold it detaches the billing
account from the project, stopping all billable resources. Recovery:

```bash
gcloud beta billing projects link $PROJECT_ID --billing-account=BILLING_ACCOUNT_ID
```

**Idle floor (~$3/mo):**
- Cloud Run scales to zero when idle (no compute billing)
- Cloud SQL f1-micro: ~$7/mo (cheapest tier; stop the instance manually if not needed)
- GCS storage: ~$0.02/GB/mo
- `make cloud-up` / `make cloud-down` start and stop the GKE cluster and KubeRay operator
  to avoid GPU node billing between training runs

## Teardown

```bash
# Remove all GCP resources
unset GOOGLE_APPLICATION_CREDENTIALS
terraform -chdir=infra destroy

# Empty buckets first if destroy fails on non-empty bucket errors
gsutil -m rm -r gs://${PROJECT_ID}-raw-data gs://${PROJECT_ID}-processed-data \
  gs://${PROJECT_ID}-sample-data gs://${PROJECT_ID}-artifacts

# Check for orphaned resources
gcloud run services list --project $PROJECT_ID
gcloud container clusters list --project $PROJECT_ID
gcloud sql instances list --project $PROJECT_ID
```
