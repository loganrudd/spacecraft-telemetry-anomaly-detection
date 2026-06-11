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
