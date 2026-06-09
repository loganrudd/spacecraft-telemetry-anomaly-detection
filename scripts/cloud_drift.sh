#!/usr/bin/env bash
# Run Evidently batch drift monitoring against cloud data (GCS + Cloud SQL MLflow).
#
# Drift is serial (no Ray), so this runs the CLI locally with cloud env vars
# pointing at GCS processed data and the Cloud SQL–backed MLflow server.
# It is the cloud equivalent of `make drift-batch-mission`.
#
# Usage:
#   ./scripts/cloud_drift.sh [--mission MISSION] [--subsystem SUBSYSTEM]
#                            [--channel CHANNEL] [--max-channels N]
#
# Required environment variables:
#   PROJECT_ID   GCP project ID
#   MLFLOW_URL   Internal Cloud Run URL for the MLflow tracking server
#                  (from `terraform output -raw mlflow_url` or
#                   `gcloud run services describe mlflow --format='value(status.url)'`)
#
# Optional environment variables:
#   REGION       GCP region (default: us-central1)
#   MISSION      Mission name (default: ESA-Mission2); overridden by --mission
#
# Example:
#   export PROJECT_ID=my-gcp-project
#   export REGION=us-central1
#   export MLFLOW_URL=$(gcloud run services describe mlflow --region $REGION --format='value(status.url)')
#   ./scripts/cloud_drift.sh --mission ESA-Mission2
#   ./scripts/cloud_drift.sh --mission ESA-Mission2 --subsystem subsystem_6

set -euo pipefail

MISSION="${MISSION:-ESA-Mission2}"
SUBSYSTEM=""
CHANNEL=""
MAX_CHANNELS=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --mission)      MISSION="$2"; shift 2 ;;
    --subsystem)    SUBSYSTEM="$2"; shift 2 ;;
    --channel)      CHANNEL="$2"; shift 2 ;;
    --max-channels) MAX_CHANNELS="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

: "${PROJECT_ID:?PROJECT_ID must be set}"
: "${MLFLOW_URL:?MLFLOW_URL must be set}"
REGION="${REGION:-us-central1}"

# All other cloud-* scripts submit GKE RayJobs where Workload Identity handles
# auth automatically.  This script runs locally, so we must authenticate to the
# Cloud Run MLflow service explicitly.  Pre-fetch a GCP ID token via gcloud and
# set MLFLOW_TRACKING_TOKEN before the Python process starts.  The Python code
# also attempts this via google.oauth2.id_token.fetch_id_token, but that path
# can silently fail when ADC uses user credentials rather than a service account
# key — leaving requests to Cloud Run unauthenticated (→ 403).
#
# Requires: gcloud auth login && gcloud auth application-default login,
#           and the caller's account must have roles/run.invoker on the MLflow
#           Cloud Run service.
# --audiences is only valid for service accounts; user accounts (gcloud auth login)
# require the bare command with no --audiences flag.  Cloud Run accepts user
# tokens as long as the account has roles/run.invoker on the service.
if MLFLOW_TRACKING_TOKEN="$(gcloud auth print-identity-token 2>/dev/null)"; then
    export MLFLOW_TRACKING_TOKEN
else
    echo "WARNING: 'gcloud auth print-identity-token' failed." >&2
    echo "         Ensure 'gcloud auth login' has been run and the account has" >&2
    echo "         roles/run.invoker on the MLflow Cloud Run service." >&2
fi

CMD=(
  uv run spacecraft-telemetry --env cloud
  drift batch-mission
  --mission "$MISSION"
)
[[ -n "$SUBSYSTEM" ]]    && CMD+=(--subsystem "$SUBSYSTEM")
[[ -n "$CHANNEL" ]]      && CMD+=(--channel "$CHANNEL")
[[ -n "$MAX_CHANNELS" ]] && CMD+=(--max-channels "$MAX_CHANNELS")

echo "==> Running cloud drift batch-mission (mission=${MISSION}${SUBSYSTEM:+, subsystem=${SUBSYSTEM}}${CHANNEL:+, channel=${CHANNEL}})"

SSL_CERT_FILE="$(uv run python -m certifi)" \
SPACECRAFT_PREPROCESS__PROCESSED_DATA_DIR="gs://${PROJECT_ID}-processed-data" \
SPACECRAFT_MLFLOW__TRACKING_URI="$MLFLOW_URL" \
MLFLOW_ARTIFACTS_DESTINATION="gs://${PROJECT_ID}-artifacts/mlflow" \
  "${CMD[@]}"

echo "==> Done."
