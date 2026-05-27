#!/usr/bin/env bash
# Submit a spacecraft-train RayJob to the GKE cluster and tail its logs.
#
# Usage:
#   ./scripts/cloud_train.sh [--mission MISSION] [--no-wait] [--delete-after]
#
# Required environment variables:
#   PROJECT_ID   GCP project ID
#   REGION       GCP region (default: us-central1)
#   MLFLOW_URL   Internal Cloud Run URL for the MLflow tracking server
#                  (from `terraform output -raw mlflow_url` or
#                   `gcloud run services describe mlflow --format='value(status.url)'`)
#
# Example:
#   export PROJECT_ID=my-gcp-project
#   export REGION=us-central1
#   export MLFLOW_URL=$(gcloud run services describe mlflow --region $REGION --format='value(status.url)')
#   ./scripts/cloud_train.sh --mission ESA-Mission2

set -euo pipefail

MISSION="${MISSION:-ESA-Mission2}"
NO_WAIT=false
DELETE_AFTER=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --mission)     MISSION="$2"; shift 2 ;;
    --no-wait)     NO_WAIT=true; shift ;;
    --delete-after) DELETE_AFTER=true; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

: "${PROJECT_ID:?PROJECT_ID must be set}"
: "${MLFLOW_URL:?MLFLOW_URL must be set}"
REGION="${REGION:-us-central1}"
export PROJECT_ID REGION MLFLOW_URL MISSION

echo "==> Submitting spacecraft-train RayJob (mission=${MISSION})"

# Delete any existing job with the same name so kubectl apply is idempotent.
if kubectl get rayjob spacecraft-train -n ray &>/dev/null; then
  echo "==> Deleting existing spacecraft-train RayJob"
  kubectl delete rayjob spacecraft-train -n ray
  kubectl wait --for=delete rayjob/spacecraft-train -n ray --timeout=120s
fi

envsubst < "$(dirname "$0")/../deploy/ray/cluster_train.yaml" | kubectl apply -f -

if $NO_WAIT; then
  echo "==> RayJob submitted. Monitor with:"
  echo "    kubectl get rayjob spacecraft-train -n ray -w"
  exit 0
fi

echo "==> Waiting for RayJob to complete (timeout: 6h)..."
kubectl wait --for=jsonpath='{.status.jobDeploymentStatus}'=Complete rayjob/spacecraft-train -n ray --timeout=21600s

echo "==> Job complete. Fetching tail of head-pod logs..."
HEAD_POD=$(kubectl get pods -n ray -l ray.io/node-type=head -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
if [[ -n "$HEAD_POD" ]]; then
  kubectl logs "$HEAD_POD" -n ray --tail=50
fi

if $DELETE_AFTER; then
  echo "==> Deleting RayJob..."
  kubectl delete rayjob spacecraft-train -n ray
fi

echo "==> Done."
