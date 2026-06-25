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

CPU="${CPU:-0}"

while [[ $# -gt 0 ]]; do
  case $1 in
    --mission)       MISSION="$2"; shift 2 ;;
    --channels-from) CHANNELS_FROM="$2"; shift 2 ;;
    --cpu)           CPU="1"; shift ;;
    --no-wait)       NO_WAIT=true; shift ;;
    --delete-after)  DELETE_AFTER=true; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

: "${PROJECT_ID:?PROJECT_ID must be set}"
: "${MLFLOW_URL:?MLFLOW_URL must be set}"
REGION="${REGION:-us-central1}"
# ISS: 6-way L4 packing (floor(1/0.16)=6). ESA: 8-way (floor(1/0.125)=8).
# 0.167 rounds to floor(5.99)=5 under floating point; 0.16 is the safe 6-way value.
# Pass NUM_GPUS=1 to run one channel at a time (large channels / preemption issues).
if [[ "${MISSION}" = "ISS" ]]; then
  NUM_GPUS="${NUM_GPUS:-0.16}"
else
  NUM_GPUS="${NUM_GPUS:-0.125}"
fi

# Build the channel-selection argument for the RayJob entrypoint.
# Precedence: --channels (inline CSV) > --channels-from (GCS file) > full channel list.
if [[ -n "${CHANNELS:-}" ]]; then
  CHANNELS_ARG="--channels ${CHANNELS}"
elif [[ -n "${CHANNELS_FROM:-}" ]]; then
  CHANNELS_ARG="--channels-from ${CHANNELS_FROM}"
else
  CHANNELS_ARG="--channels-from gs://${PROJECT_ID}-processed-data/${MISSION}/channels.txt"
fi

# ISS LOS fragmentation limits contiguous segments to <240 rows on the 30 s grid.
# W=250 requires 251 consecutive rows → 0 valid test windows → scoring crash.
# W=128 (64 min ≈ 0.7 orbits) fits comfortably; ESA keeps the 250 global default.
if [[ "${MISSION}" = "ISS" ]]; then
  WINDOW_SIZE_OVERRIDE="128"
else
  WINDOW_SIZE_OVERRIDE="250"
fi
export PROJECT_ID REGION MLFLOW_URL MISSION CHANNELS_ARG NUM_GPUS WINDOW_SIZE_OVERRIDE

echo "==> Submitting spacecraft-train RayJob (mission=${MISSION})"

# Delete any existing job with the same name so kubectl apply is idempotent.
if kubectl get rayjob spacecraft-train -n ray &>/dev/null; then
  echo "==> Deleting existing spacecraft-train RayJob"
  kubectl delete rayjob spacecraft-train -n ray
  kubectl wait --for=delete rayjob/spacecraft-train -n ray --timeout=120s
fi

if [[ "${CPU}" = "1" ]]; then
  CLUSTER_YAML="deploy/ray/cluster_train_cpu.yaml"
  echo "==> CPU mode: skipping GPU node provisioning"
else
  CLUSTER_YAML="deploy/ray/cluster_train.yaml"
fi
envsubst < "$(dirname "$0")/../${CLUSTER_YAML}" | kubectl apply -f -

if $NO_WAIT; then
  echo "==> RayJob submitted. Monitor with:"
  echo "    kubectl get rayjob spacecraft-train -n ray -w"
  exit 0
fi

echo "==> Waiting for RayJob to complete (timeout: 12h)..."
kubectl wait --for=jsonpath='{.status.jobDeploymentStatus}'=Complete rayjob/spacecraft-train -n ray --timeout=43200s

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
