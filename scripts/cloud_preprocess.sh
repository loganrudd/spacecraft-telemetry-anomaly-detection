#!/usr/bin/env bash
# Submit a spacecraft-preprocess RayJob to the GKE cluster and tail its logs.
#
# Usage:
#   ./scripts/cloud_preprocess.sh [--mission MISSION] [--no-wait] [--delete-after]
#
# Required environment variables:
#   PROJECT_ID   GCP project ID
#   REGION       GCP region (default: us-central1)
#
# Example:
#   export PROJECT_ID=my-gcp-project
#   export REGION=us-central1
#   ./scripts/cloud_preprocess.sh --mission ESA-Mission1

set -euo pipefail

MISSION="${MISSION:-ESA-Mission1}"
CHANNELS="${CHANNELS:-}"   # optional comma-separated list, e.g. S1000003,P1000003
NO_WAIT=false
DELETE_AFTER=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --mission)      MISSION="$2";   shift 2 ;;
    --channels)     CHANNELS="$2";  shift 2 ;;
    --no-wait)      NO_WAIT=true;   shift ;;
    --delete-after) DELETE_AFTER=true; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

: "${PROJECT_ID:?PROJECT_ID must be set}"
REGION="${REGION:-us-central1}"
export PROJECT_ID REGION MISSION CHANNELS

echo "==> Submitting spacecraft-preprocess RayJob (mission=${MISSION}${CHANNELS:+, channels=${CHANNELS}})"

if kubectl get rayjob spacecraft-preprocess -n ray &>/dev/null; then
  echo "==> Deleting existing spacecraft-preprocess RayJob"
  kubectl delete rayjob spacecraft-preprocess -n ray
  kubectl wait --for=delete rayjob/spacecraft-preprocess -n ray --timeout=120s
fi

envsubst < "$(dirname "$0")/../deploy/ray/cluster_preprocess.yaml" | kubectl apply -f -

if $NO_WAIT; then
  echo "==> RayJob submitted. Monitor with:"
  echo "    kubectl get rayjob spacecraft-preprocess -n ray -w"
  exit 0
fi

echo "==> Waiting for RayJob to complete (timeout: 2h)..."
kubectl wait --for=jsonpath='{.status.jobDeploymentStatus}'=Complete \
  rayjob/spacecraft-preprocess -n ray --timeout=7200s

echo "==> Job complete. Fetching tail of head-pod logs..."
HEAD_POD=$(kubectl get pods -n ray -l ray.io/node-type=head -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
if [[ -n "$HEAD_POD" ]]; then
  kubectl logs "$HEAD_POD" -n ray --tail=50
fi

if $DELETE_AFTER; then
  echo "==> Deleting RayJob..."
  kubectl delete rayjob spacecraft-preprocess -n ray
fi

echo "==> Done."
