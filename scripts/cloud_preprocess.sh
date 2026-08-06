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
# Optional environment variables:
#   TRAIN_FRACTION  Fraction of the time range used for training (default: 0.8,
#                   matching configs/cloud.yaml).
#   TRAIN_LOOKBACK  Pandas offset alias capping how far back training data
#                   reaches (default: 730D, matching configs/cloud.yaml).
#                   There is no way to express "no cap" via an env var —
#                   SPACECRAFT_PREPROCESS__TRAIN_LOOKBACK= yields the string ''
#                   and 'null' yields 'null', neither of which pydantic coerces
#                   to None. Pass a span longer than the mission instead
#                   (e.g. 36500D), which is provably equivalent to None.
#
# Example:
#   export PROJECT_ID=my-gcp-project
#   export REGION=us-central1
#   ./scripts/cloud_preprocess.sh --mission ESA-Mission1
#
#   # ESA-ADB replication (Plan 019 Stage B) — the paper's chronological
#   # 50/50 halves, training on the full first half:
#   TRAIN_FRACTION=0.5 TRAIN_LOOKBACK=36500D \
#     ./scripts/cloud_preprocess.sh --mission ESA-Mission1-ADB \
#       --channels channel_41,channel_42,channel_43,channel_44,channel_45,channel_46

set -euo pipefail

MISSION="${MISSION:-ESA-Mission1}"
CHANNELS="${CHANNELS:-}"   # optional comma-separated list, e.g. S1000003,P1000003
# Defaults mirror configs/cloud.yaml. Resolved here rather than in the YAML
# because envsubst has no default-value syntax (same reason as CHANNELS_ARG).
TRAIN_FRACTION="${TRAIN_FRACTION:-0.8}"
TRAIN_LOOKBACK="${TRAIN_LOOKBACK:-730D}"
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

# Build the channel-selection argument for the RayJob entrypoint.
# envsubst does not support ${VAR:+...} conditional expansion — resolve it here
# on the host so the YAML receives a plain ${CHANNELS_ARG} substitution.
if [[ -n "${CHANNELS:-}" ]]; then
  CHANNELS_ARG="--channels ${CHANNELS}"
else
  CHANNELS_ARG=""
fi

export PROJECT_ID REGION MISSION CHANNELS_ARG TRAIN_FRACTION TRAIN_LOOKBACK

echo "==> Submitting spacecraft-preprocess RayJob (mission=${MISSION}${CHANNELS:+, channels=${CHANNELS}})"
echo "    train_fraction=${TRAIN_FRACTION}  train_lookback=${TRAIN_LOOKBACK}"

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
