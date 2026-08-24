#!/usr/bin/env bash
# Submit a spacecraft-tune RayJob to the GKE cluster and tail its logs.
#
# Usage:
#   ./scripts/cloud_tune.sh [--mission MISSION] [--variant VARIANT] [--no-wait] [--delete-after]
#
# Required environment variables:
#   PROJECT_ID   GCP project ID
#   REGION       GCP region (default: us-central1)
#   MLFLOW_URL   Internal Cloud Run URL for the MLflow tracking server
#
# Optional environment variables:
#   VARIANT      Experiment variant (default: unset = None; see
#                docs/plans/020-experiment-variant-axis.md).
#
# Example:
#   export PROJECT_ID=my-gcp-project
#   export REGION=us-central1
#   export MLFLOW_URL=$(gcloud run services describe mlflow --region $REGION --format='value(status.url)')
#   ./scripts/cloud_tune.sh --mission ESA-Mission2

set -euo pipefail

MISSION="${MISSION:-ESA-Mission2}"
VARIANT="${VARIANT:-}"
INJECTED="${INJECTED:-0}"
CHANNELS="${CHANNELS:-}"
NO_WAIT=false
DELETE_AFTER=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --mission)     MISSION="$2"; shift 2 ;;
    --variant)     VARIANT="$2"; shift 2 ;;
    --injected)    INJECTED="1"; shift ;;
    --channels)    CHANNELS="$2"; shift 2 ;;
    --no-wait)     NO_WAIT=true; shift ;;
    --delete-after) DELETE_AFTER=true; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

: "${PROJECT_ID:?PROJECT_ID must be set}"
: "${MLFLOW_URL:?MLFLOW_URL must be set}"
REGION="${REGION:-us-central1}"
# envsubst has no conditional-expansion syntax (same reason CHANNELS_ARG is
# resolved here — see cloud_preprocess.sh), so the optional "/{variant}" path
# segment must be a single pre-resolved variable the YAML can interpolate
# verbatim rather than each YAML site re-deriving the conditional itself.
VARIANT_SEG="${VARIANT:+/${VARIANT}}"

# --injected and --variant cannot be combined: INJECTED=1 below points
# PROCESSED_DATA_DIR at a mission-only _injected root (injection/generate.py
# writes no variant segment), so composing it with VARIANT_SEG would resolve
# to _injected/{mission}/{variant}/test/... — a path nothing ever writes.
# No caller needs the combination today (ISS injection runs at variant=None),
# so this is a guard rather than plumbing variant support through
# injection/generate.py's five path sites for an unused case. See
# docs/reviews/020-experiment-variant-axis.md §3.6.
if [[ "${INJECTED}" = "1" && -n "${VARIANT}" ]]; then
  echo "ERROR: --injected and --variant cannot be combined — injection writes a" >&2
  echo "mission-only layout (injection/generate.py). See" >&2
  echo "docs/reviews/020-experiment-variant-axis.md §3.6." >&2
  exit 1
fi

# INJECTED=1 tunes against the manufactured-label dataset (ISS injection-driven
# HPO). `inject run` writes no channels.txt, so fall back to the base channels.txt
# (injected data covers exactly the same channels as the preprocessed dataset).
# The channels.txt default path is variant-aware — preprocess writes it under
# {mission}/{variant}/ when VARIANT is set. The _injected root itself is a
# separate mechanism that cannot be combined with a variant (guarded above).
if [[ "${INJECTED}" = "1" ]]; then
  PROCESSED_DATA_DIR="gs://${PROJECT_ID}-processed-data/_injected"
  if [[ -n "${CHANNELS:-}" ]]; then
    CHANNELS_ARG="--channels ${CHANNELS}"
  else
    CHANNELS_ARG="--channels-from gs://${PROJECT_ID}-processed-data/${MISSION}${VARIANT_SEG}/channels.txt"
  fi
else
  PROCESSED_DATA_DIR="gs://${PROJECT_ID}-processed-data"
  CHANNELS_ARG="--channels-from gs://${PROJECT_ID}-processed-data/${MISSION}${VARIANT_SEG}/channels.txt"
fi
# ISS W=128 override — see cloud_train.sh for rationale.
if [[ "${MISSION}" = "ISS" ]]; then
  WINDOW_SIZE_OVERRIDE="128"
else
  WINDOW_SIZE_OVERRIDE="250"
fi
# Image tag to run the sweep with. Defaults to :latest; pin to a commit SHA to
# reproduce a past sweep, or to run a controlled A/B where the only variable is
# the code itself. Used for the Plan 019 min_error_value ablation: the sweep
# with the error floor and the sweep without it differ ONLY by image SHA, so
# neither needs a config switch.
RAY_IMAGE_TAG="${RAY_IMAGE_TAG:-latest}"

# Defaulted here, not left to the pod's config, for the same reason
# cloud_train.sh does it: envsubst would otherwise emit an empty string and
# ModelConfig rejects it at container start. HPO builds no forecast head, but it
# rebuilds window labels, whose span depends on the horizon — see the comment on
# SPACECRAFT_MODEL__FORECAST_STEPS in cluster_tune.yaml. MUST match the horizon
# the arm was SCORED with or the sweep aborts on a shape mismatch.
FORECAST_STEPS="${FORECAST_STEPS:-${SPACECRAFT_MODEL__FORECAST_STEPS:-1}}"

export PROJECT_ID REGION MLFLOW_URL MISSION VARIANT VARIANT_SEG PROCESSED_DATA_DIR \
  CHANNELS_ARG WINDOW_SIZE_OVERRIDE RAY_IMAGE_TAG FORECAST_STEPS

echo "==> Submitting spacecraft-tune RayJob (mission=${MISSION}${VARIANT:+, variant=${VARIANT}}, image tag=${RAY_IMAGE_TAG})"

if kubectl get rayjob spacecraft-tune -n ray &>/dev/null; then
  echo "==> Deleting existing spacecraft-tune RayJob"
  kubectl delete rayjob spacecraft-tune -n ray
  kubectl wait --for=delete rayjob/spacecraft-tune -n ray --timeout=120s
fi

envsubst < "$(dirname "$0")/../deploy/ray/cluster_tune.yaml" | kubectl apply -f -

if $NO_WAIT; then
  echo "==> RayJob submitted. Monitor with:"
  echo "    kubectl get rayjob spacecraft-tune -n ray -w"
  exit 0
fi

echo "==> Waiting for RayJob to complete (timeout: 2h)..."
kubectl wait --for=jsonpath='{.status.jobDeploymentStatus}'=Complete rayjob/spacecraft-tune -n ray --timeout=7200s

echo "==> Job complete. Fetching tail of head-pod logs..."
HEAD_POD=$(kubectl get pods -n ray -l ray.io/node-type=head -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
if [[ -n "$HEAD_POD" ]]; then
  kubectl logs "$HEAD_POD" -n ray --tail=50
fi

if $DELETE_AFTER; then
  echo "==> Deleting RayJob..."
  kubectl delete rayjob spacecraft-tune -n ray
fi

echo "==> Done."
