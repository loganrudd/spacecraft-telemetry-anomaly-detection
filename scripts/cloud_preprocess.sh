#!/usr/bin/env bash
# Submit a spacecraft-preprocess RayJob to the GKE cluster and tail its logs.
#
# Usage:
#   ./scripts/cloud_preprocess.sh [--mission MISSION] [--variant VARIANT] [--no-wait] [--delete-after]
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
#   VARIANT         Experiment variant (default: unset = None; see
#                   docs/plans/020-experiment-variant-axis.md). Writes output
#                   under {processed}/{mission}/{variant}/ instead of
#                   {processed}/{mission}/, without touching raw input data.
#   GRID_INTERVAL   Common time grid in seconds for this mission (default:
#                   unset = native timestamps, today's behaviour). Resamples
#                   every channel onto that grid with gap-preserving semantics
#                   so multivariate groups share timestamps — see
#                   docs/plans/023-channel-time-grid.md stage .3. Pair with a
#                   VARIANT so the native tree stays intact for comparison.
#   CHANNEL_GRIDS   JSON object of per-channel overrides of GRID_INTERVAL,
#                   e.g. '{"channel_12": 90}'. The useful rate is a property of
#                   a channel GROUP (bounded by its coarsest member), and groups
#                   are disjoint, so a channel -> rate map expresses per-group
#                   rates in one preprocessing pass.
#
# Example:
#   export PROJECT_ID=my-gcp-project
#   export REGION=us-central1
#   ./scripts/cloud_preprocess.sh --mission ESA-Mission1
#
#   # An ESA-Mission1 experiment arm, output-isolated via the variant axis
#   # instead of a pseudo-mission — one copy of raw data serves every variant:
#   TRAIN_FRACTION=0.5 TRAIN_LOOKBACK=36500D \
#     ./scripts/cloud_preprocess.sh --mission ESA-Mission1 --variant adb-24m \
#       --channels channel_41,channel_42,channel_43,channel_44,channel_45,channel_46

set -euo pipefail

MISSION="${MISSION:-ESA-Mission1}"
VARIANT="${VARIANT:-}"
CHANNELS="${CHANNELS:-}"   # optional comma-separated list, e.g. S1000003,P1000003
# Defaults mirror configs/cloud.yaml. Resolved here rather than in the YAML
# because envsubst has no default-value syntax (same reason as CHANNELS_ARG).
TRAIN_FRACTION="${TRAIN_FRACTION:-0.8}"
TRAIN_LOOKBACK="${TRAIN_LOOKBACK:-730D}"
# Empty string is the env-var spelling of None for both (config.py coerces the
# "" / "null" sentinels), so unset reproduces today's native-timestamp output.
GRID_INTERVAL="${GRID_INTERVAL:-}"
CHANNEL_GRIDS="${CHANNEL_GRIDS:-{\}}"
NO_WAIT=false
DELETE_AFTER=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --mission)      MISSION="$2";   shift 2 ;;
    --variant)      VARIANT="$2";   shift 2 ;;
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
# Same reasoning as CHANNELS_ARG above — resolved here so any YAML site can
# interpolate the optional "/{variant}" path segment verbatim. This script has
# no channels.txt path of its own; exported for symmetry with the other
# cloud_*.sh scripts and for any future YAML site that needs it.
VARIANT_SEG="${VARIANT:+/${VARIANT}}"

export PROJECT_ID REGION MISSION VARIANT VARIANT_SEG CHANNELS_ARG TRAIN_FRACTION TRAIN_LOOKBACK
export GRID_INTERVAL CHANNEL_GRIDS

echo "==> Submitting spacecraft-preprocess RayJob (mission=${MISSION}${VARIANT:+, variant=${VARIANT}}${CHANNELS:+, channels=${CHANNELS}})"
echo "    train_fraction=${TRAIN_FRACTION}  train_lookback=${TRAIN_LOOKBACK}"
echo "    grid_interval=${GRID_INTERVAL:-native}  channel_grids=${CHANNEL_GRIDS}"

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
