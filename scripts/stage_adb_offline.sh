#!/usr/bin/env bash
# Stage everything the ESA-ADB report (Plan 019) needs to run OFFLINE — i.e.
# without the MLflow tracking server. Downloads from GCS via `gcloud storage`.
#
# Why this exists: the run -> channel mapping normally comes from MLflow run
# tags stored in the Postgres backend. When that server is down, the mapping is
# still fully recoverable from GCS alone:
#   1. hpo_channel_manifest.json (HPO experiment) gives channel -> BASELINE
#      scoring_run_id directly.
#   2. threshold_config.json {window, z} separates baseline runs (Hundman
#      defaults 250/3.0) from tuned ones (subsystem_5: 351/4.8408, per
#      tuned_configs.json).
#   3. All 6 subsystem_5 channels share M=3,092,810 test windows, so file size
#      cannot disambiguate the tuned runs — they are matched to channels by
#      correlating errors.npy content against the known baseline run for each
#      channel (baseline EWMA span=30 vs tuned span=13 over the SAME raw error
#      series correlate at ~0.92, versus <0.7 across channels).
# See docs/plans/019 for the full derivation.
#
# Usage:  ./scripts/stage_adb_offline.sh
# Output: data/adb_offline/  (gitignored, ~750MB)

set -euo pipefail

ART=gs://spacecraft-telemetry-ads-artifacts/mlflow/3
PROC=gs://spacecraft-telemetry-ads-processed-data/ESA-Mission1
DEST=data/adb_offline
CHANNELS=(41 42 43 44 45 46)

# channel baseline_run tuned_run
read -r -d '' ROWS <<'EOF' || true
channel_41 39ca24352369491d902a7696c551e390 57b8031968bb4de78dd96800cee0b418
channel_42 be56a2e9248040d5aa9ba01517496c7a 47c7a7595c5847e58a7aa5b9c621121b
channel_43 01b94af1ecd841659c7389600f4c20db 7d54300158124f4c9e84fd6225123e32
channel_44 6cb4e6c5b09f407f827921521f09ac5b ac04a071670d4e738e92db324c3319b4
channel_45 aca87e2a58bb4038b9b38a93cfe73d47 16c183c9976e48b390750248e9a7cc1b
channel_46 c5d20b9bacd14cffb26e3ec9547cc335 beed481396a042bebb48d5dacf4b4247
EOF

mkdir -p "$DEST/runs" "$DEST/processed/ESA-Mission1/metadata" "$DEST/sample/ESA-Mission1"

echo "==> ground truth (labels + anomaly_types) from local raw"
cp data/raw/ESA-Mission1/labels.csv        "$DEST/sample/ESA-Mission1/"
cp data/raw/ESA-Mission1/anomaly_types.csv "$DEST/sample/ESA-Mission1/"
cp data/raw/ESA-Mission1/channels.csv      "$DEST/sample/ESA-Mission1/"

echo "==> processed test partitions"
for ch in "${CHANNELS[@]}"; do
  d="$DEST/processed/ESA-Mission1/test/mission_id=ESA-Mission1/channel_id=channel_$ch"
  mkdir -p "$d"
  [ -f "$d/part.parquet" ] || gcloud storage cp \
    "$PROC/test/mission_id=ESA-Mission1/channel_id=channel_$ch/part.parquet" \
    "$d/part.parquet"
done

echo "==> normalization params + subsystem metadata"
gcloud storage cp "$PROC/normalization_params.json" \
  "$DEST/processed/ESA-Mission1/normalization_params.json" 2>/dev/null || true
gcloud storage cp "$PROC/metadata/channel_subsystems.json" \
  "$DEST/processed/ESA-Mission1/metadata/channel_subsystems.json" 2>/dev/null || true

echo "==> run artifacts (errors.npy + threshold.npy)"
while read -r ch base tuned; do
  [ -z "$ch" ] && continue
  for run in "$base" "$tuned"; do
    mkdir -p "$DEST/runs/$run"
    for f in errors.npy threshold.npy threshold_config.json; do
      [ -f "$DEST/runs/$run/$f" ] || gcloud storage cp "$ART/$run/artifacts/$f" "$DEST/runs/$run/$f"
    done
  done
  echo "  staged $ch"
done <<< "$ROWS"

echo
echo "Done -> $DEST  ($(du -sh "$DEST" | cut -f1))"
