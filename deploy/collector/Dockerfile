# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# Phase 12 ISS collector image.
#
# Slim single-stage image — no model weights, no frontend, no torch.
# Installs only the collect + gcp extras so the image stays under ~300 MB.
#
# Data is written to GCS (SPACECRAFT_COLLECT__RAW_TICKS_DIR overrides the
# local default at runtime via environment variable).
#
# Reconnection to Lightstreamer is handled by the client library. The VM
# runs this container with `--restart unless-stopped` so a crash results in
# a fresh container within seconds, losing at most flush_interval_seconds
# of buffered ticks (default 300 s / 5 min).
# ---------------------------------------------------------------------------

FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install ".[collect,gcp]"

COPY configs/ ./configs/

# Cloud environment: JSON structured logging, GCS raw-data bucket.
# SPACECRAFT_COLLECT__RAW_TICKS_DIR must be set at deploy time, e.g.:
#   SPACECRAFT_COLLECT__RAW_TICKS_DIR=gs://{project}-raw-data
ENV SPACECRAFT_ENV=cloud \
    SPACECRAFT_LOGGING__FORMAT=json

CMD ["spacecraft-telemetry", "collect"]
