# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# Phase 10 serving image.
#
# Layout:
#   Stage 1 `frontend-builder` — builds the React/Vite dashboard.
#     VITE_API_BASE_URL is set to "" so the SPA hits the same origin
#     (same-origin lets us mount the bundle under FastAPI without CORS).
#
#   Stage 2 `runtime` — slim Python image with FastAPI + the dashboard.
#     Installs the serving/tracking/gcp/inference/collect extras — `[ml]` (ray,
#     hyperopt) and `[spark]` are not needed at inference time and would bloat
#     the image by ~2 GB. `[collect]` carries lightstreamer-client-lib, which the
#     ISS live pump (api-iss) needs to open its Lightstreamer session and archive
#     ticks; without it the pump logs `pump.lightstreamer_not_installed` and
#     silently produces no telemetry. ESA `api` ignores it (LIVE=false) but shares
#     this image, and the lib is small pure-Python — cheap to include everywhere.
#
# No data is bundled. Replay Parquet + Evidently reference profiles are
# fetched from GCS at startup (per cloud.yaml gs:// env overrides).
# ---------------------------------------------------------------------------

# =========================================================================
# Stage 1 — frontend bundle
# =========================================================================
FROM node:20-slim AS frontend-builder
WORKDIR /fe

# Install dependencies first to maximize cache reuse across source edits.
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund

COPY frontend/ ./
# Same-origin deployment: the SPA loads from the same Cloud Run service that
# serves /api/*, so an empty base URL keeps fetches relative.
ENV VITE_API_BASE_URL=""
RUN npm run build

# =========================================================================
# Stage 2 — Python runtime
# =========================================================================
FROM python:3.12-slim AS runtime

# libgomp1 — pyarrow runtime dep on slim Debian.
# curl — used by HEALTHCHECK below.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# pip cache is a build-time concern only; --no-cache-dir trims final image.
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Package metadata + source. Install editable=False so the wheel is built once
# and importable without the source tree being writable.
COPY pyproject.toml README.md ./
COPY src/ ./src/
# Pin mlflow to match the MLflow server + Ray training images. Installed first so
# the .[tracking] extra below won't resolve the loose pyproject range up to mlflow
# 3.14, whose pytorch flavor defaults to 'pt2' serialization. Training logs models
# in pickle format (3.13.0); keep the serving loader on the same major/minor.
# See deploy/ray/Dockerfile for the full rationale.
RUN pip install "mlflow==3.13.0"
# Cloud Run is CPU-only. Pin the CPU-only torch wheel to avoid pulling ~2 GB
# of unused CUDA libraries from the default PyPI index.
RUN pip install ".[serving,tracking,gcp,inference,collect]" \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Application config + dashboard bundle.
COPY configs/ ./configs/
COPY --from=frontend-builder /fe/dist ./frontend/dist

# Cloud Run defaults. Cloud Run injects PORT at runtime (usually 8080); the
# shell-form CMD below expands ${PORT:-8080} so local docker runs work too.
ENV SPACECRAFT_ENV=cloud \
    SPACECRAFT_LOGGING__FORMAT=json \
    PORT=8080

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT:-8080}/health" || exit 1

# spacecraft-telemetry is the console script declared in pyproject.toml.
# Shell form so ${PORT} expands at container start (Cloud Run convention).
CMD ["sh", "-c", "exec spacecraft-telemetry api serve --host 0.0.0.0 --port ${PORT:-8080}"]
