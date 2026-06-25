FROM python:3.12-slim

WORKDIR /app

# gcc is required to build some Ray/PyTorch C extensions on slim base.
# wget is required by KubeRay's default head-pod readiness/liveness probes.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ src/
COPY configs/ configs/

# load_settings() resolves the config dir relative to the INSTALLED package
# location (site-packages), not WORKDIR — so without this it looks in
# /usr/local/lib/python3.12/configs (nonexistent) and silently falls back to
# model defaults, dropping every YAML-only setting (e.g. train_lookback).
# Point it explicitly at the baked configs so cloud.yaml is actually read.
ENV SPACECRAFT_CONFIG_DIR=/app/configs

# Install CUDA-enabled PyTorch wheel before the rest of the package so pip
# does not pull the CPU-only wheel from PyPI.  PyTorch CUDA wheels bundle
# their own CUDA runtime libs; the host only needs to supply libcuda.so
# (provided by GKE's nvidia device plugin on GPU nodes).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124

# Pin mlflow to match the MLflow server image (deploy/mlflow/Dockerfile). Installed
# BEFORE the .[tracking] extra so pip sees the constraint already satisfied and does
# NOT resolve the loose pyproject range (mlflow>=3.0,<4) up to the latest release.
# mlflow 3.14 flipped the pytorch flavor's default serialization_format to 'pt2'
# (a torch.export traced graph that REQUIRES an input_example); register_pytorch_model
# passes none, so an unpinned 3.14 image fails model logging with
# "If `serialization_format` is set to 'pt2', then input_example is required".
# 3.13.0 defaults to 'pickle' and needs no input_example. Same torch-first pattern
# used for the CUDA wheel above.
RUN pip install --no-cache-dir "mlflow==3.13.0"

# ml: torch (already installed above), ray[tune], hyperopt
# tracking: mlflow, evidently
# gcp: google-cloud-storage, gcsfs, google-auth
RUN pip install --no-cache-dir ".[ml,tracking,gcp]"

ENTRYPOINT ["spacecraft-telemetry"]
