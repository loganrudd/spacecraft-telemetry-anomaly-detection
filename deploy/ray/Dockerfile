# CUDA 12.4 runtime — required for torch.cuda.is_available() on L4/T4 GPU nodes.
# python:3.12-slim installs the CPU-only PyTorch wheel; this base has the CUDA
# runtime libs that pip's CUDA wheel needs to load on the GPU node.
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3-pip \
    gcc \
    wget \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ src/
COPY configs/ configs/

# ml: torch, ray[tune], hyperopt
# tracking: mlflow, evidently
# gcp: google-cloud-storage, gcsfs, google-auth
RUN pip install --no-cache-dir ".[ml,tracking,gcp]"

ENTRYPOINT ["spacecraft-telemetry"]
