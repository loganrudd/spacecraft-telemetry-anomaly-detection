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

# Install CUDA-enabled PyTorch wheel before the rest of the package so pip
# does not pull the CPU-only wheel from PyPI.  PyTorch CUDA wheels bundle
# their own CUDA runtime libs; the host only needs to supply libcuda.so
# (provided by GKE's nvidia device plugin on GPU nodes).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124

# ml: torch (already installed above), ray[tune], hyperopt
# tracking: mlflow, evidently
# gcp: google-cloud-storage, gcsfs, google-auth
RUN pip install --no-cache-dir ".[ml,tracking,gcp]"

ENTRYPOINT ["spacecraft-telemetry"]
