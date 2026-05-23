FROM python:3.12-slim

WORKDIR /app

# gcc is required to build some Ray/PyTorch C extensions on slim base.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ src/
COPY configs/ configs/

# ml: torch, ray[tune], hyperopt
# tracking: mlflow, evidently
# gcp: google-cloud-storage, gcsfs, google-auth
RUN pip install --no-cache-dir ".[ml,tracking,gcp]"

ENTRYPOINT ["spacecraft-telemetry"]
