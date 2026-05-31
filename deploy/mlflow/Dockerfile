FROM python:3.12-slim

RUN pip install --no-cache-dir \
    "mlflow>=3.0,<4" \
    "psycopg2-binary>=2.9" \
    "google-cloud-storage>=2.10" \
    "gunicorn>=21.0"

# mlflow server reads these from env — injected by Terraform cloud_run.tf:
#   BACKEND_STORE_URI   postgresql+psycopg2://...?host=/cloudsql/...
#   DEFAULT_ARTIFACT_ROOT  gs://{project}-artifacts/mlflow
#   MLFLOW_ARTIFACTS_DESTINATION  gs://{project}-artifacts/mlflow
ENV PORT=8080

# 3 workers × (pool_size=3 + max_overflow=4) = 21 connections — under f1-micro's 25 hard cap.
# gthread worker class required for --threads to actually enable concurrency;
# default sync workers ignore the threads flag entirely.
CMD mlflow server \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --backend-store-uri "${BACKEND_STORE_URI}" \
    --default-artifact-root "${DEFAULT_ARTIFACT_ROOT}" \
    --no-serve-artifacts \
    --workers 3 \
    --gunicorn-opts "--threads 4 --worker-class gthread" \
    --allowed-hosts "*"
