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

# MLflow 3.x serves via uvicorn by default; each --workers process runs an async
# event loop, so 3 workers gives ample concurrency for I/O-bound (DB-waiting) UI
# requests. NOTE: --gunicorn-opts/--threads cannot be combined with --allowed-hosts
# (a uvicorn-only security flag) — doing so makes the container exit(2) at startup.
# 3 workers × (pool_size=3 + max_overflow=4) = 21 connections — under f1-micro's 25 cap.
CMD mlflow server \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --backend-store-uri "${BACKEND_STORE_URI}" \
    --default-artifact-root "${DEFAULT_ARTIFACT_ROOT}" \
    --no-serve-artifacts \
    --workers 3 \
    --allowed-hosts "*"
