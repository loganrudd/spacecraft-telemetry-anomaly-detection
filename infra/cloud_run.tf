locals {
  # Placeholder used at creation time; replaced by deploy.yml on first push to main.
  placeholder_image = "us-docker.pkg.dev/cloudrun/container/hello"

  # Cloud SQL Unix socket DSN — psycopg2 connects via the mounted socket volume.
  mlflow_dsn = "postgresql+psycopg2://mlflow:${random_password.mlflow_db.result}@/mlflow?host=/cloudsql/${google_sql_database_instance.mlflow.connection_name}"
}

# ---------------------------------------------------------------------------
# MLflow tracking server — internal-only ingress
# ---------------------------------------------------------------------------

resource "google_cloud_run_v2_service" "mlflow" {
  name     = "mlflow"
  location = var.region

  # All-traffic ingress: allUsers invoker below handles auth.
  # For a hardened prod setup, revert to INGRESS_TRAFFIC_INTERNAL_ONLY + ID-token auth hook.
  ingress = "INGRESS_TRAFFIC_ALL"

  template {
    service_account = google_service_account.mlflow.email

    scaling {
      # min=1 keeps one warm instance (cpu_idle=false default: request-based CPU
      # billing only, ~$8-10/mo).  Removes container-start latency from the API
      # cold-start chain; the API depends_on this service at Terraform level.
      min_instance_count = 1
      max_instance_count = 2
    }

    volumes {
      name = "cloudsql"
      cloud_sql_instance {
        instances = [google_sql_database_instance.mlflow.connection_name]
      }
    }

    containers {
      image = local.placeholder_image

      resources {
        limits = {
          # 3 uvicorn workers need ~1.1 GiB (each ~360 MiB); 1Gi OOMs at startup.
          # 2 vCPU lets workers run in parallel instead of time-slicing one core.
          # NOTE: applied via `terraform apply` (Makefile), NOT deploy.yml — CI
          # only updates the image. Memory/CPU changes require a manual TF apply.
          cpu    = "2"
          memory = "2Gi"
        }
      }

      volume_mounts {
        name       = "cloudsql"
        mount_path = "/cloudsql"
      }

      env {
        name  = "BACKEND_STORE_URI"
        value = local.mlflow_dsn
      }

      env {
        name  = "DEFAULT_ARTIFACT_ROOT"
        value = "gs://${var.project_id}-artifacts/mlflow"
      }

      env {
        name  = "MLFLOW_ARTIFACTS_DESTINATION"
        value = "gs://${var.project_id}-artifacts/mlflow"
      }

      # Constrain SQLAlchemy pool to stay under Cloud SQL's max_connections=25.
      # Default pool_size=5, max_overflow=10 → 15 connections per gunicorn worker.
      # 1 instance × 2 workers × (3+4) = 14 connections max — well under 25.
      env {
        name  = "MLFLOW_SQLALCHEMYSTORE_POOL_SIZE"
        value = "3"
      }

      env {
        name  = "MLFLOW_SQLALCHEMYSTORE_MAX_OVERFLOW"
        value = "4"
      }
    }
  }

  depends_on = [
    google_project_service.apis,
    google_sql_database_instance.mlflow,
    google_project_iam_member.mlflow_cloudsql_client,
  ]

  # CI/CD (deploy.yml) owns the container image after initial creation.
  # Without this, any `terraform apply` resets the image to the placeholder,
  # breaking the running MLflow server. Terraform manages everything except
  # the image; CI/CD manages the image exclusively.
  lifecycle {
    ignore_changes = [template[0].containers[0].image]
  }
}

# ---------------------------------------------------------------------------
# API + dashboard — public ingress
# ---------------------------------------------------------------------------

resource "google_cloud_run_v2_service" "api" {
  name     = "api"
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    service_account = google_service_account.api.email

    # M1.5 measurements: 18s cold-start, 586 MiB at 100 channels.
    # Defaults: 2 vCPU / 2 GiB, scale-to-zero.  Override via TF vars if needed.
    scaling {
      min_instance_count = var.api_min_instances
      max_instance_count = 3
    }

    # Cap concurrent SSE connections to 10/instance × 3 instances = 30 max.
    # Default Cloud Run concurrency is 80 — too many open SSE streams per instance.
    max_instance_request_concurrency = 10

    # Gen2 execution environment for better networking + longer timeouts.
    execution_environment = "EXECUTION_ENVIRONMENT_GEN2"

    # 3600s = 1 hour — required for long-lived SSE streams.
    timeout = "3600s"

    containers {
      image = local.placeholder_image

      resources {
        limits = {
          cpu    = var.api_cpu
          memory = var.api_memory
        }
        # cpu_idle=false: CPU only allocated during request processing (default).
        # startup_cpu_boost: free extra CPU during cold-start; confirmed worthwhile
        # even at 18s (no cost, reduces P95 cold-start for slower real-model loads).
        cpu_idle          = var.api_cpu_idle
        startup_cpu_boost = true
      }

      env {
        name  = "SPACECRAFT_ENV"
        value = "cloud"
      }

      env {
        name  = "SPACECRAFT_MLFLOW__TRACKING_URI"
        value = google_cloud_run_v2_service.mlflow.uri
      }

      env {
        name  = "SPACECRAFT_PREPROCESS__PROCESSED_DATA_DIR"
        value = "gs://${var.project_id}-processed-data"
      }

      env {
        name  = "SPACECRAFT_DRIFT__REFERENCE_PROFILES_DIR"
        value = "gs://${var.project_id}-artifacts/reference_profiles"
      }

      env {
        name  = "MLFLOW_ARTIFACTS_DESTINATION"
        value = "gs://${var.project_id}-artifacts/mlflow"
      }
    }
  }

  depends_on = [
    google_project_service.apis,
    google_cloud_run_v2_service.mlflow,
    google_project_iam_member.api_cloudsql_client,
  ]
}

# ---------------------------------------------------------------------------
# Cloud Run IAM
# ---------------------------------------------------------------------------

# Public access to the api service.
resource "google_cloud_run_v2_service_iam_member" "api_public" {
  name     = google_cloud_run_v2_service.api.name
  location = var.region
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# Public access — MLflow tracking server is open for demo purposes.
# Re-add INGRESS_TRAFFIC_INTERNAL_ONLY + remove this to lock down for prod.
resource "google_cloud_run_v2_service_iam_member" "mlflow_public" {
  name     = google_cloud_run_v2_service.mlflow.name
  location = var.region
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# sa-api and sa-ray invoke the mlflow service.
resource "google_cloud_run_v2_service_iam_member" "mlflow_api_invoker" {
  name     = google_cloud_run_v2_service.mlflow.name
  location = var.region
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.api.email}"
}

resource "google_cloud_run_v2_service_iam_member" "mlflow_ray_invoker" {
  name     = google_cloud_run_v2_service.mlflow.name
  location = var.region
  role     = "roles/run.invoker"
  member   = "serviceAccount:${google_service_account.ray.email}"
}

# Autopilot 1.24+ pods authenticate as the Direct WIF principal, not the GSA above.
# Same principalSet wildcard pattern used for GCS bindings in iam.tf.
resource "google_cloud_run_v2_service_iam_member" "mlflow_ray_wif_invoker" {
  name     = google_cloud_run_v2_service.mlflow.name
  location = var.region
  role     = "roles/run.invoker"
  member   = "principalSet://iam.googleapis.com/projects/${data.google_project.project.number}/locations/global/workloadIdentityPools/${var.project_id}.svc.id.goog/*"
}
