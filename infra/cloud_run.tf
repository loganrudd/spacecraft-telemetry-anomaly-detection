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

  # Internal-only: accessible from sa-api and sa-ray via ID-token auth hook.
  # Use `gcloud run services proxy mlflow` to access the UI locally.
  ingress = "INGRESS_TRAFFIC_INTERNAL_ONLY"

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
          cpu    = "1"
          memory = "1536Mi"
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
    }
  }

  depends_on = [
    google_project_service.apis,
    google_sql_database_instance.mlflow,
    google_project_iam_member.mlflow_cloudsql_client,
  ]
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

# sa-api and sa-ray invoke the internal mlflow service.
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
