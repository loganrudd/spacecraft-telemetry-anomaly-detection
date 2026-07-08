locals {
  # Placeholder used at creation time; replaced by deploy.yml on first push to main.
  placeholder_image = "us-docker.pkg.dev/cloudrun/container/hello"

  # Cloud SQL Unix socket DSN — psycopg2 connects via the mounted socket volume.
  mlflow_dsn = "postgresql+psycopg2://mlflow:${random_password.mlflow_db.result}@/mlflow?host=/cloudsql/${google_sql_database_instance.mlflow.connection_name}"
}

# ---------------------------------------------------------------------------
# MLflow tracking server — authenticated control surface
# ---------------------------------------------------------------------------

resource "google_cloud_run_v2_service" "mlflow" {
  name     = "mlflow"
  location = var.region

  # Keep internet ingress so authenticated operators can use
  # `gcloud run services proxy mlflow` from local machines. Privacy is
  # enforced by Cloud Run IAM — there is no public allUsers binding.
  ingress = "INGRESS_TRAFFIC_ALL"

  template {
    service_account = google_service_account.mlflow.email

    scaling {
      # min=0: scale-to-zero. MLflow cold-start adds ~10-20s to the first API
      # request after idle — acceptable for a portfolio demo. min=1 triggers
      # instance-based CPU billing ($0.000018/vCPU-s always-on) which costs
      # ~$3/day for 2 vCPU regardless of traffic; request-based billing does not
      # apply to minimum instances.
      min_instance_count = 0
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
          # 1 vCPU is sufficient for single-user read-only UI access; 2 vCPU was
          # only needed for parallel workers under experiment write load.
          # NOTE: applied via `terraform apply` (Makefile), NOT deploy.yml — CI
          # only updates the image. Memory/CPU changes require a manual TF apply.
          cpu    = "1"
          memory = "2Gi"
        }
        startup_cpu_boost = true
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
# MLflow DB upgrade job — run before each MLflow image deploy to apply any
# schema migrations introduced by the new image.  The workflow executes this
# job via `gcloud run jobs execute --wait` before `gcloud run deploy mlflow`.
# The job uses the same Cloud SQL volume + DSN as the MLflow service so the
# upgrade runs against the real PostgreSQL backend.
# CI/CD owns the image (ignore_changes) — Terraform owns everything else.
# ---------------------------------------------------------------------------

resource "google_cloud_run_v2_job" "mlflow_db_upgrade" {
  name     = "mlflow-db-upgrade"
  location = var.region

  template {
    template {
      service_account = google_service_account.mlflow.email

      max_retries = 1

      volumes {
        name = "cloudsql"
        cloud_sql_instance {
          instances = [google_sql_database_instance.mlflow.connection_name]
        }
      }

      containers {
        image   = local.placeholder_image
        command = ["mlflow"]
        args    = ["db", "upgrade", local.mlflow_dsn]

        volume_mounts {
          name       = "cloudsql"
          mount_path = "/cloudsql"
        }

        resources {
          limits = {
            cpu    = "1"
            memory = "512Mi"
          }
        }
      }
    }
  }

  depends_on = [
    google_sql_database_instance.mlflow,
    google_project_iam_member.mlflow_cloudsql_client,
  ]

  lifecycle {
    ignore_changes = [template[0].template[0].containers[0].image]
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

    # Defaults: 1 vCPU / 2.5 GiB, scale-to-zero.  Override via TF vars if needed.
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
        name  = "SPACECRAFT_DATA__SAMPLE_DATA_DIR"
        value = "gs://${var.project_id}-sample-data"
      }

      env {
        name  = "SPACECRAFT_DRIFT__REFERENCE_PROFILES_DIR"
        value = "gs://${var.project_id}-artifacts/reference_profiles"
      }

      env {
        name  = "SPACECRAFT_MONITORING__REFERENCE_PROFILES_DIR"
        value = "gs://${var.project_id}-artifacts/reference_profiles"
      }

      env {
        name  = "MLFLOW_ARTIFACTS_DESTINATION"
        value = "gs://${var.project_id}-artifacts/mlflow"
      }

      env {
        name  = "SPACECRAFT_API__STATIC_DIR"
        value = "/app/frontend/dist"
      }

      env {
        name  = "SPACECRAFT_API__REPLAY_WARMUP_ROWS"
        value = "-1350"
      }

      env {
        name  = "SPACECRAFT_API__REPLAY_MAX_ROWS"
        value = "1350"
      }

      # Serving scope — restricts which subsystems are loaded at startup.
      # null (default) = whole-mission.  Override via api_subsystems in tfvars.
      dynamic "env" {
        for_each = var.api_subsystems != null ? [1] : []
        content {
          name  = "SPACECRAFT_API__SUBSYSTEMS"
          value = jsonencode(var.api_subsystems)
        }
      }

      # Mission switcher: activated once api_iss_url is populated (two-pass
      # apply — see variable "api_iss_url" in variables.tf for the workflow).
      dynamic "env" {
        for_each = var.api_iss_url != "" ? [1] : []
        content {
          name = "SPACECRAFT_API__AVAILABLE_MISSIONS"
          value = jsonencode([
            { id = "ISS", label = "NASA ISS", url = var.api_iss_url }
          ])
        }
      }
    }
  }

  depends_on = [
    google_project_service.apis,
    google_cloud_run_v2_service.mlflow,
    google_project_iam_member.api_cloudsql_client,
  ]

  # CI/CD (deploy.yml) owns the container image after initial creation; Terraform
  # owns everything else (env, scaling, IAM). Without this, `terraform apply`
  # reverts the image to the placeholder, breaking the running service. Mirrors
  # the same guard on the mlflow service above.
  lifecycle {
    ignore_changes = [template[0].containers[0].image]
  }
}

# ---------------------------------------------------------------------------
# ISS API service — second mission, shares the api container image.
# Live pump subscribes to and archives all 18 ISS PUIs (SPACECRAFT_COLLECT__
# CHANNEL_SET=all below), but only the 6 curated power+thermal channels
# (2 subsystems) carry a @champion telemanom-ISS-* model post the July drift
# review (see .claude/rules/iss.md "Demo tiering") — the solar_array BGA
# angles and attitude quaternions are demoted (non-stationary / degenerate
# std) and so are archived but never reach the SSE stream (champion-gated,
# see api/live/pump.py _served_channels). Anomalies are demonstrated via the
# on-demand Inject Fault button.
# ---------------------------------------------------------------------------

resource "google_cloud_run_v2_service" "api_iss" {
  name     = "api-iss"
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    service_account = google_service_account.api.email

    scaling {
      min_instance_count = var.iss_min_instances
      # Single instance: one Lightstreamer session, one broadcaster, one GCS
      # archive writer. Multiple instances would cause split-brain SSE state and
      # duplicate tick archival. SSE fan-out within the instance is O(1) viewers.
      max_instance_count = 1
    }

    # Same caps as the ESA api service — SSE streams are long-lived.
    max_instance_request_concurrency = 10
    execution_environment            = "EXECUTION_ENVIRONMENT_GEN2"
    timeout                          = "3600s"

    containers {
      image = local.placeholder_image

      resources {
        limits = {
          # 6 champion models fit comfortably in 1 vCPU / 2 GiB — the pump's
          # memory footprint scales with the 18 archived/subscribed channels
          # (unaffected by the champion count), while inference load scales
          # with the 6 modeled channels. Revisit only if the raw-tick/archive
          # side (not inference) shows pressure (same measure-first discipline
          # as ESA sizing in variables.tf).
          cpu    = "1"
          memory = "2Gi"
        }
        cpu_idle          = false
        startup_cpu_boost = true
      }

      env {
        name  = "SPACECRAFT_ENV"
        value = "cloud"
      }

      env {
        name  = "SPACECRAFT_API__MISSION"
        value = "ISS"
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
        name  = "SPACECRAFT_DATA__SAMPLE_DATA_DIR"
        value = "gs://${var.project_id}-sample-data"
      }

      env {
        name  = "SPACECRAFT_DRIFT__REFERENCE_PROFILES_DIR"
        value = "gs://${var.project_id}-artifacts/reference_profiles"
      }

      env {
        name  = "SPACECRAFT_MONITORING__REFERENCE_PROFILES_DIR"
        value = "gs://${var.project_id}-artifacts/reference_profiles"
      }

      # ISS-only real-time drift calibration. The reference profile's rate_of_change
      # is Δvalue/Δt_seconds; the live monitor only sees per-tick values with no
      # timestamps, so it must be told the tick's wall-clock interval (ISS's fixed
      # 30s grid) to reproduce the same units — otherwise rate_of_change is off by
      # 30x and reads as permanent drift. feature_drift_threshold is raised from the
      # generic Evidently default (0.10) because ISS's strongly periodic orbital
      # signal makes short (256-tick) live windows diverge from the multi-day
      # reference far more than Evidently's default assumes. drift_confirm_windows
      # requires 3 consecutive alerting runs before flagging, suppressing transient
      # spikes so nominal windows stop flagging constantly. Both values are a first
      # pass from a read-only sweep (scratchpad/week_sweep.py: value-drift median
      # 0.13-0.27 for the first ~3 days with brief spikes to ~1.0; genuine sustained
      # drift develops day ~5-11, medians 0.5-1.8) against the train-split reference
      # (see `make seed-reference-profiles ... MISSION=ISS` with --split train) --
      # worth re-tuning with more banked data.
      env {
        name  = "SPACECRAFT_DRIFT__REALTIME_RATE_INTERVAL_SECONDS"
        value = "30"
      }

      env {
        name  = "SPACECRAFT_DRIFT__FEATURE_DRIFT_THRESHOLD"
        value = "1.0"
      }

      env {
        name  = "SPACECRAFT_DRIFT__DRIFT_CONFIRM_WINDOWS"
        value = "3"
      }

      env {
        name  = "MLFLOW_ARTIFACTS_DESTINATION"
        value = "gs://${var.project_id}-artifacts/mlflow"
      }

      env {
        name  = "SPACECRAFT_API__STATIC_DIR"
        value = "/app/frontend/dist"
      }

      env {
        name  = "SPACECRAFT_API__REPLAY_WARMUP_ROWS"
        value = "-1350"
      }

      env {
        name  = "SPACECRAFT_API__REPLAY_MAX_ROWS"
        value = "1350"
      }

      # Phase 17: live Lightstreamer pump mode.
      env {
        name  = "SPACECRAFT_API__LIVE"
        value = "true"
      }

      # Archive raw ticks for all 18 channels to GCS (replaces standalone VM).
      env {
        name  = "SPACECRAFT_API__ARCHIVE_TO_GCS"
        value = "true"
      }

      env {
        name  = "SPACECRAFT_COLLECT__CHANNEL_SET"
        value = "all"
      }

      env {
        name  = "SPACECRAFT_COLLECT__RAW_TICKS_DIR"
        value = "gs://${var.project_id}-raw-data"
      }

      # Flush raw-tick buffers every 5 min — bounds loss on Cloud Run recycle
      # to an interval indistinguishable from a normal LOS gap (~53 s median).
      env {
        name  = "SPACECRAFT_COLLECT__FLUSH_INTERVAL_SECONDS"
        value = "300"
      }

      # Mission switcher: ISS service always references the ESA service URI
      # directly (cross-resource ref, no circular dependency). api.uri is
      # already known since the api service exists before api_iss is created.
      env {
        name = "SPACECRAFT_API__AVAILABLE_MISSIONS"
        value = jsonencode([
          { id = "ESA-Mission1", label = "ESA Mission 1", url = google_cloud_run_v2_service.api.uri }
        ])
      }
    }
  }

  depends_on = [
    google_project_service.apis,
    google_cloud_run_v2_service.mlflow,
    google_project_iam_member.api_cloudsql_client,
  ]

  lifecycle {
    ignore_changes = [template[0].containers[0].image]
  }
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

# Public access to the api-iss service (same audience as api — portfolio demo).
# sa-api already has mlflow invoker access (mlflow_api_invoker below), so no
# new IAM is needed for the shared service account.
resource "google_cloud_run_v2_service_iam_member" "api_iss_public" {
  name     = google_cloud_run_v2_service.api_iss.name
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

# Human/operator access to the MLflow control surface.
# Prefer a Google Group here so access rotation does not require code changes.
resource "google_cloud_run_v2_service_iam_member" "mlflow_admin_invoker" {
  for_each = var.mlflow_admin_invokers

  name     = google_cloud_run_v2_service.mlflow.name
  location = var.region
  role     = "roles/run.invoker"
  member   = each.value
}
