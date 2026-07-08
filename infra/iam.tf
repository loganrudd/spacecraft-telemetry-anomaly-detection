# ---------------------------------------------------------------------------
# Service accounts
# ---------------------------------------------------------------------------

resource "google_service_account" "api" {
  account_id   = "sa-api"
  display_name = "spacecraft-telemetry API (Cloud Run)"
}

resource "google_service_account" "mlflow" {
  account_id   = "sa-mlflow"
  display_name = "spacecraft-telemetry MLflow (Cloud Run)"
}

resource "google_service_account" "deployer" {
  account_id   = "sa-deployer"
  display_name = "spacecraft-telemetry CI/CD deployer (GitHub Actions WIF)"
}

resource "google_service_account" "ray" {
  account_id   = "sa-ray"
  display_name = "spacecraft-telemetry Ray training (GKE Workload Identity)"
}

# ---------------------------------------------------------------------------
# Project-level IAM
# ---------------------------------------------------------------------------

resource "google_project_iam_member" "deployer_run_admin" {
  project = var.project_id
  role    = "roles/run.admin"
  member  = "serviceAccount:${google_service_account.deployer.email}"
}

resource "google_project_iam_member" "deployer_ar_writer" {
  project = var.project_id
  role    = "roles/artifactregistry.writer"
  member  = "serviceAccount:${google_service_account.deployer.email}"
}

resource "google_project_iam_member" "deployer_container_dev" {
  project = var.project_id
  role    = "roles/container.developer"
  member  = "serviceAccount:${google_service_account.deployer.email}"
}

resource "google_project_iam_member" "api_cloudsql_client" {
  project = var.project_id
  role    = "roles/cloudsql.client"
  member  = "serviceAccount:${google_service_account.api.email}"
}

resource "google_project_iam_member" "mlflow_cloudsql_client" {
  project = var.project_id
  role    = "roles/cloudsql.client"
  member  = "serviceAccount:${google_service_account.mlflow.email}"
}

# ---------------------------------------------------------------------------
# Bucket-level IAM
# ---------------------------------------------------------------------------

# api: read-only access to processed Parquet (replay data) and model artifacts.
resource "google_storage_bucket_iam_member" "api_artifacts_viewer" {
  bucket = google_storage_bucket.artifacts.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.api.email}"
}

resource "google_storage_bucket_iam_member" "api_processed_viewer" {
  bucket = google_storage_bucket.processed_data.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.api.email}"
}

# api: read channels.csv (channel→subsystem map) from the sample bucket. The
# processed bucket has no metadata/channel_subsystems.json, so the API falls
# back to sample-data/{mission}/channels.csv (see core/metadata.py lookup order).
resource "google_storage_bucket_iam_member" "api_sample_viewer" {
  bucket = google_storage_bucket.sample_data.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.api.email}"
}

# api-iss (Phase 17): live pump archives raw ticks for all 18 ISS channels to
# raw-data/ISS/ticks/. objectCreator + objectViewer (list) mirrors the collector
# SA grants and is the minimum needed for flush_buffer to write Parquet shards.
resource "google_storage_bucket_iam_member" "api_raw_data_creator" {
  bucket = google_storage_bucket.raw_data.name
  role   = "roles/storage.objectCreator"
  member = "serviceAccount:${google_service_account.api.email}"
}

resource "google_storage_bucket_iam_member" "api_raw_data_viewer" {
  bucket = google_storage_bucket.raw_data.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.api.email}"
}

# mlflow: write MLflow artifacts (model files, reports) to the artifacts bucket.
resource "google_storage_bucket_iam_member" "mlflow_artifacts_admin" {
  bucket = google_storage_bucket.artifacts.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.mlflow.email}"
}

# ray: read sample Parquet (ESA preprocessing input), write to processed and artifacts buckets.
resource "google_storage_bucket_iam_member" "ray_sample_viewer" {
  bucket = google_storage_bucket.sample_data.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.ray.email}"
}

# ray: read raw ISS Lightstreamer ticks. ESA preprocessing reads from sample-data,
# but run_iss_preprocessing reads raw ticks from raw-data/ISS/ticks/ (list + get),
# so the Ray SA needs objectViewer on the raw-data bucket. Without this, the RayJob
# fails with "storage.objects.list denied on …-raw-data". Read-only: the pipeline
# only consumes ticks here; all writes go to processed-data.
resource "google_storage_bucket_iam_member" "ray_raw_viewer" {
  bucket = google_storage_bucket.raw_data.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.ray.email}"
}

resource "google_storage_bucket_iam_member" "ray_artifacts_admin" {
  bucket = google_storage_bucket.artifacts.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.ray.email}"
}

resource "google_storage_bucket_iam_member" "ray_processed_admin" {
  bucket = google_storage_bucket.processed_data.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.ray.email}"
}

# ---------------------------------------------------------------------------
# Direct Workload Identity Federation — GKE Autopilot 1.24+
# On GKE Autopilot, pods authenticate as the WIF pool principal directly
# rather than impersonating a GSA.  The principalSet:// wildcard covers all
# KSAs in the cluster's WIF pool (safe for a single-purpose training cluster).
# ---------------------------------------------------------------------------

locals {
  ray_wif_pool = "principalSet://iam.googleapis.com/projects/${data.google_project.project.number}/locations/global/workloadIdentityPools/${var.project_id}.svc.id.goog/*"
}

resource "google_storage_bucket_iam_member" "ray_wif_sample_viewer" {
  bucket = google_storage_bucket.sample_data.name
  role   = "roles/storage.objectViewer"
  member = local.ray_wif_pool
}

resource "google_storage_bucket_iam_member" "ray_wif_raw_viewer" {
  bucket = google_storage_bucket.raw_data.name
  role   = "roles/storage.objectViewer"
  member = local.ray_wif_pool
}

resource "google_storage_bucket_iam_member" "ray_wif_processed_admin" {
  bucket = google_storage_bucket.processed_data.name
  role   = "roles/storage.objectAdmin"
  member = local.ray_wif_pool
}

resource "google_storage_bucket_iam_member" "ray_wif_artifacts_admin" {
  bucket = google_storage_bucket.artifacts.name
  role   = "roles/storage.objectAdmin"
  member = local.ray_wif_pool
}

# ---------------------------------------------------------------------------
# Secret Manager IAM
# ---------------------------------------------------------------------------

resource "google_secret_manager_secret_iam_member" "mlflow_db_password_mlflow" {
  secret_id = google_secret_manager_secret.mlflow_db_password.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.mlflow.email}"
}

resource "google_secret_manager_secret_iam_member" "mlflow_db_password_api" {
  secret_id = google_secret_manager_secret.mlflow_db_password.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.api.email}"
}

# ---------------------------------------------------------------------------
# Service account impersonation (deployer needs to act as api and mlflow SAs
# when deploying Cloud Run revisions)
# ---------------------------------------------------------------------------

resource "google_service_account_iam_member" "deployer_act_as_api" {
  service_account_id = google_service_account.api.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.deployer.email}"
}

resource "google_service_account_iam_member" "deployer_act_as_mlflow" {
  service_account_id = google_service_account.mlflow.name
  role               = "roles/iam.serviceAccountUser"
  member             = "serviceAccount:${google_service_account.deployer.email}"
}
