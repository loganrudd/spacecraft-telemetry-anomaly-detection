# GCS bucket layout per .claude/rules/gcp.md:
#   {project}-raw-data        — Raw tick archive: ESA zips + ISS Lightstreamer ticks; immutable
#   {project}-sample-data     — Parquet-converted ESA sample (zip→pickle→Parquet via ingest/sample.py);
#                               immutable once uploaded, never deleted by lifecycle rule
#   {project}-processed-data  — Pandas + Ray Core output, partitioned by mission/channel; pruned after 90 days
#   {project}-artifacts       — MLflow artifacts, model files, reference profiles; versioned

resource "google_storage_bucket" "raw_data" {
  name          = "${var.project_id}-raw-data"
  location      = var.region
  force_destroy = false

  uniform_bucket_level_access = true

  # Raw ticks are the irreplaceable source of truth. No lifecycle rule — never auto-delete.

  depends_on = [google_project_service.apis]
}

resource "google_storage_bucket" "sample_data" {
  name          = "${var.project_id}-sample-data"
  location      = var.region
  force_destroy = false

  uniform_bucket_level_access = true

  depends_on = [google_project_service.apis]
}

resource "google_storage_bucket" "processed_data" {
  name          = "${var.project_id}-processed-data"
  location      = var.region
  force_destroy = false

  uniform_bucket_level_access = true

  # Processed Parquet is cheap to regenerate from raw; prune after 90 days.
  lifecycle_rule {
    condition { age = 90 }
    action { type = "Delete" }
  }

  depends_on = [google_project_service.apis]
}

resource "google_storage_bucket" "artifacts" {
  name          = "${var.project_id}-artifacts"
  location      = var.region
  force_destroy = false

  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  depends_on = [google_project_service.apis]
}
