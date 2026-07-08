# ---------------------------------------------------------------------------
# Phase 12: ISS Live telemetry collector VM
#
# Phase 17: The standalone VM is RETIRED.  Collection now runs inside the
# api-iss Cloud Run service (LivePump subscribes all 18 PUIs and archives raw
# ticks via flush_buffer — same path, no separate process).  sa-collector and
# its IAM bindings are retained as a zero-cost emergency backstop in case the
# pump needs to be manually bypassed.
#
# Retirement sequence at deploy time (manual, once pump is confirmed):
#   1. Deploy api-iss with SPACECRAFT_API__LIVE=true + SPACECRAFT_API__ARCHIVE_TO_GCS=true
#   2. Confirm GCS raw-data shards for all 18 channels are arriving fresh
#   3. terraform apply  # destroys the VM; SA + IAM remain
#   4. Brief tick overlap during the window is harmless (same shard filenames).
#
# To re-enable the VM for emergency collection (without Terraform):
#   gcloud compute instances create iss-collector ... --service-account=sa-collector@...
# ---------------------------------------------------------------------------

# Kept for emergency manual collection (no VM attached = no cost).
locals {
  collector_image = "${var.region}-docker.pkg.dev/${var.project_id}/spacecraft-telemetry/collector:latest"
}

# Service account — minimal permissions: read+write objects on raw-data bucket.
resource "google_service_account" "collector" {
  account_id   = "sa-collector"
  display_name = "spacecraft-telemetry ISS Collector (GCE)"
}

# Create new shard objects. Shard filenames are second-unique, so the collector
# never overwrites — create is all the *write* path needs (no delete grant).
resource "google_storage_bucket_iam_member" "collector_raw_writer" {
  bucket = google_storage_bucket.raw_data.name
  role   = "roles/storage.objectCreator"
  member = "serviceAccount:${google_service_account.collector.email}"
}

# gcsfs (under pyarrow's flush) does a list/stat on the target path before
# writing a shard. Without storage.objects.list every flush 403s with
# "does not have storage.objects.list access" and the buffer backs up to the
# overflow cap. objectViewer grants list+get only — still no delete.
resource "google_storage_bucket_iam_member" "collector_raw_viewer" {
  bucket = google_storage_bucket.raw_data.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.collector.email}"
}

# Allow the Artifact Registry reader role so the VM can pull the image.
resource "google_project_iam_member" "collector_ar_reader" {
  project = var.project_id
  role    = "roles/artifactregistry.reader"
  member  = "serviceAccount:${google_service_account.collector.email}"
}

# Allow the SA to write Cloud Logging entries (structured JSON logs).
resource "google_project_iam_member" "collector_log_writer" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.collector.email}"
}

# google_compute_instance.collector removed in Phase 17.
# The VM is destroyed via `terraform apply` after confirming the live pump
# is archiving all 18 ISS channels to GCS (see retirement sequence above).
# SA + IAM bindings above are kept as an emergency backstop.
