# ---------------------------------------------------------------------------
# Phase 12: ISS Live telemetry collector VM
#
# Always-on, non-preemptible e2-small instance.  Spot/preemptible is
# intentionally NOT used — a mid-run eviction breaks the 9-10 day collection
# window.  e2-small runs ~$12/mo; the ~10-day collection window costs ~$4-5.
#
# The VM pulls and runs the collector image from Artifact Registry using a
# dedicated service account with write-only access to the raw-data bucket.
# Workload Identity Federation is not used here (GCE, not GKE); instead the
# SA is attached directly to the instance so the metadata server vends tokens
# without any key files in the image or container.
#
# To start collection:
#   terraform apply -target=google_compute_instance.collector
#   (Then SSH in and check: sudo docker logs collector -f)
#
# To stop and destroy:
#   terraform destroy -target=google_compute_instance.collector
# ---------------------------------------------------------------------------

locals {
  collector_image = "${var.region}-docker.pkg.dev/${var.project_id}/spacecraft-telemetry/collector:latest"
}

# Service account — minimal permissions: write to raw-data bucket only.
resource "google_service_account" "collector" {
  account_id   = "sa-collector"
  display_name = "spacecraft-telemetry ISS Collector (GCE)"
}

resource "google_storage_bucket_iam_member" "collector_raw_writer" {
  bucket = google_storage_bucket.raw_data.name
  role   = "roles/storage.objectCreator"
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

resource "google_compute_instance" "collector" {
  name         = "iss-collector"
  machine_type = "e2-small"
  zone         = "${var.region}-a"

  tags = ["iss-collector"]

  labels = {
    owner     = "spacecraft-telemetry"
    component = "iss-collector"
    phase     = "12"
  }

  # Container-Optimized OS — ships with Docker, handles auto-updates.
  boot_disk {
    initialize_params {
      image = "cos-cloud/cos-stable"
      size  = 20 # GB — holds the image layers and log buffers; data goes to GCS
      type  = "pd-standard"
    }
  }

  network_interface {
    network = "default"
    # Ephemeral public IP so the VM can reach push.lightstreamer.com without
    # a NAT gateway (saves ~$3/mo vs Cloud NAT for a single egress-only VM).
    access_config {}
  }

  # Workload Identity is a GKE concept; for GCE we attach the SA directly.
  service_account {
    email  = google_service_account.collector.email
    scopes = ["cloud-platform"]
  }

  scheduling {
    # Non-preemptible: a mid-run eviction breaks the collection window.
    preemptible         = false
    on_host_maintenance = "MIGRATE"
    automatic_restart   = true
  }

  metadata = {
    # cloud-init startup script: pull and run the collector container.
    # Environment variables configure GCS output and structured logging.
    # --restart unless-stopped ensures recovery from crashes within seconds.
    startup-script = <<-SCRIPT
      #!/bin/bash
      set -euo pipefail

      # Authenticate Docker with Artifact Registry. Container-Optimized OS does
      # NOT ship gcloud, so we use docker-credential-gcr (preinstalled on COS),
      # which vends tokens from the attached service account via the metadata
      # server. gcloud here would fail with "command not found" and, under
      # `set -euo pipefail`, abort the script before docker pull.
      docker-credential-gcr configure-docker --registries=${var.region}-docker.pkg.dev

      # Stop any existing container from a previous startup.
      docker stop collector 2>/dev/null || true
      docker rm   collector 2>/dev/null || true

      docker pull ${local.collector_image}

      docker run --name collector \
        --restart unless-stopped \
        --detach \
        -e SPACECRAFT_COLLECT__RAW_TICKS_DIR=gs://${var.project_id}-raw-data \
        -e SPACECRAFT_COLLECT__CHANNEL_SET=all \
        ${local.collector_image}
    SCRIPT
  }

  depends_on = [
    google_project_service.apis,
    google_service_account.collector,
    google_storage_bucket_iam_member.collector_raw_writer,
    google_project_iam_member.collector_ar_reader,
  ]
}
