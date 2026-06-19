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

      # COS mounts / read-only, so the default $HOME (/root) is not writable and
      # docker-credential-gcr can't create /root/.docker. Point HOME at a
      # writable path; both the credential helper (writes $HOME/.docker/config.json)
      # and the docker CLI (reads it) then agree. /var is writable on COS.
      export HOME=/var/lib/collector
      mkdir -p "$HOME"

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
    google_storage_bucket_iam_member.collector_raw_viewer,
    google_project_iam_member.collector_ar_reader,
    google_project_iam_member.collector_log_writer,
  ]
}
