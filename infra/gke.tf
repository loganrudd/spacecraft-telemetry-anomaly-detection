# GKE Autopilot — free-tier credit ($74.40/mo) covers the zonal control plane,
# so idle cost is ~$0.  Spot GPU worker nodes are only provisioned when a
# RayJob is submitted (scale-to-zero via KubeRay autoscaler).

resource "google_container_cluster" "ray" {
  name     = "ray-cluster"
  location = var.region

  enable_autopilot = true

  release_channel {
    channel = "REGULAR"
  }

  # Workload Identity lets Ray pods authenticate as sa-ray without key files.
  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  # Limit metrics collection to system components only (node/pod health).
  # Managed Prometheus cannot be disabled on Autopilot 1.25+ — but restricting
  # enable_components prevents workload-level metric ingestion, which is the
  # expensive part. Observability is handled by MLflow + Evidently instead.
  monitoring_config {
    enable_components = ["SYSTEM_COMPONENTS"]
  }

  # Suppress workload logs (Ray worker verbosity); retain system component logs
  # for node/pod lifecycle events only.
  logging_config {
    enable_components = ["SYSTEM_COMPONENTS"]
  }

  # Allow `terraform destroy` without manual cluster deletion.
  deletion_protection = false

  depends_on = [google_project_service.apis]
}
