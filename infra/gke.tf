# GKE Autopilot — free-tier credit ($74.40/mo) covers the zonal control plane,
# so idle cost is ~$0.  On-demand GPU worker nodes are only provisioned when a
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

  # Capture Ray worker stdout/stderr (app + per-channel training logs) so a run
  # is debuggable in Cloud Logging AFTER the cluster is torn down — kubectl logs
  # die with the head pod, but ingested logs persist at the project level for the
  # 30-day retention. WORKLOADS adds ingestion volume (Ray is verbose) but stays
  # within the 50 GiB/mo free tier for portfolio-scale runs. If volume climbs,
  # add a log exclusion filter for noisy Ray system logs rather than disabling.
  # Metrics stay SYSTEM-only (monitoring_config above) — that is the costly part.
  logging_config {
    enable_components = ["SYSTEM_COMPONENTS", "WORKLOADS"]
  }

  # Allow `terraform destroy` without manual cluster deletion.
  deletion_protection = false

  depends_on = [google_project_service.apis]
}

# ---------------------------------------------------------------------------
# Cloud NAT — internet egress for GKE Autopilot pods
# ---------------------------------------------------------------------------
# Autopilot pods have no external IPs; without NAT they can only reach Google
# services via Private Google Access (GCS, BigQuery, etc.).  Public endpoints
# like Cloud Run (*.run.app) require a route through the internet — Cloud NAT
# provides that without assigning external IPs to individual nodes.

resource "google_compute_router" "nat_router" {
  name    = "nat-router"
  network = "default"
  region  = var.region
}

resource "google_compute_router_nat" "nat" {
  name                               = "ray-nat"
  router                             = google_compute_router.nat_router.name
  region                             = var.region
  nat_ip_allocate_option             = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"
}
