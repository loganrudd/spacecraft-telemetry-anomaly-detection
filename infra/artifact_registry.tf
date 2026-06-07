resource "google_artifact_registry_repository" "spacecraft" {
  repository_id = "spacecraft-telemetry"
  format        = "DOCKER"
  location      = var.region
  description   = "Docker images for spacecraft-telemetry-anomaly-detection (api, mlflow, training)"

  # Keep the 5 most recent tagged images per repo. Untagged images (intermediate
  # CI layers that were never promoted) are deleted immediately. Without this,
  # every push accumulates; Python+PyTorch images are 4-8 GB each.
  cleanup_policy_dry_run = false

  cleanup_policies {
    id     = "keep-last-5-tagged"
    action = "KEEP"
    most_recent_versions {
      keep_count = 5
    }
  }

  cleanup_policies {
    id     = "delete-untagged"
    action = "DELETE"
    condition {
      tag_state = "UNTAGGED"
    }
  }

  depends_on = [google_project_service.apis]
}
