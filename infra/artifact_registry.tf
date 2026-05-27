resource "google_artifact_registry_repository" "spacecraft" {
  repository_id = "spacecraft-telemetry"
  format        = "DOCKER"
  location      = var.region
  description   = "Docker images for spacecraft-telemetry-anomaly-detection (api, mlflow, training)"

  depends_on = [google_project_service.apis]
}
