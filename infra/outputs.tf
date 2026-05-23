output "api_url" {
  description = "Public URL of the api Cloud Run service"
  value       = google_cloud_run_v2_service.api.uri
}

output "mlflow_url" {
  description = "Internal-only URL of the MLflow Cloud Run service (use `gcloud run services proxy mlflow` to access locally)"
  value       = google_cloud_run_v2_service.mlflow.uri
}

output "wif_provider" {
  description = "Workload Identity Federation provider resource name — paste into GitHub Actions workflow"
  value       = google_iam_workload_identity_pool_provider.github.name
}

output "deployer_sa_email" {
  description = "Email of the deployer service account — paste into GitHub Actions workflow"
  value       = google_service_account.deployer.email
}

output "artifact_repo" {
  description = "Artifact Registry Docker repository URL prefix"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/spacecraft-telemetry"
}

output "gke_cluster_name" {
  description = "Name of the GKE Autopilot cluster for Ray training"
  value       = google_container_cluster.ray.name
}

output "cloudsql_connection_name" {
  description = "Cloud SQL connection name — used in DSN and Cloud Run --add-cloud-sql-instances"
  value       = google_sql_database_instance.mlflow.connection_name
}
