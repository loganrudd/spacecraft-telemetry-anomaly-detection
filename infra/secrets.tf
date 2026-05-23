resource "random_password" "mlflow_db" {
  length  = 32
  special = false # avoid URL-encoding issues in DSN strings
}

resource "google_secret_manager_secret" "mlflow_db_password" {
  secret_id = "mlflow-db-password"

  replication {
    auto {}
  }

  depends_on = [google_project_service.apis]
}

resource "google_secret_manager_secret_version" "mlflow_db_password" {
  secret      = google_secret_manager_secret.mlflow_db_password.id
  secret_data = random_password.mlflow_db.result
}
