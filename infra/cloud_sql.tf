# Postgres 15 on db-f1-micro (~$10/mo) — MLflow backend store.
# Cloud Run connects via Cloud SQL Auth Proxy Unix socket (no VPC connector needed).
# The MLflow service mounts the socket volume at /cloudsql and uses the DSN:
#   postgresql+psycopg2://mlflow:{password}@/mlflow?host=/cloudsql/{connection_name}

resource "google_sql_database_instance" "mlflow" {
  name             = "mlflow-pg"
  database_version = "POSTGRES_15"
  region           = var.region

  settings {
    tier = "db-f1-micro"

    backup_configuration {
      enabled = false
    }

    ip_configuration {
      # Public IP required by GCP even when using the Cloud SQL Auth Proxy connector.
      # The connector handles auth + encryption; the DB is not reachable without it.
      ipv4_enabled = true
    }

    database_flags {
      name  = "max_connections"
      value = "25"
    }
  }

  deletion_protection = false

  depends_on = [google_project_service.apis]
}

resource "google_sql_database" "mlflow" {
  name     = "mlflow"
  instance = google_sql_database_instance.mlflow.name
}

resource "google_sql_user" "mlflow" {
  name     = "mlflow"
  instance = google_sql_database_instance.mlflow.name
  password = random_password.mlflow_db.result
}
