variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region for all resources"
  type        = string
  default     = "us-central1"
}

variable "github_repo" {
  description = "GitHub repository in owner/name format — used to scope WIF to this repo"
  type        = string
  default     = "loganrudd/spacecraft-telemetry-anomaly-detection"
}

variable "notification_email" {
  description = "Email address for billing budget alerts"
  type        = string
  default     = "loganrudd@gmail.com"
}

variable "billing_account" {
  description = "GCP billing account ID (format: XXXXXX-XXXXXX-XXXXXX)"
  type        = string
}

# Cloud Run api service sizing — populated from M1.5 measurements.
# Defaults reflect the Plan 009b target confirmed by profiling:
#   18s cold-start, 586 MiB RSS at 100 channels → 2 vCPU / 2 GiB, min=0.

variable "api_cpu" {
  description = "vCPU allocation for the api Cloud Run service"
  type        = string
  default     = "2"
}

variable "api_memory" {
  description = "Memory allocation for the api Cloud Run service"
  type        = string
  # 2.5Gi: baseline usage is ~1.85 GiB (30 LSTM engines + drift references +
  # pre-cached 3k-row replay slices). The full-series lazy load that SIGKILLed
  # the 2Gi instance is gone — replay is now pre-cached at startup from slices
  # (~1 MB total vs 9M+ rows). 2.5Gi gives ~650 MB headroom for inference
  # spikes and GC pressure. 2Gi is too tight at ~92% baseline utilisation.
  default     = "2.5Gi"
}

variable "api_min_instances" {
  description = "Minimum instances (0 = scale-to-zero, 1 = always-on at ~$80/mo)"
  type        = number
  default     = 0
}

variable "api_cpu_idle" {
  description = "Allocate CPU when the instance is idle (true = always-on CPU billing)"
  type        = bool
  default     = false
}

variable "api_subsystems" {
  description = <<-EOT
    Serving scope: list of subsystem names to load on startup.
    null (default) = whole-mission mode, all promoted channels.
    Set to a non-empty list to restrict serving to those subsystems only
    — no image rebuild needed, just a `terraform apply`.
    Example: ["subsystem_1"] or ["subsystem_1", "subsystem_2"]
  EOT
  type        = list(string)
  default     = null
}

variable "mlflow_admin_invokers" {
  description = "IAM principals allowed to invoke the private MLflow Cloud Run service for operator access (for example, user:alice@example.com or group:mlflow-admins@example.com)"
  type        = set(string)
  default     = []
}
