# KSA used by Ray pods.  On GKE Autopilot 1.24+, pods authenticate as the
# Direct WIF principal (principal://...) without impersonating a GSA — the
# iam.gke.io annotation and workloadIdentityUser binding below are kept for
# compatibility with standard GKE node pools.  Actual bucket access grants are
# in iam.tf (ray_wif_* resources).

resource "kubernetes_service_account" "ray" {
  metadata {
    name      = "ray-sa"
    namespace = kubernetes_namespace.ray.metadata[0].name
    annotations = {
      # MUST be "gcp-service-account" — GKE Workload Identity only recognizes
      # this exact key. "google-service-account" is silently ignored, leaving
      # the KSA unlinked so the metadata server cannot mint ID tokens (causing
      # 403s when Ray pods call the private MLflow Cloud Run service).
      "iam.gke.io/gcp-service-account" = google_service_account.ray.email
    }
  }
}

# Legacy GSA impersonation binding (no-op on Autopilot 1.24+).
resource "google_service_account_iam_member" "ray_workload_identity" {
  service_account_id = google_service_account.ray.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "serviceAccount:${var.project_id}.svc.id.goog[${kubernetes_namespace.ray.metadata[0].name}/${kubernetes_service_account.ray.metadata[0].name}]"
}
