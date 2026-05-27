# Bind the Kubernetes service account used by Ray pods to the GCP sa-ray service
# account.  Pods annotated with this KSA get sa-ray credentials via the GKE
# metadata server — no key files needed in the container.

resource "kubernetes_service_account" "ray" {
  metadata {
    name      = "ray-sa"
    namespace = kubernetes_namespace.ray.metadata[0].name
    annotations = {
      "iam.gke.io/google-service-account" = google_service_account.ray.email
    }
  }
}

# GCP side: allow the KSA to impersonate sa-ray.
resource "google_service_account_iam_member" "ray_workload_identity" {
  service_account_id = google_service_account.ray.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "serviceAccount:${var.project_id}.svc.id.goog[${kubernetes_namespace.ray.metadata[0].name}/${kubernetes_service_account.ray.metadata[0].name}]"
}
