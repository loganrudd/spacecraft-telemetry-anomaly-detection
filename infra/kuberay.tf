# KubeRay operator — manages RayCluster and RayJob CRDs on the GKE cluster.
#
# TWO-STEP APPLY REQUIRED:
#   The Helm and Kubernetes providers below need the GKE cluster endpoint,
#   which is only known after the cluster exists.  On a fresh project:
#
#     terraform apply -target=google_container_cluster.ray
#     terraform apply
#
#   On subsequent applies (cluster already in state) a single `terraform apply`
#   works fine.

data "google_client_config" "default" {}

provider "helm" {
  kubernetes {
    host                   = "https://${google_container_cluster.ray.endpoint}"
    token                  = data.google_client_config.default.access_token
    cluster_ca_certificate = base64decode(google_container_cluster.ray.master_auth[0].cluster_ca_certificate)
  }
}

provider "kubernetes" {
  host                   = "https://${google_container_cluster.ray.endpoint}"
  token                  = data.google_client_config.default.access_token
  cluster_ca_certificate = base64decode(google_container_cluster.ray.master_auth[0].cluster_ca_certificate)
}

resource "kubernetes_namespace" "ray_system" {
  metadata {
    name = "ray-system"
  }
}

resource "kubernetes_namespace" "ray" {
  metadata {
    name = "ray"
  }
}

# KubeRay 1.1.1 supports Ray >= 2.31 — pinned to match deploy/training/Dockerfile.
locals {
  kuberay_version = "1.1.1"
}

# KubeRay's CRDs are installed OUT OF BAND, not by Helm — see skip_crds below.
#
# The three CRDs are 750KB–1.1MB each (their schemas embed full RayCluster pod
# specs). On Autopilot every write also traverses the Warden admission webhooks,
# so the apiserver needs ~17–27s to process a single one. That is longer than
# the Helm provider's per-request REST timeout, which `helm_release.timeout`
# does NOT control (that governs the post-install resource wait, not individual
# API calls) and which the provider exposes no knob for.
#
# The failure mode is nastier than a plain timeout: Helm installs CRDs with a
# bare Create and treats AlreadyExists as "skip". Here the request times out
# *before* the apiserver reaches its storage-layer conflict check, so Helm never
# sees AlreadyExists and errors out — meaning a retry fails identically even
# once every CRD is present, just on whichever CRD it happens to reach first.
#
# kubectl sets no client-side request timeout, so it simply waits out the ~20s
# and succeeds. Applying the CRDs here and setting skip_crds keeps Helm off that
# path entirely, leaving it only the operator Deployment to install.
resource "null_resource" "kuberay_crds" {
  triggers = {
    kuberay_version = local.kuberay_version
    cluster         = google_container_cluster.ray.id
  }

  provisioner "local-exec" {
    command = <<-EOT
      set -euo pipefail
      gcloud container clusters get-credentials ${google_container_cluster.ray.name} \
        --region ${var.region} --project ${var.project_id}
      tmp=$(mktemp -d)
      trap 'rm -rf "$tmp"' EXIT
      helm pull kuberay-operator \
        --repo https://ray-project.github.io/kuberay-helm/ \
        --version ${local.kuberay_version} --untar --untardir "$tmp"
      # --server-side is required, not stylistic: client-side apply records the
      # object in a last-applied-configuration annotation, and these CRDs are far
      # past the 256KB ceiling on annotation size.
      kubectl apply --server-side --force-conflicts -f "$tmp/kuberay-operator/crds/"
    EOT
  }

  depends_on = [google_container_cluster.ray]
}

resource "helm_release" "kuberay_operator" {
  name       = "kuberay-operator"
  repository = "https://ray-project.github.io/kuberay-helm/"
  chart      = "kuberay-operator"
  version    = local.kuberay_version
  namespace  = kubernetes_namespace.ray_system.metadata[0].name

  # CRDs come from null_resource.kuberay_crds above — see its comment.
  skip_crds = true

  # 15 min, up from the provider default of 300s. This one IS the resource wait:
  # on Autopilot the operator rollout must provision a node from zero, which can
  # take several minutes on a cluster `cloud-up` created moments earlier.
  timeout = 900

  set {
    name  = "batchScheduler.enabled"
    value = "false"
  }

  depends_on = [null_resource.kuberay_crds]
}
