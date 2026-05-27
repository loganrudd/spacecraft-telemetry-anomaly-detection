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
resource "helm_release" "kuberay_operator" {
  name       = "kuberay-operator"
  repository = "https://ray-project.github.io/kuberay-helm/"
  chart      = "kuberay-operator"
  version    = "1.1.1"
  namespace  = kubernetes_namespace.ray_system.metadata[0].name

  set {
    name  = "batchScheduler.enabled"
    value = "false"
  }

  depends_on = [google_container_cluster.ray]
}
