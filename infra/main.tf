provider "google" {
  project = var.project_id
  region  = var.region
}

# Needed to resolve the numeric project number for Direct WIF principal URIs.
data "google_project" "project" {}

provider "google-beta" {
  project = var.project_id
  region  = var.region
}

# Helm and Kubernetes providers are configured in kuberay.tf because they need
# the GKE cluster endpoint, which is only known after `terraform apply
# -target=google_container_cluster.ray`.  See kuberay.tf for the two-step
# apply note.

# Enable all required GCP APIs.  disable_on_destroy=false avoids breaking the
# project if infra is torn down and recreated.
resource "google_project_service" "apis" {
  for_each = toset([
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "sqladmin.googleapis.com",
    "secretmanager.googleapis.com",
    "iamcredentials.googleapis.com",
    "sts.googleapis.com",
    "cloudbilling.googleapis.com",
    "billingbudgets.googleapis.com",
    "container.googleapis.com",
    "compute.googleapis.com",
    "dataproc.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "iam.googleapis.com",
  ])

  service            = each.value
  disable_on_destroy = false
}
