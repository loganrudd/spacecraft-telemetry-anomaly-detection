# Workload Identity Federation — keyless auth for GitHub Actions.
# The deployer SA is only accessible from pushes to refs/heads/main in the
# configured repo.  PRs and other branches get no GCP credentials.

resource "google_iam_workload_identity_pool" "github" {
  workload_identity_pool_id = "github-actions"
  display_name              = "GitHub Actions"
  description               = "WIF pool for spacecraft-telemetry CI/CD"

  depends_on = [google_project_service.apis]
}

resource "google_iam_workload_identity_pool_provider" "github" {
  workload_identity_pool_id          = google_iam_workload_identity_pool.github.workload_identity_pool_id
  workload_identity_pool_provider_id = "github-provider"
  display_name                       = "GitHub Actions OIDC"

  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }

  attribute_mapping = {
    "google.subject"       = "assertion.sub"
    "attribute.actor"      = "assertion.actor"
    "attribute.repository" = "assertion.repository"
    "attribute.ref"        = "assertion.ref"
  }

  # Restrict to this specific repository — prevents other GitHub repos from
  # using this pool even if they somehow obtained the provider ID.
  attribute_condition = "assertion.repository == '${var.github_repo}'"
}

# Bind deployer SA → WIF principal set scoped to refs/heads/main only.
# Workflow dispatch and PRs from forks cannot impersonate this SA.
resource "google_service_account_iam_member" "deployer_wif" {
  service_account_id = google_service_account.deployer.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.github.name}/attribute.ref/refs/heads/main"
}

# Also allow the long-lived ISS feature branch. Phases 12-18 build the collector
# image from iss_ext, well before merge to main, so it needs to push to Artifact
# Registry. Scoped to this exact ref — PRs and other branches still get nothing.
# NOTE: this grants iss_ext pushers the full deployer SA (same as main); acceptable
# for a single-maintainer repo. Tighten to a push-only SA if collaborators are added.
resource "google_service_account_iam_member" "deployer_wif_iss_ext" {
  service_account_id = google_service_account.deployer.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.github.name}/attribute.ref/refs/heads/iss_ext"
}
