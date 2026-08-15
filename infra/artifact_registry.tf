resource "google_artifact_registry_repository" "spacecraft" {
  repository_id = "spacecraft-telemetry"
  format        = "DOCKER"
  location      = var.region
  description   = "Docker images for spacecraft-telemetry-anomaly-detection (api, mlflow, training)"

  # Keep the 5 most recent images per package; delete anything older than 30d
  # beyond that. Without this, every push accumulates; Python+PyTorch images are
  # ~2 GB each.
  #
  # Policy semantics (the reason an earlier keep-5 + delete-untagged pair never
  # collected anything): KEEP policies only PROTECT versions — they never delete
  # the remainder. Deletion happens only where a DELETE policy matches and no
  # KEEP policy covers the version. A DELETE rule scoped to UNTAGGED matched
  # zero versions here because CI tags every image with its commit SHA, so
  # untagged versions never exist. The delete rule must therefore use
  # tag_state = "ANY", with the KEEP rules below as the guardrail.
  cleanup_policy_dry_run = false

  cleanup_policies {
    id     = "keep-last-5-tagged"
    action = "KEEP"
    most_recent_versions {
      keep_count = 5
    }
  }

  # Protect the floating `latest` tag. Cloud Run revisions and the Ray cluster
  # images pin by digest, but `latest` is what local/manual pulls resolve to.
  cleanup_policies {
    id     = "keep-latest"
    action = "KEEP"
    condition {
      tag_state    = "TAGGED"
      tag_prefixes = ["latest"]
    }
  }

  cleanup_policies {
    id     = "delete-old"
    action = "DELETE"
    condition {
      tag_state  = "ANY"
      older_than = "2592000s" # 30d
    }
  }

  depends_on = [google_project_service.apis]
}
