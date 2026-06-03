# GCP Architecture Decisions

Key architectural decisions made during Phase 10 (GCP deployment), with rationale.

## Two Cloud Run services (api + mlflow) instead of one

MLflow is a long-running stateful service backed by Cloud SQL; the API is a stateless
inference server that scales to zero. Bundling them would force MLflow's always-on
requirement onto the API (preventing scale-to-zero) or vice versa. Separate services
also let each scale independently and be deployed from different images on different
cadences — MLflow changes rarely, the API rebuilds on every `src/**` push.

## GKE Autopilot over Standard or single-VM

Autopilot eliminates node pool management: no under-provisioned VMs sitting idle between
training runs, no over-provisioned nodes when a job finishes early. The RayJob CRD
autoscaler requests exactly the CPUs/GPUs needed for the current trial count and releases
them immediately on completion. For an infrequent batch workload this is materially
cheaper than a persistent node pool.

## RayJob CRD over manual `ray submit` / port-forward

`kubectl apply -f cluster_train.yaml` submits a self-contained job with known resource
specs, health checks, and TTL cleanup. Port-forwarding to a Ray head node is fragile and
requires a persistent local connection. The CRD also gives structured status
(`jobDeploymentStatus`) that `kubectl wait` can poll, enabling clean `make cloud-train`
shell scripts without busy-looping on log scraping.

## Workload Identity Federation over JSON service-account keys

GitHub Actions authenticates to GCP using short-lived OIDC tokens exchanged via WIF —
no JSON key file ever touches the Actions environment or git history. The WIF binding is
scoped to `refs/heads/main` so PRs cannot authenticate with GCP credentials. This
matches the GCP-recommended keyless CI/CD pattern.

## GCS-at-startup over image-bundled replay data

Bundling 31 GB of ESA Parquet into the API image would make cold starts impractical and
force a full image rebuild whenever new mission data is added. The API reads preprocessed
Parquet from `gs://{project}-processed-data` at startup via gcsfs/UPath, which adds
~5–15s to cold-start time but keeps the image small (~2 GB) and the data lifecycle
independent of the image lifecycle.

## ID-token auth on internal MLflow over basic auth or VPC connector

Cloud Run services are internet-accessible by default when `ingress=INGRESS_TRAFFIC_ALL`.
Basic auth bakes credentials into the image or env vars. A VPC connector adds ~$40/mo
for a fixed private IP. GCP ID tokens (minted via `google.oauth2.id_token.fetch_id_token`
from the attached service account) are zero-cost, rotate automatically every 60 minutes,
and are enforced by Cloud Run's IAM layer — no application-level auth code needed on
the MLflow side. The auth interceptor in `mlflow_tracking/runs.py` installs transparently
whenever the tracking URI ends in `.run.app`.

## Billing kill-switch over Cloud Armor rate limiting

Cloud Armor rate limiting caps requests but doesn't stop Cloud SQL or GKE billing.
A Pub/Sub billing alert → Cloud Function → `billing.projects.updateBillingInfo` pipeline
detaches the billing account if the $150 threshold is hit, which stops ALL compute
billing within minutes. The recovery step (`gcloud beta billing projects link`) is a
one-liner. This is a stronger guarantee for a portfolio project with no on-call rotation.

## `cloud-up` / `cloud-down` lifecycle for the Ray cluster

The GKE Autopilot cluster itself has a ~$0.10/hr management fee even when idle. Wrapping
`helm install kuberay-operator` / `helm uninstall` in `make cloud-up` / `make cloud-down`
Makefile targets means the operator and its namespace only exist during active training
runs. Terraform manages the cluster (which persists cheaply at ~$3/mo for the management
fee) while the operator lifecycle is handled imperatively — this matches the operational
model where the cluster is a long-lived but mostly-idle resource and the operator is
ephemeral.

## Explicit `TUNED` flag on `cloud-score` over GCS bucket sniffing

The original design had `cloud-score` auto-detect `tuned_configs.json` in the artifacts
bucket and apply it when present. This created an implicit dependency on bucket state:
a re-training cycle would silently apply stale HPO params from the previous cycle as the
"baseline", making the HPO comparison meaningless. The explicit `TUNED=1` flag mirrors
local `make ray-score TUNED_CONFIGS=...` semantics — the operator decides which pass is
the baseline and which uses HPO params. Cloud Run services add `TUNED_CONFIGS` via env
var; the Makefile target auto-derives it.

## Single `mlflow-promote` target over separate local/cloud variants

The original `mlflow-promote` / `mlflow-promote-all` split required the operator to
remember which target to use for which environment. Unifying under `ENV=local` (default)
/ `ENV=cloud` flag keeps the mental model consistent with other cloud targets, makes the
help text self-documenting, and removes the duplication of `--env cloud` being hardcoded
into one target.

## Terraform `lifecycle { ignore_changes }` on both Cloud Run service images

Both the `api` and `mlflow` Cloud Run service resources declare `image =
local.placeholder_image` in Terraform so the services can be created on `terraform apply`
before any Docker image exists. After the first CI/CD deploy, the image is owned by
`deploy.yml` — Terraform should never revert it to the placeholder. The `ignore_changes`
lifecycle guard makes this contract explicit and prevents `terraform apply` from
accidentally breaking a running service.

## `SPACECRAFT_ENV` as Click `envvar` on `--env`

The container CMD is `spacecraft-telemetry api serve` with no `--env` flag. Without
`envvar="SPACECRAFT_ENV"` on the Click option, the `SPACECRAFT_ENV=cloud` env var set by
Cloud Run is ignored and the container loads `local.yaml`, silently skipping all
cloud-specific settings (`static_dir`, `cors_allowed_origins`, etc.) that have no
env-var override. Setting `envvar="SPACECRAFT_ENV"` makes the Click option respect the
env var while still allowing explicit `--env cloud` flags in Ray entrypoints to take
precedence (flag > envvar is Click's default priority).

## ESA-Mission1 as the production serving mission

The original plan targeted ESA-Mission2. ESA-Mission1 was chosen instead because the
smoke-test training pipeline was validated end-to-end on Mission1 channels first and all
cloud infrastructure (GCS bucket paths, channels.csv, preprocessed Parquet) was populated
for Mission1. The serving mission is set via `cloud.yaml` (`api.mission`) and
`SPACECRAFT_API__MISSION` env var; switching missions is a Terraform + config change
with no code changes required.
