.DEFAULT_GOAL := help
SHELL         := bash
MISSION       ?= ESA-Mission1
CHANNEL       ?=
SUBSYSTEM     ?=

# Detect the Python / uv binary so the Makefile works in CI and local dev.
UV := uv
RUN := $(UV) run

.PHONY: help setup test test-all lint format typecheck \
        download-sample explore \
        profile preprocess \
        model-train model-score model-evaluate model-test \
        ray-train ray-score ray-tune ray-train-smoke ray-tune-smoke ray-test \
        mlflow-server mlflow-ui mlflow-promote mlflow-promote-all cloud-deploy \
        serve \
        frontend-install frontend-dev frontend-build frontend-test \
        docker-build docker-build-ray docker-run-local \
        tf-init tf-plan tf-apply tf-destroy \
        cloud-up cloud-down cloud-db-start cloud-db-stop \
        cloud-preprocess cloud-train cloud-tune cloud-score \
        seed-reference-profiles \
        smoke-cloud \
        clean clean-processed clean-models clean-data clean-all

help:          ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

setup:         ## Install all dependency groups (dev + tracking + ml)
	$(UV) sync --extra dev --extra tracking --extra ml

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

test:          ## Run fast tests
	$(RUN) pytest -m "not slow" -q

test-all:      ## Run the full test suite including slow tests
	$(RUN) pytest -q

# ---------------------------------------------------------------------------
# Code quality
# ---------------------------------------------------------------------------

lint:          ## Check code with ruff (no fixes)
	$(RUN) ruff check src/ tests/

format:        ## Auto-format with ruff
	$(RUN) ruff format src/ tests/
	$(RUN) ruff check --fix src/ tests/

typecheck:     ## Run mypy type checker
	$(RUN) mypy src/

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

download-sample: ## Download ESA dataset sample from Zenodo (MISSION=…, SUBSYSTEM=…, CHANNEL=…)
	$(RUN) spacecraft-telemetry download \
		--mission $(MISSION) \
		--sample \
		$(if $(filter command line,$(origin CHANNEL)),--channel $(CHANNEL),) \
		$(if $(SUBSYSTEM),--subsystem $(SUBSYSTEM),)

explore:       ## Print dataset exploration report (MISSION=ESA-Mission1)
	$(RUN) spacecraft-telemetry explore \
		--mission $(MISSION)

# ---------------------------------------------------------------------------
# Preprocessing (Phase 10.5 — pandas + Ray)
# ---------------------------------------------------------------------------

profile:          ## Profile raw channel suitability, write channel_suitability.json (MISSION=…)
	$(RUN) spacecraft-telemetry preprocess profile --mission $(MISSION)

preprocess:       ## Run pandas + Ray preprocessing pipeline on sample data (MISSION=…, SUBSYSTEM=…, CHANNEL=…)
	$(RUN) spacecraft-telemetry preprocess run \
		--mission $(MISSION) \
		$(if $(filter command line,$(origin CHANNEL)),--channel $(CHANNEL),) \
		$(if $(SUBSYSTEM),--subsystem $(SUBSYSTEM),)

# ---------------------------------------------------------------------------
# Telemanom model (Phase 4)
# ---------------------------------------------------------------------------

model-train:      ## Train Telemanom LSTM on one channel (MISSION=…, CHANNEL=channel_1)
	$(RUN) spacecraft-telemetry model train \
		--mission $(MISSION) --channel $(CHANNEL)

model-score:      ## Score a trained model against its test split (MISSION=…, CHANNEL=…)
	$(RUN) spacecraft-telemetry model score \
		--mission $(MISSION) --channel $(CHANNEL)

model-evaluate:   ## Train + score end-to-end (Phase 4 single-channel demo)
	$(RUN) spacecraft-telemetry model train \
		--mission $(MISSION) --channel $(CHANNEL) && \
	$(RUN) spacecraft-telemetry model score \
		--mission $(MISSION) --channel $(CHANNEL)

model-test:       ## Run only model tests (fast; excludes @pytest.mark.slow)
	$(RUN) pytest tests/model/ -m "not slow" -v

# ---------------------------------------------------------------------------
# Ray parallel training (Phase 5)
# ---------------------------------------------------------------------------

ray-train:        ## Train channels in parallel with Ray (MISSION=…, SUBSYSTEM=subsystem_6)
	$(RUN) spacecraft-telemetry ray train --mission $(MISSION) \
		$(if $(SUBSYSTEM),--subsystem $(SUBSYSTEM),)

# TUNED=1 auto-derives the canonical tuned_configs path (mirrors cloud-score).
# TUNED_CONFIGS=<path> is an explicit override that takes precedence.
_TUNED_PATH := $(if $(TUNED_CONFIGS),$(TUNED_CONFIGS),$(if $(filter 1,$(TUNED)),models/$(MISSION)/tuned_configs.json,))
ray-score:        ## Score channels in parallel with Ray (MISSION=…, SUBSYSTEM=…, [TUNED=1 | TUNED_CONFIGS=…])
	$(RUN) spacecraft-telemetry ray score --mission $(MISSION) \
		$(if $(SUBSYSTEM),--subsystem $(SUBSYSTEM),) \
		$(if $(_TUNED_PATH),--tuned-configs $(_TUNED_PATH),)

ray-tune:         ## Run Ray Tune HPO (MISSION=…, SUBSYSTEM=subsystem_6 for one subsystem)
	$(RUN) spacecraft-telemetry ray tune --mission $(MISSION) \
		$(if $(SUBSYSTEM),--subsystem $(SUBSYSTEM),)

ray-train-smoke:  ## Smoke test: train 1 channel via Ray (fast local check)
	$(RUN) spacecraft-telemetry ray train --mission $(MISSION) --max-channels 1

ray-tune-smoke:   ## Smoke test: tune one subsystem with 3 trials (fast local check)
	$(RUN) spacecraft-telemetry ray tune --mission $(MISSION) --subsystem subsystem_1 --num-samples 3

ray-test:         ## Run Ray training/tuning unit tests (fast only)
	$(RUN) pytest tests/ray_fanout/ -m "not slow" -v

# ---------------------------------------------------------------------------
# MLflow (Phase 7)
# ---------------------------------------------------------------------------

STAGE         ?= Production
TUNED_CONFIGS ?=
TUNED         ?=
ENV           ?= local

mlflow-server:    ## Start MLflow tracking server — required before parallel training (port 5001)
	$(RUN) spacecraft-telemetry mlflow ui

mlflow-promote:   ## Set @champion alias (MISSION=…, [CHANNEL=…, SUBSYSTEM=…, ENV=cloud])
	$(if $(filter cloud,$(ENV)), \
	  SSL_CERT_FILE=$(SSL_CERT_FILE) \
	  SPACECRAFT_MLFLOW__TRACKING_URI=$$(gcloud run services describe mlflow --region $(REGION) --project $(PROJECT_ID) --format='value(status.url)') \
	  MLFLOW_TRACKING_TOKEN=$$(gcloud auth print-identity-token) \
	  SPACECRAFT_PREPROCESS__PROCESSED_DATA_DIR=gs://$(PROJECT_ID)-processed-data \
	  SPACECRAFT_DATA__SAMPLE_DATA_DIR=gs://$(PROJECT_ID)-sample-data,) \
	$(RUN) spacecraft-telemetry --env $(ENV) mlflow promote \
		--mission $(MISSION) \
		$(if $(CHANNEL),--channels $(CHANNEL),) \
		$(if $(SUBSYSTEM),--subsystem $(SUBSYSTEM),)

cloud-deploy:     ## Redeploy Cloud Run API so it cold-starts and loads the newly promoted Production model
	$(eval _API_IMAGE := $(shell gcloud run services describe api \
		--region $(REGION) --project $(PROJECT_ID) \
		--format='value(spec.template.spec.containers[0].image)' 2>/dev/null))
	@if [ -z "$(_API_IMAGE)" ]; then echo "ERROR: Cloud Run service 'api' not found. Run terraform apply first."; exit 1; fi
	gcloud run deploy api \
		--image $(_API_IMAGE) \
		--region $(REGION) \
		--project $(PROJECT_ID)

# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------

clean:           ## Remove build artifacts and Python caches (safe, instant)
	rm -rf dist/ .ruff_cache/ .mypy_cache/ .pytest_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

clean-processed: ## Remove preprocessed data (data/processed/) — re-run preprocess to rebuild
	rm -rf data/processed/

clean-models:    ## Remove trained model artifacts (models/) — re-run model-train to rebuild
	rm -rf models/

clean-data:      ## Remove downloaded raw + sample data — requires re-running download-sample
	rm -rf data/raw/ data/sample/

clean-all:       ## Remove everything: caches + processed + models + downloaded data
	$(MAKE) clean clean-processed clean-models clean-data

# ---------------------------------------------------------------------------
# FastAPI serving (Phase 8)
# ---------------------------------------------------------------------------

serve:            ## Start the FastAPI serving layer locally (SUBSYSTEM=subsystem_6)
	$(RUN) spacecraft-telemetry --env local api serve \
		$(if $(SUBSYSTEM),--subsystem $(SUBSYSTEM),)

# ---------------------------------------------------------------------------
# React dashboard (Phase 9)
# ---------------------------------------------------------------------------

frontend-install: ## Install dashboard npm dependencies
	cd frontend && npm install

frontend-dev:     ## Run the Vite dev server on :5173 (requires `make serve` in another terminal)
	cd frontend && npm run dev

frontend-build:   ## Build the dashboard static bundle to frontend/dist/
	cd frontend && npm run build

frontend-test:    ## Run dashboard unit tests (Vitest)
	cd frontend && npm run test

# ---------------------------------------------------------------------------
# Docker (Phase 10)
# ---------------------------------------------------------------------------

IMAGE_TAG     ?= dev
PROJECT_ID    ?=
REGION        ?= us-central1
# Python (via uv) uses its own CA bundle and ignores the macOS system keychain.
# Pointing SSL_CERT_FILE at certifi fixes HTTPS to Cloud Run / GCS without
# requiring a system-level certificate install.
SSL_CERT_FILE ?= $(shell uv run python -m certifi 2>/dev/null)
AR_REPO        = $(REGION)-docker.pkg.dev/$(PROJECT_ID)/spacecraft-telemetry

docker-build:     ## Build the API serving image locally (IMAGE_TAG=dev)
	docker build -f deploy/api/Dockerfile -t st-api:$(IMAGE_TAG) .

docker-build-ray:  ## Build the Ray image locally (IMAGE_TAG=dev)
	docker build -f deploy/ray/Dockerfile -t st-ray:$(IMAGE_TAG) .

docker-run-local: ## Run the API container locally against local MLflow (IMAGE_TAG=dev)
	docker run --rm -p 8080:8080 \
		-e SPACECRAFT_ENV=local \
		-e SPACECRAFT_MLFLOW__TRACKING_URI=http://host.docker.internal:5001 \
		-v "$(PWD)/models:/app/models" \
		-v "$(PWD)/data:/app/data" \
		st-api:$(IMAGE_TAG)

# ---------------------------------------------------------------------------
# Terraform (Phase 10)
# ---------------------------------------------------------------------------

tf-init:          ## Initialize Terraform providers (run once per checkout)
	cd infra && terraform init

tf-plan:          ## Show Terraform plan (PROJECT_ID=… BILLING_ACCOUNT=… required)
	cd infra && terraform plan \
		-var="project_id=$(PROJECT_ID)" \
		-var="billing_account=$(BILLING_ACCOUNT)"

tf-apply:         ## Apply Terraform changes (PROJECT_ID=… BILLING_ACCOUNT=… required)
	cd infra && terraform apply \
		-var="project_id=$(PROJECT_ID)" \
		-var="billing_account=$(BILLING_ACCOUNT)"

tf-destroy:       ## Destroy all Terraform-managed resources (irreversible)
	cd infra && terraform destroy \
		-var="project_id=$(PROJECT_ID)" \
		-var="billing_account=$(BILLING_ACCOUNT)"

# ---------------------------------------------------------------------------
# Cloud session lifecycle — run cloud-up before training, cloud-down after
# ---------------------------------------------------------------------------
# cloud-up:   starts Cloud SQL + provisions GKE + installs KubeRay (~10 min)
# cloud-down: destroys GKE + NAT to eliminate idle billing (leaves Cloud SQL up)
#
# Cloud SQL is intentionally decoupled from cloud-down: MLflow + the API
# (Cloud Run, scale-to-zero) depend on it, so stopping it breaks deploys, the
# smoke test, and the live demo. The db-f1-micro tier costs ~$10/month running
# 24/7 (~$8/month more than stopped, since 10GB storage is billed either way) —
# cheap enough to leave on. The expensive infra is GKE Autopilot + GPU VMs,
# which is what cloud-down tears down. Use cloud-db-stop only for long breaks
# with no deploys or demos.

cloud-db-start:   ## Start Cloud SQL (mlflow-pg) and wait for RUNNABLE. Needed for MLflow/API.
	gcloud sql instances patch mlflow-pg --activation-policy=ALWAYS --async --quiet
	@echo "Waiting for Cloud SQL to reach RUNNABLE state..."
	@until [ "$$(gcloud sql instances describe mlflow-pg --format='value(state)' --quiet)" = "RUNNABLE" ]; do \
		sleep 10; echo "  still waiting..."; \
	done
	@echo "Cloud SQL is RUNNABLE."

cloud-db-stop:    ## Stop Cloud SQL (mlflow-pg). Breaks MLflow/API — only for long idle breaks.
	gcloud sql instances patch mlflow-pg --activation-policy=NEVER --async --quiet
	@echo "Cloud SQL stop requested. MLflow + API will fail until cloud-db-start."

cloud-up: cloud-db-start  ## Start Cloud SQL + provision GKE. Run before cloud-preprocess/train/tune.
	terraform -chdir=infra apply \
		-target=google_container_cluster.ray \
		-target=google_compute_router.nat_router \
		-target=google_compute_router_nat.nat \
		-refresh=false \
		-auto-approve
	terraform -chdir=infra apply \
		-target=kubernetes_namespace.ray_system \
		-target=kubernetes_namespace.ray \
		-target=helm_release.kuberay_operator \
		-target=kubernetes_service_account.ray \
		-refresh=false \
		-auto-approve
	terraform -chdir=infra apply \
		-target=google_service_account_iam_member.ray_workload_identity \
		-target=google_storage_bucket_iam_member.ray_sample_viewer \
		-target=google_storage_bucket_iam_member.ray_processed_admin \
		-target=google_storage_bucket_iam_member.ray_artifacts_admin \
		-target=google_storage_bucket_iam_member.ray_wif_sample_viewer \
		-target=google_storage_bucket_iam_member.ray_wif_processed_admin \
		-target=google_storage_bucket_iam_member.ray_wif_artifacts_admin \
		-refresh=false \
		-auto-approve
	gcloud container clusters get-credentials ray-cluster --region=$(REGION) --project=$(PROJECT_ID)

cloud-down:       ## Destroy GKE + NAT to stop training billing. Leaves Cloud SQL up (see cloud-db-stop).
	terraform -chdir=infra destroy \
		-target=helm_release.kuberay_operator \
		-target=kubernetes_namespace.ray_system \
		-target=kubernetes_namespace.ray \
		-auto-approve
	@# Block until the ray namespace has fully terminated before destroying the
	@# cluster. Namespace deletion waits on KubeRay finalizers and can lag the
	@# terraform destroy return; a later cloud-up then races a still-Terminating
	@# namespace and fails with "being terminated". Must run while the cluster
	@# still exists (kubectl is unreachable once the cluster is gone).
	kubectl wait --for=delete namespace/ray --timeout=300s || true
	terraform -chdir=infra destroy \
		-target=google_container_cluster.ray \
		-auto-approve
	terraform -chdir=infra destroy \
		-target=google_compute_router_nat.nat \
		-auto-approve
	terraform -chdir=infra destroy \
		-target=google_compute_router.nat_router \
		-auto-approve
	@echo "GKE destroyed. Cloud NAT destroyed. Cloud SQL left RUNNING for MLflow/API."
	@echo "To stop Cloud SQL as well (breaks deploys/demos): make cloud-db-stop"

# ---------------------------------------------------------------------------
# GCP one-time operations (Phase 10 — M6)
# ---------------------------------------------------------------------------

# MLFLOW_URL is fetched live so the Makefile works without storing it.
_mlflow_url = $(shell gcloud run services describe mlflow --region $(REGION) --project $(PROJECT_ID) --format='value(status.url)' 2>/dev/null)

cloud-preprocess: ## Submit preprocessing RayJob to GKE (PROJECT_ID=… REGION=… MISSION=…)
	PROJECT_ID=$(PROJECT_ID) REGION=$(REGION) MISSION=$(MISSION) \
		./scripts/cloud_preprocess.sh

cloud-train:      ## Submit Ray training RayJob to GKE (PROJECT_ID=… REGION=… MISSION=… [CHANNELS=ch1,ch2 | CHANNELS_FROM=gs://…] [NUM_GPUS=1])
	PROJECT_ID=$(PROJECT_ID) REGION=$(REGION) MLFLOW_URL=$(_mlflow_url) MISSION=$(MISSION) \
		CHANNELS=$(CHANNELS) CHANNELS_FROM=$(CHANNELS_FROM) NUM_GPUS=$(NUM_GPUS) \
		./scripts/cloud_train.sh

cloud-tune:       ## Submit Ray Tune RayJob to GKE (PROJECT_ID=… REGION=… MISSION=…)
	PROJECT_ID=$(PROJECT_ID) REGION=$(REGION) MLFLOW_URL=$(_mlflow_url) MISSION=$(MISSION) \
		./scripts/cloud_tune.sh

cloud-score:      ## Score models on GKE (PROJECT_ID=… REGION=… MISSION=… [TUNED=1] [NUM_GPUS=0.2] [EVAL_SPLIT=full_test]). Baseline by default; TUNED=1 applies HPO params (run after cloud-tune).
	PROJECT_ID=$(PROJECT_ID) REGION=$(REGION) MLFLOW_URL=$(_mlflow_url) MISSION=$(MISSION) TUNED=$(TUNED) NUM_GPUS=$(NUM_GPUS) EVAL_SPLIT=$(EVAL_SPLIT) \
		./scripts/cloud_score.sh

seed-reference-profiles: ## Build + upload Evidently reference profiles to GCS (PROJECT_ID=… MISSION=…)
	SSL_CERT_FILE=$$($(RUN) python -m certifi) \
	SPACECRAFT_PREPROCESS__PROCESSED_DATA_DIR=gs://$(PROJECT_ID)-processed-data \
	$(RUN) python scripts/build_reference_profiles.py \
		--env cloud \
		--mission $(MISSION) \
		--upload gs://$(PROJECT_ID)-artifacts/reference_profiles \
		--upload-only

smoke-cloud:      ## Smoke-test the deployed API (PROJECT_ID=… REGION=…)
	@API_URL=$$(gcloud run services describe api --region $(REGION) --project $(PROJECT_ID) \
		--format='value(status.url)'); \
	echo "==> GET $${API_URL}/health"; \
	for i in 1 2 3 4 5; do \
		STATUS=$$(curl -fsS "$${API_URL}/health" 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))" 2>/dev/null); \
		[ "$$STATUS" = "ok" ] && echo "==> API status: ok" && exit 0; \
		echo "  attempt $$i/5 — retrying in 12s..."; sleep 12; \
	done; \
	echo "ERROR: API did not return status=ok after 5 attempts"; exit 1
