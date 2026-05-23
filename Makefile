.DEFAULT_GOAL := help
SHELL         := bash
MISSION       ?= ESA-Mission1
CHANNEL       ?= channel_1
SUBSYSTEM     ?=

# Detect the Python / uv binary so the Makefile works in CI and local dev.
UV := uv
RUN := $(UV) run

# Auto-detect JDK 21 for PySpark (macOS Homebrew path).
# Override by setting JAVA_HOME externally (e.g. on Linux CI).
JAVA_HOME_21 ?= $(shell brew --prefix openjdk@21 2>/dev/null)
# Prepended to commands that need JDK 21; empty string if not found (tests skip gracefully).
_SPARK_ENV   := $(if $(JAVA_HOME_21),JAVA_HOME=$(JAVA_HOME_21))

.PHONY: help setup test test-all lint format typecheck \
        download-sample explore \
        spark-test spark-preprocess \
        model-train model-score model-evaluate model-test \
        ray-train ray-score ray-tune ray-train-smoke ray-tune-smoke ray-test \
        mlflow-server mlflow-ui mlflow-promote \
        serve \
        frontend-install frontend-dev frontend-build frontend-test \
        docker-build docker-build-ray docker-run-local \
        tf-init tf-plan tf-apply tf-destroy \
        dataproc-preprocess \
        cloud-train cloud-tune \
        seed-reference-profiles \
        smoke-cloud \
        clean clean-processed clean-models clean-data clean-all

help:          ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

setup:         ## Install all dependency groups (dev + spark + tracking + ml)
	$(UV) sync --extra dev --extra spark --extra tracking --extra ml

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

test:          ## Run fast tests (Spark tests skip automatically if JDK 21 absent)
	$(_SPARK_ENV) $(RUN) pytest -m "not slow" -q

test-all:      ## Run the full test suite including slow tests
	$(_SPARK_ENV) $(RUN) pytest -q

spark-test:    ## Run only PySpark tests (requires JDK 21 — brew install openjdk@21)
	$(_SPARK_ENV) $(RUN) pytest tests/spark/ -v

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
# Spark preprocessing (Phase 2)
# ---------------------------------------------------------------------------

spark-preprocess: ## Run Spark preprocessing pipeline on sample data (MISSION=…, SUBSYSTEM=…, CHANNEL=…)
	$(_SPARK_ENV) $(RUN) spacecraft-telemetry spark preprocess \
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

ray-score:        ## Score channels in parallel with Ray (MISSION=…, SUBSYSTEM=…, TUNED_CONFIGS=…)
	$(RUN) spacecraft-telemetry ray score --mission $(MISSION) \
		$(if $(SUBSYSTEM),--subsystem $(SUBSYSTEM),) \
		$(if $(TUNED_CONFIGS),--tuned-configs $(TUNED_CONFIGS),)

ray-tune:         ## Run Ray Tune HPO (MISSION=…, SUBSYSTEM=subsystem_6 for one subsystem)
	$(RUN) spacecraft-telemetry ray tune --mission $(MISSION) \
		$(if $(SUBSYSTEM),--subsystem $(SUBSYSTEM),)

ray-train-smoke:  ## Smoke test: train 1 channel via Ray (fast local check)
	$(RUN) spacecraft-telemetry ray train --mission $(MISSION) --max-channels 1

ray-tune-smoke:   ## Smoke test: tune one subsystem with 3 trials (fast local check)
	$(RUN) spacecraft-telemetry ray tune --mission $(MISSION) --subsystem subsystem_1 --num-samples 3

ray-test:         ## Run Ray training/tuning unit tests (fast only)
	$(RUN) pytest tests/ray_training/ -m "not slow" -v

# ---------------------------------------------------------------------------
# MLflow (Phase 7)
# ---------------------------------------------------------------------------

STAGE         ?= Production
TUNED_CONFIGS ?=

mlflow-server:    ## Start MLflow tracking server — required before parallel training (port 5001)
	$(RUN) spacecraft-telemetry mlflow ui

mlflow-promote:   ## Promote a registered model to STAGE (MISSION=…, CHANNEL=…, STAGE=Production)
	$(RUN) spacecraft-telemetry mlflow promote \
		--name telemanom-$(MISSION)-$(CHANNEL) \
		--stage $(STAGE)

# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------

clean:           ## Remove build artifacts and Python caches (safe, instant)
	rm -rf dist/ .ruff_cache/ .mypy_cache/ .pytest_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

clean-processed: ## Remove Spark output (data/processed/) — re-run spark-preprocess to rebuild
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
AR_REPO        = $(REGION)-docker.pkg.dev/$(PROJECT_ID)/spacecraft-telemetry

docker-build:     ## Build the API serving image locally (IMAGE_TAG=dev)
	docker build -t st-api:$(IMAGE_TAG) .

docker-build-ray:  ## Build the Ray training/tuning image locally (IMAGE_TAG=dev)
	docker build -f deploy/ray/Dockerfile -t st-training:$(IMAGE_TAG) .

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
# GCP one-time operations (Phase 10 — M6)
# ---------------------------------------------------------------------------

# MLFLOW_URL is fetched live so the Makefile works without storing it.
_mlflow_url = $(shell gcloud run services describe mlflow --region $(REGION) --project $(PROJECT_ID) --format='value(status.url)' 2>/dev/null)

dataproc-preprocess: ## Run Spark preprocessing on Dataproc (MISSION=…, PROJECT_ID=…)
	gcloud dataproc workflow-templates instantiate spark-preprocess \
		--region $(REGION) \
		--project $(PROJECT_ID) \
		--parameters MISSION=$(MISSION)

cloud-train:      ## Submit Ray training RayJob to GKE (PROJECT_ID=… REGION=… MISSION=…)
	PROJECT_ID=$(PROJECT_ID) REGION=$(REGION) MLFLOW_URL=$(_mlflow_url) MISSION=$(MISSION) \
		./scripts/cloud_train.sh

cloud-tune:       ## Submit Ray Tune RayJob to GKE (PROJECT_ID=… REGION=… MISSION=…)
	PROJECT_ID=$(PROJECT_ID) REGION=$(REGION) MLFLOW_URL=$(_mlflow_url) MISSION=$(MISSION) \
		./scripts/cloud_tune.sh

seed-reference-profiles: ## Build + upload Evidently reference profiles to GCS (PROJECT_ID=… MISSION=…)
	$(RUN) python scripts/build_reference_profiles.py \
		--env cloud \
		--mission $(MISSION) \
		--channels-from gs://$(PROJECT_ID)-processed-data/$(MISSION)/channels.txt \
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
