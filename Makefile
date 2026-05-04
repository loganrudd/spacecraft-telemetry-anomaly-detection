.DEFAULT_GOAL := help
SHELL         := bash
MISSION       ?= ESA-Mission1
CHANNEL       ?= channel_1

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
        feast-apply feast-materialize feast-test \
        model-train model-score model-evaluate model-test \
        ray-train ray-score ray-tune ray-train-smoke ray-tune-smoke ray-test \
        mlflow-ui mlflow-promote \
        clean clean-processed clean-models clean-feast clean-data clean-all

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

download-sample: ## Download ESA dataset sample from Zenodo (MISSION=ESA-Mission1)
	$(RUN) spacecraft-telemetry download \
		--mission $(MISSION) \
		--sample

explore:       ## Print dataset exploration report (MISSION=ESA-Mission1)
	$(RUN) spacecraft-telemetry explore \
		--mission $(MISSION)

# ---------------------------------------------------------------------------
# Spark preprocessing (Phase 2)
# ---------------------------------------------------------------------------

spark-preprocess: ## Run Spark preprocessing pipeline on sample data (MISSION=ESA-Mission1)
	$(_SPARK_ENV) $(RUN) spacecraft-telemetry spark preprocess \
		--mission $(MISSION)

# ---------------------------------------------------------------------------
# Feast feature store (Phase 3)
# ---------------------------------------------------------------------------

feast-apply:      ## Register Feast definitions to local registry
	$(RUN) spacecraft-telemetry feast apply --mission $(MISSION)

feast-materialize: ## Materialize features to online store — incremental (MISSION=ESA-Mission1)
	$(RUN) spacecraft-telemetry feast materialize --mission $(MISSION)

feast-test:       ## Run only Feast tests
	$(RUN) python -m pytest tests/feast_client/ -v



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

ray-train:        ## Train channels in parallel with Ray (MISSION=…)
	$(RUN) spacecraft-telemetry ray train --mission $(MISSION)

ray-score:        ## Score channels in parallel with Ray (MISSION=…)
	$(RUN) spacecraft-telemetry ray score --mission $(MISSION)

ray-tune:         ## Run Ray Tune HPO (all discovered subsystems)
	$(RUN) spacecraft-telemetry ray tune --mission $(MISSION)

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

mlflow-ui:        ## Open MLflow UI against the local SQLite tracking store (port 5001)
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

clean-feast:     ## Wipe Feast local registry and online store
	rm -rf feature_repo/data/

clean-data:      ## Remove downloaded raw + sample data — requires re-running download-sample
	rm -rf data/raw/ data/sample/

clean-all:       ## Remove everything: caches + processed + models + Feast + downloaded data
	$(MAKE) clean clean-processed clean-models clean-feast clean-data
