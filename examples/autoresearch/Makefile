PYTHON ?= python
UV ?= uv

MODEL ?= gpt-5
GPU ?= L4
PARALLEL ?= 8
MAX_SUBMISSIONS ?= 32
MAX_ITERS ?= 40
CHILD_ITERS ?= 8
CREATED_TRIAL_TIMEOUT_S ?= 1800
SUBMITTED_TRIAL_TIMEOUT_S ?=
EAGER_CHILDREN ?= 1
AGENT_RUNTIME ?= docker
DOCKER_IMAGE ?= rlmflow:local
OUT ?=
NO_LIVE ?=

.PHONY: help sync prepare train run smoke lint

help:
	@echo "Autoresearch targets:"
	@echo "  make sync        Install task deps with uv"
	@echo "  make prepare     Download/tokenize TinyStories"
	@echo "  make train       Run one local train.py experiment"
	@echo "  make run         Run rflow coordinator with Modal trials"
	@echo "  make smoke       Run with MAX_SUBMISSIONS=1 and --no-live"
	@echo "  make lint        Lint runner files"
	@echo ""
	@echo "Common overrides:"
	@echo "  MODEL=$(MODEL) GPU=$(GPU) PARALLEL=$(PARALLEL) MAX_SUBMISSIONS=$(MAX_SUBMISSIONS)"
	@echo "  CREATED_TRIAL_TIMEOUT_S=$(CREATED_TRIAL_TIMEOUT_S) SUBMITTED_TRIAL_TIMEOUT_S=$(SUBMITTED_TRIAL_TIMEOUT_S)"
	@echo "  OUT=/path/to/run NO_LIVE=1"
	@echo "  AGENT_RUNTIME=docker DOCKER_IMAGE=rlmflow:local"

sync:
	$(UV) sync

prepare:
	$(UV) run prepare.py

train:
	$(UV) run train.py

run:
	$(PYTHON) run.py \
		--model "$(MODEL)" \
		--gpu "$(GPU)" \
		--parallel "$(PARALLEL)" \
		--max-submissions "$(MAX_SUBMISSIONS)" \
		--max-iters "$(MAX_ITERS)" \
		--created-trial-timeout-s "$(CREATED_TRIAL_TIMEOUT_S)" \
		--agent-runtime "$(AGENT_RUNTIME)" \
		--docker-image "$(DOCKER_IMAGE)" \
		$(if $(SUBMITTED_TRIAL_TIMEOUT_S),--submitted-trial-timeout-s "$(SUBMITTED_TRIAL_TIMEOUT_S)",) \
		$(if $(OUT),--out "$(OUT)",) \
		$(if $(NO_LIVE),--no-live,)

smoke:
	$(MAKE) run MAX_SUBMISSIONS=1 NO_LIVE=1

lint:
	cd ../.. && $(PYTHON) -m ruff check examples/autoresearch/run.py examples/autoresearch/modal_runner.py examples/autoresearch/prepare.py examples/autoresearch/train.py
