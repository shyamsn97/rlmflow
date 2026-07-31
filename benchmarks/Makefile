.PHONY: help eval-help eval-smoke eval-test eval-run eval-wandb eval-benchmark eval-clean
.DEFAULT_GOAL := help

# Commands run from the repo root because the harness uses absolute imports
# (`from benchmarks.eval ...`). This lets `make -C benchmarks ...` or
# `cd benchmarks && make ...` both work.
REPO_ROOT := $(realpath $(dir $(realpath $(firstword $(MAKEFILE_LIST))))/..)

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z0-9_/-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

EVAL_PROVIDER ?= openai
EVAL_MODEL ?= gpt-5-mini
EVAL_DATASETS ?= oolong
EVAL_RUNNERS ?= vanilla rlmflow-local official-rlm
EVAL_SEEDS ?= 0:10
EVAL_BENCHMARK_DATASETS_FULL ?= oolong official_browsecomp official_longbench_v2 official_livecodebench official_sudoku_extreme
EVAL_BENCHMARK_DATASETS ?= oolong official_browsecomp official_longbench_v2 official_livecodebench official_sudoku_extreme
EVAL_BENCHMARK_RUNNERS ?= rlmflow-local vanilla official-rlm
EVAL_BENCHMARK_SEEDS ?= 0:20
EVAL_MAX_ITERS ?= 20
EVAL_MAX_DEPTH ?= 1
EVAL_INCLUDE_LLM_QUERY ?= false
EVAL_EXECUTOR ?= local
EVAL_PARALLEL ?= 1
EVAL_BEST_OF_N ?= 1
EVAL_ATTEMPT_TIMEOUT ?= 900
EVAL_MODAL_APP_NAME ?= rlmflow-benchmarks
EVAL_MODAL_CPU ?= 1
EVAL_MODAL_TIMEOUT ?= 3600
EVAL_OUT_DIR ?= benchmarks/eval/runs
EVAL_DATASET_PARAMS ?=
EVAL_RUNNER_PARAMS ?= --runner-param rlmflow-local.max_iters=$(EVAL_MAX_ITERS) --runner-param rlmflow-local.max_depth=$(EVAL_MAX_DEPTH) --runner-param rlmflow-local.use_llm_query=$(EVAL_INCLUDE_LLM_QUERY)
EVAL_ARGS ?=
EVAL_WANDB_PROJECT ?= rlmflow-eval
EVAL_WANDB_ENTITY ?=
EVAL_WANDB_ENTITY_ARG := $(if $(EVAL_WANDB_ENTITY),--wandb-entity $(EVAL_WANDB_ENTITY),)

help: ## Show this help message.
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

eval-help: ## Show eval harness CLI help.
	cd $(REPO_ROOT) && python -m benchmarks.eval --help

eval-smoke: ## Run local fake eval smoke across fake/vanilla/rlmflow.
	cd $(REPO_ROOT) && python -m benchmarks.eval \
		--model fake \
		--dataset synthetic_needle \
		--runner fake vanilla rlmflow-local \
		--seeds 0:3 \
		--dataset-param synthetic_needle.records=8 \
		--dataset-param synthetic_needle.filler_words=2 \
		--runner-param rlmflow-local.max_iters=3 \
		--runner-param rlmflow-local.max_depth=1 \
		--runner-param rlmflow-local.live_save=false \
		--out-dir $(EVAL_OUT_DIR) \
		$(EVAL_ARGS)

eval-test: eval-smoke ## Alias for the deterministic local eval smoke.

eval-run: ## Run real eval harness. Override EVAL_PROVIDER/MODEL/TASKS/RUNNERS/SEEDS/ARGS.
	cd $(REPO_ROOT) && python -m benchmarks.eval \
		--model $(EVAL_PROVIDER):$(EVAL_MODEL) \
		--dataset $(EVAL_DATASETS) \
		--runner $(EVAL_RUNNERS) \
		--seeds $(EVAL_SEEDS) \
		--out-dir $(EVAL_OUT_DIR) \
		--executor $(EVAL_EXECUTOR) \
		--parallel $(EVAL_PARALLEL) \
		--best-of-n $(EVAL_BEST_OF_N) \
		--attempt-timeout $(EVAL_ATTEMPT_TIMEOUT) \
		--modal-app-name $(EVAL_MODAL_APP_NAME) \
		--modal-cpu $(EVAL_MODAL_CPU) \
		--modal-timeout $(EVAL_MODAL_TIMEOUT) \
		$(EVAL_DATASET_PARAMS) \
		$(EVAL_RUNNER_PARAMS) \
		$(EVAL_ARGS)

eval-wandb: ## Run real eval harness with W&B logging enabled.
	cd $(REPO_ROOT) && python -m benchmarks.eval \
		--model $(EVAL_PROVIDER):$(EVAL_MODEL) \
		--dataset $(EVAL_DATASETS) \
		--runner $(EVAL_RUNNERS) \
		--seeds $(EVAL_SEEDS) \
		--out-dir $(EVAL_OUT_DIR) \
		--executor $(EVAL_EXECUTOR) \
		--parallel $(EVAL_PARALLEL) \
		--best-of-n $(EVAL_BEST_OF_N) \
		--attempt-timeout $(EVAL_ATTEMPT_TIMEOUT) \
		--modal-app-name $(EVAL_MODAL_APP_NAME) \
		--modal-cpu $(EVAL_MODAL_CPU) \
		--modal-timeout $(EVAL_MODAL_TIMEOUT) \
		--wandb \
		--wandb-project $(EVAL_WANDB_PROJECT) \
		$(EVAL_WANDB_ENTITY_ARG) \
		$(EVAL_DATASET_PARAMS) \
		$(EVAL_RUNNER_PARAMS) \
		$(EVAL_ARGS)

eval-benchmark:
	cd $(REPO_ROOT) && python -m benchmarks.eval \
		--model $(EVAL_PROVIDER):$(EVAL_MODEL) \
		--dataset $(EVAL_BENCHMARK_DATASETS) \
		--runner $(EVAL_BENCHMARK_RUNNERS) \
		--seeds $(EVAL_BENCHMARK_SEEDS) \
		--out-dir $(EVAL_OUT_DIR) \
		--executor $(EVAL_EXECUTOR) \
		--parallel $(EVAL_PARALLEL) \
		--best-of-n $(EVAL_BEST_OF_N) \
		--attempt-timeout $(EVAL_ATTEMPT_TIMEOUT) \
		--modal-app-name $(EVAL_MODAL_APP_NAME) \
		--modal-cpu $(EVAL_MODAL_CPU) \
		--modal-timeout $(EVAL_MODAL_TIMEOUT) \
		--wandb \
		--wandb-project $(EVAL_WANDB_PROJECT) \
		$(EVAL_WANDB_ENTITY_ARG) \
		$(EVAL_DATASET_PARAMS) \
		$(EVAL_RUNNER_PARAMS) \
		$(EVAL_ARGS)

eval-benchmark-oolong:
	cd $(REPO_ROOT) && python -m benchmarks.eval \
		--model $(EVAL_PROVIDER):$(EVAL_MODEL) \
		--dataset oolong \
		--runner $(EVAL_BENCHMARK_RUNNERS) \
		--seeds 0:20 \
		--out-dir $(EVAL_OUT_DIR) \
		--executor $(EVAL_EXECUTOR) \
		--parallel $(EVAL_PARALLEL) \
		--best-of-n $(EVAL_BEST_OF_N) \
		--attempt-timeout $(EVAL_ATTEMPT_TIMEOUT) \
		--modal-app-name $(EVAL_MODAL_APP_NAME) \
		--modal-cpu $(EVAL_MODAL_CPU) \
		--modal-timeout $(EVAL_MODAL_TIMEOUT) \
		$(EVAL_DATASET_PARAMS) \
		$(EVAL_RUNNER_PARAMS) \
		$(EVAL_ARGS)

eval-clean: ## Remove local eval run artifacts.
	rm -rf $(REPO_ROOT)/benchmarks/eval/runs/
