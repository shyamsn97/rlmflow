.PHONY: help eval-help eval-smoke eval-test eval-run eval-wandb eval-benchmark eval-benchmark-full eval-benchmark-oolong eval-delegation eval-delegation-five eval-delegation-five-compare eval-delegation-five-matrix eval-delegation-ten-compare eval-delegation-regression eval-delegation-canary eval-delegation-phase1 eval-delegation-official eval-delegation-report eval-clean
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
EVAL_SEEDS ?= 0
EVAL_BENCHMARK_DATASETS ?= rlm-core
EVAL_BENCHMARK_RUNNERS ?= rlmflow-local vanilla official-rlm
EVAL_BENCHMARK_SEEDS ?= 0
EVAL_SAMPLE_ARGS ?= --limit 5
EVAL_MAX_ITERS ?= 20
EVAL_CHILD_MAX_ITERS ?= 20
EVAL_MAX_DEPTH ?= 2
EVAL_MAX_BUDGET ?= 100000
EVAL_INCLUDE_LLM_QUERY ?= true
EVAL_EXECUTOR ?= local
EVAL_PARALLEL ?= 1
EVAL_BEST_OF_N ?= 1
EVAL_ATTEMPT_TIMEOUT ?= 900
EVAL_MODAL_APP_NAME ?= rlmflow-benchmarks
EVAL_MODAL_CPU ?= 1
EVAL_MODAL_TIMEOUT ?= 3600
EVAL_OUT_DIR ?= benchmarks/eval/runs
EVAL_DATASET_PARAMS ?=
EVAL_RUNNER_PARAMS ?= --runner-param rlmflow-local.max_iters=$(EVAL_MAX_ITERS) --runner-param rlmflow-local.child_max_iters=$(EVAL_CHILD_MAX_ITERS) --runner-param rlmflow-local.max_depth=$(EVAL_MAX_DEPTH) --runner-param rlmflow-local.max_budget=$(EVAL_MAX_BUDGET) --runner-param rlmflow-local.use_llm_query=$(EVAL_INCLUDE_LLM_QUERY) --runner-param official-rlm.max_iters=$(EVAL_MAX_ITERS) --runner-param official-rlm.max_depth=$(EVAL_MAX_DEPTH) --runner-param official-rlm.max_budget=$(EVAL_MAX_BUDGET)
EVAL_ARGS ?=
EVAL_WANDB_PROJECT ?= rlmflow-eval
EVAL_WANDB_ENTITY ?=
EVAL_DELEGATION_CONDITION ?= current_policy
EVAL_DELEGATION_DATASET ?= delegation-suite
EVAL_DELEGATION_SAMPLE_ARGS ?= --full
EVAL_RLMFLOW_RUN ?=
EVAL_OFFICIAL_RUN ?=
EVAL_COMPARISON_OUT ?= benchmarks/eval/comparisons/latest
EVAL_RUN_ID ?=
EVAL_RESUME ?= false
EVAL_WANDB_ENTITY_ARG := $(if $(EVAL_WANDB_ENTITY),--wandb-entity $(EVAL_WANDB_ENTITY),)
EVAL_RUN_ID_ARG := $(if $(EVAL_RUN_ID),--run-id $(EVAL_RUN_ID),)
EVAL_RESUME_ARG := $(if $(filter true,$(EVAL_RESUME)),--resume,)

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
		$(EVAL_SAMPLE_ARGS) \
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
		$(EVAL_SAMPLE_ARGS) \
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

eval-benchmark: ## Run a five-example canary of the minimal RLM comparison suite.
	cd $(REPO_ROOT) && python -m benchmarks.eval \
		--model $(EVAL_PROVIDER):$(EVAL_MODEL) \
		--dataset $(EVAL_BENCHMARK_DATASETS) \
		--runner $(EVAL_BENCHMARK_RUNNERS) \
		--seeds $(EVAL_BENCHMARK_SEEDS) \
		$(EVAL_SAMPLE_ARGS) \
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

eval-benchmark-full: ## Run every exposed example in the minimal RLM comparison suite.
	cd $(REPO_ROOT) && python -m benchmarks.eval \
		--model $(EVAL_PROVIDER):$(EVAL_MODEL) \
		--dataset $(EVAL_BENCHMARK_DATASETS) \
		--runner $(EVAL_BENCHMARK_RUNNERS) \
		--seed 0 \
		--full \
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

eval-benchmark-oolong: ## Run the 20-seed OOLONG comparison.
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

eval-delegation: ## Run the frozen 20-problem task-graph delegation suite.
	cd $(REPO_ROOT) && python -m benchmarks.eval \
		--model $(EVAL_PROVIDER):$(EVAL_MODEL) \
		--dataset $(EVAL_DELEGATION_DATASET) \
		--runner rlmflow-local \
		--seeds $(EVAL_SEEDS) \
		$(EVAL_DELEGATION_SAMPLE_ARGS) \
		--out-dir $(EVAL_OUT_DIR) \
		--executor $(EVAL_EXECUTOR) \
		--parallel $(EVAL_PARALLEL) \
		--best-of-n $(EVAL_BEST_OF_N) \
		--attempt-timeout $(EVAL_ATTEMPT_TIMEOUT) \
		--runner-param rlmflow-local.max_iters=$(EVAL_MAX_ITERS) \
		--runner-param rlmflow-local.child_max_iters=$(EVAL_CHILD_MAX_ITERS) \
		--runner-param rlmflow-local.max_depth=$(EVAL_MAX_DEPTH) \
		--runner-param rlmflow-local.max_budget=$(EVAL_MAX_BUDGET) \
		--runner-param rlmflow-local.use_llm_query=$(EVAL_INCLUDE_LLM_QUERY) \
		--runner-param rlmflow-local.prompt_condition=$(EVAL_DELEGATION_CONDITION) \
		$(EVAL_RUN_ID_ARG) \
		$(EVAL_RESUME_ARG) \
		$(EVAL_ARGS)

eval-delegation-five: ## Run five stable delegation and local-control tasks.
	$(MAKE) eval-delegation \
		EVAL_DELEGATION_DATASET=delegation-iteration-five \
		EVAL_DELEGATION_SAMPLE_ARGS=--full \
		EVAL_PARALLEL=5

eval-delegation-five-compare: ## Compare rlmflow and official RLM on the five-task suite.
	cd $(REPO_ROOT) && python -m benchmarks.eval \
		--model $(EVAL_PROVIDER):$(EVAL_MODEL) \
		--dataset delegation-iteration-five \
		--runner rlmflow-local official-rlm \
		--seeds $(EVAL_SEEDS) \
		--full \
		--out-dir $(EVAL_OUT_DIR) \
		--executor local \
		--parallel 5 \
		--best-of-n $(EVAL_BEST_OF_N) \
		--attempt-timeout $(EVAL_ATTEMPT_TIMEOUT) \
		--runner-param rlmflow-local.max_iters=$(EVAL_MAX_ITERS) \
		--runner-param rlmflow-local.child_max_iters=$(EVAL_CHILD_MAX_ITERS) \
		--runner-param rlmflow-local.max_depth=$(EVAL_MAX_DEPTH) \
		--runner-param rlmflow-local.max_budget=$(EVAL_MAX_BUDGET) \
		--runner-param rlmflow-local.use_llm_query=$(EVAL_INCLUDE_LLM_QUERY) \
		--runner-param rlmflow-local.prompt_condition=$(EVAL_DELEGATION_CONDITION) \
		--runner-param official-rlm.max_iters=$(EVAL_MAX_ITERS) \
		--runner-param official-rlm.max_depth=$(EVAL_MAX_DEPTH) \
		--runner-param official-rlm.max_budget=$(EVAL_MAX_BUDGET) \
		$(EVAL_ARGS)

eval-delegation-five-matrix: ## Run the five-task comparison with gpt-5-mini and gpt-5.
	$(MAKE) eval-delegation-five-compare EVAL_MODEL=gpt-5-mini
	$(MAKE) eval-delegation-five-compare EVAL_MODEL=gpt-5

eval-delegation-ten-compare: ## Compare rlmflow and official RLM on ten stable tasks.
	cd $(REPO_ROOT) && python -m benchmarks.eval \
		--model $(EVAL_PROVIDER):$(EVAL_MODEL) \
		--dataset delegation-iteration-ten \
		--runner rlmflow-local official-rlm \
		--seeds $(EVAL_SEEDS) \
		--full \
		--out-dir $(EVAL_OUT_DIR) \
		--executor local \
		--parallel 5 \
		--best-of-n $(EVAL_BEST_OF_N) \
		--attempt-timeout $(EVAL_ATTEMPT_TIMEOUT) \
		--runner-param rlmflow-local.max_iters=$(EVAL_MAX_ITERS) \
		--runner-param rlmflow-local.child_max_iters=$(EVAL_CHILD_MAX_ITERS) \
		--runner-param rlmflow-local.max_depth=$(EVAL_MAX_DEPTH) \
		--runner-param rlmflow-local.max_budget=$(EVAL_MAX_BUDGET) \
		--runner-param rlmflow-local.use_llm_query=$(EVAL_INCLUDE_LLM_QUERY) \
		--runner-param rlmflow-local.prompt_condition=$(EVAL_DELEGATION_CONDITION) \
		--runner-param official-rlm.max_iters=$(EVAL_MAX_ITERS) \
		--runner-param official-rlm.max_depth=$(EVAL_MAX_DEPTH) \
		--runner-param official-rlm.max_budget=$(EVAL_MAX_BUDGET) \
		$(EVAL_ARGS)

eval-delegation-regression: ## Rerun the five failures targeted by current fixes.
	cd $(REPO_ROOT) && python -m benchmarks.eval \
		--model $(EVAL_PROVIDER):$(EVAL_MODEL) \
		--dataset delegation-regression-five \
		--runner rlmflow-local \
		--seeds $(EVAL_SEEDS) \
		--full \
		--out-dir $(EVAL_OUT_DIR) \
		--executor local \
		--parallel 5 \
		--best-of-n $(EVAL_BEST_OF_N) \
		--attempt-timeout $(EVAL_ATTEMPT_TIMEOUT) \
		--runner-param rlmflow-local.max_iters=$(EVAL_MAX_ITERS) \
		--runner-param rlmflow-local.child_max_iters=$(EVAL_CHILD_MAX_ITERS) \
		--runner-param rlmflow-local.max_depth=$(EVAL_MAX_DEPTH) \
		--runner-param rlmflow-local.max_budget=$(EVAL_MAX_BUDGET) \
		--runner-param rlmflow-local.use_llm_query=$(EVAL_INCLUDE_LLM_QUERY) \
		--runner-param rlmflow-local.prompt_condition=$(EVAL_DELEGATION_CONDITION) \
		$(EVAL_ARGS)

eval-delegation-canary: ## Run exactly one DABstep problem before a full delegation run.
	$(MAKE) eval-delegation \
		EVAL_DELEGATION_DATASET=delegation_dabstep \
		EVAL_DELEGATION_SAMPLE_ARGS="--limit 1" \
		EVAL_PARALLEL=1

eval-delegation-phase1: ## Run the seven-source delegation canary.
	$(MAKE) eval-delegation EVAL_DELEGATION_DATASET=delegation-suite-phase1

eval-delegation-official: ## Run all frozen delegation problems with official RLM.
	cd $(REPO_ROOT) && python -m benchmarks.eval \
		--model $(EVAL_PROVIDER):$(EVAL_MODEL) \
		--dataset delegation-suite \
		--runner official-rlm \
		--seeds $(EVAL_SEEDS) \
		--full \
		--out-dir $(EVAL_OUT_DIR) \
		--parallel $(EVAL_PARALLEL) \
		--attempt-timeout $(EVAL_ATTEMPT_TIMEOUT) \
		--runner-param official-rlm.max_iters=$(EVAL_MAX_ITERS) \
		$(EVAL_ARGS)

eval-delegation-report: ## Write JSON and Markdown comparisons for two completed runs.
	@test -n "$(EVAL_RLMFLOW_RUN)" || (echo "EVAL_RLMFLOW_RUN is required" >&2; exit 1)
	@test -n "$(EVAL_OFFICIAL_RUN)" || (echo "EVAL_OFFICIAL_RUN is required" >&2; exit 1)
	cd $(REPO_ROOT) && python -m benchmarks.eval.compare \
		--run rlmflow=$(EVAL_RLMFLOW_RUN) \
		--run official=$(EVAL_OFFICIAL_RUN) \
		--output-dir $(EVAL_COMPARISON_OUT)

eval-clean: ## Remove local eval run artifacts.
	rm -rf $(REPO_ROOT)/benchmarks/eval/runs/
