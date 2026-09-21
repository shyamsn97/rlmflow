.PHONY: help eval-help eval-list-suites eval-run eval-smoke eval-conformance eval-core eval-interactive
.DEFAULT_GOAL := help

REPO_ROOT := $(realpath $(dir $(realpath $(firstword $(MAKEFILE_LIST))))/..)

define PRINT_HELP_PYSCRIPT
import re, sys
for line in sys.stdin:
	match = re.match(r'^([a-zA-Z0-9_/-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-22s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

EVAL_PROVIDER ?= openai
EVAL_MODEL ?= gpt-5-mini
EVAL_SUITE ?= smoke
EVAL_SPLIT ?= pilot
EVAL_SEEDS ?= 0
EVAL_SAMPLE_ARGS ?= --full
EVAL_MAX_ITERS ?= 20
EVAL_MAX_BUDGET ?= 100000
EVAL_OFFICIAL_MAX_DEPTH ?= 3
EVAL_PARALLEL ?= 1
EVAL_ATTEMPT_TIMEOUT ?= 900
EVAL_OUT_DIR ?= benchmarks/eval/runs
EVAL_ARGS ?=

define EVAL_DEPTH_RUNNER
--runner-param rlmflow-depth-$(1).max_depth=$(1) --runner-param rlmflow-depth-$(1).max_iters=$(EVAL_MAX_ITERS) --runner-param rlmflow-depth-$(1).max_budget=$(EVAL_MAX_BUDGET)
endef
EVAL_DEPTH_PARAMS := $(foreach depth,0 1 2,$(call EVAL_DEPTH_RUNNER,$(depth)))
EVAL_SHARED_PARAMS := $(EVAL_DEPTH_PARAMS) --runner-param official-rlm.max_iters=$(EVAL_MAX_ITERS) --runner-param official-rlm.max_depth=$(EVAL_OFFICIAL_MAX_DEPTH) --runner-param official-rlm.max_budget=$(EVAL_MAX_BUDGET)

EVAL_LAUNCH = cd $(REPO_ROOT) && uv run --locked --python 3.12 --extra eval --extra eval-environments --extra official-rlm python -m benchmarks.eval --model $(EVAL_PROVIDER):$(EVAL_MODEL) --dataset $(EVAL_SUITE) --split $(EVAL_SPLIT) --seeds $(EVAL_SEEDS) --parallel $(EVAL_PARALLEL) --attempt-timeout $(EVAL_ATTEMPT_TIMEOUT) --out-dir $(EVAL_OUT_DIR) $(EVAL_SAMPLE_ARGS) $(EVAL_SHARED_PARAMS) $(EVAL_ARGS)

help: ## Show the minimal benchmark commands.
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

eval-help: ## Show the suite CLI help.
	cd $(REPO_ROOT) && python -m benchmarks.eval --help

eval-list-suites: ## Print the four public suites.
	cd $(REPO_ROOT) && python -m benchmarks.eval --list-suites

eval-run: ## Run EVAL_SUITE with the shared four-arm matrix.
	$(EVAL_LAUNCH)

eval-smoke: ## Run the deterministic offline pipeline smoke test.
	cd $(REPO_ROOT) && python -m benchmarks.eval --model fake --dataset smoke --split pilot --seeds 0 --full --out-dir $(EVAL_OUT_DIR) $(EVAL_ARGS)

eval-conformance: ## Run MiniMaze and Sudoku; set EVAL_SPLIT=full for held-out tasks.
	$(MAKE) eval-run EVAL_SUITE=conformance EVAL_SEEDS=0:3

eval-core: ## Run paper-aligned OOLONG-Pairs 32K and OOLONG Yahoo 128K.
	$(MAKE) eval-run EVAL_SUITE=core EVAL_SEEDS=0:3

eval-interactive: ## Run the official MazeBench visual environment.
	$(MAKE) eval-run EVAL_SUITE=interactive EVAL_SEEDS=0
