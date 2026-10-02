.PHONY: help list image run smoke delegation long-context interactive
.DEFAULT_GOAL := help

ROOT := $(realpath $(dir $(realpath $(firstword $(MAKEFILE_LIST))))/..)
BENCHMARK ?= smoke
SPLIT ?= pilot
RUNNERS ?=
MODEL ?= openai:gpt-5-mini
OUTPUT ?=
EXECUTOR ?= local
PARALLELISM ?= 1
ARGS ?=

RUN = cd $(ROOT) && uv run python benchmarks/run_benchmark.py $(BENCHMARK) \
	--split $(SPLIT) --model $(MODEL) --executor $(EXECUTOR) \
	--parallelism $(PARALLELISM) \
	$(if $(RUNNERS),--runners $(RUNNERS)) \
	$(if $(OUTPUT),--output $(OUTPUT)) $(ARGS)

help: ## Show benchmark commands.
	@echo "make -f benchmarks/Makefile {list,smoke,delegation,long-context,interactive}"

list: ## List the four benchmarks.
	cd $(ROOT) && uv run python benchmarks/run_benchmark.py --list

image: ## Build the benchmark Docker image.
	cd $(ROOT) && docker build -f benchmarks/Dockerfile -t rlmflow-eval:local .

run: ## Run BENCHMARK with explicit variables.
	$(RUN)

smoke: ## Run deterministic needle/haystack offline.
	$(MAKE) -f benchmarks/Makefile run BENCHMARK=smoke RUNNERS=fake MODEL=fake:fake

delegation: ## Run swarm fan-out on every compatible method.
	$(MAKE) -f benchmarks/Makefile run BENCHMARK=delegation \
		RUNNERS=prime-agent,official-rlm,jaz,rlmflow-depth-1

long-context: ## Run OOLONG on Official RLM and rlmflow.
	$(MAKE) -f benchmarks/Makefile run BENCHMARK=long-context \
		RUNNERS=official-rlm,rlmflow-depth-1

interactive: ## Run MiniMaze, Sudoku, and MazeBench.
	$(MAKE) -f benchmarks/Makefile run BENCHMARK=interactive \
		RUNNERS=official-rlm,rlmflow-depth-1
