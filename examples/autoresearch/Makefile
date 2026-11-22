PYTHON ?= python
UV ?= uv

MODEL ?= gpt-5
GPU ?= A10G
PARALLEL ?= 12
MAX_SUBMISSIONS ?= 48
MINUTES ?=
MAX_USD ?=
PLANNER ?= claude_code
IMPLEMENTER ?= claude_code
CLAUDE_MODEL ?=
OUT ?=

.PHONY: help sync prepare train run smoke lint

help:
	@echo "Autoresearch targets:"
	@echo "  make sync        Install task deps with uv"
	@echo "  make prepare     Download/tokenize TinyStories"
	@echo "  make train       Run one local train.py experiment"
	@echo "  make run         Run rlmflow coordinator with Modal trials"
	@echo "  make smoke       Run with MAX_SUBMISSIONS=1"
	@echo "  make lint        Lint runner files"
	@echo ""
	@echo "Common overrides:"
	@echo "  MODEL=$(MODEL) GPU=$(GPU) PARALLEL=$(PARALLEL) MAX_SUBMISSIONS=$(MAX_SUBMISSIONS)"
	@echo "  MINUTES=120 MAX_USD=20 PLANNER=repl IMPLEMENTER=repl CLAUDE_MODEL=opus OUT=/path/to/run"

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
		--planner "$(PLANNER)" \
		--implementer "$(IMPLEMENTER)" \
		$(if $(CLAUDE_MODEL),--claude-model "$(CLAUDE_MODEL)",) \
		$(if $(MINUTES),--minutes "$(MINUTES)",) \
		$(if $(MAX_USD),--max-usd "$(MAX_USD)",) \
		$(if $(OUT),--out "$(OUT)",)

smoke:
	$(MAKE) run MAX_SUBMISSIONS=1

lint:
	cd ../.. && $(PYTHON) -m ruff check examples/autoresearch/run.py examples/autoresearch/modal_runner.py examples/autoresearch/prepare.py examples/autoresearch/train.py
