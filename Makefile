.PHONY: clean clean-build clean-pyc clean-test help install lint format-md lint-md serve-docs test test-all examples examples-list examples-optional examples-live examples-sandbox examples-all examples-slow showcase-shepherd-assets showcase-needle-assets showcase-boids-assets animation animation-preview animation-mp4 animation-gif animation-gif-small animation-clean bump-version release-check release-tag release-push release
.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z0-9_/-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

define BUMP_VERSION_PYSCRIPT
from pathlib import Path
import re
import sys

part = sys.argv[1]
requested = sys.argv[2].strip()
if requested and not re.fullmatch(r"\d+\.\d+\.\d+", requested):
    raise SystemExit("VERSION must be in MAJOR.MINOR.PATCH form, e.g. 0.4.1")
if not requested and part not in {"major", "minor", "patch"}:
    raise SystemExit("BUMP must be one of: major, minor, patch")

path = Path("pyproject.toml")
text = path.read_text(encoding="utf-8")
match = re.search(r'(?m)^version = "(\d+)\.(\d+)\.(\d+)"$$', text)
if not match:
    raise SystemExit("Could not find [project] version in pyproject.toml")

major, minor, patch = map(int, match.groups())
if requested:
    new_version = requested
else:
    if part == "major":
        major, minor, patch = major + 1, 0, 0
    elif part == "minor":
        minor, patch = minor + 1, 0
    else:
        patch += 1
    new_version = f"{major}.{minor}.{patch}"

old_version = match.group(0).split('"')[1]
text = text[: match.start(1)] + new_version + text[match.end(3) :]
path.write_text(text, encoding="utf-8")
print(f"{path}: {old_version} -> {new_version}")
endef
export BUMP_VERSION_PYSCRIPT

BUMP ?= patch
VERSION ?=


help: ## Show this help message.
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr coverage/
	rm -fr .pytest_cache

lint: ## Check Python style with Ruff, matching CI.
	python -m ruff check .

# ── Markdown ─────────────────────────────────────────────────────────
# Scoped to public-facing docs only.
# Install once:  pip install mdformat mdformat-gfm
MD_FILES := README.md CHANGELOG.md UPGRADING.md $(wildcard docs/*.md)

format-md: ## Auto-format public markdown (one paragraph per line, no hard wrap).
	python -m mdformat --wrap no $(MD_FILES)

lint-md: ## Check public markdown formatting without writing changes.
	python -m mdformat --wrap no --check $(MD_FILES)

install: clean lint
	python -m pip install . --upgrade

serve-docs:
	mkdocs serve

test: ## Run the default test suite.
	python -m pytest

test-all: build-docker-image ## Run all tests, including Docker-gated integration tests.
	RECURSIVE_FLOW_DOCKER_TEST=1 python -m pytest

# Every examples target writes a pass/fail/skip report to
# examples/_runs/examples_report.md (with the output of each failure). Override
# the path or add flags with EXAMPLES_ARGS, e.g.
#   make examples-live EXAMPLES_ARGS="--report /tmp/live.md --fail-fast"
EXAMPLES_ARGS ?=

examples: ## Run deterministic/offline examples (report: examples/_runs/examples_report.md).
	python examples/run_examples.py $(EXAMPLES_ARGS)

examples-list: ## List all examples and skip reasons.
	python examples/run_examples.py --all --include-slow --list $(EXAMPLES_ARGS)

examples-optional: ## Run offline + optional-dependency examples.
	python examples/run_examples.py --include-optional $(EXAMPLES_ARGS)

examples-live: ## Run offline + live LLM examples.
	python examples/run_examples.py --include-live $(EXAMPLES_ARGS)

examples-sandbox: ## Run offline + sandbox runtime examples.
	python examples/run_examples.py --include-sandbox $(EXAMPLES_ARGS)

examples-slow: ## Run offline + the long opt-in examples that --all skips.
	python examples/run_examples.py --include-slow $(EXAMPLES_ARGS)

examples-all: ## Run every example category except the slow opt-in ones.
	python examples/run_examples.py --all $(EXAMPLES_ARGS)

SHEPHERD_RUN ?= examples/_runs/shepherd
SHEPHERD_DOC_ASSETS := docs/static/shepherd

showcase-shepherd-assets: ## Publish Shepherd SVGs and GIFs from SHEPHERD_RUN.
	@test -d "$(SHEPHERD_RUN)/shepherd" || (echo "missing $(SHEPHERD_RUN)/shepherd; run the Shepherd example first" >&2; exit 1)
	mkdir -p $(SHEPHERD_DOC_ASSETS)
	uv run --locked --python 3.12 --extra image python examples/shepherd/render_graph.py \
		--run-dir $(SHEPHERD_RUN) \
		--out docs/static/shepherd_graph.svg \
		--nodes-out docs/static/shepherd_nodes.svg
	uv run --locked --python 3.12 --extra image python examples/shepherd/render_rewind.py \
		--run-dir $(SHEPHERD_RUN) \
		--out $(SHEPHERD_DOC_ASSETS)/rewind.gif
	cp "$(SHEPHERD_RUN)/traces/worker.gif" "$(SHEPHERD_DOC_ASSETS)/jam.gif"
	rm -f $(SHEPHERD_DOC_ASSETS)/branch*.gif
	cp "$(SHEPHERD_RUN)"/traces/branch*.gif $(SHEPHERD_DOC_ASSETS)/

NEEDLE_RUN ?= examples/_runs/needle-haystack
NEEDLE_DOC_ASSETS := docs/static

showcase-needle-assets: ## Render and publish the saved million-line Needle graph.
	@test -f "$(NEEDLE_RUN)/graph.json" || (echo "missing $(NEEDLE_RUN)/graph.json; run the Needle example first" >&2; exit 1)
	uv run --locked --python 3.12 rlmflow render svg \
		$(NEEDLE_RUN) $(NEEDLE_DOC_ASSETS)/needle-haystack.svg
	uv run --locked --python 3.12 rlmflow render html \
		$(NEEDLE_RUN) $(NEEDLE_RUN)/run.html
	uv run --locked --python 3.12 --extra image python examples/needle/render_parallel.py \
		--run-dir $(NEEDLE_RUN) \
		--out $(NEEDLE_DOC_ASSETS)/needle-haystack.gif

BOIDS_RUN ?= examples/_runs/coding/boids-rlmflow
BOIDS_DOC_ASSETS := docs/static/boids

showcase-boids-assets: ## Validate Boids and publish its browser and graph visuals.
	@test -f "$(BOIDS_RUN)/graph/graph.json" || (echo "missing $(BOIDS_RUN)/graph/graph.json; run the Boids example first" >&2; exit 1)
	mkdir -p $(BOIDS_DOC_ASSETS)
	uv run --locked --python 3.12 python examples/coding/boids/validate.py \
		$(BOIDS_RUN) \
		--browser \
		--screenshot $(BOIDS_DOC_ASSETS)/simulation.png \
		--report $(BOIDS_RUN)/validation.json
	uv run --locked --python 3.12 --extra image python examples/coding/boids/render_simulation.py \
		--run-dir $(BOIDS_RUN) \
		--gif $(BOIDS_DOC_ASSETS)/simulation.gif \
		--mp4 $(BOIDS_DOC_ASSETS)/simulation.mp4
	uv run --locked --python 3.12 rlmflow render svg \
		$(BOIDS_RUN)/graph $(BOIDS_DOC_ASSETS)/graph.svg
	uv run --locked --python 3.12 --extra image python examples/coding/boids/render_parallel.py \
		--run-dir $(BOIDS_RUN)/graph \
		--out $(BOIDS_DOC_ASSETS)/graph.gif

build-docker-image:
	docker build -t rlmflow:local .

# ── Eval harness ─────────────────────────────────────────────────────
# Public suites and recipes live in benchmarks/Makefile. `make eval-long-context`
# from the repo root forwards there. See docs/benchmarks.md.
eval-%:
	$(MAKE) -C benchmarks $@

# ── Animation (manim) ────────────────────────────────────────────────
ANIMATION_SRC := docs/static/rlm_animation.py
ANIMATION_SCENE := RecursiveFlowHero
ANIMATION_OUT_DIR := media/videos/rlm_animation/1080p60

animation: animation-mp4 animation-gif-small ## Render the rlmflow animation: MP4 + share-friendly GIF.

animation-preview: ## Quick low-res manim preview of the rlmflow animation.
	manim -pql $(ANIMATION_SRC) $(ANIMATION_SCENE)

animation-mp4: ## Render docs/static/rlm_animation.mp4 (1080p60).
	manim -qh $(ANIMATION_SRC) $(ANIMATION_SCENE)
	cp $(ANIMATION_OUT_DIR)/$(ANIMATION_SCENE).mp4 docs/static/rlm_animation.mp4

animation-gif: ## Raw 1080p60 GIF from manim — WARNING: ~1GB+ output.
	manim -qh --format=gif $(ANIMATION_SRC) $(ANIMATION_SCENE)
	cp "$$(ls -t $(ANIMATION_OUT_DIR)/$(ANIMATION_SCENE)*.gif | head -n 1)" docs/static/rlm_animation.gif

animation-gif-small: animation-mp4 ## Share-friendly GIF (ffmpeg, ~5MB) — replaces the README hero.
	ffmpeg -y -i docs/static/rlm_animation.mp4 \
		-vf "fps=20,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer:bayer_scale=5" \
		docs/static/rlm_animation.gif

animation-clean: ## Remove manim render artifacts (media/).
	rm -rf media/

bump-version: ## Bump pyproject.toml version. Use BUMP=minor or VERSION=0.4.1.
	python -c "$$BUMP_VERSION_PYSCRIPT" "$(BUMP)" "$(VERSION)"

release-check: ## Validate VERSION matches pyproject and working tree is clean.
	@test -n "$(VERSION)" || (echo "VERSION is required, e.g. make release VERSION=0.4.1" >&2; exit 1)
	@python -c "import re,sys; v='$(VERSION)'; sys.exit(0 if re.fullmatch(r'\d+\.\d+\.\d+', v) else 'VERSION must be MAJOR.MINOR.PATCH, e.g. 0.4.1')"
	@python -c "import pathlib,sys,tomllib; v=tomllib.loads(pathlib.Path('pyproject.toml').read_text())['project']['version']; exp='$(VERSION)'; sys.exit(0 if v == exp else f'pyproject version {v} != VERSION {exp}')"
	@test -z "$$(git status --porcelain --untracked-files=all)" || (echo "working tree has tracked or untracked changes; commit the complete release first" >&2; exit 1)
	@! git rev-parse -q --verify "refs/tags/v$(VERSION)" >/dev/null || (echo "tag v$(VERSION) already exists locally" >&2; exit 1)

release-tag: release-check ## Create git tag v$(VERSION) for the current commit.
	git tag v$(VERSION)

release-push: ## Push current branch and tag v$(VERSION) to trigger release workflow.
	@test -n "$(VERSION)" || (echo "VERSION is required, e.g. make release-push VERSION=0.4.1" >&2; exit 1)
	@git rev-parse -q --verify "refs/tags/v$(VERSION)" >/dev/null || (echo "tag v$(VERSION) does not exist locally; run make release-tag VERSION=$(VERSION)" >&2; exit 1)
	git push origin HEAD
	git push origin v$(VERSION)

release: release-tag ## Create and push v$(VERSION), triggering the GitHub release workflow.
	$(MAKE) release-push VERSION=$(VERSION)
