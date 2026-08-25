.PHONY: clean clean-build clean-pyc clean-test coverage dist docs help install lint lint/flake8 format-md lint-md test test-all test-live test-live-boids examples examples-list examples-optional examples-live examples-sandbox examples-all eval-help eval-smoke eval-test eval-run eval-wandb eval-benchmark eval-benchmark-full eval-benchmark-oolong eval-clean animation animation-preview animation-mp4 animation-gif animation-gif-small animation-clean bump-version release-check release-tag release-push release
	{%- if cookiecutter.use_black == 'y' %} lint/black{% endif %}
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

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

BROWSER := python -c "$$BROWSER_PYSCRIPT"
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

lint: ## check style with flake8
	isort --profile black rlmflow
	black rlmflow
	flake8 rlmflow
	python -m ruff check .

# ── Markdown ─────────────────────────────────────────────────────────
# Scoped to public-facing docs only.
# Install once:  pip install mdformat mdformat-gfm
MD_FILES := README.md CHANGELOG.md $(wildcard docs/*.md)

format-md: ## Auto-format public markdown (one paragraph per line, no hard wrap).
	python -m mdformat --wrap no $(MD_FILES)

lint-md: ## Check public markdown formatting without writing changes.
	python -m mdformat --wrap no --check $(MD_FILES)

install: clean lint
	python -m pip install . --upgrade

doc:
	rm -r docs/reference/
	pdocs as_markdown rlmflow -o docs/reference
	rm docs/reference/rlmflow/index.md
	cp examples/*.ipynb docs/examples/
	cp README.md docs/index.md

serve-docs:
	mkdocs serve

commit: install test doc
	git add .
	git commit -a

test: ## Run the default test suite.
	python -m pytest

test-all: build-docker-image ## Run all tests, including Docker-gated integration tests.
	RECURSIVE_FLOW_DOCKER_TEST=1 python -m pytest

# Behavioral checks against a real model: does the prompt delegate substantial
# independent work, and does it stay local on trivial work? Pinned to one model
# so results stay comparable across prompt changes. Override with
#   make test-live LIVE_MODEL=gpt-5 LIVE_ARGS="-k explainers"
LIVE_MODEL ?= gpt-5-mini
LIVE_ARGS ?= -m live

# These runs cost money, so name each scenario as it goes rather than printing
# dots. `-vv` because the config's `-q` (pyproject `addopts`) cancels the first
# `-v`; the second one is what actually reaches verbose. Names of tests a filter
# dropped come from the `pytest_deselected` hook in `tests/conftest.py`.
LIVE_PYTEST = RLMFLOW_LIVE_TESTS=1 RLMFLOW_LIVE_MODEL=$(LIVE_MODEL) \
	python -m pytest tests/test_delegation_behavior.py

test-live: ## Run every live delegation-behavior check, boids included (needs an API key, costs money).
	$(LIVE_PYTEST) $(LIVE_ARGS) -s -vv -ra

# Selected by node id rather than by marker, so the one scenario you asked for is
# the only thing collected and nothing is reported as deselected.
test-live-boids: ## Run only the boids delegation check (slow: an 8-module task).
	$(LIVE_PYTEST)::test_boids_delegates_the_substantial_modules -s -vv -ra

test-html: test
	$(BROWSER) tests/cov-report/index.html

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

build-docker-image:
	docker build -t rlmflow:local .

# ── Eval harness ─────────────────────────────────────────────────────
# Eval/benchmark targets now live in benchmarks/Makefile. Run them with
# `make -C benchmarks <target>` (or `cd benchmarks && make <target>`).
eval-help eval-smoke eval-test eval-run eval-wandb eval-benchmark eval-benchmark-full eval-benchmark-oolong eval-clean:
	$(MAKE) -C benchmarks $@

# ── Animation (manim) ────────────────────────────────────────────────
ANIMATION_SRC := docs/rlm_animation.py
ANIMATION_SCENE := RecursiveFlowHero
ANIMATION_OUT_DIR := media/videos/rlm_animation/1080p60

animation: animation-mp4 animation-gif-small ## Render the rlmflow animation: MP4 + share-friendly GIF.

animation-preview: ## Quick low-res manim preview of the rlmflow animation.
	manim -pql $(ANIMATION_SRC) $(ANIMATION_SCENE)

animation-mp4: ## Render docs/rlm_animation.mp4 (1080p60).
	manim -qh $(ANIMATION_SRC) $(ANIMATION_SCENE)
	cp $(ANIMATION_OUT_DIR)/$(ANIMATION_SCENE).mp4 docs/rlm_animation.mp4

animation-gif: ## Raw 1080p60 GIF from manim — WARNING: ~1GB+ output.
	manim -qh --format=gif $(ANIMATION_SRC) $(ANIMATION_SCENE)
	cp "$$(ls -t $(ANIMATION_OUT_DIR)/$(ANIMATION_SCENE)*.gif | head -n 1)" docs/rlm_animation.gif

animation-gif-small: animation-mp4 ## Share-friendly GIF (ffmpeg, ~5MB) — replaces the README hero.
	ffmpeg -y -i docs/rlm_animation.mp4 \
		-vf "fps=20,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer:bayer_scale=5" \
		docs/rlm_animation.gif

animation-clean: ## Remove manim render artifacts (media/).
	rm -rf media/

bump-version: ## Bump pyproject.toml version. Use BUMP=minor or VERSION=0.4.1.
	python -c "$$BUMP_VERSION_PYSCRIPT" "$(BUMP)" "$(VERSION)"

release-check: ## Validate VERSION matches pyproject and working tree is clean.
	@test -n "$(VERSION)" || (echo "VERSION is required, e.g. make release VERSION=0.4.1" >&2; exit 1)
	@python -c "import re,sys; v='$(VERSION)'; sys.exit(0 if re.fullmatch(r'\d+\.\d+\.\d+', v) else 'VERSION must be MAJOR.MINOR.PATCH, e.g. 0.4.1')"
	@python -c "import pathlib,sys,tomllib; v=tomllib.loads(pathlib.Path('pyproject.toml').read_text())['project']['version']; exp='$(VERSION)'; sys.exit(0 if v == exp else f'pyproject version {v} != VERSION {exp}')"
	@git diff --quiet || (echo "working tree has unstaged changes; commit the release first" >&2; exit 1)
	@git diff --cached --quiet || (echo "working tree has staged changes; commit the release first" >&2; exit 1)
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
