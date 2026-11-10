# rlmflow for coding agents

- Writing code that uses rlmflow: the [README](README.md) has the usage, and [`docs/`](docs) the details.
- Orchestrating agents, inside a run or from host code: read [`rlmflow/plugin/skills/orchestrate/SKILL.md`](rlmflow/plugin/skills/orchestrate/SKILL.md). Agents that can launch others already get it in their prompt.

Working on rlmflow itself:

- `uv run pytest -q`, `uv run ruff check`, `uv run ruff format`.
- Don't regenerate `uv.lock` unless asked.
- Docs are in `docs/`; the README's code must run against the current API.
