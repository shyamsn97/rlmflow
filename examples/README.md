# Examples

Every example lives in a folder; [`common.py`](common.py) holds the helpers they share. Generated runs go under [`_runs/`](_runs/).

## Flagship examples

The public story is focused:

1. [Needle](needle/) proves recursive decomposition: a root agent divides a million-line in-memory value among fresh-context workers and combines their findings.
1. [Shepherd](shepherd/) shows why retaining the execution graph matters: a meta-agent can inspect a failed run, rewind it, fork several recoveries, re-instruct them, and select the best result.
1. [Boids](coding/) applies recursive execution to practical software construction: a coding agent chooses the architecture and graph for a dependency-free browser simulation from product-level constraints.

All three are live examples:

```bash
uv run --locked --python 3.12 --extra openai --extra image \
  python examples/shepherd/shepherd.py \
  --model gpt-5 --fast-model gpt-5 --branches 8 --time-budget 1800

uv run --locked --python 3.12 --extra openai \
  python examples/needle/haystack.py \
  --num-lines 1000000 --model gpt-5-mini --fast-model gpt-5-nano

uv run --locked --python 3.12 --extra openai --extra image \
  python examples/coding/boids/boids.py
```

OOLONG is in [`benchmarks/`](../benchmarks/README.md), which compares Claude Code, Claude Code with rlmflow, and rlmflow's own harness on the same tasks.

## Everything else

| Example | What it shows | Needs a key |
| --- | --- | --- |
| [`needle/`](needle/) | Fan-out: the million-line in-memory needle, and the file needle on any harness (`filesystem.py --all`) | yes |
| [`autoresearch/`](autoresearch/) | A planner launching implementers that each train a variant on Modal GPUs, each in its own copy of the files | yes |
| [`graph/`](graph/) | The graph API on its own: query, navigate, edit, save and load, replay, fork, launch from host code, streaming, and a full tour (`09_tour.py`) | no |
| [`patterns/`](patterns/) | One small run per shape: tree, pipeline, review loop, and peers with no manager (messaging, a shared task board) | yes |
| [`features/`](features/) | One script per feature: skills, structured output, MCP tools, Docker and Modal sandboxes | mostly |

[`shepherd/README.md`](shepherd/README.md) has the canonical run and artifact commands. [`docs/shepherd.md`](../docs/shepherd.md) is the worked-run walkthrough: the jam, the plans the meta-agent wrote against it, what each recovery branch did, and forking a Claude Code worker. After a Shepherd run, `make showcase-shepherd-assets` redraws its figures and animations.

Run `--help` on any script for its flags. Most take `--model`, `--max-iters`, `--out-dir`, and `--docker-image IMAGE` to run agent code in a container (`docker build -t rlmflow:local .` builds one with `rlmflow` installed).

A finished run is saved under [`_runs/`](_runs/). The directory holds `graph.json`, the graph's nodes and edges (plus `blobs/` for large inputs). Read it back as the same graph the run held in memory:

```python
from rlmflow import Graph
from rlmflow.view.text import kind

g = Graph.load("examples/_runs/tour")
for node in g.nodes():
    print(node.agent.path(), kind(node))
```

Each agent reads its `RLMFLOW_*` metadata from its own per-REPL `ENV` mapping (e.g. `ENV["RLMFLOW_AGENT_ID"]`), so that metadata is isolated per agent in every runtime. `LocalRuntime` runs each agent in a worker subprocess; use `SubprocessRuntime` when you also need a selected Python executable, and `DockerRuntime` or `ModalRuntime` when you need container-level isolation.

## Last verified run

Every example, run live on 2026-11-19 with each script's default models (`make examples` flags for the manifest rows). Claude Code used its own default model.

| Example | Models | Result | Time |
| --- | --- | --- | --- |
| `graph/` (11 scripts), `skills` | scripted, no model | pass | under 3s each |
| `skills-live` | gpt-4o-mini | pass | 12s |
| `structured-output` | gpt-5-mini | pass | 78s |
| `mcp-weather` | gpt-5-mini | pass | 48s |
| `sandbox-docker` | gpt-5, gpt-5-mini children | pass | 2m03s |
| `sandbox-modal` | gpt-5, gpt-5-mini children | pass | 3m59s |
| `patterns/peers` | gpt-5-mini | pass | 32s |
| `patterns/board` | gpt-5-mini | pass | 20s |
| `patterns/tree` | gpt-5-mini | pass | 59s |
| `patterns/review` | gpt-5-mini | pass | 34s |
| `patterns/pipeline` | gpt-5-mini | pass | 1m30s |
| `needle/haystack` (2,000 lines) | gpt-5-mini root, gpt-5-nano workers | pass | 65s |
| `needle/filesystem` (50 files) | gpt-5-mini root, gpt-5-nano workers | pass | 50s |
| `needle/claude_plugin` (20 files) | Claude Code root and searchers | pass | 30s |
| `shepherd` | gpt-5 meta-agent and workers | pass, 6 of 8 branches solved | 16m41s |
| `coding/boids` | gpt-5 peers | pass, all 5 checks | 5m40s |
| `autoresearch` (A10G, 8 parallel, 48 submissions, $20) | Claude Code planner and implementers | 32 trials, best val_bpb 0.764 (baseline 0.903) | 31m |
| Live tests (`RLMFLOW_LIVE=1 uv run pytest tests/live`) | gpt-5-mini, gpt-5-nano, Claude Code | 26 passed | 2m53s |

## Smoke runner

`run_examples.py` is the manifest-driven smoke runner. By default it runs the deterministic/offline examples; use `--include-optional`, `--include-live`, `--include-sandbox`, `--include-manual`, or `--all --list` to expand or inspect the suite. Notebooks are documented separately and are not executed by this runner.

`--all` covers every category except `slow`. Those examples (currently `shepherd`, which lets a meta-agent shepherd a worker through Sokoban) run for long enough that a full-suite run should not block on them, so they need an explicit `--include-slow` — `make examples-slow` or `make examples-list` to see them.

Each run writes a markdown report to `examples/_runs/examples_report.md` (`--report PATH` to move it, `--no-report` to skip it) listing every example as pass/fail/skip, with the command and captured output of each failure — useful when a long live run scrolls its own summary off screen.

See each subdirectory README for the examples in that group.
