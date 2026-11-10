# Examples

Single-file scripts live at the root. Multi-file tasks and API tours live in named folders. Generated runs go under [`_runs/`](_runs/).

## Flagship examples

The public story is focused:

1. [Needle](needle/) proves recursive decomposition: a root agent divides a million-line in-memory value among fresh-context workers and combines their findings.
1. [Shepherd](shepherd/) shows why retaining the execution graph matters: a meta-agent can inspect a failed run, rewind it, fork several recoveries, re-instruct them, and select the best result.
1. [Boids](coding/) applies recursive execution to practical software construction: a coding agent chooses the architecture and graph for a dependency-free browser simulation from product-level constraints.

All three are live examples:

```bash
uv run --locked --python 3.12 --extra openai --extra image \
  python examples/shepherd/shepherd.py \
  --model gpt-5 --fast-model gpt-5-mini --branches 8 --time-budget 1800

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
| [`architectures/`](architectures/) | One small run per graph shape (tree, pipeline, review loop), plus where every shape in the README grid comes from | yes |
| [`autoresearch/`](autoresearch/) | A planner running TinyStories training trials on Modal GPUs | yes |
| [`collaboration/`](collaboration/) | Peers with no manager: messaging (`decentralized_peers.py`) and a shared task board (`shared_board.py`) | yes |
| [`showcase.py`](showcase.py) | One end-to-end tour: stepping a run, saving and loading it, replaying it, gym-style rewards | no |
| [`graph/`](graph/) | The graph API on its own: query, navigate, edit, save and load, replay, fork, launch from host code, stop and resume a run, a deep tree | no |
| [`skills.py`](skills.py) | Skills as files in a live prompt section, and an `add_skill` tool the agent extends itself with (`--scripted` runs offline) | optional |
| [`structured_output.py`](structured_output.py) | `output_schema` on the root and its children | yes |
| [`providers/mcp_weather.py`](providers/mcp_weather.py) | Every tool on an MCP server, as REPL tools | yes |
| [`sandboxes/platformer.py`](sandboxes/platformer.py) | Every agent's code in a Docker container or a Modal sandbox (`--runtime docker\|modal`) | yes |
| [`notebooks/coding_agent.ipynb`](notebooks/coding_agent.ipynb) | The coding agent as a notebook, step by step | yes |

[`shepherd/README.md`](shepherd/README.md) has the canonical run and artifact commands. [`docs/shepherd.md`](../docs/shepherd.md) is the worked-run walkthrough: the jam, the plans the meta-agent wrote against it, what each recovery branch did, and forking a Claude Code worker. After a Shepherd run, `make showcase-shepherd-assets` redraws its figures and animations.

Run `--help` on any script for its flags. Most take `--model`, `--max-iters`, `--out-dir`, and `--docker-image IMAGE` to run agent code in a container (`docker build -t rlmflow:local .` builds one with `rlmflow` installed).

A finished run is saved under [`_runs/`](_runs/). The directory holds `graph.json`, the graph's nodes and edges (plus `blobs/` for large inputs). Read it back as the same graph the run held in memory:

```python
from rlmflow import Graph
from rlmflow.view.text import kind

g = Graph.load("examples/_runs/showcase")
for node in g.nodes():
    print(node.agent.path(), kind(node))
```

Each agent reads its `RLMFLOW_*` metadata from its own per-REPL `ENV` mapping (e.g. `ENV["RLMFLOW_AGENT_ID"]`), so that metadata is isolated per agent in every runtime. `LocalRuntime` runs each agent in a worker subprocess; use `SubprocessRuntime` when you also need a selected Python executable, and `DockerRuntime` or `ModalRuntime` when you need container-level isolation.

## Smoke runner

`run_examples.py` is the manifest-driven smoke runner. By default it runs the deterministic/offline examples; use `--include-optional`, `--include-live`, `--include-sandbox`, `--include-manual`, or `--all --list` to expand or inspect the suite. Notebooks are documented separately and are not executed by this runner.

`--all` covers every category except `slow`. Those examples (currently `shepherd`, which lets a meta-agent shepherd a worker through Sokoban) run for long enough that a full-suite run should not block on them, so they need an explicit `--include-slow` — `make examples-slow` or `make examples-list` to see them.

Each run writes a markdown report to `examples/_runs/examples_report.md` (`--report PATH` to move it, `--no-report` to skip it) listing every example as pass/fail/skip, with the command and captured output of each failure — useful when a long live run scrolls its own summary off screen.

See each subdirectory README for the examples in that group.
