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
  python examples/coding/boids/boids.py \
  --runner rlmflow --model gpt-5 --fast-model gpt-5-mini --force
```

OOLONG and MazeBench remain in the separate [benchmark harness](../docs/benchmarks.md), where their results can be compared across the same runtime matrix.

## Scripts

| Script                                                                       | What it shows                                                         |
| ---------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| [`showcase.py`](showcase.py)                                                 | End-to-end `Flow` run, persistence, and gym-style stepping            |
| [`drop_in_llm.py`](drop_in_llm.py)                                           | Minimal `FlowLLM` as a drop-in LLM                                    |
| [`llm_query_fanout.py`](llm_query_fanout.py)                                 | Concurrent `llm_query` calls with `asyncio.gather`                    |
| [`skills.py`](skills.py)                                                     | On-disk skills + dynamic prompt section                               |
| [`structured_output.py`](structured_output.py)                               | Root + child `output_schema` validation                               |
| [`summarizer.py`](summarizer.py)                                             | Recursive map-reduce summarization                                    |
| [`providers/claude_code_delegation.py`](providers/claude_code_delegation.py) | Claude Code children through a named `claude_code` config |
| [`collaboration/shared_board.py`](collaboration/shared_board.py)             | Shared task board: every task is an agent, `AGENTS` is the context   |

```bash
python examples/showcase.py
python examples/skills.py --model gpt-4o-mini
python examples/summarizer.py --sections 10
python examples/collaboration/shared_board.py \
  --model gpt-5-mini --task "API" --task "Tests"
python examples/providers/claude_code_delegation.py
```

## Tasks

| Folder                           | What it shows                                                                                                                                       |
| -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`needle/`](needle/)             | Needle-in-haystack (in-memory + filesystem variants)                                                                                                |
| [`coding/`](coding/)             | Interactive file-editing agent plus the flagship model-designed Boids browser build                                                                 |
| [`autoresearch/`](autoresearch/) | TinyStories autoresearch loop with Modal GPU trials                                                                                                 |
| [`research/textcraft/`](research/textcraft/) | Gymnasium TextCraft-Synth environment, grouped recursive rollouts, exact per-agent credit, and Tinker RLOO training |
| [`shepherd/`](shepherd/)         | Live meta-agent recovers a jammed Sokoban worker by forking its graph and replaying recovery branches in parallel, with board panels + a node trace |

[`shepherd/README.md`](shepherd/README.md) has the canonical run and artifact commands. [`docs/shepherd.md`](../docs/shepherd.md) is the worked-run walkthrough: the jam, the plans the meta-agent wrote against it, what each recovery branch did, and how a profile user-prompt modifier adds the live board to the current worker turn.

After a Shepherd run, `make showcase-shepherd-assets` redraws the agent summary and every-node graph as SVG, animates the jam → rewind → recovery transition, and publishes all branch trace GIFs to the documentation paths.

## Tours & integrations

| Folder                     | What it shows                                  |
| -------------------------- | ---------------------------------------------- |
| [`graph/`](graph/)         | Offline Node API (inspect, append, fork, attach subtrees, save) |
| [`control/`](control/)     | Delegation, branching, injection               |
| [`sandboxes/`](sandboxes/) | Docker and Modal remote execution              |
| [`providers/`](providers/) | DSPy, MCP, Tinker adapters                     |

The nonblocking delegation example is deterministic and needs no API key:

```bash
python examples/control/delegation/nonblocking.py
```

______________________________________________________________________

Most compute examples (`summarizer.py`, `needle/haystack.py`, `showcase.py`, `coding/agent.py`) share the same flags. Run `--help` on any script for defaults.

| Flag                   | Default                 | Meaning                                                                                                                                                                                           |
| ---------------------- | ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--model MODEL`        | varies                  | Main LLM. Prefix decides client (`claude*` → Anthropic, else OpenAI).                                                                                                                             |
| `--fast-model MODEL`   | varies                  | Optional cheap secondary model registered as `fast` for delegates.                                                                                                                                |
| `--docker-image IMAGE` | unset                   | If set, run agent code inside this Docker image via a `DockerRuntime`. Must have `rlmflow` installed. Leaving this unset uses `LocalRuntime`, which runs code in a worker subprocess on the host. |
| `--max-depth N`        | `1`                     | Max delegation depth. FILE_TOOLS writes bind on agents that cannot spawn, so `1` lets children write.                                                                                             |
| `--max-iters N`        | `15`                    | Max LLM turns per agent.                                                                                                                                                                          |
| `--out-dir PATH`       | `_runs/<example-name>/` | Save the final run here. Defaults use flat example names under [`_runs/`](_runs/).                                                                                                                |

## Running under Docker

Build the image once:

```bash
docker build -t rlmflow:local .
```

Then pass `--docker-image rlmflow:local` to any example that supports it:

```bash
python examples/summarizer.py                 --docker-image rlmflow:local
python examples/needle/haystack.py            --docker-image rlmflow:local
python examples/needle/filesystem.py          --docker-image rlmflow:local
python examples/coding/agent.py --workdir ./proj --docker-image rlmflow:local
```

Examples that use file tools pass them to the flow (`Flow(tools=[FILE_TOOLS])`) and set `working_directory` on the runtime, so relative paths resolve into that directory the same way in local and Docker modes.

Each agent reads its `RLMFLOW_*` metadata from its own per-REPL `ENV` mapping (e.g. `ENV["RLMFLOW_AGENT_ID"]`), so that metadata is isolated per agent in every runtime. `LocalRuntime` already runs each agent in a worker subprocess; use `SubprocessRuntime` when you also need a selected Python executable, and `DockerRuntime` or a cloud sandbox when you need container-level isolation.

A finished run is saved automatically under `_runs/`:

```bash
python examples/summarizer.py        # saves to examples/_runs/summarizer/
```

The saved directory holds `graph.json`, the graph's item log (plus `blobs/` for large arguments). Read it back as the same graph the run held in memory:

```python
from rlmflow import Graph

g = Graph.load("examples/_runs/summarizer")
for node in g.nodes():
    print(g.path(g.node(node.agent_id)), node.type)
```

## Docker And Modal

Remote sandbox examples live under [`sandboxes/`](sandboxes/). They run a small platformer-building task, so set `OPENAI_API_KEY` plus any provider credentials:

```bash
python examples/sandboxes/docker_agent.py --docker-image rlmflow:local
python examples/sandboxes/modal_agent.py --model gpt-5
```

The Docker runtime needs only the core package and the host `docker` CLI. Install `rlmflow[modal]` for Modal; `rlmflow[sandbox]` is an equivalent convenience extra.

For fully locked-down local runs, pass a `DockerRuntime`:

```python
from rlmflow import DockerRuntime, Flow
from rlmflow.llm import OpenAIClient

runtime = DockerRuntime("rlmflow:local", working_directory="./proj")
flow = Flow(OpenAIClient(model="gpt-4o"), runtime=runtime)
```

## Smoke runner

`run_examples.py` is the manifest-driven smoke runner. By default it runs the deterministic/offline examples; use `--include-optional`, `--include-live`, `--include-sandbox`, `--include-manual`, or `--all --list` to expand or inspect the suite. Notebooks are documented separately and are not executed by this runner.

`--all` covers every category except `slow`. Those examples (currently `shepherd`, which lets a meta-agent shepherd a worker through Sokoban) run for long enough that a full-suite run should not block on them, so they need an explicit `--include-slow` — `make examples-slow` or `make examples-list` to see them.

Each run writes a markdown report to `examples/_runs/examples_report.md` (`--report PATH` to move it, `--no-report` to skip it) listing every example as pass/fail/skip, with the command and captured output of each failure — useful when a long live run scrolls its own summary off screen.

See each subdirectory README for the examples in that group.
