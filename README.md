# rlmflow

<p align="center">
  <a href="https://pypi.org/project/rlmflow/"><img src="https://img.shields.io/pypi/v/rlmflow.svg?label=pypi" alt="PyPI" /></a>
  <a href="https://github.com/shyamsn97/rlmflow/pkgs/container/rlmflow"><img src="https://img.shields.io/badge/ghcr.io-rlmflow-2496ED?logo=docker&logoColor=white" alt="Docker" /></a>
</p>

Read the blog post: [rlmflow](https://shyamsn97.github.io/blog/rlmflow/).

**rlmflow** is a Python library for building **recursive agents** — agents that spawn other agents — as live execution graphs. Every query, action, observation, child agent, and result is a typed node you can inspect, step through, save, fork, and append to mid-run.

It gives an LLM a stateful Python REPL and a graph-native way to spawn agents with fresh context.

The release has two flagship demonstrations: **Shepherd** is the primary product story — rewind, fork, recover, and select from a retained execution graph — while **Needle** is the compact proof of recursive decomposition. Everything else is an API tour or supporting benchmark evidence.

## Flagship examples

### Shepherd: recover the graph, not just the answer

The example first writes a reproducible failed worker trace by scripting eight myopic Sokoban pushes. GPT-5 then reads that execution, previews earlier graph states, and proposes eight differently instructed GPT-5-mini recoveries across several rewind depths. The host forks and runs them as the shepherd's child agents, then selects the highest-scoring result. Planning and recovery play use live model calls; the fixed jam makes runs comparable, and the local deterministic game makes graph replay exact.

<p align="center">
  <img src="https://raw.githubusercontent.com/shyamsn97/rlmflow/main/docs/shepherd_graph.svg" alt="The Shepherd execution graph: one jammed worker, eight rewind branches, six solved recoveries, and one selected winner" width="900" />
</p>
<p align="center"><em>One failed run becomes eight recoverable branches inside the same retained graph.</em></p>

<p align="center">
  <img src="https://raw.githubusercontent.com/shyamsn97/rlmflow/main/docs/static/shepherd/rewind.gif" alt="A live Shepherd run progressing from the deliberate Sokoban jam through graph rewind to the selected recovery" width="460" />
</p>
<p align="center"><em>Recorded board replay: deliberate jam → rewind → selected live-model recovery.</em></p>

```bash
uv run --locked --python 3.12 --extra openai --extra image \
  python examples/shepherd/shepherd.py \
  --model gpt-5 --fast-model gpt-5-mini --branches 8 --time-budget 1800
```

In the checked-in run, six of eight recoveries solved; Shepherd selected `branch2`, which rewound seven pushes and solved in 11 total pushes. The run is saved under `examples/_runs/shepherd/`; `make showcase-shepherd-assets` reproduces the checked-in SVG and GIF assets. See the [example instructions](examples/shepherd/) and [worked walkthrough](docs/shepherd.md).

### Needle: decompose a million-line value

The root receives one million lines through `INPUTS["haystack"]`, partitions the value in its REPL, delegates chunks to GPT-5-nano workers with fresh contexts, and checks their combined answer against the generated secret.

<p align="center">
  <img src="https://raw.githubusercontent.com/shyamsn97/rlmflow/main/docs/static/needle-haystack.gif" alt="The million-line Needle run growing from one root into parallel search workers and returning the generated magic number" width="900" />
</p>
<p align="center"><em>Parallel graph replay: same-level nodes across the 13 worker branches light together. The recorded run reached 10-way wall-clock concurrency.</em></p>

```bash
uv run --locked --python 3.12 --extra openai \
  python examples/needle/haystack.py \
  --num-lines 1000000 --model gpt-5-mini --fast-model gpt-5-nano
```

The checked-in run generated `5335239` at line 466,831 and returned the same value in 3m 26s using 13 child workers, with peak concurrency of 10. Its saved graph at `examples/_runs/needle-haystack/` renders directly to a steppable HTML trace, static graph, and level-synchronized parallel animation; `make showcase-needle-assets` reproduces the checked-in assets. See the [Needle example](examples/needle/).

### Benchmarks: long context and visual interaction

The evaluation harness compares the original `official-rlm` runtime with rlmflow at depths 0, 1, and 2 under one contract. OOLONG-Pairs 32K and OOLONG Yahoo 128K are the primary long-context tasks. MazeBench is the official interactive benchmark: agents explore a persistent ASCII world, collect gems, and produce replay artifacts under a shared action budget. See [Evaluation suites](docs/benchmarks.md).

## Recursive agents as graphs

Recursive agents let a model split work across fresh sub-agents, each with its own context, tools, and execution history. Those sub-agents can delegate again, so one root task can quickly become a tree of parallel work.

For example, the root agent can split a haystack into chunks. Every launch returns a future-like handle immediately:

```python
handles = [
    await launch_subagent(
        "scan first third", model="default", name="chunk_0", inputs={"chunk": chunk_0}
    ),
    await launch_subagent(
        "scan middle third", model="default", name="chunk_1", inputs={"chunk": chunk_1}
    ),
    await launch_subagent(
        "scan final third", model="default", name="chunk_2", inputs={"chunk": chunk_2}
    ),
]
results = [await handle.wait_for_result() for handle in handles]
finish(extract_answer(results))
```

Then `chunk_2` can recursively delegate again:

```python
hits = find_candidate_windows(INPUTS["chunk"])
handles = [
    await launch_subagent(
        "inspect window A",
        model="default",
        name="candidate_a",
        inputs={"window": hits[0]},
    ),
    await launch_subagent(
        "inspect window B",
        model="default",
        name="candidate_b",
        inputs={"window": hits[1]},
    ),
]
results = [await handle.wait_for_result() for handle in handles]
finish(select_candidate(results))
```

That code creates an agent graph:

```text
root  "Find the code in the haystack"
├── chunk_0      "scan first third"   -> "not found"
├── chunk_1      "scan middle third"  -> "decoy, no code"
└── chunk_2      "scan final third"   -> "candidate code 84721"
    ├── candidate_a  "inspect window A" -> "decoy"
    └── candidate_b  "inspect window B" -> "code 84721"
```

That parent API is useful: children return simple values the parent can compose. The problem is when those return values are the only surviving record of the work:

```text
results == ["not found", "decoy, no code", "candidate code 84721"]
```

If `chunk_2` launched `candidate_a` and `candidate_b` internally, the parent can still receive `"candidate code 84721"` as its normal result. But for debugging, evaluation, supervision, or reuse, you also want the execution state that produced it: which agents ran, what they saw, what failed, and where you could intervene.

**rlmflow** keeps that structure alive. Every recursive call is a child agent inside one typed `Node` tree:

<p align="center">
  <img src="docs/recursive_agents_graph.svg" alt="rlmflow graph showing root, child agents, grandchild agents, and final result states" />
</p>

The graph is the run itself:

```python
from rlmflow import Flow

flow = Flow(client)
root = flow.start(query)

async for node in flow.run_streaming(root):
    print(node.parent_agent.config.path, node.type)
```

`run_streaming` mutates the root in place and yields each node as it lands, so the live tree is always just `root`. The Node tree *is* the durable state: you can inspect a branch, save a checkpoint, fork from an old node, append controller feedback, and continue from the edited graph.

See [`docs/internals.md`](docs/internals.md), [`docs/node_model.md`](docs/node_model.md), and [`docs/control.md`](docs/control.md) for the full engine and graph API.

## Install

```
pip install rlmflow               # core
pip install rlmflow[openai]       # + OpenAI client
pip install rlmflow[anthropic]    # + Anthropic client
pip install rlmflow[tinker]       # + Tinker inference client
pip install rlmflow[dspy]         # + DSPy adapter
pip install rlmflow[modal]        # + Modal runtime
pip install rlmflow[tui]          # + full-screen TUI
pip install rlmflow[viewer]       # + browser viewer
pip install rlmflow[image]        # + PNG / GIF export
pip install rlmflow[all]          # all of the above
```

From source:

```
git clone https://github.com/shyamsn97/rlmflow && cd rlmflow
pip install -e .
```

For local development, `make install` runs cleanup, formatting/lint checks including `ruff check .`, then installs the package.

> **Security warning — `LocalRuntime` is not a sandbox.** Agent code runs as full Python in a separate worker process, but with the same filesystem, network, environment, and subprocess privileges as the host. LLM-generated code can be wrong or malicious (prompt injection, model errors, supply-chain risk). **Use `LocalRuntime` only for code you would run yourself.** For untrusted agents or anything exposed to the internet, use [`DockerRuntime`](docs/runtimes.md) or a remote sandbox ([`ModalRuntime`](docs/runtimes.md)). See [`docs/security.md`](docs/security.md). **Use at your own risk.**

## Quick start

This example builds a simple coding agent with file tools in a local working directory. See [`examples/coding/agent.py`](./examples/coding/agent.py) for the full interactive version.

```python
from pathlib import Path

from rlmflow.llm import OpenAIClient
from rlmflow import FILE_TOOLS, AgentConfig, Flow, LocalRuntime

workdir = Path("examples/_runs/quickstart")

flow = Flow(
    OpenAIClient(model="gpt-5"),
    runtime=LocalRuntime(working_directory=workdir),
    tools=[FILE_TOOLS],
    root_config=AgentConfig(max_depth=2, max_iters=20),
    llm_clients={"fast": OpenAIClient(model="gpt-5-mini")},
)

query = "Build a Python text-based adventure game with combat and inventory."
root = flow.start(query)

async for node in flow.run_streaming(root):
    print(node.parent_agent.config.path, node.type)

print(root.result())
root.save(workdir / "graph")
```

Delegated children fan out as ordinary asyncio tasks that step alongside their parent. See [`examples/control/delegation/step_until.py`](./examples/control/delegation/step_until.py) for a deterministic demo of `Flow.run_streaming(..., until=...)` boundaries while child work runs concurrently.

A saved run is a directory rooted at `graph.json` plus `agents/` logs. Reopen it later with `AgentStart.load(path)` and keep running it.

## Command line

`rlmflow tui` is the coding agent above, without writing any of that. It edits `--workdir` (`.` by default), checkpoints to `<workdir>/graph`, and opens the dashboard — install it with `pip install "rlmflow[tui]"`.

```bash
pip install "rlmflow[tui]"

rlmflow tui                                   # the dashboard, waiting for a query
rlmflow tui --query "add a test for parse_args"
rlmflow tui --workdir ./proj --model gpt-5 --docker-image rlmflow:local
rlmflow tui --resume ./proj/graph             # pick a saved run back up
rlmflow tui --agent mypkg.agents:build        # your own Flow instead of the coding one

rlmflow run print "fix the failing test"      # no dashboard: stream it, print the answer
```

`--model` / `--fast-model` pick the clients (`claude*` → Anthropic, else OpenAI), `--reasoning-effort` reaches OpenAI reasoning models, and `--tools {files,none}` chooses whether the agent gets `FILE_TOOLS`. Anything the flags cannot say — custom prompts, tools, runtimes — stays in Python: point `--agent` at a `module:factory` that returns a `Flow`.

```python
# mypkg/agents.py
from rlmflow import FILE_TOOLS, AgentConfig, Flow, LocalRuntime
from rlmflow.llm import client_for

def build() -> Flow:
    return Flow(
        client_for("gpt-5", reasoning_effort="low"),
        runtime=LocalRuntime(working_directory="."),
        tools=[FILE_TOOLS],
    root_config=AgentConfig(max_depth=2, max_iters=40),
    )
```

Settings resolve flags first, then `RLMFLOW_*`, then `./rlmflow.toml`, then `~/.config/rlmflow/config.toml`:

```bash
rlmflow config show    # every setting, its value, which source won
rlmflow config path    # the files that are read
rlmflow config init    # write a starter rlmflow.toml
```

```toml
# rlmflow.toml
[run]
model = "gpt-5"
fast_model = "gpt-5-mini"
reasoning_effort = "low"
max_iters = 30
```

A finished run is what `rlmflow view` and `rlmflow render` read — the tree and timeline, or a figure / stepper / GIF / browser viewer. See [Visualization](#visualization) below, and [`docs/cli.md`](docs/cli.md) for the full flag list. `python -m rlmflow …` works the same; `rlmflow --help` lists every command.

## Drop-in `LLMClient`

`FlowLLM` wraps a `Flow` in the `LLMClient` `chat(messages)` interface, so a full recursive agent run is a drop-in replacement for any raw LLM.

```python
import rlmflow
from rlmflow.llm import LLMClient, OpenAIClient
from rlmflow.adapters import FlowLLM

def ask(llm: LLMClient, q: str) -> str:
    return llm.chat([{"role": "user", "content": q}])

ask(OpenAIClient(model="gpt-4o-mini"), "2+2?")                      # one LLM call
ask(FlowLLM(rlmflow.Flow(OpenAIClient(model="gpt-4o-mini"))), "2+2?")  # full agent loop
```

Nest agents by passing one `FlowLLM(inner_flow)` as another flow's `llm`. See [`examples/drop_in_llm.py`](examples/drop_in_llm.py).

## Stream and inspect

`Flow` gives you three ways to drive the same durable Node tree:

- `flow.run(query)` — run to completion and return the result.
- `await flow.arun(root)` — async completion.
- `async for node in flow.run_streaming(root, until=...)` — stream landed Nodes to a boundary.

```python
root = flow.start(query)
seen = 0

def two_steps(node, _root):
    global seen
    seen += 1
    return seen == 2

async for node in flow.run_streaming(root, until=two_steps):
    pass
print(root.frontier.type)             # ...inspect...
async for node in flow.run_streaming(root, until="done"):
    pass                              # ...then run to completion
print(root.result())
```

Every transition follows the same obs → action → obs shape:

```text
LLMOutput  -> ExecAction -> ExecOutput    (REPL output, normal continuation)
                         -> DoneOutput    (code called finish())
                         -> ErrorOutput   (code raised, or the REPL died)
```

A turn that delegates hangs its children off the `ExecAction` that launched them, and lands one `ExecOutput` when the block finishes — so a parent's transcript stays a single chain whether or not it spawned anyone.

The graph is queryable in plain Python:

```python
root.frontier                           # current Node for this agent
root.transcript()                       # this agent's typed history
root.sub_agents                         # the agents it launched
root.leaves()                           # the frontier of every agent in the tree
list(root.walk())                       # every Node, preorder
root.result()                           # root DoneOutput result
root.usage                              # O(1) whole-run usage
root.stats                              # immutable counts and usage snapshot

from rlmflow import persistence
persistence.to_document(root)           # flat JSON-serializable v3 document
```

## Inject controller Nodes

Because the Node tree is the control surface, an external controller can append to any agent's frontier — as its node lands, or between streaming calls — and let the run continue. This is useful for human feedback, budget nudges, and forced finalization without losing traceability:

```python
from rlmflow import ExecOutput, UserQuery

# nudge a worker the moment it comes up for air, then let the run continue
async for node in flow.run_streaming(root):
    agent = node.parent_agent
    if agent is not root and isinstance(node, ExecOutput):
        agent.frontier.append(UserQuery(content="Answer with the best evidence."))
```

Injected nodes become ordinary graph nodes with the same shape as organic ones. An append lands on an agent's frontier or it raises, so history stays append-only, and forking is how you change what an agent already did. See [`docs/injections.md`](docs/injections.md) and [`examples/control/injection/`](examples/control/injection/).

## Save, load, and fork

The Node tree is the durable run:

```python
root = flow.start(query)
flow.run(root)
run_dir = root.save("runs/shepherd")

latest = rlmflow.AgentStart.load(run_dir)
```

Fork an independent branch from any node and continue it with a `Flow`:

```python
branch = node.fork()
flow.run(branch)
branch.save("runs/shepherd_repair")
```

A `Flow` that has not run a tree — loaded, forked, or from another process — rebuilds each unfinished agent's REPL from its recorded code before stepping it, or with `Flow(restore="lazy")` tells the agent its variables are gone instead. See [`examples/showcase.py`](examples/showcase.py), [`docs/control.md`](docs/control.md), and [`docs/injections.md`](docs/injections.md).

## Visualization

Because the run *is* a typed graph, every view is a render of that graph. They all live in `rlmflow.view`, re-exported from `rlmflow`, and all work the same on a live tree, a loaded one, or a fork.

### Live terminal tree

`render_tree(root)` is the whole agent tree as ASCII — reprint it each tick to watch a run unfold:

```python
from rlmflow import render_tree

async for node in flow.run_streaming(root):
    print("\033[2J\033[H" + render_tree(root))   # clear + redraw
```

```text
The Sokoban worker in `jam` walked itself into a bad irreversible plan.
└── root: children running 1/2 (2 turns)
    ├── root.branch0: done stuck (16 turns)
    └── root.branch1: planning (5 turns)
```

`LiveTreeRenderer` and `LiveGraphTree` package that as stream consumers if you would rather hand them nodes as they land.

### Full-screen TUI

```bash
pip install "rlmflow[tui]"
rlmflow tui --query "add a test for parse_args"
```

Chat on the left with a tab per agent; clickable tree on the right. Send (or Ctrl+S) submits query and context. Ctrl+R continues a paused run, Ctrl+T steps once.

<p align="center">
  <img src="docs/static/tui.png" alt="The rlmflow TUI: a chat pane on the left showing the root agent's system prompt and query, and a side panel on the right with a Tree tab listing every step of the root and its two sub-agents" width="820" />
</p>

Or hand `FlowTUI` a flow yourself:

```python
from rlmflow.consumers import FlowTUI, GraphCheckpointer

ui = FlowTUI(sink=GraphCheckpointer("runs/coding/graph"))
ui.init(flow)
ui.run()                       # ui.run(query="fix the tests") starts one immediately
```

It is a stream consumer like the rest, so the checkpointer sees every node, and feeding it from your own `async for node in flow.run_streaming(...)` loop still works.

### Browser viewer

`open_viewer(source)` puts a step slider over the figure, an agent picker beside it, and that agent's transcript underneath — all as of the step you are on:

```python
from rlmflow import open_viewer

# the shepherd example's own run, once you have run it
open_viewer("examples/_runs/shepherd/shepherd")   # needs: pip install rlmflow[viewer]
```

<p align="center">
  <img src="docs/static/gradio_ui.png" alt="The browser viewer on the shepherd run: the node graph with the shepherd trunk fanning into eight branch chains, the final done_output beside it reporting the picked branch, a step slider over the full run, and an agent picker above the selected agent's transcript" width="820" />
</p>

### Figures, steppers, and frames

The same graph, without a server:

```python
from rlmflow import graph_svg, render_steps, replay, save_html, save_svg, steps, timeline

for step in steps(root):              # the run in order, with content
    print(step.index, step.title, step.summary)

for snap in replay(root):             # the tree as it stood at each step
    print(render_tree(snap))

save_svg(root, "graph.svg")           # the figure
save_html(root, "run.html")           # a single-file stepper, arrow keys and all
```

The stepper is one self-contained page — the graph with the node you are on ringed, everything after it faded back, its content beside it. It embeds the figure once and reveals it a node at a time, so a 500-node run is a few hundred KB you can mail, not a flipbook of duplicated drawings.

Or from the shell, on any saved run — here the one the shepherd example wrote:

```bash
RUN=examples/_runs/shepherd/shepherd
rlmflow view show $RUN               # the agent tree, then the timeline
rlmflow view show $RUN --step 12     # one step, with its content
rlmflow view show $RUN --frames-only # every step as an ASCII tree
rlmflow render svg $RUN g.svg        # the figure
rlmflow render html $RUN run.html    # steppable single file
rlmflow render browser $RUN          # the viewer above
rlmflow render gif $RUN run.gif      # animated (needs [image])
rlmflow render frames $RUN out/      # a PNG per step (needs [image])
```

`python -m rlmflow view show …` works the same. Everything here needs nothing beyond the standard library except the two that say otherwise: the viewer wants `rlmflow[viewer]`, and PNG/GIF export wants `rlmflow[image]`. See [`docs/observability.md`](docs/observability.md).

## How Shepherd uses the graph

Because a run is a durable graph rather than a one-way transcript, a second agent can be programmed against it. [Shepherd](https://arxiv.org/abs/2605.10913) (Yu et al., 2026) motivates this experimental example, but rlmflow does not implement Shepherd's reversible environment model. It reads an earlier graph node, forks from it, and re-instructs the copy; external state is rebuilt by the example.

[`examples/shepherd/`](examples/shepherd/) does that to a concrete failed trace. A fixed script records a worker shoving one box past saving; a larger model reads the trace, reverts to eight different earlier pushes, and gives each live worker fork its own plan to try in parallel. The forks are its own children, so the search is one agent tree. A runtime-bound current-frontier renderer adds each worker's live board without putting runtime state on graph nodes or closing over the `Flow`.

[`docs/shepherd.md`](docs/shepherd.md) is the full walkthrough: the tools the meta-agent gets, the plans it wrote, and what each branch did.

## DSPy Adapter

`DSPyFlow` lets DSPy use a `Flow` agent anywhere it expects a language model — every DSPy "LM call" becomes a full recursive agent run:

```python
import dspy
import rlmflow
from rlmflow import AgentConfig
from rlmflow.llm import OpenAIClient
from rlmflow.adapters import DSPyFlow, FlowLLM

flow = rlmflow.Flow(
    OpenAIClient(model="gpt-4o-mini"),
    root_config=AgentConfig(max_depth=1, max_iters=5),
)

dspy.configure(lm=DSPyFlow(FlowLLM(flow), model="rlmflow/gpt-4o-mini"))
qa = dspy.ChainOfThought("question -> answer")
print(qa(question="What is 17 * 23?").answer)
```

Install it with `pip install rlmflow[openai,dspy]`. See [`examples/providers/dspy_drop_in.py`](examples/providers/dspy_drop_in.py) for the runnable version.

## Customizable skills

Give agents reusable know-how without baking it into the core prompt. Put project conventions, domain playbooks, child-agent contracts, benchmark heuristics, or lessons from previous runs in `SKILL.md` files, then load the right ones for each root or child agent.

See [`docs/skills.md`](docs/skills.md) for examples of always-on skills, query-selected skills, child-only skills, and run-memory skills. [`examples/skills.py`](examples/skills.py) is the runnable minimal version.

## Examples

Run the offline smoke suite with `python examples/run_examples.py`. Every run writes a pass/fail/skip report to `examples/_runs/examples_report.md`. Add `--include-optional`, `--include-live`, `--include-sandbox`, `--include-manual`, or `--include-slow` as needed. Most live examples share flags like `--docker-image rlmflow:local`, `--max-depth`, and `--max-iters`; see [`examples/README.md`](examples/README.md).

| Example                                                       | What it shows                                                                                    |
| ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| [`showcase.py`](examples/showcase.py)                         | Stepping to boundaries, save/load, and forking a run.                                            |
| [`coding/agent.py`](examples/coding/agent.py)                 | Coding agent over a working directory with `FILE_TOOLS`.                                         |
| [`structured_output.py`](examples/structured_output.py)       | Root and child results validated with JSON Schema / Pydantic.                                    |
| [`drop_in_llm.py`](examples/drop_in_llm.py)                   | Minimal `FlowLLM` adapter, including nested flows.                                               |
| [`skills.py`](examples/skills.py)                             | On-disk skill files loaded through a dynamic prompt section.                                     |
| [`llm_query_batched.py`](examples/llm_query_batched.py)       | Fan out one-shot model calls from agent code without spawning children.                          |
| [`dspy_drop_in.py`](examples/providers/dspy_drop_in.py)       | Use a `Flow` agent as the LM behind a DSPy program.                                              |
| [`mcp_weather.py`](examples/providers/mcp_weather.py)         | Start a local MCP weather server, delegate city forecasts, and combine advice.                   |
| [`tinker_agent.py`](examples/providers/tinker_agent.py)       | Run an agent against `TinkerClient` inference.                                                   |
| [`sandboxes/`](examples/sandboxes/)                           | Build a small web app while Python code runs inside Docker or Modal.                             |
| [`needle/haystack.py`](examples/needle/haystack.py)           | Needle-in-a-haystack over a massive in-memory `INPUTS["haystack"]`.                              |
| [`needle/filesystem.py`](examples/needle/filesystem.py)       | Needle-in-a-haystack across many files with `FILE_TOOLS` and runtime working directories.        |
| [`summarizer.py`](examples/summarizer.py)                     | Recursive map-reduce summarization over a long document.                                         |
| [`step_until.py`](examples/control/delegation/step_until.py)  | Minimal `Flow.run_streaming(..., until=...)` boundaries while delegated child work fans out.     |
| [`control/injection/`](examples/control/injection/)           | Generate a baseline run, append controller Nodes to copies of it, and continue the variants.     |
| [`fork_repair.py`](examples/control/branching/fork_repair.py) | Fork graph/workdir snapshots into independent repair branches and compare results.               |
| [`best_of_n.py`](examples/control/branching/best_of_n.py)     | Run N independent branches and pick the best result.                                             |
| [`shepherd/`](examples/shepherd/)                             | Rewind a jammed Sokoban worker to several depths, race the recovery branches, and keep the best. |
| [`autoresearch/`](examples/autoresearch/)                     | TinyStories autoresearch loop with custom `@tool`s, delegation, and Modal GPU trials.            |
| [`graph/`](examples/graph/)                                   | Offline tour of the Node API: query, navigate, save/load, timing, fork.                          |
| [`run_examples.py`](examples/run_examples.py)                 | Manifest-driven smoke runner for offline, optional, live, sandbox, manual, and slow examples.    |

## Benchmarks (research beta)

Eval is organized around four public **suites** — one question each, not a mixed average.

```bash
make -f benchmarks/Makefile eval-smoke
make -f benchmarks/Makefile eval-conformance
make -f benchmarks/Makefile eval-core
make -f benchmarks/Makefile eval-interactive
python -m benchmarks.eval --list-suites
```

| Suite         | Question                                                       |
| ------------- | -------------------------------------------------------------- |
| `smoke`       | Does the pipeline run end to end without network access?       |
| `conformance` | Do runtimes obey the same offline environment/search contract? |
| `core`        | How do harnesses manage paper-aligned long-context tasks?      |
| `interactive` | How do harnesses scale on persistent visual environments?      |

See [docs/benchmarks.md](docs/benchmarks.md) for the fixed runner matrix, pilot/full splits, the bounded ARC visual showcase, and result artifacts. The repository does not claim comparative quality, cost, or latency beyond checked-in or reproducibly generated run evidence.

## Roadmap

- [~] Four public eval suites (`smoke`, `conformance`, `core`, `interactive`) [benchmarks](docs/benchmarks.md)
- [ ] Additional remote sandbox providers (E2B, Daytona)
- [x] Data-only worker-to-host protocol; `LocalRuntime` remains trusted-code only
- [~] [Experimental RAO utilities](docs/research/rao_implementation_plan.md): rollout collection, per-node rewards, leave-one-out advantages, depth weighting, and trainer export exist; validation results do not
- [ ] [DeLM-style coordination](docs/research/delm_vs_rlmflow.md): verified shared context and multi-worker coordination over Node trees

## Docs

The top-level docs are short, user-facing guides. The deep dive lives in [`docs/internals.md`](docs/internals.md). Research notes live under [`docs/research/`](docs/research/).

- [**Upgrading to 0.6**](UPGRADING.md): breaking API and persistence changes.
- [**Internals**](docs/internals.md): Node structure, Flow transitions, the task queue and pools, Runtime identity, replay, forks, and persistence.
- [Blog post](https://shyamsn97.github.io/blog/rlmflow/): long-form pitch — recursive agents, why graphs beat flat traces, and walkthroughs.
- [Positioning](docs/positioning.md): when to use rlmflow vs rlm-minimal, ypi, LangGraph, CrewAI, AutoGen, SWE-agent, Aider.
- [Command line](docs/cli.md): `rlmflow tui` for a coding agent in the dashboard, `run print` headless, `view` / `render` for a saved run, and how settings resolve.
- [Control](docs/control.md): streaming loop, per-agent limits, multi-turn runs, save/load resume, forks, `INPUTS`, `launch_subagent`, custom tools.
- [Streaming and scheduling](docs/streaming.md): `run_streaming(..., until=...)`, `TaskQueue`, transition diagrams, delegation, parallel roots, boundaries, cancellation, and Pool/Runtime placement.
- [Node model](docs/node_model.md): the typed node hierarchy, the transitions between them, and how delegation is recorded.
- [Skills](docs/skills.md): workspace `SKILL.md` files, query-selected skills, child-only skills, and run-memory skills.
- [Node injection](docs/injections.md): append controller Nodes between streaming calls and continue the same root.
- [Observability](docs/observability.md): querying the Node tree, run layout, stream consumers, reading a saved run, and stepping through one with `rlmflow view`.
- [Runtimes](docs/runtimes.md): `Runtime` protocol, shipped runtimes (Local / subprocess / Docker / Modal), writing your own.
- [Prompt customization](docs/prompt_customization.md): `PromptBuilder`, `Flow.build_system_prompt`, `UserQuery.build_system_prompt`, `PromptProfile` / `prompt_profile` / `prompt_router`.
- [Security](docs/security.md): trust model, Docker isolation knobs, engine-level caps, proxied tools, approval gates.
- [Benchmarks](docs/benchmarks.md): the `smoke`, `conformance`, `core`, and `interactive` suites, with OOLONG as the primary long-context evidence and MazeBench as the visual environment.
- [Changelog](CHANGELOG.md): release-by-release changes.

## References

- [Original recursive-agent work](https://github.com/alexzhang13/rlm): the paper and implementation that inspired this project.
- [rlm-minimal](https://github.com/alexzhang13/rlm-minimal): the single-file reference rlmflow grew from.
- [Tau](https://github.com/huggingface/tau): Hugging Face's minimalist coding-agent harness. rlmflow's separation of execution, durable state, and consumers follows Tau's clean harness design.
- [Scaling Managed Agents: Decoupling the brain from the hands](https://www.anthropic.com/engineering/managed-agents): Anthropic's writeup on separating harness, session, and sandbox interfaces for long-horizon agents.
- [ypi](https://github.com/rawwerks/ypi): recursive coding agent built on Pi. Our session layout and much of the default prompt (size-up → delegate → combine, guardrails, aggressive delegation) come from ypi's `SYSTEM_PROMPT.md`.

## License

See [LICENSE](LICENSE).

## Citation

```bibtex
@misc{sudhakaran2025rlmflow,
  author = {Sudhakaran, Shyam},
  title = {rlmflow},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/shyamsn97/rlmflow}},
}
```
