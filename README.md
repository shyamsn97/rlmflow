# rlmflow

<p align="center">
  <a href="https://pypi.org/project/rlmflow/"><img src="https://img.shields.io/pypi/v/rlmflow.svg?label=pypi" alt="PyPI" /></a>
  <a href="https://github.com/shyamsn97/rlmflow/pkgs/container/rlmflow"><img src="https://img.shields.io/badge/ghcr.io-rlmflow-2496ED?logo=docker&logoColor=white" alt="Docker" /></a>
</p>

Read the blog post: [rlmflow](https://shyamsn97.github.io/blog/rlmflow/).

**rlmflow** is a Python library for building **recursive agents** — agents
that spawn other agents — as live execution graphs. Every query, action,
observation, child call, wait, resume, and result is a typed node you can
inspect, step through, save, fork, and edit mid-run.

It gives an LLM a stateful Python REPL and a graph-native way to spawn agents
with fresh context.

<p align="center">
  <img src="docs/rlm_animation.gif" alt="rlmflow animation" />
</p>

## Recursive agents as graphs

Recursive agents let a model split work across fresh sub-agents, each with its
own context, tools, and execution history. Those sub-agents can delegate again,
so one root task can quickly become a tree of parallel work.

For example, the root agent can split a haystack into chunks:

```python
results = await launch_subagents([
    {"name": "chunk_0", "query": "scan first third", "inputs": {"chunk": chunk_0}},
    {"name": "chunk_1", "query": "scan middle third", "inputs": {"chunk": chunk_1}},
    {"name": "chunk_2", "query": "scan final third", "inputs": {"chunk": chunk_2}},
])
done(extract_answer(results))
```

Then `chunk_2` can recursively delegate again:

```python
hits = find_candidate_windows(INPUTS["chunk"])
results = await launch_subagents([
    {"name": "candidate_a", "query": "inspect window A", "inputs": {"window": hits[0]}},
    {"name": "candidate_b", "query": "inspect window B", "inputs": {"window": hits[1]}},
])
done(select_candidate(results))
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

That parent API is useful: children return simple values the parent can compose.
The problem is when those return values are the only surviving record of the
work:

```text
results == ["not found", "decoy, no code", "candidate code 84721"]
```

If `chunk_2` launched `candidate_a` and `candidate_b` internally, the parent can
still receive `"candidate code 84721"` as its normal result. But for debugging,
evaluation, supervision, or reuse, you also want the execution state that
produced it: which agents ran, what they saw, where they waited, what failed,
and where you could intervene.

**rlmflow** keeps that structure alive. Every recursive call is a
sub-`Graph`, and every turn inside an agent is a typed `Node`:

<p align="center">
  <img src="docs/recursive_agents_graph.svg" alt="rlmflow graph showing root, child agents, grandchild agents, supervision, resume, and final result states" />
</p>

The graph is the run itself:

```python
from rlmflow import Graph, render_tree

graph = Graph(query=query)

async for event in agent.run_streaming(graph=graph):
    print(render_tree(graph))
```

`run_streaming` mutates `graph` in place and yields one typed `Event` per commit, so
the live tree is just `render_tree(graph)`. The graph *is* the durable state: you
can inspect a branch, save a checkpoint, fork from an old snapshot, inject
controller feedback, replace a bad node, and continue from the edited graph.

See [`docs/internals.md`](docs/internals.md), [`docs/node_model.md`](docs/node_model.md),
and [`docs/control.md`](docs/control.md) for the full engine and graph API.

## Install

```
pip install rlmflow               # core
pip install rlmflow[openai]       # + OpenAI client
pip install rlmflow[anthropic]    # + Anthropic client
pip install rlmflow[tinker]       # + Tinker inference client
pip install rlmflow[dspy]         # + DSPy adapter
pip install rlmflow[sandbox]      # + Modal, E2B, and Daytona runtimes
pip install rlmflow[viewer]       # + Gradio viewer (plotly)
pip install rlmflow[image]        # + static image / GIF export (kaleido)
pip install rlmflow[all]          # all of the above
```

From source:

```
git clone https://github.com/shyamsn97/rlmflow && cd rlmflow
pip install -e .
```

For local development, `make install` runs cleanup, formatting/lint checks
including `ruff check .`, then installs the package.

> **Security warning — `LocalRuntime` is not a sandbox.**
> Agent code runs as full Python in your process: filesystem, network,
> environment variables, subprocesses — the same privileges as your interpreter.
> LLM-generated code can be wrong or malicious (prompt injection, model errors,
> supply-chain risk). **Use `LocalRuntime` only for code you would run yourself.**
> For untrusted agents or anything exposed to the internet, use
> [`DockerRuntime`](docs/runtimes.md) or a remote sandbox
> ([`ModalRuntime`](docs/runtimes.md) / [`E2BRuntime`](docs/runtimes.md) /
> [`DaytonaRuntime`](docs/runtimes.md)). See [`docs/security.md`](docs/security.md).
> **Use at your own risk.**

## Quick start

This example builds a simple coding agent with file tools in a local working
directory. See [`examples/coding/agent.py`](./examples/coding/agent.py) for the
full interactive version.

```python
from pathlib import Path

from rlmflow.clients import OpenAIClient
from rlmflow import FILE_TOOLS, Flow, Graph, LocalRuntime, open_viewer, render_tree

workdir = Path("examples/_runs/quickstart")
runtime = LocalRuntime(working_directory=workdir)
runtime.register_tools(FILE_TOOLS)

flow = Flow(
    OpenAIClient(model="gpt-5"),
    runtime=runtime,
    max_depth=2,
    max_iters=20,
    llm_clients={"fast": OpenAIClient(model="gpt-5-mini")},
)

query = "Build a Python text-based adventure game with combat and inventory."
graph = Graph(query=query)

async for event in flow.run_streaming(graph=graph):
    print(render_tree(graph))

print(graph.result())
graph.save(workdir / "graph")
open_viewer(workdir / "graph", launch=False)
```

With the minimal event loop, delegated children fan out through the shared async
pool by default. See
[`examples/control/delegation/step_until.py`](./examples/control/delegation/step_until.py)
for a deterministic demo of `Flow.run_streaming(..., until=...)` boundaries while
child work runs concurrently.

A saved run is a directory rooted at `graph.json` plus `agents/` logs. Reopen it
later with `Graph.load(path)` or `open_viewer(path)`.

## Drop-in `LLMClient`

`FlowLLM` wraps a `Flow` in the `LLMClient` `chat(messages)` interface, so a full
recursive agent run is a drop-in replacement for any raw LLM.

```python
import rlmflow
from rlmflow import FlowLLM
from rlmflow.clients import LLMClient, OpenAIClient

def ask(llm: LLMClient, q: str) -> str:
    return llm.chat([{"role": "user", "content": q}])

ask(OpenAIClient(model="gpt-4o-mini"), "2+2?")                      # one LLM call
ask(FlowLLM(rlmflow.Flow(OpenAIClient(model="gpt-4o-mini"))), "2+2?")  # full agent loop
```

Nest agents by passing one `FlowLLM(inner_flow)` as another flow's `llm`. See
[`examples/drop_in_llm.py`](examples/drop_in_llm.py).

## Stream and inspect

`Flow` gives you three ways to drive a run, all over the same durable `Graph`:

- `flow.run(query=query)` — run to completion and return the result string.
- `async for event in flow.run_streaming(graph=graph, until=..., n=...)` — stream to a
  boundary; the graph is mutated in place.
- `async for event in flow.run_streaming(graph=graph)` — stream every commit as it happens.

```python
from rlmflow import Graph, render_tree

graph = Graph(query=query)

async for event in agent.run_streaming(graph=graph, until="next", n=2):
    pass                              # advance two global steps
print(render_tree(graph))             # ...inspect...
async for event in agent.run_streaming(graph=graph, until="done"):
    pass                              # ...then run to completion
print(render_tree(graph))
```

```text
scan the repo for vulnerabilities
└── root: done audited 3 modules (4 turns)
    ├── root.scanner_auth: done found SQL injection in login.py (2 turns)
    ├── root.scanner_api: waiting on 2 children (3 turns)
    └── root.scanner_db: done no issues found (2 turns)
```

Every transition follows the same obs → action → obs shape:

```text
LLMOutput  -> ExecAction -> ExecOutput          (REPL output, normal continuation)
                         -> DoneOutput          (code called done())
                         -> ErrorOutput         (code raised / no code block)
                         -> SupervisingOutput   (awaited a launcher — waiting on children)
SupervisingOutput -> ResumeAction -> ...        (children settled — supervisor resumes)
```

The graph is queryable in plain Python:

```python
render_tree(graph)              # ASCII tree render (rlmflow.render_tree)
graph["root.scanner_api"]       # sub-Graph rooted at that agent
graph["root.scanner_api"].nodes # node trajectory for one agent
graph.children                  # dict[str, Graph] of direct children
graph.agents                    # dict[str, Graph] of every agent in the tree
list(graph.walk())              # every agent, preorder
graph.current()                 # latest node on the root agent
graph.result()                  # terminal DoneOutput result
graph.tokens(), graph.total_tokens()  # usage accounting
graph.to_dict()                 # full JSON-serializable payload
```

## Inject controller events

Because `Graph` is the control surface, external controllers can edit it in place
and continue the run. `graph.inject(text, agent_id=...)` appends a user-turn
observation to any agent; its `mode`/`truncate` options and the
`append`/`prepend`/`replace` helpers, plus `rewind`, `checkpoint`/`revert`, and
`remove_child`, rewrite history. This is useful for human feedback, budget nudges,
and forced finalization without losing traceability:

```python
# nudge one agent, then continue the run from the edited graph
graph.inject(
    "Answer with the best current evidence.",
    agent_id="root.scanner_api",
)
agent.run(graph=graph)  # or: async for _ in agent.run_streaming(graph=graph): ...
```

Injected nodes become ordinary graph nodes with the same shape as organic ones,
stamped with ``metadata["injected"] = True``.
See [`docs/injections.md`](docs/injections.md) and
[`examples/control/controller_injection.py`](examples/control/controller_injection.py).

## Save, load, fork

`Graph` is the durable run object. Save a run directory with `graph.save(...)`
and reopen it with `Graph.load(...)`:

```python
graph = rlmflow.Graph(query=query)
agent.run(graph=graph)
run_dir = graph.save("runs/deep_research")

latest = rlmflow.Graph.load(run_dir)
```

Fork an independent branch from any point and continue it with a `Flow`:

```python
branch = latest.fork(session="isolated")   # deep copy with a fresh graph_id
agent.run(branch)
branch.save("runs/deep_research_repair")
```

Controller edits use the same graph surface (`inject`/`append`/`prepend`/
`replace`, `rewind`, `checkpoint`/`revert`) and then continue through
`agent.run(graph=graph)` or `agent.run_streaming(graph=graph)`. See [`examples/showcase.py`](examples/showcase.py),
[`docs/control.md`](docs/control.md), [`docs/injections.md`](docs/injections.md),
and the live graph-controller pool example in
[`examples/control/graph_controller_agent.py`](examples/control/graph_controller_agent.py).

## Visualization

Because the run is a typed graph, every view is just a render of that graph. It
all lives in `rlmflow.view` (re-exported from `rlmflow`) and works on a saved run
directory, a single `Graph`, or a list of step snapshots.

### Live terminal tree

`render_tree(graph)` renders a compact ASCII tree of the whole agent tree from
any snapshot — reprint it each tick to watch a run unfold:

```python
from rlmflow import render_tree

async for event in agent.run_streaming(graph=graph):
    print("\033[2J\033[H" + render_tree(graph))  # clear + redraw
```

`LiveTreeRenderer` packages the same event-consuming pattern if you'd rather hand
it events as they arrive.

### Gradio viewer

![](docs/static/gradio_ui.png)

`open_viewer(source)` launches a small browser app with a step slider, an agent
selector, and a per-agent transcript over a swimlane of the run:

```python
from rlmflow import open_viewer

open_viewer("runs/deep_research")   # needs: pip install rlmflow[viewer]
```

### Full-screen TUI

![rlmflow TUI showing chat, query/context inputs, and execution tree](docs/static/tui.png)

`FlowTUI` is a stream consumer: feed it events from your own
`async for event in flow.run_streaming(...)` loop (or pass that loop as `drive`
to `FlowTUI().run(drive)` for the interactive prompt UI). It shows separate
query/context inputs, live chat bubbles, and side tabs for the execution tree,
agents, counts, waiting supervisors, errors, and latest nodes. Ctrl+S sends,
Ctrl+R continues a paused run, Ctrl+T steps once. See
[`examples/coding/agent.py`](examples/coding/agent.py) for a runnable example.

### Replay / render a saved run

Point anything at a graph directory (or a `Graph`) and walk the sequence:

```python
from rlmflow import replay, render_steps, render_tree

# Graph snapshots, one per visible step
for snap in replay("runs/coding/graph"):
    print(render_tree(snap))

# Or just the ASCII frames
for i, frame in enumerate(render_steps("runs/coding/graph"), 1):
    print(f"=== step {i} ===\n{frame}")
```

Or from the shell (after `pip install -e .`):

```bash
rlmflow view runs/coding/graph              # print every ASCII step
rlmflow view runs/coding/graph --step 3     # one step
rlmflow view runs/coding/graph --browser    # Gradio stepper
rlmflow view runs/coding/graph --frames out/  # PNG strip (needs [image])
rlmflow view runs/coding/graph --gif run.gif
```

`python -m rlmflow view …` works the same.

### Static frames for docs and blogs

`save_image` writes one PNG/SVG/PDF of a run; `save_steps` writes one frame per
execution step with a stable layout, ideal for GIFs or blog strips. Both accept
a saved run directory, a `Graph`, or a list of snapshots, and need
`pip install rlmflow[image]` (Plotly + kaleido):

```python
from rlmflow import save_image, save_steps

save_image("runs/deep_research", "hero.png")
save_steps(
    "runs/deep_research",
    "blog/frames/",           # writes step_000.png, step_001.png, ...
    width=1600, height=1200, scale=2,
    marker_mult=3.5,          # fatter node dots + edges
    text_mult=2.2,            # shrink labels so they don't collide
)
```

See [`examples/view_demo.py`](./examples/view_demo.py) for a runnable viewer demo.

## DSPy Adapter

`DSPyFlow` lets DSPy use a `Flow` agent anywhere it expects a language model —
every DSPy "LM call" becomes a full recursive agent run:

```python
import dspy
import rlmflow
from rlmflow import DSPyFlow, FlowLLM
from rlmflow.clients import OpenAIClient

flow = rlmflow.Flow(OpenAIClient(model="gpt-4o-mini"), max_depth=1, max_iters=5)

dspy.configure(lm=DSPyFlow(FlowLLM(flow), model="rlmflow/gpt-4o-mini"))
qa = dspy.ChainOfThought("question -> answer")
print(qa(question="What is 17 * 23?").answer)
```

Install it with `pip install rlmflow[openai,dspy]`. See
[`examples/providers/dspy_drop_in.py`](examples/providers/dspy_drop_in.py) for the runnable
version.

## Customizable skills

Give agents reusable know-how without baking it into the core prompt. Put
project conventions, domain playbooks, child-agent contracts, benchmark
heuristics, or lessons from previous runs in `SKILL.md` files, then load the
right ones for each root or child agent.

See [`docs/skills.md`](docs/skills.md) for examples of always-on skills,
query-selected skills, child-only skills, and run-memory skills.
[`examples/skills.py`](examples/skills.py) is the runnable minimal version.

## Examples

Run the offline smoke suite with `python examples/run_examples.py`.
Add `--include-optional`, `--include-live`, `--include-sandbox`, or
`--include-manual` as needed. Most live examples share flags like `--no-viz`,
`--docker-image rlmflow:local`, `--max-depth`, and `--max-iters`; see
[`examples/README.md`](examples/README.md).

| Example | What it shows |
|---|---|
| [`showcase.py`](examples/showcase.py) | Functional stepping, snapshots, save/load, and live terminal visualization. |
| [`coding/agent.py`](examples/coding/agent.py) | Coding agent in the rich TUI (or `--cli` for the REPL + live tree). |
| [`structured_output.py`](examples/structured_output.py) | Root and child results validated with JSON Schema / Pydantic. |
| [`drop_in_llm.py`](examples/drop_in_llm.py) | Minimal `FlowLLM` adapter, including nested flows. |
| [`skills.py`](examples/skills.py) | On-disk skill files loaded through a dynamic prompt section. |
| [`dspy_drop_in.py`](examples/providers/dspy_drop_in.py) | Use a minimal `Flow` agent as the LM behind a DSPy program. |
| [`mcp_weather.py`](examples/providers/mcp_weather.py) | Start a local MCP weather server, delegate city forecasts, and combine advice. |
| [`tinker_agent.py`](examples/providers/tinker_agent.py) | Run the live terminal graph view with `TinkerClient` inference. |
| [`sandboxes/`](examples/sandboxes/) | Build a small web app while Python code runs inside Docker or Modal. |
| [`needle/haystack.py`](examples/needle/haystack.py) | Needle-in-a-haystack over a massive in-memory `INPUTS["haystack"]`. |
| [`needle/filesystem.py`](examples/needle/filesystem.py) | Needle-in-a-haystack across many files with `FILE_TOOLS` and runtime working directories. |
| [`summarizer.py`](examples/summarizer.py) | Recursive map-reduce summarization over a long document. |
| [`step_until.py`](examples/control/delegation/step_until.py) | Minimal `Flow.run_streaming(..., until=...)` boundaries while delegated child work fans out. |
| [`graph_controller_agent.py`](examples/control/graph_controller_agent.py) | Live controller agent creates a diversified worker pool with `create_worker(...)`, inspects query/graph diversity, advances named worker graphs, and saves all runs under `examples/_runs/graph_controller_runs/`. |
| [`control/injection/`](examples/control/injection/) | Generate a baseline run, edit copies with graph injection/replacement, and continue variants. |
| [`fork_repair.py`](examples/control/branching/fork_repair.py) | Fork graph/workdir snapshots into independent repair branches and compare results. |
| [`best_of_n.py`](examples/control/branching/best_of_n.py) | Run N independent branches and pick the best result. |
| [`autoresearch/`](examples/autoresearch/) | TinyStories autoresearch loop with custom `@tool`s, delegation, and Modal GPU trials. |
| [`shepherd/`](examples/shepherd/) | Meta-agent recovers a jammed Sokoban worker: rewind + warm `launch_subgraphs` children, `LiveGraphTree` + board panels, play-by-play GIF/JSON traces. |
| [`graph/`](examples/graph/) | Offline tour of the `Graph` API: query, navigate, mutate, save/load, rewind, fork, render. |
| [`run_examples.py`](examples/run_examples.py) | Manifest-driven smoke runner for offline, optional, live, sandbox, and manual examples. |
| [`view_demo.py`](examples/view_demo.py) | Build synthetic minimal `Graph` snapshots and launch the lightweight viewer. |

## Benchmarks

The shared eval harness lives under [`benchmarks/eval/`](benchmarks/eval/).
It uses a task/runner registry, writes `results.jsonl` + `summary.json`, records
rlmflow graph-shape metrics, shows tqdm progress bars, and can log per-row metrics
to W&B. Real runs can compare `vanilla`, `rlmflow`, and the upstream official RLM
runner ported from [`avilum/minrlm/eval`](https://github.com/avilum/minrlm/tree/master/eval).
It also writes model-oriented reports under `eval-runs/<model>/<benchmark>/`,
including per-question JSON files with prompt, inputs, expected answer, and each
runner's solution.

```bash
make eval-benchmark EVAL_MODEL=gpt-5-mini
```

See [`benchmarks/eval/README.md`](benchmarks/eval/README.md) for task/runner
extension points and W&B usage.

## Roadmap
- [~] OOLONG, LongBench-v2, CodeQA, SWE-bench, etc. benchmarks [benchmarks](benchmarks/eval/)
- [~] Remote sandbox support (modal, e2b, daytona)
- [ ] **REPL security (local)**
- [ ] [RAO library module](docs/research/rao_implementation_plan.md): `rlmflow.rao` rollout collection, per-node rewards, leave-one-out advantages, depth weighting, trainer export
- [ ] [DeLM-style coordination](docs/research/delm_vs_rlmflow.md): shared task queue, verified shared context, multi-worker coordinator over `Flow` graphs

## Docs

The top-level docs are short, user-facing guides. The deep dive lives
in [`docs/internals.md`](docs/internals.md). Research notes live under
[`docs/research/`](docs/research/).

- [**Internals**](docs/internals.md): deep reference — the `Flow`/`Graph`/`Run`
  split, the two phases of a run, the per-agent scheduler, exec turns,
  delegation, REPL lifecycle, runtime backends, graph persistence, and extension
  seams.
- [Blog post](https://shyamsn97.github.io/blog/rlmflow/): long-form pitch —
  recursive agents, why graphs beat flat traces, and walkthroughs.
- [Positioning](docs/positioning.md): when to use rlmflow vs
  rlm-minimal, ypi, LangGraph, CrewAI, AutoGen, SWE-agent, Aider.
- [Control](docs/control.md): streaming loop, save/load resume, rewind,
  forks, `INPUTS`, `launch_subagents`, inline-first strategy, custom tools.
- [Streaming and scheduling](docs/streaming.md): detailed guide to
  `run_streaming(..., until=...)`, per-agent queues, boundaries, and delegation.
- [Skills](docs/skills.md): workspace `SKILL.md` files, query-selected
  skills, child-only skills, and run-memory skills.
- [Node injection](docs/injections.md): append typed controller events to a
  running graph and continue it with `agent.run(graph=graph)` / `agent.run_streaming(graph=graph)`.
- [Observability](docs/observability.md): querying the `Graph`,
  run layout, the live terminal tree, the Gradio viewer, and static
  image/step exports.
- [Runtimes](docs/runtimes.md): `Runtime` protocol, shipped runtimes
  (Local / Docker / Modal / E2B / Daytona), writing your own.
- [Prompt customization](docs/prompt_customization.md): `SystemPromptBuilder`
  sections, callable dynamic sections, deriving from the default prompt,
  full replacement.
- [Security](docs/security.md): trust model, Docker isolation knobs,
  engine-level caps, proxied tools, approval gates.
- [Changelog](CHANGELOG.md): release-by-release changes.

## References

- [Original recursive-agent work](https://github.com/alexzhang13/rlm): the
  paper and implementation that inspired this project.
- [rlm-minimal](https://github.com/alexzhang13/rlm-minimal): the
  single-file reference rlmflow grew from.
- [Tau](https://github.com/huggingface/tau): Hugging Face's minimalist
  coding-agent harness. Much of rlmflow's event-driven loop — typed
  events as the contract between engine, consumers, and frontends —
  follows Tau's clean separation of harness, session, and rendering.
- [Scaling Managed Agents: Decoupling the brain from the hands](https://www.anthropic.com/engineering/managed-agents):
  Anthropic's writeup on separating harness, session, and sandbox
  interfaces for long-horizon agents.
- [ypi](https://github.com/rawwerks/ypi): recursive coding agent built
  on Pi. Our session layout and much of the default prompt
  (size-up → delegate → combine, guardrails, aggressive delegation) come
  from ypi's `SYSTEM_PROMPT.md`.

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
