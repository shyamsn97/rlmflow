# rlmflow

<p align="center">
  <a href="https://pypi.org/project/rlmflow/"><img src="https://img.shields.io/pypi/v/rlmflow.svg?label=pypi" alt="PyPI" /></a>
  <a href="https://github.com/shyamsn97/rlmflow/pkgs/container/rlmflow"><img src="https://img.shields.io/badge/ghcr.io-rlmflow-2496ED?logo=docker&logoColor=white" alt="Docker" /></a>
</p>

Read the blog post: [rlmflow](https://shyamsn97.github.io/blog/rlmflow/).

**rlmflow** is a Python library for building **recursive agents** — agents
that spawn other agents — as live execution graphs. Every query, action,
observation, child agent, and result is a typed node you can inspect, step
through, save, fork, and append to mid-run.

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
produced it: which agents ran, what they saw, what failed, and where you could
intervene.

**rlmflow** keeps that structure alive. Every recursive call is a child agent
inside one typed `Node` tree:

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

`run_streaming` mutates the root in place and yields each node as it lands, so
the live tree is always just `root`. The Node tree *is* the durable state: you
can inspect a branch, save a checkpoint, fork from an old node, append
controller feedback, and continue from the edited graph.

See [`docs/internals.md`](docs/internals.md), [`docs/node_model.md`](docs/node_model.md),
and [`docs/control.md`](docs/control.md) for the full engine and graph API.

## Install

```
pip install rlmflow               # core
pip install rlmflow[openai]       # + OpenAI client
pip install rlmflow[anthropic]    # + Anthropic client
pip install rlmflow[tinker]       # + Tinker inference client
pip install rlmflow[dspy]         # + DSPy adapter
pip install rlmflow[modal]        # + Modal runtime
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
> ([`ModalRuntime`](docs/runtimes.md)). See [`docs/security.md`](docs/security.md).
> **Use at your own risk.**

## Quick start

This example builds a simple coding agent with file tools in a local working
directory. See [`examples/coding/agent.py`](./examples/coding/agent.py) for the
full interactive version.

```python
from pathlib import Path

from rlmflow.llm import OpenAIClient
from rlmflow import FILE_TOOLS, AgentConfig, Flow, LocalRuntime

workdir = Path("examples/_runs/quickstart")

flow = Flow(
    OpenAIClient(model="gpt-5"),
    runtime=LocalRuntime(working_directory=workdir),
    tools=FILE_TOOLS,
    config=AgentConfig(max_depth=2, max_iters=20),
    llm_clients={"fast": OpenAIClient(model="gpt-5-mini")},
)

query = "Build a Python text-based adventure game with combat and inventory."
root = flow.start(query)

async for node in flow.run_streaming(root):
    print(node.parent_agent.config.path, node.type)

print(root.result())
root.save(workdir / "graph")
```

Delegated children fan out as ordinary asyncio tasks that step alongside their
parent. See
[`examples/control/delegation/step_until.py`](./examples/control/delegation/step_until.py)
for a deterministic demo of `Flow.run_streaming(..., until=...)` boundaries while
child work runs concurrently.

A saved run is a directory rooted at `graph.json` plus `agents/` logs. Reopen it
later with `AgentStart.load(path)` and keep running it.

## Drop-in `LLMClient`

`FlowLLM` wraps a `Flow` in the `LLMClient` `chat(messages)` interface, so a full
recursive agent run is a drop-in replacement for any raw LLM.

```python
import rlmflow
from rlmflow.llm import LLMClient, OpenAIClient
from rlmflow.adapters import FlowLLM

def ask(llm: LLMClient, q: str) -> str:
    return llm.chat([{"role": "user", "content": q}])

ask(OpenAIClient(model="gpt-4o-mini"), "2+2?")                      # one LLM call
ask(FlowLLM(rlmflow.Flow(OpenAIClient(model="gpt-4o-mini"))), "2+2?")  # full agent loop
```

Nest agents by passing one `FlowLLM(inner_flow)` as another flow's `llm`. See
[`examples/drop_in_llm.py`](examples/drop_in_llm.py).

## Stream and inspect

`Flow` gives you three ways to drive the same durable Node tree:

- `flow.run(query)` — run to completion and return the result.
- `await flow.arun(root)` — async completion.
- `async for node in flow.run_streaming(root, until=...)` — stream landed
  Nodes to a boundary.

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
                         -> DoneOutput    (code called done())
                         -> ErrorOutput   (code raised, or the REPL died)
```

A turn that delegates hangs its children off the `ExecAction` that launched
them, and lands one `ExecOutput` when the block finishes — so a parent's
transcript stays a single chain whether or not it spawned anyone.

The graph is queryable in plain Python:

```python
root.frontier                           # current Node for this agent
root.transcript()                       # this agent's typed history
root.sub_agents                         # the agents it launched
root.leaves()                           # the frontier of every agent in the tree
list(root.walk())                       # every Node, preorder
root.result()                           # root DoneOutput result
root.tokens()                           # recursive usage accounting
persistence.to_dict(root)               # JSON-serializable payload
```

## Inject controller Nodes

Because the Node tree is the control surface, an external controller can append
to any agent's frontier — as its node lands, or between streaming calls — and
let the run continue. This is useful for human feedback, budget nudges, and
forced finalization without losing traceability:

```python
from rlmflow import ExecOutput, UserQuery

# nudge a worker the moment it comes up for air, then let the run continue
async for node in flow.run_streaming(root):
    agent = node.parent_agent
    if agent is not root and isinstance(node, ExecOutput):
        agent.frontier.append(UserQuery(content="Answer with the best evidence."))
```

Injected nodes become ordinary graph nodes with the same shape as organic ones.
An append lands on an agent's frontier or it raises, so history stays
append-only, and forking is how you change what an agent already did.
See [`docs/injections.md`](docs/injections.md) and
[`examples/control/injection/`](examples/control/injection/).

## Save, load, and fork

The Node tree is the durable run:

```python
root = flow.start(query)
flow.run(root)
run_dir = root.save("runs/deep_research")

latest = rlmflow.AgentStart.load(run_dir)
```

Fork an independent branch from any node and continue it with a `Flow`:

```python
branch = node.fork()
flow.run(branch)
branch.save("runs/deep_research_repair")
```

A `Flow` that has not run a tree — loaded, forked, or from another process —
rebuilds each unfinished agent's REPL from its recorded code before stepping it,
or with `Flow(restore="lazy")` tells the agent its variables are gone instead.
See [`examples/showcase.py`](examples/showcase.py),
[`docs/control.md`](docs/control.md), and [`docs/injections.md`](docs/injections.md).

## Rewind a stuck agent, then branch

[`examples/shepherd/`](examples/shepherd/) runs that fork API on a real failure. A
small model plays Sokoban one push at a time and shoves a box flat against the
wall. You push a box by standing on the opposite side, and there is nowhere to
stand inside a wall, so that box can never move again. It never reached a goal,
so nothing the worker does from here solves the board:

<p align="center">
  <img src="docs/shepherd_jam.gif" alt="Sokoban worker shoving a box east until it sits against the wall, unpushable" width="340" />
</p>

The board cannot be recovered, but the transcript can. A larger model reads the
stuck worker, previews what the board looked like at each earlier push, and forks
it into eight recovery branches — each rewound to a different depth and handed a
different box-to-goal plan. They run in parallel under one root, and the best
scoring branch is kept. Every box below is one agent, showing the board it
stopped on:

<p align="center">
  <img src="docs/shepherd_graph.svg" alt="Agent graph of one shepherd run: the jammed worker's final board, the shepherd that rewound it, and eight recovery branches each drawn with its rewind depth, final Sokoban board, and solved or stuck outcome, with the picked winner highlighted" />
</p>

Rewind depth is a trade-off. A shallow rewind keeps finished work but leaves the
bad plan in the worker's visible history, where it tends to imitate it: `branch1`
kept seven of the eight jammed pushes and needed 27 turns to dig out. The winner
threw all eight away and solved the board in 11 pushes:

<p align="center">
  <img src="docs/shepherd_recovery.gif" alt="The winning branch solving the same Sokoban board, locking every box on a goal" width="340" />
</p>

The worker and the shepherd are different models with different prompts on one
flow, so the cheap model plays and the expensive one supervises:

```python
flow = Flow(
    build_client("gpt-5"),                               # the shepherd
    llm_clients={"worker": build_client("gpt-5-mini")},  # the players
    prompt_profiles={
        "worker": PromptProfile(system=WORKER_SYSTEM, user=UserPromptBuilder(board_prompt))
    },
)
worker = flow.start(WORKER_QUERY, model="worker", prompt_profile="worker", max_depth=0)
```

The jam, the planning, all eight branches, and the pick are 526 nodes in one
tree, so the losing branches are still there to read afterwards:

```
python examples/shepherd/shepherd.py       # --gradio for the live board dashboard
python examples/shepherd/render_graph.py   # redraw the figure above from the saved run
```

## DSPy Adapter

`DSPyFlow` lets DSPy use a `Flow` agent anywhere it expects a language model —
every DSPy "LM call" becomes a full recursive agent run:

```python
import dspy
import rlmflow
from rlmflow import AgentConfig
from rlmflow.llm import OpenAIClient
from rlmflow.adapters import DSPyFlow, FlowLLM

flow = rlmflow.Flow(
    OpenAIClient(model="gpt-4o-mini"),
    config=AgentConfig(max_depth=1, max_iters=5),
)

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

Run the offline smoke suite with `python examples/run_examples.py`. Every run
writes a pass/fail/skip report to `examples/_runs/examples_report.md`. Add
`--include-optional`, `--include-live`, `--include-sandbox`, `--include-manual`,
or `--include-slow` as needed. Most live examples share flags like
`--docker-image rlmflow:local`, `--max-depth`, and `--max-iters`; see
[`examples/README.md`](examples/README.md).

| Example | What it shows |
|---|---|
| [`showcase.py`](examples/showcase.py) | Stepping to boundaries, save/load, and forking a run. |
| [`coding/agent.py`](examples/coding/agent.py) | Coding agent over a working directory with `FILE_TOOLS`. |
| [`structured_output.py`](examples/structured_output.py) | Root and child results validated with JSON Schema / Pydantic. |
| [`drop_in_llm.py`](examples/drop_in_llm.py) | Minimal `FlowLLM` adapter, including nested flows. |
| [`skills.py`](examples/skills.py) | On-disk skill files loaded through a dynamic prompt section. |
| [`llm_query_batched.py`](examples/llm_query_batched.py) | Fan out one-shot model calls from agent code without spawning children. |
| [`dspy_drop_in.py`](examples/providers/dspy_drop_in.py) | Use a `Flow` agent as the LM behind a DSPy program. |
| [`mcp_weather.py`](examples/providers/mcp_weather.py) | Start a local MCP weather server, delegate city forecasts, and combine advice. |
| [`tinker_agent.py`](examples/providers/tinker_agent.py) | Run an agent against `TinkerClient` inference. |
| [`sandboxes/`](examples/sandboxes/) | Build a small web app while Python code runs inside Docker or Modal. |
| [`needle/haystack.py`](examples/needle/haystack.py) | Needle-in-a-haystack over a massive in-memory `INPUTS["haystack"]`. |
| [`needle/filesystem.py`](examples/needle/filesystem.py) | Needle-in-a-haystack across many files with `FILE_TOOLS` and runtime working directories. |
| [`summarizer.py`](examples/summarizer.py) | Recursive map-reduce summarization over a long document. |
| [`step_until.py`](examples/control/delegation/step_until.py) | Minimal `Flow.run_streaming(..., until=...)` boundaries while delegated child work fans out. |
| [`control/injection/`](examples/control/injection/) | Generate a baseline run, append controller Nodes to copies of it, and continue the variants. |
| [`fork_repair.py`](examples/control/branching/fork_repair.py) | Fork graph/workdir snapshots into independent repair branches and compare results. |
| [`best_of_n.py`](examples/control/branching/best_of_n.py) | Run N independent branches and pick the best result. |
| [`shepherd/`](examples/shepherd/) | Rewind a jammed Sokoban worker to several depths, race the recovery branches, and keep the best. |
| [`autoresearch/`](examples/autoresearch/) | TinyStories autoresearch loop with custom `@tool`s, delegation, and Modal GPU trials. |
| [`graph/`](examples/graph/) | Offline tour of the Node API: query, navigate, save/load, timing, fork. |
| [`run_examples.py`](examples/run_examples.py) | Manifest-driven smoke runner for offline, optional, live, sandbox, manual, and slow examples. |

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
- [ ] Additional remote sandbox providers (E2B, Daytona)
- [ ] **REPL security (local)**
- [ ] [RAO library module](docs/research/rao_implementation_plan.md): `rlmflow.rao` rollout collection, per-node rewards, leave-one-out advantages, depth weighting, trainer export
- [ ] [DeLM-style coordination](docs/research/delm_vs_rlmflow.md): verified shared context and multi-worker coordination over Node trees

## Docs

The top-level docs are short, user-facing guides. The deep dive lives
in [`docs/internals.md`](docs/internals.md). Research notes live under
[`docs/research/`](docs/research/).

- [**Internals**](docs/internals.md): Node structure, Flow transitions,
  the task queue and pools, Runtime identity, replay, forks, and persistence.
- [Blog post](https://shyamsn97.github.io/blog/rlmflow/): long-form pitch —
  recursive agents, why graphs beat flat traces, and walkthroughs.
- [Positioning](docs/positioning.md): when to use rlmflow vs
  rlm-minimal, ypi, LangGraph, CrewAI, AutoGen, SWE-agent, Aider.
- [Control](docs/control.md): streaming loop, per-agent limits, multi-turn runs,
  save/load resume, forks, `INPUTS`, `launch_subagents`, custom tools.
- [Streaming and scheduling](docs/streaming.md): `run_streaming(..., until=...)`,
  `TaskQueue`, transition diagrams, delegation, parallel roots, boundaries,
  cancellation, and Pool/Runtime placement.
- [Node model](docs/node_model.md): the seven node types, the transitions
  between them, and how delegation is recorded.
- [Skills](docs/skills.md): workspace `SKILL.md` files, query-selected
  skills, child-only skills, and run-memory skills.
- [Node injection](docs/injections.md): append controller Nodes between
  streaming calls and continue the same root.
- [Observability](docs/observability.md): querying the Node tree, run
  layout, stream consumers, and reading a saved run.
- [Runtimes](docs/runtimes.md): `Runtime` protocol, shipped runtimes
  (Local / subprocess / Docker / Modal), writing your own.
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
  coding-agent harness. rlmflow's separation of execution, durable state,
  and consumers follows Tau's clean harness design.
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
