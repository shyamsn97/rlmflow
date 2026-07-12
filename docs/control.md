# Control

`Graph` is the control surface. `Flow.run_streaming(graph, until=...)` advances
it in place and yields the `Event`s it emitted. Save/load, rewind, fork, inject,
and resume are all graph operations.

## Step Loop

```python
import asyncio
from rflow import render_tree

agent = rflow.Flow(rflow.OpenAIClient(model="gpt-5"), max_depth=2)
graph = agent.start(query)

async def drive():
    async for _event in agent.run_streaming(graph):
        print(render_tree(graph))

asyncio.run(drive())
```

`agent.run(query)` drives the same loop synchronously and returns
`graph.result()`; `await agent.arun(query)` is the async form. `agent.chat(messages)`
is the `LLMClient` interface — the latest user message becomes the query and the
recursive loop runs under the hood.

`run_streaming` yields one typed `Event` per commit and mutates `graph` in place.
Every model turn is an `LLMOutput` (the model's code) followed by exactly one
observation: `ExecOutput`, `DoneOutput`, `ErrorOutput`, or `SupervisingOutput`.
See [`node_model.md`](node_model.md) for the typed node flow.

## Stream Boundaries

`Flow.run_streaming(graph, until=...)` streams until a boundary and **halts there**
— the driver does not enqueue more work past what you observe, so edits you make
between streaming calls are seen when the run resumes. Pass the graph on the first
call; omit it on later calls to continue the active run.

Two boundary families:

**Global steps** advance every active agent in parallel, then halt the whole
frontier:

```python
async for event in flow.run_streaming(graph, until="next"):
    ...
async for event in flow.run_streaming(until="idle"):
    ...
```

- `next` surfaces everything, including errors — it stops *on* an `error_output`.
- `idle` runs each agent to a clean rest (`exec_output`/`done_output`), healing
  errors on the way (an `error_output` is not a rest point, so the agent keeps
  going and fixes it). A supervising parent with live children just no-ops.

Other boundaries stop when that event is observed:

```python
async for event in flow.run_streaming(graph, until="supervising"):
    ...
async for event in flow.run_streaming(until=lambda event, graph: graph.finished):
    ...
```

Because the run halts at the boundary, reactive control is deterministic:

```python
async for _event in flow.run_streaming(graph, until="idle"):
    pass
graph.inject("Finalize now with the best current evidence.")
async for _event in flow.run_streaming(until="done"):
    pass
```

See [`examples/control/controller_injection.py`](../examples/control/controller_injection.py)
and [`examples/control/delegation/step_until.py`](../examples/control/delegation/step_until.py).
For the full scheduler model, see [`streaming.md`](streaming.md).

## Save And Resume

A saved graph directory is the durable run:

```python
graph.save("runs/deep_research")

resumed = rflow.Graph.load("runs/deep_research")
agent.run(resumed)   # or: async for _ in agent.run_streaming(resumed): ...
```

For live checkpointing, call `graph.save(...)` inside the stream loop. The same
path is overwritten with the latest complete graph/run layout.

## Rewind And Branch

`Graph` mutates in place, so keep restore points with `checkpoint()` /
`revert(...)`, or drop everything after a node with `rewind(node_id)`:

```python
cp = graph.checkpoint()
agent.run(graph)
graph.revert(cp)        # back to the checkpoint
graph.rewind(node_id)   # or truncate history after a specific node
```

Fork an independent branch (deep copy with a fresh `graph_id`) and continue it
somewhere else:

```python
branch = graph.fork(session="isolated")
agent.run(branch)
branch.save("runs/repair-branch")
```

## Node Injection

Controllers can append a user-turn observation to any agent and continue the
run. This is useful for budget nudges, human feedback, and forced finalization:

```python
graph.inject("Answer now with the best current evidence.", agent_id="root.worker")
agent.run(graph)   # or: async for _ in agent.run_streaming(graph): ...
```

`replace_node`, `rewind`, and `remove_child` rewrite existing history the same
way. See [`injections.md`](injections.md) and
[`examples/control/controller_injection.py`](../examples/control/controller_injection.py).

## Delegation

Agents delegate through one launcher, which must be awaited:

```python
# One child — still pass a one-item list of dict specs, and unpack the result.
[answer] = await launch_subagents([
    {"name": "single", "query": query, "inputs": {"data": data}},
])

# Many children in parallel — returns answers in spec order.
results = await launch_subagents([
    {"name": "a", "query": "...", "inputs": {"chunk": chunk_a}},
    {"name": "b", "query": "...", "inputs": {"chunk": chunk_b}},
])
```

- **Sequential dependent steps:** chain one-item `await launch_subagents([...])`
  calls, feeding each result into the next child's `inputs`.
- **Parallel independent work:** pass every spec in one call so the engine
  schedules them concurrently.
- **Child data:** put payloads in each spec's `inputs` dict. The child sees only
  its query and its own `INPUTS`.

## Custom Runtime

Subclass `Runtime` and implement `open(agent)` to mint a backend:

```python
class MyRuntime(rflow.Runtime):
    def open(self, agent: rflow.Graph) -> rflow.ReplBackend:
        return MyBackend(...)
```

Most users should pass `LocalRuntime`, `DockerRuntime`, or a sandbox runtime.
See [`runtimes.md`](runtimes.md).

## Custom Tools

Register tools on the runtime before constructing or stepping the flow:

```python
@rflow.tool("Search files for a regex.")
def search(pattern: str, path: str = ".") -> str:
    ...

runtime = rflow.LocalRuntime(working_directory=".")
runtime.register_tool(search)
runtime.register_tools(rflow.FILE_TOOLS)
agent = rflow.Flow(rflow.OpenAIClient(model="gpt-5"), runtime=runtime)
```

## Custom Prompt

For a fuller guide, see [`prompt_customization.md`](prompt_customization.md).

```python
from rflow.prompts import DEFAULT_BUILDER

GUARDRAILS = """
- Verify before `done()`. Empty/zero/surprising results -> one sanity check first.
- Ask children for structured output when shape matters.
"""

agent = rflow.Flow(rflow.OpenAIClient(model="gpt-5"))
agent.prompt_builder = (
    DEFAULT_BUILDER
    .section("role", "You are a security auditor.", title="Role")
    .section("guardrails", GUARDRAILS, title="Guardrails", after="strategy")
)
```

You can also subclass `Flow` and override `build_system_prompt` (or set
`agent.prompt_builder` directly) to fully control the prompt.

## Walkthroughs

- [`examples/showcase.py`](../examples/showcase.py) — stepping, snapshots,
  save/load, and live terminal visualization.
- [`examples/graph/`](../examples/graph/) — querying, mutating, saving, forking,
  and rendering minimal graphs.
