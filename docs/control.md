# Control

`Graph` is the control surface. `Flow.run_streaming(graph=graph, until=...)` advances
it in place and yields the `Event`s it emitted. Save/load, rewind, fork, inject,
and resume are all graph operations.

## Step Loop

```python
import asyncio
from rflow import render_tree

agent = rflow.Flow(rflow.OpenAIClient(model="gpt-5"), max_depth=2)
graph = rflow.Graph(query=query)

async def drive():
    async for _event in agent.run_streaming(graph=graph):
        print(render_tree(graph))

asyncio.run(drive())
```

`agent.run(query=query)` drives the same loop synchronously and returns
`graph.result()`; `await agent.arun(query=query)` is the async form. To use a
flow anywhere an `LLMClient` is expected, wrap it in `FlowLLM(agent)` — its
`chat(messages)` projects the messages to a query and runs the recursive loop
under the hood (see the drop-in section in the README).

`run`, `arun`, and `run_streaming` are keyword-only. Pass `query=` to start a
fresh graph, or `graph=` to resume an existing one; passing both appends `query`
as a new turn on that graph (see [Multi-turn runs](#multi-turn-runs)). `inputs=`
and `output_schema=` are applied to the graph before driving (`inputs` merges by
default; pass `merge_inputs=False` to replace).

`run_streaming` yields one typed `Event` per commit and mutates `graph` in place.
Every model turn is an `LLMOutput` (the model's code) followed by exactly one
observation: `ExecOutput`, `DoneOutput`, `ErrorOutput`, or `SupervisingOutput`.
See [`node_model.md`](node_model.md) for the typed node flow.

## Stream Boundaries

`Flow.run_streaming(graph=graph, until=...)` streams until a boundary and **halts there**
— the driver does not enqueue more work past what you observe, so edits you make
between streaming calls are seen when the run resumes. Pass the graph on the first
call; omit it on later calls to continue the active run.

Two boundary families:

**Global steps** advance every active agent in parallel, then halt the whole
frontier:

```python
async for event in flow.run_streaming(graph=graph, until="next"):
    ...
async for event in flow.run_streaming(graph=graph, until="idle"):
    ...
```

- `next` surfaces everything, including errors — it stops *on* an `error_output`.
- `idle` runs each agent to a clean rest (`exec_output`/`done_output`), healing
  errors on the way (an `error_output` is not a rest point, so the agent keeps
  going and fixes it). A supervising parent with live children just no-ops.

Other boundaries stop when that event is observed:

```python
async for event in flow.run_streaming(graph=graph, until="supervising"):
    ...
async for event in flow.run_streaming(graph=graph, until=lambda event, graph: graph.finished):
    ...
```

Because the run halts at the boundary, reactive control is deterministic:

```python
async for _event in flow.run_streaming(graph=graph, until="idle"):
    pass
graph.inject("Finalize now with the best current evidence.")
async for _event in flow.run_streaming(graph=graph, until="done"):
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
agent.run(graph=resumed)   # or: async for _ in agent.run_streaming(graph=resumed): ...
```

For live checkpointing, call `graph.save(...)` inside the stream loop. The same
path is overwritten with the latest complete graph/run layout.

## Multi-turn Runs

One graph can serve a long-running, multi-turn agent. When a run finishes,
`graph.finished` is true; appending a new `UserQuery` flips it back to unfinished,
so the next `run`/`run_streaming` re-drives the *same* agent with its full history
and warm REPL (variables from earlier turns are still in scope).

The ergonomic path is to pass `query=` alongside `graph=`:

```python
agent.run(query="Audit the repo.")                 # turn 1 (fresh graph)
graph = ...  # keep a handle to the graph you drove

agent.run(graph=graph, query="Now write the fixes.")   # turn 2, same graph
agent.run(graph=graph, query="Summarize what changed.", # turn 3, new inputs/schema
          inputs={"format": "markdown"}, output_schema=Report)
```

`inputs` merges into the graph's existing `INPUTS` (and is re-synced into the warm
REPL) unless `merge_inputs=False` replaces it; a truthy `output_schema` becomes the
contract for that turn's `done(...)`. The bare graph op is `graph.append_query(...)`
(same keywords), for when you want to stage the next turn without driving yet.

## Rewind And Branch

`Graph` mutates in place, so keep restore points with `checkpoint()` /
`revert(...)`, or drop everything after a node with `rewind(node_id)`:

```python
cp = graph.checkpoint()
agent.run(graph=graph)
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
agent.run(graph=graph)   # or: async for _ in agent.run_streaming(graph=graph): ...
```

`inject`'s `mode`/`truncate` options (and the `append`/`prepend`/`replace`
helpers), plus `rewind` and `remove_child`, rewrite existing history the same
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
- **Child prompt profile:** optional `prompt_profile` per spec selects a named
  `PromptProfile` on the flow (see [`prompt_customization.md`](prompt_customization.md)).

### Warm launch (host-side)

`launch_subagents` is the model-facing tool (cold children from `query` strings).
Host code that already has prepared graphs — typically `fork` / `flow.rewind`
results — should use the warm counterpart:

```python
# Structural only: reparent a prepared graph under `parent` (remap ids, move its
# REPL, emit AddChild). Does not run or await the child.
child = flow.adopt(parent, fork, name="b0")

# Run prepared graphs as children of `parent`, in parallel, and await results.
# Thin wrapper over launch_subagents' warm path — the `graph` spec key stays
# internal so example / model-facing code never has to write it.
await flow.launch_subgraphs(
    parent,
    [fork_a, fork_b],
    queries=["recover with plan A", "recover with plan B"],
    names=["b0", "b1"],
)
```

Use this for best-of-N / rewind-and-branch patterns (see
[`examples/shepherd/`](../examples/shepherd/)): prepare forks on the host, then
one `launch_subgraphs` call attaches and runs them as real children under the
orchestrator.

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
from rflow import SystemPromptBuilder

GUARDRAILS = """
- Verify before `done()`. Empty/zero/surprising results -> one sanity check first.
- Ask children for structured output when shape matters.
"""

prompt = SystemPromptBuilder()
prompt.sections.update("role", "You are a security auditor.")
prompt.sections.add("guardrails", GUARDRAILS, title="Guardrails", after="strategy")

agent = rflow.Flow(rflow.OpenAIClient(model="gpt-5"), system_prompt=prompt)
```

`agent.system_prompt` accepts a `SystemPromptBuilder`, a string, or a
`(flow, graph) -> str` function; subclass `SystemPromptBuilder` (override
`default_sections` or `__call__`) for reusable customization.

## Walkthroughs

- [`examples/showcase.py`](../examples/showcase.py) — stepping, snapshots,
  save/load, and live terminal visualization.
- [`examples/graph/`](../examples/graph/) — querying, mutating, saving, forking,
  and rendering minimal graphs.
