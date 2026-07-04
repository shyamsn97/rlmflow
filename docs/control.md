# Control

`Graph` is the control surface. `Flow.start(...)` creates a graph, and every
`Flow.step(graph)` returns a fresh advanced snapshot. Save/load, rewind, branch,
inject, and resume are all graph operations.

## Step Loop

```python
agent = rflow.Flow(rflow.OpenAIClient(model="gpt-5"), max_depth=2)
graph = agent.start(query)
while not graph.finished:
    graph = agent.step(graph)
```

`agent.run(query)` drives the same loop and returns `graph.result()`.
`agent.chat(messages)` is the `LLMClient` interface; the latest user message
becomes the query and the recursive loop runs under the hood.

Each `step(graph)` advances one observation-to-observation transition for every
agent that is ready to move. A model turn is usually two steps: LLM call
(`obs -> LLMAction -> LLMOutput`) and code execution
(`LLMOutput -> ExecAction -> CodeObservation`). See [`node_model.md`](node_model.md)
for the typed node flow.

## Minimal Step Boundaries

In the minimal event loop, delegated children fan out through the shared async
pool once the parent reaches `await launch_subagents([...])`. The caller controls
how much of that graph event stream to observe with `Flow.step(..., until=...)`.

Common boundaries:

```python
await flow.step(graph)  # default: until="node"
await flow.step(until="supervising")
await flow.step(until="node", n=3)
await flow.step(until=lambda event, graph: graph.finished)
```

See [`examples/control/delegation/step_until.py`](../examples/control/delegation/step_until.py)
for a deterministic offline demo.

## Save And Resume

A saved graph directory is the durable run:

```python
graph.save("runs/deep_research")

resumed = rflow.Graph.load("runs/deep_research")
while not resumed.finished:
    resumed = agent.step(resumed)
```

For live checkpointing, save after every step. The same path is overwritten with
the latest complete graph/run layout.

## Rewind And Branch

Keep every `Graph` snapshot in a list and resume any one of them:

```python
history = [agent.start(query)]
while not history[-1].finished:
    history.append(agent.step(history[-1]))

graph = history[-5]
while not graph.finished:
    graph = agent.step(graph)
```

Branch by copying or loading a graph and saving the result somewhere else:

```python
branch = history[-5].copy(deep=True)
while not branch.finished:
    branch = agent.step(branch)
branch.save("runs/repair-branch")
```

## Node Injection

Controllers can append typed nodes to a graph and commit them through the normal
step loop. This is useful for budget nudges, human feedback, and forced
finalization:

```python
graph = graph.inject(
    target="root.worker",
    node=rflow.ExecOutput(
        output="Injected controller observation: answer now.",
        content="Injected controller observation: answer now.",
    ),
)
graph = agent.step(graph)

graph = graph.inject(
    target="root.worker",
    node=rflow.ExecAction(code='done("best available answer")'),
)
graph = agent.step(graph)
```

See [`injections.md`](injections.md) and
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

You can also subclass `Flow` and override `build_system_prompt`,
`build_messages`, `format_exec_output`, `first_prompt`, or `step`.

## Walkthroughs

- [`examples/showcase.py`](../examples/showcase.py) — stepping, snapshots,
  save/load, and live terminal visualization.
- [`examples/graph/`](../examples/graph/) — querying, mutating, saving, forking,
  and rendering minimal graphs.
