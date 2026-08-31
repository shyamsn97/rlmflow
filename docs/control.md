# Control

The root Node is the control surface. Flow mutates that tree in place and `run_streaming` yields the nodes as they land.

## Run and step

```python
import asyncio
from rlmflow import Flow

flow = Flow(client)
root = flow.start("Audit this repository.", max_depth=2)

async def drive():
    async for node in flow.run_streaming(root):
        print(node.parent_agent.config.path, node.type)

asyncio.run(drive())
print(root.result())
```

`flow.run("query")` and `await flow.arun("query")` return the root agent's final result — a string, or the parsed value if the agent has an `output_schema`.

Pass `until="next"` for one appended Node, `until="idle"` for a clean `ExecOutput`/`DoneOutput`, or a `(node, root) -> bool` callable. See [`streaming.md`](streaming.md).

## Per-agent limits

Limits belong to the agent, not the flow. Set them on the root with `flow.start(...)`, which takes any `AgentConfig` field:

The authoritative `AgentConfig` defaults are:

- `max_depth=1`;
- `max_iters=30` model turns per agent;
- `max_budget=100_000` tokens across the complete run;
- `child_max_iters=None`, meaning children inherit `max_iters`;
- `keep_n_messages=None`, meaning prompts retain the full transcript;
- `max_output_length=4_000` and `max_query_chars=2_000`.

Passing `None` explicitly opts out of the iteration or token limit.

```python
root = flow.start(
    "Audit this repository.",
    inputs={"tree": listing},
    model="fast",
    max_depth=2,
    max_iters=20,
    max_budget=200_000,
    output_schema=schema,
)
```

Give the flow a default `AgentConfig` to stop repeating the limits it should apply anyway. It copies them onto every root it builds — from a bare query string, or from `flow.start(...)` when you want to override a field or two:

```python
from rlmflow import AgentConfig, Flow

flow = Flow(client, root_config=AgentConfig(max_depth=2, max_iters=20))
flow.run("Audit this repository.")

root = flow.start("Audit this repository.", max_iters=40)  # max_depth still 2
```

For a stronger coordinator with cheaper workers, register both choices under clear names:

```python
flow = Flow(
    root_client,
    llm_clients={"fast": worker_client},
    root_config=AgentConfig(max_depth=2),
)
```

The system prompt shows every registered key and marks the current model. Agent code must choose one explicitly with `model=` for each launch; there is no host-selected child-model fallback.

The module-level `start(...)` builds a root with no flow involved, which is what loading, forking, and tests do. It has nothing to inherit from, so it carries `AgentConfig`'s own defaults; pass `config=` to build on something else.

A loaded v3 run keeps the complete effective `AgentConfig`, including all limits. Changing a loaded config before resuming is an explicit continuation override:

```python
from rlmflow import AgentStart

loaded = AgentStart.load("runs/audit")
loaded.config.max_iters = 40
```

Children inherit the parent's config through `config.child(name)`, with `child_max_iters` as the per-child iteration default and the launch spec's `inputs`, required `model`, `prompt_profile`, and `output_schema` layered on top.

## Multi-turn runs

An agent that answered is finished only because its frontier is a `DoneOutput`. Append a new query to that frontier and the same root runs again, with its REPL namespace and its whole history intact:

```python
from rlmflow import UserQuery

flow.run(root)

root.frontier.append(UserQuery(content="Now implement the fixes."))
flow.run(root)
```

`max_iters` counts an agent's model turns across the whole transcript. The default is `None` (no cap). When you set a number, a root you intend to drive for several turns needs headroom for all of them.

To change the inputs an agent sees, set them on its config and let the next step re-seed the REPL:

```python
root.config.inputs = {"format": "markdown"}
```

## Reactive edits

Edit after a streaming call returns, when its pass has settled:

```python
async for _ in flow.run_streaming(root, until="idle"):
    pass

root.frontier.append(UserQuery(content="Finalize with current evidence."))

async for _ in flow.run_streaming(root):
    pass
```

An append lands on the agent's frontier or it raises, so an edit cannot silently rewrite history: to change what an agent already did, fork the run at that point instead.

Two rules keep this clean in a run that delegates. Stop on the agent's own output rather than on a child's, since closing a stream cancels the parent's block and resuming re-runs it, costing a turn. And to reach a worker while its parent is still inside that block, append from inside the loop as the worker's node lands. See [`injections.md`](injections.md).

## Fork and selection

`node.fork()` copies the whole run and cuts everything after that node. The copy is an independent root, with fresh ids by default, that any `Flow` can continue:

```python
from rlmflow import ExecAction

action = next(n for n in root.transcript() if isinstance(n, ExecAction))
branch = action.fork()

async for _ in flow.run_streaming(branch):
    pass
```

The fork's agent is back at that node with the nodes below it gone, so the next step re-runs from there. A `Flow` that has not run the branch rebuilds its REPL first — see the restore modes below.

There is no merge operation. Continue the selected branch and discard the alternatives.

## Delegation

The only model-facing delegation function is:

```python
async def launch_subagent(
    goal: str,
    *,
    model: str,
    name: str | None = None,
    inputs: dict[str, str] | None = None,
    output_schema=None,
    prompt_profile: str | None = None,
    reuse_repl: bool = False,
) -> AgentHandle: ...
```

Agent code awaits each launch to receive its persistent handle. Launch all independent children before waiting for any result:

```python
a = await launch_subagent(
    "...", model="fast", name="a", inputs={"chunk": chunk_a}
)
b = await launch_subagent(
    "...", model="default", name="b", inputs={"chunk": chunk_b}
)
results = [
    await a.wait_for_result(),
    await b.wait_for_result(),
]
```

`launch_subagent` always starts background work and returns immediately after the child is attached and queued. Awaiting handles sequentially does not serialize the children: both are already running. Child `name` is one ASCII identifier segment containing only letters, digits, `_`, or `-`, and is unique among its siblings. Every call must pass `model`; `prompt_profile`, `output_schema`, and `reuse_repl` are optional. Invalid model, depth, or goal length raises an execution error without spawning a child.

The handle exposes identity and current status:

```python
child = await launch_subagent(
    "run a long check", model="fast", name="check"
)
print(child.name, child.status)
```

The child remains in the same `run_streaming()` queue and continues appending nodes to the graph. A later block can wait on the persistent handle without launching anything again:

```python
result = await child.wait_for_result()
```

If the root becomes terminal first, the stream still drains queued descendants before returning. This is not a detached daemon job.

Children hang off the `ExecAction` that launched them, so the parent's transcript stays a single chain: `root.sub_agents` is where the children are, and the launching step lands one `ExecOutput` when the block finishes.

## Agent tree inspection

Opt into a read-only snapshot of the recursive tree:

```python
flow = Flow(client, use_agent_tree=True)
```

Every REPL action then receives `AGENTS` alongside `INPUTS`:

```python
AGENTS.get()                    # this agent
AGENTS.get("root.researcher")   # by path, id, or unique name
AGENTS.get_parent()
AGENTS.get_siblings()
AGENTS.get_children()
AGENTS.get("researcher").result()
await AGENTS.get("researcher").wait_for_result()
AGENTS.print_graph(show_results=True)
```

Each `AgentInfo` includes identity, parent/children, frontier metadata, status, and a completed result. Status is one of `running`, `waiting`, `idle`, or `completed`.

`AGENTS` is an immutable snapshot refreshed before each REPL action. Local queries do not become fresher within one action; `wait_for_result()` is the one operation that waits on live graph state. The option defaults off, so disabled flows add no prompt text, snapshot work, or remote serialization. Process-isolated runtimes require the core `cloudpickle` dependency to inject the custom snapshot object by value; it is installed automatically with `rlmflow`.

## Save and load

```python
from rlmflow import AgentStart

run_dir = root.save("runs/audit")
loaded = AgentStart.load("runs/audit")

async for _ in flow.run_streaming(loaded):
    pass
```

Loading is a cold boundary: the tree is restored, but no REPL is. A tree handed to a `Flow` that has not run it — loaded, forked, or from another process — gets its namespaces back one of two ways:

- `Flow(restore="replay")`, the default, re-runs each unfinished agent's recorded code once before the run loop starts. Nothing is appended and no child is relaunched; `ENV["RLMFLOW_REPLAY"]` is `"1"` while it happens, so agent code can skip work it should not repeat.
- `Flow(restore="lazy")` defers the same replay until immediately before each unfinished agent's first new `ExecAction`.

To checkpoint as a run goes, use the periodic consumer instead of serializing the whole graph after every Node:

```python
from rlmflow import GraphCheckpointer

checkpoint = GraphCheckpointer("runs/audit", interval_s=5, interval_nodes=25)
async for node in flow.run_streaming(root):
    checkpoint.handle(node)
checkpoint.close()
```

## Tools

Pass fixed tools to the flow, which describes them in the system prompt and seeds them into every REPL:

```python
from rlmflow import FILE_TOOLS, Flow, LocalRuntime

flow = Flow(client, tools=[FILE_TOOLS], runtime=LocalRuntime(working_directory="."))
```

Add or remove them later; both reach REPLs that are already open:

```python
flow.add_tool(my_tool)
flow.inject("LOOKUP", {"a": 1})   # any object, not just callables
flow.remove_tool("my_tool")
```

`finish`, `launch_subagent`, `INPUTS`, and `AGENTS` are reserved: framework values are rebuilt for each step and cannot be injected over.

`Flow(use_llm_query=True)` adds `llm_query_batched` to the REPL, for agents that want to fan out one-shot model calls without spawning children. It is available to host code as `flow.llm_query_batched(...)` either way.

## Cleanup

A REPL lives until something closes it, and finishing does not close it. An agent that answered keeps its namespace, including a child that answered inside a parent's run, because that namespace is the only record of what its code built: read JSON-compatible finished-child state with `flow.runtime.get_var(child, "name")`, or append a new query and run the same root again without a replay to rebuild it.

The flip side is that a large fan-out holds a worker per child until you say otherwise. Close one run's REPLs as the stream exits:

```python
async for _ in flow.run_streaming(root, close_repls=True):
    pass
```

`run(root, close_repls=True)` and `arun(root, close_repls=True)` do the same, and `flow.runtime.close_repl(agent)` closes exactly one.

Close all Flow resources — queued work, thread pool, and every REPL — with `await flow.aclose()`.
