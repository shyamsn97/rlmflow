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

`flow.run("query")` and `await flow.arun("query")` return the root agent's final strict JSON value. With an `output_schema`, that value has also been parsed and validated against the schema.

Pass `until="next"` for one appended Node, `until="idle"` for a clean `ExecOutput`/`DoneOutput`, or a `(node, root) -> bool` callable. See [`streaming.md`](streaming.md).

## Per-agent limits

Limits belong to the agent, not the flow. Set them on the root with `flow.start(...)`, which takes any `AgentConfig` field:

The authoritative `AgentConfig` defaults are:

- `max_depth=1`;
- `max_iters=None` and `max_budget=None` (unbounded turns and tokens);
- `child_output_schema=None`, meaning launched children are untyped unless the launch passes `output_schema`;
- `keep_n_messages=None`, meaning prompts retain the full transcript;
- `max_output_length=20_000` and `max_query_chars=20_000` (official RLM's per-block REPL cap);
- `resume_on_child_done=True`, so a parent that has already `finish()`ed still observes later child results; pass `False` on scored evals that must stay terminal.

Passing `None` explicitly opts out of the iteration or token limit.

`max_budget` is a soft threshold over recorded model usage for the complete root tree. Once crossed, new child creation stops and every unfinished agent gets one final turn. Descendants finalize first, then their parents synthesize the returned results. Calls already in flight plus one final turn per unfinished agent can exceed the threshold; use `workers` and per-provider output limits when a tighter billing bound matters.

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

Children inherit the parent's `max_iters` through `config.child(name)`. `child_output_schema` becomes the child's `output_schema` unless the launch spec passes one. The launch spec's `inputs`, required `model`, `prompt_profile`, and `output_schema` layer on top.

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
    inputs: dict[str, str | AgentHandle] | None = None,
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

`launch_subagent` attaches the child and returns its handle immediately. A child whose inputs are all strings is queued at once; a child with unfinished handle-valued inputs is queued when its final source completes. Awaiting independently launched handles sequentially does not serialize the children because they are already running. Child `name` is one ASCII identifier segment containing only letters, digits, `_`, or `-`, and is unique among its siblings. Every call must pass `model`; `prompt_profile`, `output_schema`, and `reuse_repl` are optional. Invalid model, input, depth, or goal length raises an execution error without spawning a child.

Pass handles as named inputs when a downstream child has substantial work that requires upstream results:

```python
requirements = await launch_subagent(
    "Determine the requirements.", model="fast", name="requirements"
)
constraints = await launch_subagent(
    "Investigate the constraints.", model="fast", name="constraints"
)
implementation = await launch_subagent(
    "Implement and test using INPUTS.",
    model="default",
    name="implementation",
    inputs={"requirements": requirements, "constraints": constraints},
)
result = await implementation
```

The downstream child waits until every handle-valued input completes. String inputs are available immediately. Handle results are exposed to the child as strings under the mapping's explicit keys; non-string structured results are JSON-encoded. Only top-level values may be handles.

Passing a handle does not move either agent. All three agents above remain siblings beneath the action that launched them. Their dependency edges are derived from the handle-valued inputs and are visible with `AGENTS.print_graph(view="dependencies")`. `max_depth` controls recursive spawning only, not dependency-chain length, so a root at `max_depth=1` can build an arbitrary sibling dependency DAG.

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

Children stay beneath the `ExecAction` that launched them, so ownership records who created each agent and the parent's transcript remains one chain. Input references record dataflow independently. The launching step still lands one `ExecOutput` when the block finishes.

When a child `finish()`es, the launching parent also gets a durable `UserQuery` carrying that result (`[handoff {path}]` plus the child's `finish()` value), unless the result is already routed to another agent through a handle-valued input. A mapping is listed as fields; a scalar stays labeled JSON. The payload is capped. The engine does not splice it into an in-flight LLM or REPL step: it waits until that parent lands `ExecOutput`, then appends one batched query for every child that finished in the meantime and follows it with a `PlanQuery`. `ErrorOutput` and `ReplDead` do not flush; the mailbox waits for the next `ExecOutput`. `await wait_for_result()` in the launch block still works — the handoff appears after that block's `ExecOutput`.

The coding CLI and [`examples/coding/agent.py`](https://github.com/shyamsn97/rlmflow/blob/main/examples/coding/agent.py) set `AgentConfig.child_output_schema` to `CODING_HANDOFF_SCHEMA`, so a launched child must `finish` `{done, diff_summary, concerns, followups}` and the parent reads those fields on the next `PlanQuery`. The root itself is not given that schema. Other `Flow`s (evals, needle) stay untyped unless they set the field.

If the parent has already `finish()`ed, the default is to append the same `UserQuery` onto that `DoneOutput` and keep driving (`resume_on_child_done=True`). Pass `False` (on `AgentConfig`, `flow.start(...)`, or `Flow(root_config=...)`) to leave the parent terminal while descendants drain. Children inherit the flag through `config.child(name)`. The mailbox is live-only; a resumed stream rebuilds it from terminal children whose marker is not yet in the parent's transcript.

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

Each `AgentInfo` includes identity, parent/children, dependency metadata, frontier metadata, status, and a completed result. Status is one of `blocked`, `running`, `waiting`, `idle`, or `completed`.

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

- `Flow(restore="replay")`, the default, re-runs each unfinished agent's recorded code once before the run loop starts. Each recorded launch call uses its `ExecAction` launch receipt to return the existing agent immediately, without revalidating inputs or submitting work. Nothing is appended and no child is relaunched; `ENV["RLMFLOW_REPLAY"]` is `"1"` while it happens, so agent code can skip work it should not repeat.
- `Flow(restore="lazy")` defers the same replay until immediately before each unfinished agent's first new `ExecAction`.

Replay follows physical worker ownership. An unfinished `reuse_repl=True` child requires its parent's worker even when that parent has already finished, so restore rebuilds required shared sessions parent-first. It also replays terminal tenants in such a session because their mutations may remain in the shared heap. Finished sessions unrelated to unfinished work stay closed.

To checkpoint as a run goes, use the periodic consumer instead of serializing the whole graph after every Node:

```python
from rlmflow import GraphCheckpointer

checkpoint = GraphCheckpointer("runs/audit", interval_s=5, interval_nodes=25)
async for node in flow.run_streaming(root):
    checkpoint.handle(node)
checkpoint.close()
```

## Tools

Pass tools to the flow, which describes them in the system prompt and seeds them into each REPL. Visibility can depend on the agent: `@tool(..., when=at_leaf)` (or a toolset `visible_tools(flow, node)`) omits the tool from both the prompt and the namespace. `FILE_TOOLS` writes are leaf-only — spawners keep reads.

```python
from rlmflow import FILE_TOOLS, Flow, LocalRuntime, at_leaf, tool


@tool("Run a migration.", when=at_leaf)
def migrate(sql: str) -> str: ...


flow = Flow(
    client,
    tools=[FILE_TOOLS, migrate],
    runtime=LocalRuntime(working_directory="."),
)
```

`at_leaf` is true when `can_spawn` is false (`depth >= max_depth`). A `max_depth=1` root can launch; its children can write. Agents that still have remaining depth keep spawn and lose writes.

Decorate a standalone callable with `@tool`; standalone tools share a `### Tools` prompt subsection. Use `@toolset("Title")` when several related callables need their own subsection and shared guidance:

```python
from rlmflow import tool, toolset


@toolset("Database")
class DatabaseTools:
    @tool("List available database tables.")
    def list_tables(self) -> list[str]: ...

    @tool("Execute read-only SQL and return rows.")
    def query(self, sql: str) -> list[dict]: ...

    def guidance(self, flow, node) -> str:
        return "Inspect table names before writing SQL. Queries are read-only."


flow = Flow(client, tools=[DatabaseTools()])
```

Toolset methods remain flat REPL globals—the model calls `query(...)`, not `DatabaseTools.query(...)`. A toolset subsection contains each active member's signature and `@tool` description followed by optional `guidance(flow, node)`. Removing its last active member removes the whole subsection and its guidance.

Add or remove them later; both reach REPLs that are already open:

```python
flow.add_tool(my_tool)
flow.inject("LOOKUP", {"a": 1})   # any object, not just callables
flow.remove_tool("my_tool")
```

`finish`, `launch_subagent`, `INPUTS`, and `AGENTS` are reserved: framework values are rebuilt for each step and cannot be injected over.

`Flow(use_llm_query=True)` adds `llm_query` and `llm_query_batched` to the REPL. Pass `use_llm_query_batched=False` to keep the single-query tool while removing fan-out from the agent. Both remain available to host code as `flow.llm_query(...)` and `flow.llm_query_batched(...)` either way.

## Cleanup

A REPL lives until something closes it, and finishing does not close it. An agent that answered keeps its namespace, including a child that answered inside a parent's run, because that namespace is the only record of what its code built: read JSON-compatible finished-child state with `flow.runtime.get_var(child, "name")`, or append a new query and run the same root again without a replay to rebuild it.

The flip side is that a large fan-out holds a worker per child until you say otherwise. Close one run's REPLs as the stream exits:

```python
async for _ in flow.run_streaming(root, close_repls=True):
    pass
```

`run(root, close_repls=True)` and `arun(root, close_repls=True)` do the same, and `flow.runtime.close_repl(agent)` closes exactly one.

Close all Flow resources — queued work, thread pool, and every REPL — with `await flow.aclose()`.
