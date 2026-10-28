# Control

The root Node is the control surface. Flow mutates that tree in place and `run_streaming` yields the nodes as they land.

## Run and step

```python
import asyncio
from rlmflow import AgentConfig, Flow, Graph, Node

g = Graph()
root = g.create_agent("Audit this repository.")

flow = Flow(configs=[AgentConfig(model="gpt-5", max_depth=2)])   # "default"

async def drive():
    async for item in flow.run_streaming(g):
        if isinstance(item, Node):
            print(g.path(g.node(item.agent_id)), item.type)

asyncio.run(drive())
print(g.result(root))
```

`flow.run("query")` and `await flow.arun("query")` return the root agent's final strict JSON value. With an `output_schema`, that value has also been parsed and validated against the schema.

Pass `until="next"` for one appended Node, `until="idle"` for a clean `ExecOutput`/`DoneOutput`, or an `(item, graph) -> bool` callable. See [`streaming.md`](streaming.md).

## Per-agent limits

Limits belong to a config, and the Flow owns the configs. Roots run with the run's `root_config`; everything else names one of `flow.configs`.

The `AgentConfig` defaults are:

- `max_depth=1`;
- `max_iters=None` and `max_budget=None` (unbounded turns and tokens);
- `keep_n_messages=None`, meaning prompts retain the full transcript;
- `max_output_length=20_000` and `max_query_chars=20_000`.

`max_budget` is a soft threshold over recorded model usage for the complete root tree. Once crossed, new child creation stops and every unfinished agent gets one final turn. Descendants finalize first, then their parents synthesize the returned results. Calls already in flight plus one final turn per unfinished agent can exceed the threshold; use `workers` and per-provider output limits when a tighter billing bound matters.

```python
g = Graph()
root = g.create_agent("Audit this repository.", tree=listing)
flow.run(
    g,
    root_config=AgentConfig(model="gpt-5-mini", max_depth=2, max_iters=20, max_budget=200_000, output_schema=schema),
)
```

`root_config` is an `AgentConfig`, the name of one of the Flow's configs, or `None` for `"default"`. It applies to that call only, so the same graph can continue with different settings:

```python
g = Graph.load("runs/audit")
g.send(root, "Now check the tests too.")
flow.run(g, root_config=AgentConfig(max_depth=2, max_iters=40))
```

Each agent records the config it last ran with in `metadata["config"]`. That copy is for viewers and is never read back to drive a run.

### Named configs

Named configs live on the Flow; the model picks one with `launch_agent(task, config=NAME)` and sees each config's `description`. `launch_agent` without `config=` uses `"default"`. For a stronger coordinator with cheaper workers:

```python
flow = Flow(
    configs=[
        AgentConfig(model="gpt-5"),
        AgentConfig(name="worker", model="gpt-5-mini", description="cheap and fast"),
    ],
)
flow.run("Audit this repository.", root_config=AgentConfig(model="gpt-5", max_depth=2))
```

`model` is a model name; the Flow builds and caches its client. Register a client in `llm_clients` only when a name can't describe it (a reasoning effort, a base URL, a Tinker sampler), under the name configs use: `Flow(llm_clients={"fast": client_for("gpt-5-mini", reasoning_effort="low")})` and `AgentConfig(model="fast")`. A config without a model raises when it reaches the REPL harness; Claude Code and pi use their own default. `flow.add_config(config)` adds one later; an unknown name is a `KeyError` when the Flow reaches that agent. `g.spawn(task, config="worker")` puts a top-level agent on a named config.

### Harnesses

`AgentConfig.harness` names the step function that runs an agent: `"repl"`, the default, is rlmflow's model-and-REPL loop; `"claude_code"` and `"pi"` run one headless session turn per step. A harness config is just another named config:

```python
flow.add_config(AgentConfig(name="claude", harness="claude_code", harness_options={"mcp": True}))
```

External sessions reach the same agent tools from the shell through `rlmflow repl`, so every cell they run lands as a `code` node. They keep their own tools and permissions: Claude Code runs with your settings, and rlmflow only allows `rlmflow` commands on top. Claude Code must already be installed and authenticated. Replaying the graph resumes the recorded session, or starts a fresh one from the REPL state when the session is gone.

## Multi-turn runs

An agent that answered is finished only because its frontier is a `DoneOutput`. Append a new query to that frontier and the same root runs again, with its REPL namespace and its whole history intact:

```python
from rlmflow import UserQuery

flow.run(root)

root.frontier.append(UserQuery(content="Now implement the fixes."))
flow.run(root)
```

`max_iters` counts an agent's model turns across the whole transcript. The default is `None` (no cap). When you set a number, a root you intend to drive for several turns needs headroom for all of them.

To change an agent argument, update the agent and let the next step re-seed the REPL:

```python
root.args["format"] = "markdown"
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
def launch_subagent(
    *,
    task: str,
    name: str | None = None,
    schema=None,
    verify=None,
    **json_inputs,
) -> AgentHandle: ...
```

Launch returns a persistent handle immediately. Launch all independent children before waiting for any result:

```python
a = launch_subagent(task="...", chunk=chunk_a)
b = launch_subagent(task="...", chunk=chunk_b)
results = await asyncio.gather(a, b)
```

Use `Input(value, description=...)` when the caller already knows an argument's format or meaning:

```python
from rlmflow import Input

child = launch_subagent(
    task="Compare labels by user.",
    rows=Input(
        rows,
        description="List of records with user_id, label, and text fields.",
    ),
)
```

The child receives the underlying value as the direct `rows` variable. Only the description, name, type, and size appear in its system prompt; the value remains in the REPL. `description` must be a non-empty string. Descriptions are caller-supplied context, not generated inspection. They persist in the graph and may wrap a child handle as well as a JSON value. The same wrapper works for root arguments passed to `flow.run(...)` or `g.create_agent(...)`.

`launch_subagent` durably attaches and submits the child before returning. A child whose arguments are all JSON values is queued at once; a child with unfinished handle-valued arguments is queued when its final source completes. The parent's `AgentConfig.child_config` sets the child's model, prompt profile, placement, and backend; model code cannot override them. `schema=` validates the child's result and `verify=` may reject a candidate with repair feedback.

Pass handles as named arguments when a downstream child has substantial work that requires upstream results:

```python
requirements = launch_subagent(task="Determine the requirements.")
constraints = launch_subagent(task="Investigate the constraints.")
implementation = launch_subagent(
    task="Implement and test using requirements and constraints.",
    requirements=requirements,
    constraints=constraints,
)
result = await implementation
```

The downstream child waits until every handle-valued input completes. Other strict JSON values are available immediately. Handle references may be nested in JSON objects and arrays, and resolve to the source's validated JSON-compatible result.

Passing a handle does not move either agent. All three agents above remain siblings beneath the action that launched them. Their dependency edges are derived from the handle-valued inputs and are visible with `AGENTS.print_graph(view="dependencies")`. `max_depth` controls recursive spawning only, not dependency-chain length, so a root at `max_depth=1` can build an arbitrary sibling dependency DAG.

The handle exposes identity and current status:

```python
child = launch_subagent(task="run a long check")
print(child.name, child.status)
```

The child remains in the same `run_streaming()` queue and continues appending nodes to the graph. A later block can wait on the persistent handle without launching anything again:

```python
result = await child
```

If the root becomes terminal first, the stream still drains queued descendants before returning. This is not a detached daemon job.

Children stay beneath the `ExecAction` that launched them, so ownership records who created each agent and the parent's transcript remains one chain. Input references record dataflow independently. The launching step still lands one `ExecOutput` when the block finishes.

When a child `submit()`s, the launching parent also gets a durable `UserQuery` carrying that result (`[handoff {path}]` plus the child's `submit()` value), unless the result is already routed to another agent through a handle-valued input. A mapping is listed as fields; a scalar stays labeled JSON. The payload is capped. The engine does not splice it into an in-flight LLM or REPL step: it waits until that parent lands `ExecOutput`, then appends one batched query for every child that completed in the meantime. That query calls the model directly. `ErrorOutput` and `ReplDead` do not deliver; pending results wait for the next `ExecOutput`. `await wait_for_result()` in the launch block still works — the handoff appears after that block's `ExecOutput`.

The coding CLI and [`examples/coding/agent.py`](https://github.com/shyamsn97/rlmflow/blob/main/examples/coding/agent.py) set `AgentConfig(child_config={"output_schema": CODING_HANDOFF_SCHEMA})`, so a launched child must `submit` `{done, diff_summary, concerns, followups}` and the parent reads those fields from the handoff `UserQuery`. The root itself is not given that schema. Other `Flow`s (evals, needle) stay untyped unless they set the field.

If the parent has already submitted, the default is to append the same `UserQuery` onto that `DoneOutput` and keep driving (`resume_on_child_done=True`). Pass `False` (on the agent's `AgentConfig`, or `Flow(root_config=...)` for `flow.run("task")`) to leave the parent terminal while descendants drain. Children inherit the flag through `config.child(name)`. Nothing is held in memory: a child result counts as delivered once a `UserQuery` has a `from` edge to the child's `DoneOutput`, so a resumed stream delivers exactly the results that were not yet sent.

## Submission policy

`submit(candidate)` validates the candidate and lands a durable `DoneOutput` without another model turn. Rejected schema or verifier checks land as `VerificationFailure` and return to the same agent through `RepairQuery`. The prompt tells agents to end any block that prints evidence, read that observation, and submit only in a later block. The old `finish(...)` name remains executable as a hidden compatibility alias but is not shown in model tool documentation.

## Agent tree inspection

Opt into a read-only snapshot of the recursive tree:

```python
flow = Flow(use_agent_tree=True)
```

Every REPL action then receives `AGENTS` alongside its direct invocation variables:

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

- `Flow(restore="replay")`, the default, re-runs each unfinished agent's recorded code once before the run loop starts. Restorable tool calls return their saved results, so `launch_subagent` resolves its existing child and `llm_query` does not repeat a model request. Nothing is appended and no child is relaunched; `ENV["RLMFLOW_REPLAY"]` is `"1"` while it happens, so agent code can skip other work it should not repeat.
- `Flow(restore="lazy")` defers the same replay until immediately before each unfinished agent's first new `ExecAction`.

Replay follows physical worker ownership. An unfinished `reuse_worker=True` child requires its parent's worker even when that parent has already finished, so restore rebuilds the required session owner first. Agent namespaces remain isolated.

To checkpoint as a run goes, use the periodic consumer instead of serializing the whole graph after every Node:

```python
from rlmflow import GraphCheckpointer

checkpoint = GraphCheckpointer("runs/audit", interval_s=5, interval_nodes=25)
async for node in flow.run_streaming(root):
    checkpoint.handle(node)
checkpoint.close()
```

## Tools

Pass tools to the flow, which describes them in the system prompt and seeds them into each REPL. The prompt's `### Tools` list is rendered from exactly the callables the REPL binds, so the two cannot drift. Visibility can depend on the agent: `@tool(..., when=at_leaf)` omits the tool from both. `FILE_TOOLS` writes are leaf-only — spawners keep reads.

```python
from rlmflow import FILE_TOOLS, Flow, LocalRuntime, at_leaf, tool


@tool("Run a migration.", when=at_leaf)
def migrate(sql: str) -> str: ...


flow = Flow(
    tools=[FILE_TOOLS, migrate],
    runtime=LocalRuntime(working_directory="."),
)
```

`at_leaf` is true when `can_spawn` is false (`depth >= max_depth`). A `max_depth=1` root can launch; its children can write. Agents that still have remaining depth keep spawn and lose writes.

`@tool` is the only decorator, and `description` is a string. Group related tools in a tuple; `tools=` flattens one level:

```python
from rlmflow import tool


@tool("List available database tables.")
def list_tables() -> list[str]: ...


@tool("Execute read-only SQL and return rows. Call `list_tables()` first.")
def query(sql: str) -> list[dict]: ...


DATABASE_TOOLS = (list_tables, query)
flow = Flow(tools=[DATABASE_TOOLS])
```

Host-side tools can preserve a successful JSON result across namespace reconstruction:

```python
@tool("Fetch one remote record.", proxy=True, restore="result")
def fetch_record(key: str) -> dict: ...
```

Without `restore=`, replay calls the tool normally. Use `restore="result"` for ordinary JSON results, or pass a restore callable when the saved JSON value must be rebuilt into a runtime object. Saved calls remain internal details of their owning `ExecAction`.

Proxy tools can attach arbitrary JSON telemetry to that action:

```python
from rlmflow import record_metadata, tool

@tool("Fetch one remote record.", proxy=True, restore="result")
def fetch_record(key: str) -> dict:
    value = remote_fetch(key)
    record_metadata(
        "records.fetched",
        {"key": key, "source": "primary"},
        append=True,
    )
    return value
```

Every proxied invocation also records a compact `tools.calls` receipt with its name, call index, duration, replay status, and error status. Metadata is durable through save/load/fork and visible to stream consumers through the owning node.

Add or remove them later; both reach REPLs that are already open:

```python
flow.add_tool(my_tool)
flow.inject("LOOKUP", {"a": 1})   # any object, not just callables
flow.remove_tool("my_tool")
```

`submit`, its hidden `finish` compatibility alias, `launch_subagent`, `Input`, `asyncio`, `ENV`, `AGENTS`, and the agent inspection bindings are reserved: framework values are rebuilt for each step and cannot be injected over.

`Flow(use_llm_query=True)` adds `llm_query` to the REPL. Fan out independent queries with ordinary Python concurrency:

```python
answers = await asyncio.gather(*(llm_query(prompt) for prompt in prompts))
```

Host code can use the same pattern with `flow.llm_query(...)`.

Set `Flow(..., llm_query_cache=True)` to deduplicate identical one-shot calls within that Flow, or override one call with `llm_query(..., cache=True|False)`. The key includes model, prompt, normalized schema, and sampling options. `llm_query.calls` metadata records the key, prompt hash, usage, and whether each call hit the cache. The cache is opt-in because repeated stochastic samples are often intentional.

## Cleanup

A REPL lives until something closes it, and submitting does not close it. An agent that answered keeps its namespace, including a child that answered inside a parent's run, because that namespace is the only record of what its code built: read JSON-compatible finished-child state with `flow.runtime.get_var(child, "name")`, or append a new query and run the same root again without a replay to rebuild it.

The flip side is that a large fan-out holds a worker per child until you say otherwise. Close one run's REPLs as the stream exits:

```python
async for _ in flow.run_streaming(root, close_repls=True):
    pass
```

`run(root, close_repls=True)` and `arun(root, close_repls=True)` do the same, and `flow.runtime.close_repl(agent)` closes exactly one.

Close all Flow resources — queued work, thread pool, and every REPL — with `await flow.aclose()`.
