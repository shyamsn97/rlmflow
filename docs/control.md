# Control

The graph is the control surface. A `Flow` steps whatever the graph says is
ready, `run_streaming` yields each node as it lands, and between runs you can
read and change the graph however you like.

## Run and step

```python
import asyncio
from rlmflow import AgentConfig, Flow, Graph
from rlmflow.view.text import kind

g = Graph()
root = g.start("Audit this repository.")

flow = Flow(configs=[AgentConfig(model="gpt-5", max_depth=2)])  # "default"


async def drive():
    async for step in flow.run_streaming(g):
        print(step.agent.path(), kind(step.node))


asyncio.run(drive())
print(root.final_result())
```

`flow.run("query")` and `await flow.arun("query")` return the root's result;
`flow.run(g)` runs a graph you built and returns it. With an `output_schema`, the
result has been validated against the schema.

Pass `until="next"` for one of an agent's own nodes, `until="idle"` to stop at
the next node that ran code or ended a turn, or a `(step) -> bool` callable. See [Streaming](streaming.md).

## Per-agent limits

Limits belong to a config, and the Flow owns the configs. Roots run with the
run's `root_config`; everything else names one of `flow.configs`.

The `AgentConfig` defaults are:

- `max_depth=1`;
- `max_iters=None` and `max_budget=None` (unbounded turns and tokens);
- `keep_n_messages=None`, meaning prompts retain the full transcript;
- `max_output_length=20_000` and `max_query_chars=20_000`.

`max_budget` is a soft threshold over recorded model usage for the whole run
below the root. Once crossed, new launches stop and every unfinished agent gets one
final turn. Calls already in flight plus one final turn per unfinished agent can
exceed the threshold.

```python
g = Graph()
root = g.start("Audit this repository.", tree=listing)
flow.run(
    g,
    root_config=AgentConfig(
        model="gpt-5-mini", max_depth=2, max_iters=20, max_budget=200_000, output_schema=schema
    ),
)
```

`root_config` is an `AgentConfig`, the name of one of the Flow's configs, or
`None` for `"default"`. It applies to that call only, so the same graph can
continue with different settings:

```python
g = Graph.load("runs/audit")
g.send(g.roots()[0], "Now check the tests too.")
flow.run(g, root_config=AgentConfig(max_depth=2, max_iters=40))
```

Each agent records the config it last ran with in `agent.config_record`. That
copy is for viewers and is never read back to drive a run.

### Named configs

Named configs live on the Flow; the model picks one with
`launch_agent(task, config=NAME)` and sees each config's `description`.
`launch_agent` without `config=` uses `"default"`:

```python
flow = Flow(
    configs=[
        AgentConfig(model="gpt-5"),
        AgentConfig(name="worker", model="gpt-5-mini", description="cheap and fast"),
    ],
)
flow.run("Audit this repository.", root_config=AgentConfig(model="gpt-5", max_depth=2))
```

`model` is a model name; the Flow builds and caches its client. Register a
client in `llm_clients` only when a name can't describe it (a reasoning effort, a
base URL), under the name configs use:
`Flow(llm_clients={"fast": client_for("gpt-5-mini", reasoning_effort="low")})`
and `AgentConfig(model="fast")`. `flow.add_config(config)` adds one later; an
unknown name is a `KeyError` when the Flow reaches that agent.
`g.start(task, config="worker")` puts a top-level agent on a named config.

### Harnesses

`AgentConfig.harness` names the step function that runs an agent: `"repl"`, the
default, is rlmflow's model-and-REPL loop; `"claude_code"` and `"pi"` run one
headless session turn per step. A harness config is just another named config:

```python
flow.add_config(AgentConfig(name="claude", harness="claude_code", harness_options={"mcp": True}))
```

External sessions reach the same agent tools from the shell through
`rlmflow repl`, so all the code they run lands as their nodes. They keep their
own tools and permissions.

## Multi-turn runs

A finished agent is one whose last turn submitted (`agent.status() == "done"`).
Send it a message and it runs again, with its REPL and nodes intact:

```python
flow.run(g)
g.send(root, "Now implement the fixes.")
flow.run(g)
```

`max_iters` counts an agent's model turns across all its nodes, so a root
you drive for several turns needs headroom for all of them.

## Editing a run

Between runs the graph is plain data. Each edit follows the edges to whatever
depended on what it changed, and returns what it invalidated
(`Invalidation(rewound, removed, dropped, rerun)`); `dry_run=True` reports
without changing anything:

```python
node = next(n for n in root.nodes() if n.runs_code)

g.edit(node, code="print(len(docs))")  # rewind root to the node, with new code
g.delete(root.nodes()[-1])  # drop the last node; the agent redoes it
g.remove(root.children()[0])  # an agent, its subtree, what they caused
g.replace(node, 0, "cached")  # the node's first saved call returns this instead
flow.run(g)
```

When a child's result changes, the node that awaited it is set ready. On the
next run it runs again in place: its other calls return their saved values, so
only the changed one is new. A results message built from a changed result is
taken back and delivered again. The full rules are in
[The graph](node_model.md#editing).

To edit during a stream, either stop it first (`until="idle"` returns at a clean
point) or go through the Flow, which stops the affected agents before it
applies the edit: `await flow.edit(g, "edit", node, code=...)`.

```python
async for _ in flow.run_streaming(g, until="idle"):
    pass
g.send(root, "Finalize with current evidence.")
async for _ in flow.run_streaming(g):
    pass
```

## Fork

`g.fork(node)` returns a new graph: a copy of the run with the same ids, cut
right after `node`. The original is untouched.

```python
node = next(n for n in root.nodes() if n.runs_code)
branch = g.fork(node)
agent = branch.agent(root.id)
agent.config_name = "worker"  # optional: run the copy on another config
branch.send(agent, "Try a different approach.")
flow.run(branch)
```

What the copy keeps and drops:

- `node`'s agent ends at `node`. What came after it is gone, with what that
  caused: agents launched later are removed, messages sent later are taken
  back, and a result delivered to a parent from the dropped nodes is taken back
  too, so the code that awaited it runs again.
- If `node`'s own code launched agents or sent messages, those are gone and
  `node` is set to run again. To keep the children, fork at a later node.
- Every other agent is copied as it is, finished or not, including other
  top-level agents; running the copy runs them. To leave one out, remove it
  from the copy with `branch.remove(branch.agent(id))`.

Each agent's REPL, workspace and Claude Code or pi session is named by its
`repl_id`. By default (`repl="copy"`) the copy's agents get new ones: a REPL is
rebuilt by running its nodes again, with launches and `llm_query` calls
returning their recorded results, files come back from the latest workspace
snapshot, and a resumed session is forked. `g.fork(node, repl="same")` keeps
the live ones, so the copy carries on where the original is (agents the cut
rewound still get new ones); don't run the original and that copy at once.

There is no merge: run the copy you want, each on its own `Flow` if they run
at the same time, and `branch.save(path)` keeps one.

## Launching agents

Agent code starts another agent with `launch_agent`, which returns a handle
immediately:

```python
launch_agent(task: str, *, config: str | None = None, name: str | None = None, **args) -> AgentHandle
```

`config=` names one of the Flow's configs (default `"default"`), which sets the
child's model, harness and limits; agent code can't override them. `name=` gives
it a stable name (otherwise `<config>-<n>`). Keyword arguments become variables
in the child's REPL. Launch every independent child before waiting for any:

```python
a = launch_agent("Find the magic number in `chunk`.", chunk=chunk_a)
b = launch_agent("Find the magic number in `chunk`.", chunk=chunk_b)
results = await asyncio.gather(a, b)
```

Use `Input(value, description=...)` when the caller already knows an argument's
format or meaning. Only the description, name, type and size appear in the
child's prompt; the value stays in its REPL:

```python
child = launch_agent(
    "Compare labels by user.",
    rows=Input(rows, description="List of records with user_id, label, and text fields."),
)
```

To chain agents, await one and pass its result:

```python
requirements, constraints = await asyncio.gather(
    launch_agent("Determine the requirements."),
    launch_agent("Investigate the constraints."),
)
result = await launch_agent(
    "Implement and test using `requirements` and `constraints`.",
    requirements=requirements,
    constraints=constraints,
)
```

A child keeps running after the code that launched it ends. Wait for it in a
later turn with `await child`, or don't: when it finishes, its result reaches
its parent in a results message at the parent's next turn, bound as a variable
named after the child, unless some code already awaited it. A parent that has finished stays finished; only a message reopens it.

## Agent handles

`launch_agent` returns an `AgentHandle`, and `AGENTS[selector]` looks one up by
name, path or id. Both give the same object:

```python
agent = AGENTS["researcher"]
await agent  # its result (only for agents you launched, or they did)
await agent.send("Check 2019 too.")  # message it and wait for its reply (same rule)
agent.send("FYI: schema changed.", rows=rows)  # message any agent; rows becomes a variable there
agent.status  # "done" | "idle" | "running"
agent.done(), agent.result()  # result() raises unless it had finished
agent.parent, agent.children
agent.name, agent.path, agent.config, agent.task
```

`AGENTS` is every agent in the run:

```python
AGENTS.self  # this agent
AGENTS.parent, AGENTS.children
AGENTS.edges(kind="peer")  # edges you added with g.link
"researcher" in AGENTS, len(AGENTS), [a.name for a in AGENTS]
AGENTS.print_graph(show_results=True)
```

Everything except `await` reads a snapshot taken when the code started, so it
doesn't change while it runs, and an agent launched in this code appears in
`AGENTS` from the next code it runs. Awaiting an agent that isn't your descendant raises
`NotYourAgent`; message it and read its reply at your next turn.

## Save and load

```python
g.save("runs/audit")
g = Graph.load("runs/audit")
flow.run(g)
```

Loading restores the graph, not the REPLs. The first time a Flow steps an agent
whose REPL it doesn't hold, it rebuilds it by running the agent's nodes
again: tool calls with saved results return them, launches resolve to the
existing children, and `llm_query` doesn't repeat a request. Nothing is
appended; `ENV["RLMFLOW_REPLAY"]` is `"1"` while it happens, so agent code can
skip other work it shouldn't repeat.

To checkpoint as a run goes, use the periodic consumer:

```python
from rlmflow import GraphCheckpointer

checkpoint = GraphCheckpointer("runs/audit", interval_s=5, interval_nodes=25)
async for step in flow.run_streaming(g):
    checkpoint.handle(step, g)
checkpoint.close()
```

## Tools

Pass tools to the flow, which describes them in the system prompt and seeds them
into each REPL. Visibility can depend on the agent: `@tool(..., when=at_leaf)`
omits the tool from both. `FILE_TOOLS` writes are leaf-only; spawners keep reads.

```python
from rlmflow import FILE_TOOLS, Flow, LocalRuntime, at_leaf, tool


@tool("Run a migration.", when=at_leaf)
def migrate(sql: str) -> str: ...


flow = Flow(
    tools=[FILE_TOOLS, migrate],
    runtime=LocalRuntime(working_directory="."),
)
```

`at_leaf` is true when `can_spawn` is false (`depth >= max_depth`).

A plain function is a tool too: its docstring describes it and its signature
comes from its type hints. `@tool` adds a description, a name, `when=`, or
`where=`. Tools run on the host by default (agent code calls them
over the REPL protocol, so they can hold clients, locks, and connections);
`where="repl"` copies a pure function into the agent's REPL instead. Group
related tools in a tuple; `tools=` flattens one level:

```python
@tool("List available database tables.")
def list_tables() -> list[str]: ...


@tool("Execute read-only SQL and return rows. Call `list_tables()` first.")
def query(sql: str) -> list[dict]: ...


DATABASE_TOOLS = (list_tables, query)
flow = Flow(tools=[DATABASE_TOOLS])
```

Tools described by JSON Schema become tools with `tool_from_schema`. `call`
receives the arguments as a dict and may be async:

```python
from rlmflow import tool_from_schema

forecast = tool_from_schema(
    "forecast",
    "Weather for a city.",
    {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    call=lambda arguments: weather_api.get(arguments["city"]),
)
```

Tools that need a live resource come as a `Toolkit`: any object with
`async open() -> tools` and `async close()`. The Flow opens it when a run starts
and closes it when the run ends. `MCPTools` exposes every tool on an MCP stdio
server this way:

```python
from rlmflow.integrations.mcp import MCPTools

flow = Flow(tools=[MCPTools("python", "weather_server.py")])
```

A host tool marked `@tool(cache=True)` has its JSON result saved on the node
that called it, keyed by its arguments, and replay returns it instead of
calling again. Other tools, and results over 64KB, run again on replay. A tool
can also record metadata on the calling node:

```python
@tool("Fetch one remote record.", cache=True)
def fetch_record(key: str) -> dict:
    value = remote_fetch(key)
    record_metadata("records.fetched", {"key": key, "source": "primary"}, append=True)
    return value
```

Every saved call can be replaced from host code with `g.replace(node, index, value)`,
where `index` is the call's position in `node.calls`.

Add or remove tools later; both reach REPLs that are already open:

```python
flow.add_tool(my_tool)
flow.inject("LOOKUP", {"a": 1})  # any object, not just callables
flow.remove_tool("my_tool")
```

`submit`, `launch_agent`, `llm_query`, `Input`, `asyncio`, `ENV` and `AGENTS`
are reserved.

Fan out independent model calls with ordinary Python concurrency:

```python
answers = await asyncio.gather(*(llm_query(prompt) for prompt in prompts))
```

Host code can use the same with `flow.llm_query(...)`.

## Cleanup

A REPL lives until something closes it, and submitting does not close it: a
finished agent keeps its namespace, because a message can reopen it. Close one
with `flow.runtime.close_repl(agent)`, all of them with
`flow.runtime.close_repls()`, and every Flow resource (turns in flight, model
clients, REPLs) with `await flow.aclose()`.
