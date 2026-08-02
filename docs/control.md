# Control

The root Node is the control surface. Flow mutates that tree in place and
`run_streaming` yields the nodes as they land.

## Run and step

```python
import asyncio
from rlmflow import Flow, start

flow = Flow(client)
root = start("Audit this repository.", max_depth=2)

async def drive():
    async for node in flow.run_streaming(root):
        print(node.parent_agent.config.path, node.type)

asyncio.run(drive())
print(root.result())
```

`flow.run("query")` and `await flow.arun("query")` return the root agent's final
result — a string, or the parsed value if the agent has an `output_schema`.

Pass `until="next"` for one appended Node, `until="idle"` for a clean
`ExecOutput`/`DoneOutput`, or a `(node, root) -> bool` callable. See
[`streaming.md`](streaming.md).

## Per-agent limits

Limits belong to the agent, not the flow. Set them on the root with `start(...)`:

```python
root = start(
    "Audit this repository.",
    inputs={"tree": listing},
    model="fast",
    max_depth=2,
    max_iters=20,
    max_budget=200_000,
    keep_n_messages=8,
    output_schema=schema,
)
```

Or give the flow a default `AgentConfig`, which it copies onto every root it
builds from a bare query string:

```python
from rlmflow import AgentConfig, Flow

flow = Flow(client, config=AgentConfig(max_depth=2, max_iters=20))
flow.run("Audit this repository.")
```

A loaded run keeps its identity, inputs, model, prompt profile, and schema, but
not its limits — those come back as `AgentConfig` defaults. Set them on the
loaded root before resuming if they matter:

```python
from rlmflow import AgentStart

loaded = AgentStart.load("runs/audit")
loaded.config.max_iters = 40
```

Children inherit the parent's config through `config.child(name)`, with
`child_max_iters` as the per-child override and the spec's `inputs`, `model`,
`prompt_profile`, and `output_schema` layered on top.

## Multi-turn runs

An agent that answered is finished only because its frontier is a `DoneOutput`.
Append a new query to that frontier and the same root runs again, with its REPL
namespace and its whole history intact:

```python
from rlmflow import UserQuery

flow.run(root)

root.frontier.append(UserQuery(content="Now implement the fixes."))
flow.run(root)
```

`max_iters` counts an agent's model turns across the whole transcript, so a root
you intend to drive for several turns needs headroom for all of them.

To change the inputs an agent sees, set them on its config and let the next
step re-seed the REPL:

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

An append lands on the agent's frontier or it raises, so an edit cannot silently
rewrite history: to change what an agent already did, fork the run at that point
instead.

Two rules keep this clean in a run that delegates. Stop on the agent's own output
rather than on a child's, since closing a stream cancels the parent's block and
resuming re-runs it, costing a turn. And to reach a worker while its parent is still inside that
block, append from inside the loop as the worker's node lands. See
[`injections.md`](injections.md).

## Fork and selection

`node.fork()` copies the whole run and cuts everything after that node. The copy
is an independent root, with fresh ids by default, that any `Flow` can continue:

```python
from rlmflow import ExecAction

action = next(n for n in root.transcript() if isinstance(n, ExecAction))
branch = action.fork()

async for _ in flow.run_streaming(branch):
    pass
```

The fork's agent is back at that node with the nodes below it gone, so the next
step re-runs from there. A `Flow` that has not run the branch rebuilds its REPL
first — see the restore modes below.

There is no merge operation. Continue the selected branch and discard the
alternatives.

## Delegation

Agent code must await the launcher:

```python
[answer] = await launch_subagents([
    {"name": "single", "query": query, "inputs": {"data": data}},
])

results = await launch_subagents([
    {"name": "a", "query": "...", "inputs": {"chunk": chunk_a}},
    {"name": "b", "query": "...", "inputs": {"chunk": chunk_b}},
])
```

Specs run concurrently and results preserve spec order. Child `name` is one
ASCII identifier segment containing only letters, digits, `_`, or `-`, and is
unique among its siblings. A spec may also carry `model`, `prompt_profile`, and
`output_schema`. A spec deeper than `max_depth`, or with a query longer than
`max_query_chars`, is answered with a refusal string in place of a child.

Children hang off the `ExecAction` that launched them, so the parent's
transcript stays a single chain: `root.sub_agents` is where the children are,
and the launching step lands one `ExecOutput` when the block finishes.

## Save and load

```python
from rlmflow import AgentStart

run_dir = root.save("runs/audit")
loaded = AgentStart.load("runs/audit")

async for _ in flow.run_streaming(loaded):
    pass
```

Loading is a cold boundary: the tree is restored, but no REPL is. A tree handed
to a `Flow` that has not run it — loaded, forked, or from another process — gets
its namespaces back one of two ways:

- `Flow(restore="replay")`, the default, re-runs each unfinished agent's
  recorded code once before the run loop starts. Nothing is appended and no
  child is relaunched; `ENV["RLMFLOW_REPLAY"]` is `"1"` while it happens, so
  agent code can skip work it should not repeat.
- `Flow(restore="lazy")` skips the rebuild and tells each unfinished agent, in
  its own transcript, that its variables are gone.

To checkpoint as a run goes, save from inside the streaming loop:

```python
async for node in flow.run_streaming(root):
    root.save("runs/audit")
```

## Tools

Pass fixed tools to the flow, which describes them in the system prompt and
seeds them into every REPL:

```python
from rlmflow import FILE_TOOLS, Flow, LocalRuntime

flow = Flow(client, tools=FILE_TOOLS, runtime=LocalRuntime(working_directory="."))
```

Add or remove them later; both reach REPLs that are already open:

```python
flow.add_tool(my_tool)
flow.inject("LOOKUP", {"a": 1})   # any object, not just callables
flow.remove_tool("my_tool")
```

`done`, `launch_subagents`, and `INPUTS` are reserved: they are rebuilt for each
step and cannot be injected over.

`Flow(use_llm_query=True)` adds `llm_query_batched` to the REPL, for agents that
want to fan out one-shot model calls without spawning children. It is available
to host code as `flow.llm_query_batched(...)` either way.

## Cleanup

Close one run's REPLs as the stream exits:

```python
async for _ in flow.run_streaming(root, close_repls=True):
    pass
```

Close all Flow resources — queued work, thread pool, and every REPL — with
`await flow.aclose()`.
