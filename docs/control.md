# Control

The root Node is the control surface. Flow mutates that tree in place and
`run_streaming` yields changed Nodes.

## Run and step

```python
import asyncio
from rlmflow import Flow, render_tree, start

flow = Flow(client, max_depth=2)
root = start("Audit this repository.")

async def drive():
    async for node in flow.run_streaming(root):
        print(node.type)
        print(render_tree(root))

asyncio.run(drive())
print(root.agent_result())
```

`flow.run("query")` and `await flow.arun("query")` return the root agent's final
string result.

Pass `until="next"` for one appended Node, `until="idle"` for a clean
`ExecOutput`/`DoneOutput`, or a `(node, root) -> bool` callable. See
[`streaming.md`](streaming.md).

## Multi-turn runs

Pass a finished root back with `query=`:

```python
flow.run(root, query="Now implement the fixes.")
flow.run(
    root,
    query="Summarize the result.",
    inputs={"format": "markdown"},
    output_schema=Report,
)
```

The new `UserQuery` inherits unspecified model, prompt, inputs, and schema.
Inputs merge by default; `merge_inputs=False` replaces them. An existing warm
REPL receives the new `INPUTS`.

Stage a turn without driving:

```python
flow.append_query(root, "Review one more file.")
```

## Reactive edits

Edit only after a streaming call returns, when its current round has settled:

```python
async for _ in flow.run_streaming(root, until="idle"):
    pass

flow.append(root.tail(), "Finalize with current evidence.", injected=True)

async for _ in flow.run_streaming(root):
    pass
```

For structural edits:

```python
from rlmflow import ExecOutput, surgery

surgery.insert(root, anchor_id, "Controller instruction", mode="after")
surgery.insert(root, anchor_id, ExecOutput(output="replacement"), mode="replace")
removed = surgery.remove(root, node_id)
```

Surgery returns the affected Node. Mutation provenance is available in
`node.metadata["mutation"]`.

## Checkpoint and revert

```python
checkpoint = surgery.checkpoint(root, agent_id="root.worker")

# ...append or run more work...

removed = surgery.revert(root, checkpoint)
```

A checkpoint binds to both Node ids and durable content. Rewriting the saved
prefix makes it stale.

## Fork, rewind, and selection

Forks are independent root Nodes:

```python
branch = await flow.fork(root)
repair = await flow.rewind(root, n=2, agent_id="root.worker")

async for _ in flow.run_streaming(branch):
    pass

flow.discard(repair)
```

An isolated fork gets a new `trajectory_id` and may run concurrently with its
source. A shared-session fork retains identity and cannot stream concurrently
on the same Flow.

There is no merge operation. Continue the selected branch, discard alternatives,
or pass prepared single-agent branches to `flow.launch_branches(...)`.

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
ASCII identifier segment containing only letters, digits, `_`, or `-`.

Host code with prepared branches can use:

```python
await flow.launch_branches(
    parent,
    [fork_a, fork_b],
    queries=["recover with plan A", "recover with plan B"],
    names=["a", "b"],
)
```

This attaches each branch below a `SupervisingOutput`, transfers warm REPL
state, drives the children, and resumes the parent.

## Save and load

```python
from rlmflow import persistence

persistence.save(root, "runs/audit")
loaded = persistence.load("runs/audit")

async for _ in flow.run_streaming(loaded):
    pass
```

Loading is a cold boundary: the Node tree is restored, but REPLs and suspended
Python frames are not. Runtime state is rebuilt when execution needs it.

Use `GraphCheckpointer` for live snapshots:

```python
checkpointer = GraphCheckpointer("runs/audit")
try:
    async for node in flow.run_streaming(root):
        checkpointer.handle(node)
finally:
    checkpointer.close()
```

## Tools

Register fixed tools on Runtime before constructing Flow:

```python
runtime = LocalRuntime(working_directory=".")
runtime.register_tools(FILE_TOOLS)
flow = Flow(client, runtime=runtime)
```

Inject Flow-wide or scoped/factory tools later with `inject_tools`. Runtime
backfills rebuilt namespaces into already-open REPLs.

Enable controller branch tools with:

```python
enable_graph_ops(flow, controller=root)
```

The injected operations are `fork`, `rewind`, `run`, `step`, and `discard`.

## Termination and cleanup

Request cooperative termination before selected agents' next transitions:

```python
flow.terminate(root, agent_ids=["root.worker"])
```

Close one trajectory automatically:

```python
async for _ in flow.run_streaming(root, close_repls=True):
    pass
```

Close all Flow resources with `await flow.aclose()`.
