# Node injection

Controllers change the graph while a run streams or between streaming calls,
then continue the same run.

## Send a message

A message is how a controller gets a word in. It waits in the agent's inbox and
lands in its history at the agent's next turn:

```python
g.send(root, "Controller stop request: finalize now.")
flow.run(g)
```

A finished agent that gets a message reopens. Messages from the host have
`sender="user"` unless you pass another label, and no `message` edge, because no
cell sent them. They carry no timing, because no step produced them.

## Change history

Everything already in the graph can be edited; see
[The graph](node_model.md#editing) for the rules.

```python
cell = next(n for n in g.history(root) if n.type == "code")
print(g.edit(cell, code="print(sorted(files)[:20])", dry_run=True))
g.edit(cell, code="print(sorted(files)[:20])")  # root rewinds to this cell
flow.run(g)
```

To keep the original and try something else, fork instead:

```python
branch = g.fork(cell, name="retry")
g.send(branch, "Try a different approach.")
flow.run(g)
```

## Inject tools and values

The other injection point is the REPL namespace, which reaches agents that are
already running:

```python
flow.inject("BUDGET_REMAINING", 4)
flow.add_tool(escalate)
flow.remove_tool("escalate")
```

`inject` takes any object, not just callables; `add_tool` is the same thing with
the name read off the function. Both write into every open REPL as well as the
flow's namespace, so a tool added mid-run is visible on the next step.

## Reactive control

**Inside the loop.** `g.send` is safe at any time, from the stream loop or
another thread: the message waits for the agent's next turn.

```python
async for step in flow.run_streaming(g):
    if step.agent is not root and step.node.type == "code":
        g.send(step.agent, "Answer with the best evidence.")
```

This reaches a worker while its parent is still inside the cell that launched it.

**Between streaming calls.** Edits are refused for agents that are stepping, so
stop the stream first:

```python
async for _ in flow.run_streaming(g, until="idle"):
    pass

g.send(root, "Human feedback: verify the auth path.")

async for _ in flow.run_streaming(g):
    pass
```

A boundary that fires on a child's node stops the stream while the parent's
cell is still running. Stopping cancels that cell and undoes what it did, its
children included, so the next run runs the cell, and its children, again. To
avoid the repeat, stop on the parent's own nodes:

```python
def root_idle(step):
    return step.agent is root and step.node.type in ("code", "idle", "done")
```

See [`examples/control/injection/`](../examples/control/injection/) for runnable variants.
