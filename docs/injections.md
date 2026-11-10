# Node injection

Controllers change the graph while a run streams or between streaming calls,
then continue the same run.

## Send a message

A message is how a controller gets a word in. It is a user node added to
the agent, which the agent reads at its next turn:

```python
g.send(root, "Controller stop request: finalize now.")
flow.run(g)
```

A finished agent that gets a message reopens. Keyword arguments become
variables in the agent's REPL, and the text says what was set:

```python
g.send(root, "Use these thresholds.", limits={"p95_ms": 250})
```

Messages from the host have producer `"user"` unless you pass `sender=`, and no
`write` edge unless you pass `by=` the node they come from. They carry no
timing, because no turn produced them.

## Edit the graph

Everything already in the graph can be edited; see
[The graph](node_model.md#editing) for the rules.

```python
node = next(n for n in root.nodes() if n.runs_code)
print(g.edit(node, code="print(sorted(files)[:20])", dry_run=True))
g.edit(node, code="print(sorted(files)[:20])")  # root rewinds to this node
flow.run(g)
```

To keep the original and try something else, fork instead. The copy is cut
right after `node`, and the message is injected into it:

```python
branch = g.fork(node)
branch.send(branch.agent(node.owner), "Try a different approach.")
flow.run(branch)
```

[`examples/graph/06_fork.py`](../examples/graph/06_fork.py) forks one run twice
and injects a different instruction into each copy.

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
flow's namespace, so a tool added mid-run is visible on the next run of code.

## Reactive control

**Inside the loop.** Host code in the `async for` runs on the Flow's loop
between steps, so `g.send` there is safe; the agent reads the message at its
next turn.

```python
async for step in flow.run_streaming(g):
    if step.agent is not root and step.node.runs_code:
        g.send(step.agent, "Answer with the best evidence.")
```

This reaches a worker while its parent is still inside the code that launched
it.

**Edits during a stream.** `await flow.edit(g, op, node, ...)` stops the agents
the edit touches, then applies it, so it is safe inside the loop too:

```python
async for step in flow.run_streaming(g):
    if step.node.error:
        await flow.edit(g, "delete", step.node)  # the agent redoes that turn
```

**Between streaming calls.** Or stop the stream first:

```python
async for _ in flow.run_streaming(g, until="idle"):
    pass

g.send(root, "Human feedback: verify the auth path.")

async for _ in flow.run_streaming(g):
    pass
```

A boundary that fires on a child's node stops the stream while the parent's
code is still running. That node goes back to ready, keeping what it saved, so
the next run runs it again in place: its launches return the same children and
its finished awaits their saved results. To stop on the parent's own turns
instead:

```python
def root_idle(step):
    return step.agent is root and (step.node.runs_code or "seen" in step.node.data)
```

See [`examples/graph/08_streaming.py`](../examples/graph/08_streaming.py) for stopping a run, sending a message and resuming, and [`examples/shepherd/`](../examples/shepherd/) for forking a run and steering each fork live, rewinding Sokoban workers.
