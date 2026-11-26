# Node injection

Controllers change the graph while a run streams or between streaming calls, then continue the same run.

## Send a message

A message is how a controller gets a word in. It is a user node added to the agent, which the agent reads at its next turn:

```python
g.send(root, "Controller stop request: finalize now.")
flow.run(g)
```

A finished agent that gets a message reopens. Keyword arguments become variables in the agent's REPL, and the text says what was set:

```python
g.send(root, "Use these thresholds.", limits={"p95_ms": 250})
```

Messages from the host have producer `"user"` unless you pass `sender=`, and no `write` edge unless you pass `by=` the node they come from. They carry no timing, because no turn produced them.

## Edit the graph

Everything already in the graph can be edited; see [The graph](node_model.md#editing) for the rules.

```python
node = next(n for n in root.nodes() if n.runs_code)
print(g.edit(node, code="print(sorted(files)[:20])", dry_run=True))
g.edit(node, code="print(sorted(files)[:20])")  # root rewinds to this node
flow.run(g)
```

To keep the original and try something else, fork instead. The copy is cut right after `node`, and the message is injected into it:

```python
branch = g.fork(node)
branch.send(branch[node.root_id].root, "Try a different approach.")
flow.run(branch)
```

[`examples/graph/06_fork.py`](https://github.com/shyamsn97/rlmflow/blob/main/examples/graph/06_fork.py) forks one run twice and injects a different instruction into each copy.

## Inject tools and values

The other injection point is the REPL namespace, which reaches agents that are already running:

```python
flow.add_tool(4, name="BUDGET_REMAINING")
flow.add_tool(escalate)
flow.remove_tool("escalate")
flow.repl(worker).set(hint="check the auth path")  # one run's REPL only
```

`add_tool` takes any object with `name=`, and reads the name off a function without it. Each REPL binds exactly the flow's tools whenever it runs code, so a tool added mid-run is there on the next run of code and a removed one is gone. `flow.repl(run).set(...)` binds values and tools in that run's REPL for the rest of the run.

## Reactive control

**Inside the loop.** Host code in the `async for` runs on the Flow's loop between steps, so `g.send` there is safe; the agent reads the message at its next turn.

```python
async for node in flow.stream(g):
    if node.root_id != root.id and node.runs_code:
        g.send(node.root, "Answer with the best evidence.")
```

This reaches a worker while its parent is still inside the code that launched it.

**Edits during a stream.** `await flow.edit(g, op, node, ...)` stops the agents the edit touches, then applies it, so it is safe inside the loop too:

```python
async for node in flow.stream(g):
    if node.error:
        await flow.edit(g, "delete", node)  # the agent redoes that turn
```

**Between streaming calls.** Or stop the stream first:

```python
async for _ in flow.stream(g, until="idle"):
    pass

g.send(root, "Human feedback: verify the auth path.")

async for _ in flow.stream(g):
    pass
```

A boundary that fires on a child's node stops the stream while the parent's code is still running. That node goes back to ready, keeping what it saved, so the next run runs it again in place: its launches return the same children and its finished awaits their saved results. To stop on the parent's own turns instead:

```python
def root_idle(node):
    return node.root_id == root.id and (node.runs_code or node.ends_turn)
```

See [`examples/graph/08_streaming.py`](https://github.com/shyamsn97/rlmflow/blob/main/examples/graph/08_streaming.py) for stopping a run, sending a message and resuming, and [`examples/shepherd/`](../examples/shepherd/) for forking a run and steering each fork live, rewinding Sokoban workers.
