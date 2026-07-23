# Node Injection

Node injection lets an external controller edit a running agent graph by
appending typed nodes, then continuing the run with `agent.run(graph=graph)` or
`agent.run_streaming(graph=graph)`. It is useful for budget controls, human/controller
feedback, forced finalization, and repair nudges that should be represented in
the same trace as normal model and runtime events.

The caller owns the `Graph`, so injection is just a graph mutation followed by
more scheduling. `graph.inject(...)` is the base graph-edit op; `append`,
`prepend`, and `replace` are thin helpers over it:

```python
graph.inject("controller note")                     # append a user-turn (default)
graph.inject(node, at=anchor, mode="before")        # insert before a node
graph.append(node)                                  # mode="after"   (sugar)
graph.prepend(node)                                 # mode="before"  (sugar)
graph.replace(anchor, node, truncate="descendants") # swap + re-route a branch
```

Those edit the graph value. To inject *while a run is streaming* (so the edit is
emitted into the live event stream), go through the Flow instead:

```python
flow.append_node(graph, ExecOutput(...), injected=True)  # stamp + emit into the run
flow.apply_action(graph, action)           # apply any GraphAction (e.g. add a child)
```

All three mutate `graph` in place. Nothing is committed to a live REPL session
until the next `run_streaming`/`run`, which persists the appended nodes as ordinary graph
rows, updates the message projection, and resumes normal scheduling.

For a runnable offline demo, see
[`examples/control/controller_injection.py`](../examples/control/controller_injection.py).

## Inject A Controller Observation

Append an `ExecOutput` when you want the next LLM turn to see controller-authored
feedback without pretending the model wrote a REPL block:

```python
import asyncio
import rlmflow

graph = rlmflow.Graph(query="Wait for a controller note, then finish.")

agent.append_node(
    graph,
    rlmflow.ExecOutput(
        output="Injected controller observation: submit your final answer now.",
        content="Injected controller observation: submit your final answer now.",
    ),
    injected=True,
)

agent.run(graph=graph)  # persists the note, then continues
```

Adjacent observations are coalesced into one user-role message by the message
projection (`flow.messages(graph)`), so providers with strict role alternation
still accept the prompt. `graph.inject(text, agent_id=...)` is the shorthand for
appending a `UserQuery` observation to a specific agent (a string is wrapped as a
`UserQuery`; pass a `Node` to inject any typed node).

## Inject Mid-Run From The Stream

Because `run_streaming` mutates the graph in place, a controller can watch the
event stream and inject at a chosen boundary:

```python
async def run_with_controller_stop():
    injected = False
    async for event in agent.run_streaming(graph=graph):
        if not injected and event.type == "append_node" and event.node_type == "exec_output":
            injected = True
            agent.append_node(
                graph,
                rlmflow.UserQuery(content="Controller stop request: finalize now."),
                injected=True,
            )
```

The injected node becomes an ordinary graph row; the next model turn reads it and
can `done(...)` cleanly.

## Rules

- `inject` covers append/insert/replace via `mode`; `replace(..., truncate=
  "descendants")` also rewrites history and prunes orphaned children. `rewind`
  and `remove_child` remain for pure removal.
- Injected nodes are ordinary node rows with ``metadata["injected"] = True``.
  ``Graph.inject`` / ``append`` / ``prepend`` / ``replace`` stamp this by
  default; ``Flow.append_node`` does so when called with ``injected=True``
  (organic scheduler commits leave it unset).
- Do not inject into a finished agent.
- Multiple adjacent observation nodes are allowed and coalesce in the message
  projection.
