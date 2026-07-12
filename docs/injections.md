# Node Injection

Node injection lets an external controller edit a running agent graph by
appending typed nodes, then continuing the run with `agent.run(graph)` or
`agent.run_streaming(graph)`. It is useful for budget controls, human/controller
feedback, forced finalization, and repair nudges that should be represented in
the same trace as normal model and runtime events.

The caller owns the `Graph`, so injection is just a graph mutation followed by
more scheduling. There are three entry points:

```python
flow.append_node(graph, ExecOutput(...))   # append any typed node
flow.apply_action(graph, action)           # apply a GraphAction (e.g. add a child)
graph.inject("controller note", agent_id="root.worker")  # append a user-turn observation
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
import rflow

graph = agent.start("Wait for a controller note, then finish.")

agent.append_node(
    graph,
    rflow.ExecOutput(
        output="Injected controller observation: submit your final answer now.",
        content="Injected controller observation: submit your final answer now.",
    ),
)

agent.run(graph)  # persists the note, then continues
```

Adjacent observations are coalesced into one user-role message by the message
projection (`flow.messages(graph)`), so providers with strict role alternation
still accept the prompt. `graph.inject(text, agent_id=...)` is the shorthand for
appending a `UserQuery` observation to a specific agent.

## Inject Mid-Run From The Stream

Because `run_streaming` mutates the graph in place, a controller can watch the
event stream and inject at a chosen boundary:

```python
async def run_with_controller_stop():
    injected = False
    async for event in agent.run_streaming(graph):
        if not injected and event.type == "append_node" and event.node_type == "exec_output":
            injected = True
            agent.append_node(
                graph,
                rflow.UserQuery(content="Controller stop request: finalize now."),
            )
```

The injected node becomes an ordinary graph row; the next model turn reads it and
can `done(...)` cleanly.

## Rules

- Injection is append-only in the public API today; use `replace_node`,
  `rewind`, or `remove_child` to rewrite existing history.
- Injected nodes are stored as ordinary node rows; there is no per-node
  injection metadata.
- Do not inject into a finished agent.
- Multiple adjacent observation nodes are allowed and coalesce in the message
  projection.
