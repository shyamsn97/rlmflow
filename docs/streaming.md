# Streaming and turns

`Flow.run_streaming(g)` is the execution driver. It keeps one turn in flight for every running agent, including agents added to the graph while the stream runs, yields a `Step` for each node as it finishes, and stops at a boundary.

```python
from rlmflow import AgentConfig, Flow, Graph
from rlmflow.view.text import kind

g = Graph()
root = g.start("Audit this repository.")

flow = Flow(configs=[AgentConfig(model="gpt-5-mini")])
async for step in flow.run_streaming(g):
    print(step.agent.path(), kind(step.node), [e.kind for e in step.edges])

print(root.final_result())
```

The graph is durable state. The Flow holds only live execution state.

```text
durable                      live
─────────────────────        ───────────────────────────
Graph                        Flow.run_streaming
  agents, their nodes          flow.tasks: one turn per agent
  edges                        REPL locks, waiters
                               Runtime resources
```

## Source map

- [`rlmflow/flow.py`](https://github.com/shyamsn97/rlmflow/blob/main/rlmflow/flow.py): `run_streaming` (the driver loop), `_schedule`, `_turn` (one turn), `run_node` (run one node in the agent's REPL), `deliver_results`, `edit`.
- [`rlmflow/graph/graph.py`](https://github.com/shyamsn97/rlmflow/blob/main/rlmflow/graph/graph.py): `status`, `last_turn`, `send`.
- [`rlmflow/graph/calls.py`](https://github.com/shyamsn97/rlmflow/blob/main/rlmflow/graph/calls.py): `CallLog`, what a node's tool calls saved.

## The driver loop

```text
open toolkits
loop:
    forget finished turns; re-raise a root's error
    for every agent with no turn in flight:
        attach unseen results
        if status is "running": send the budget's final-turn message if due, start a turn
    yield a Step for every node that finished since; stop if the boundary says so
    if no turn is in flight: stop
    wait for the graph to change
finally: cancel what's in flight, attach remaining results, close toolkits
```

Every node that finishes wakes the driver, and so does `flow.wake()`, which is safe from any thread. Anything that writes to the graph during a stream (code launching an agent, a peer's message, host code calling `g.send` between steps) is scheduled on the next pass.

## One turn

A turn runs the node an edit left ready, if there is one; otherwise it calls the agent's harness. The native `repl` harness:

```text
call the model with the transcript      -> one assistant node with the reply
reply has one repl block                -> its code runs in place; terminal if it submitted
reply has no repl block                 -> the reply is the answer; terminal
reply has several, empty or open blocks -> the node records the error; nothing runs
```

The node records `seen`, how many of the agent's nodes the prompt had, so a message that arrived during the model call is still unread and the agent takes another turn.

Session harnesses (`claude_code`, `pi`, `interactive`) run one session turn per step; code the session runs through `rlmflow repl` lands as nodes during the turn, and the turn ends with a node carrying `seen` and either `terminal` or `idle` (it is waiting for agents it launched or for unread results).

A turn that raises appends a terminal node with the error as its result. Cancellation is not caught.

## Waiting

```python
flow.tasks  # agent id -> its turn in flight
flow.wake()  # ask the driver to look again; safe from any thread
await flow.wait_for(pred)  # wait until pred() holds, checked whenever the graph changes
```

`await child` in agent code is a `wait_for` on "the child is done". It runs on the Flow's event loop, so code blocked on a child holds no model slot.

## Workers versus Runtime

`Flow(workers=n)` bounds how many model calls run at once; by default there is no bound. `Runtime` places code: one REPL per agent, in a worker process, container or sandbox. A parent awaiting its children holds neither.

## Stream boundaries

| `until`            | Stops after                                                                                 |
| ------------------ | ------------------------------------------------------------------------------------------- |
| `"done"` (default) | No turn is in flight and no agent is running.                                               |
| `"next"`           | The next node an agent wrote itself (messages are written by senders, so they don't count). |
| `"idle"`           | The next node that ran code or ended a turn.                                                |
| a callable         | The first `step` for which `until(step)` is true.                                           |

Stopping cancels the turns still in flight. A node cut off mid-run goes back to ready, keeping what it saved: the agents it launched keep running on the next stream, and the next run executes it again in place, so its launches return the same agents and its finished awaits their saved results. Nothing a child did is paid for twice. Any other turn that was cut off (a model call, a session turn) runs again. A run saved mid-run loads the same way.

## Ordering

Nodes are yielded in the order they finished. Within one agent that is the order of its nodes, which is the order its REPL ran things in; across agents it is completion order, which `seq` records.

## Failure behaviour

Agent code that raises is an observation: the node records the error and the model sees it next turn. A turn that raises (a provider error, a harness crash) finishes its agent with a terminal node and an error; for a root it is also re-raised from the stream.

## Invariants

1. One turn per agent in flight, kept by `flow.tasks`.
1. Every finished node is yielded exactly once.
1. Results are delivered once, recorded by `return` edges.
1. Edits through `flow.edit` cancel the affected turns before they apply.
