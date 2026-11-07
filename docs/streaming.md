# Streaming and task scheduling

`Flow.run_streaming(g)` is the execution driver. It keeps one step in flight for
every ready agent, including agents added to the graph while the stream runs,
yields a `Step` for each node as it lands, and stops at a boundary.

```python
from rlmflow import AgentConfig, Flow, Graph

g = Graph()
root = g.add_agent("Audit this repository.")

flow = Flow(configs=[AgentConfig(model="gpt-5-mini")])
async for step in flow.run_streaming(g):
    print(g.path(step.agent), step.node.type, [e.kind for e in step.edges])

print(g.result(root))
```

The graph is durable state. The queue holds only live execution state.

```text
durable                      live
─────────────────────        ───────────────────────────
Graph                        Flow.run_streaming
  agents, histories            TaskQueue: one task per agent
  inboxes                      asyncio tasks, waiters
  edges, edits                 Pool / Runtime resources
```

## Source map

- [`rlmflow/flow.py`](https://github.com/shyamsn97/rlmflow/blob/main/rlmflow/flow.py):
  `run_streaming` (the driver loop), `_schedule`, `run_step` (one step),
  `rerun` (a cell an edit invalidated), `deliver_results`.
- [`rlmflow/engine/scheduler.py`](https://github.com/shyamsn97/rlmflow/blob/main/rlmflow/engine/scheduler.py):
  `StepResult`, `TaskQueue`, `Pool`/`ThreadPool`/`SequentialPool`, `resolve_boundary`.
- [`rlmflow/graph/graph.py`](https://github.com/shyamsn97/rlmflow/blob/main/rlmflow/graph/graph.py):
  `status`, `deliver`, `busy`, `subscribe`.

## The driver loop

```text
subscribe to the graph (every landed node is queued for the stream)
open toolkits
deliver unseen results to every agent that is between turns
schedule
loop while anything is in flight:
    result = await queue.next()          # a finished step, or a wake-up
    if a step finished: unmark busy, deliver results to its agent and its parent
    yield a Step for every node that landed since; stop if the boundary says so
    notify waiters (cells awaiting agents re-check the graph)
    re-raise a root's infrastructure error
    schedule
finally: unsubscribe, cancel what's in flight, close toolkits
```

`_schedule` goes through every agent whose `g.status` is `"running"` and has no
step in flight: it sends the budget's final-turn message if the run is nearly
out of tokens, moves the inbox into the history (`g.deliver`), marks the agent
busy and submits `run_step`.

Because the loop reacts to graph writes, anything that writes to the graph during
a stream (a cell launching an agent, a peer's message, a host thread calling
`g.send`) wakes the driver, and the new work is scheduled on the next pass.

## One step, one node

`run_step(step)` either runs a pending cell again, or calls the agent's harness
and appends the node it returns. The native `repl` harness:

```text
last node is none, a message or code  -> call the model          -> llm
last node is llm with one repl block  -> run the cell            -> code, then done if it submitted
last node is llm with no repl block   -> the reply is the answer -> done
```

A cell lands pending when it starts, and is filled in when it finishes.

Session harnesses (`claude_code`, `pi`, `interactive`) run one session turn per
step; cells the session runs through `rlmflow repl` land as `code` nodes during
the turn, and the turn ends with `idle` (it is waiting for agents it launched or
for unread results) or `done`.

A step that raises lands `done` with the error as its result. Cancellation is
not caught.

## TaskQueue

```python
queue.submit(step, fn)  # run fn(step) unless this agent already has a step in flight
await queue.next()  # the next StepResult, or None when only woken
queue.wake()  # ask the driver to look for new work; safe from any thread
await queue.join(pred)  # wait until pred() holds after some processed change
await queue.cancel()  # cancel every step (or some agents')
```

`await child` in a cell is a `join` on "the child is done". It runs on the
Flow's event loop, so a cell blocked on a child doesn't hold a pool slot.

## Pool versus Runtime

`Pool` places model compute: `ThreadPool(workers)` bounds how many model calls
run at once, `SequentialPool` runs one at a time. `Runtime` places code: one REPL
per agent, in a worker process, container or sandbox. A parent awaiting its
children holds neither.

## Stream boundaries

| `until` | Stops after |
| --- | --- |
| `"done"` (default) | Nothing is ready or in flight. |
| `"next"` | The next node a step produced (messages are written by senders, so they don't count). |
| `"idle"` | The next `code`, `idle` or `done`. |
| a callable | The first `step` for which `until(step)` is true. |

Stopping cancels the steps still in flight. A cell cut off mid-run stays
pending, keeping what it saved: the agents it launched keep running on the next
stream, and the next run executes the cell again in place, so its launches
return the same agents and its finished awaits their saved results. Nothing a
child did is paid for twice. Any other step that was cut off (a model call, a
session turn) runs again. A run saved mid-cell loads the same way.

## Ordering

Nodes are yielded in the order they landed. Within one agent that is its history
order; across agents it is completion order, which `seq` records.

## Failure behaviour

Agent code that raises is an observation: the `code` node records the error and
the model sees it next turn. A step that raises (a provider error, a harness
crash) finishes its agent with `done` and an error; for a root it is also
re-raised from the stream.

## Scheduler invariants

1. One step per agent in flight, enforced by `g.busy` and the queue.
1. Every landed node is yielded exactly once.
1. A message is delivered only at a turn boundary.
1. Results are delivered once, recorded by `return` edges.
