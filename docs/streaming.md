# Streaming and turns

`Flow.stream(g)` is the execution driver. It keeps one step in flight for every running agent, including agents added to the graph while the stream runs, yields each node as it finishes, and stops at a boundary.

```python
from rlmflow import Agent, Flow, Graph
from rlmflow.view.text import kind

g = Graph()
root = g.start("Audit this repository.")

flow = Flow(agents={"default": Agent(model="gpt-6.1-terra")})
async for node in flow.stream(g):
    print(node.path(), kind(node), [e.kind for e in g.edges(touching=node.id)])

print(root.final_result())
```

The graph is durable state. The Flow holds only live execution state.

```text
durable                      live
─────────────────────        ───────────────────────────
Graph                        Stream: one live stream call (a record)
  agents, their nodes            its turns, by agent
  edges                          its hooks, timers and toolkits
                             Runtime: REPLs and their locks
                             Flow: agents, tools, and every operation
```

## Source map

- [`rlmflow/flow/flow.py`](https://github.com/shyamsn97/rlmflow/blob/main/rlmflow/flow/flow.py): `stream` (the driver loop), `step` (one step), `deliver_results`, `launch`, `edit`, `stop`.
- [`rlmflow/flow/stream.py`](https://github.com/shyamsn97/rlmflow/blob/main/rlmflow/flow/stream.py): `Stream`, the record of one live stream: its turns, hooks, timers and toolkits. It never calls the Flow.
- [`rlmflow/flow/execution.py`](https://github.com/shyamsn97/rlmflow/blob/main/rlmflow/flow/execution.py): `run_node` (run one node in the agent's REPL) and rebuilding a REPL from the graph.
- [`rlmflow/hooks.py`](https://github.com/shyamsn97/rlmflow/blob/main/rlmflow/hooks.py): the seams hooks plug into.
- [`rlmflow/graph/graph.py`](https://github.com/shyamsn97/rlmflow/blob/main/rlmflow/graph/graph.py): `status`, `last_turn`, `send`.
- [`rlmflow/graph/queries.py`](https://github.com/shyamsn97/rlmflow/blob/main/rlmflow/graph/queries.py): `has_work`, `ready_runs`: who takes a step next.
- [`rlmflow/graph/calls.py`](https://github.com/shyamsn97/rlmflow/blob/main/rlmflow/graph/calls.py): `CallLog`, what a node's tool calls saved.

## The driver loop

```text
open this stream's toolkits; start the hooks
loop:
    run timer callbacks that came due
    forget finished turns; re-raise a root's error
    if nodes finished: hand each to the hooks, then yield it; stop if the boundary says so
    for every agent with work and no turn in flight:
        attach unseen results
        ask the hooks (a node to append, or Stop), then start a step
    if no turn is in flight and nothing finished: stop
    wait for the graph to change
finally: cancel what's in flight, attach remaining results, hand the hooks what
         was never yielded, end the hooks, close toolkits
```

An agent has work when it is running, or idle with every node done and a result it hasn't read. Every node that finishes wakes the driver, and so does `flow.wake()`, which is safe from any thread. Anything that writes to the graph during a stream (code launching an agent, a peer's message, host code calling `g.send` between steps) is scheduled on the next pass.

## One turn

A step runs the node an edit left ready, if there is one; otherwise it calls the agent's `step(node, llm, repl)`. `Agent`'s:

```text
call the model with the transcript and the repl tool -> one assistant node with the reply
reply calls repl once                    -> its code runs in place; terminal if it submitted
reply calls no tool                      -> the reply is the answer; terminal
reply calls repl twice, with no code, or another tool -> the node records the error; nothing runs
```

The Flow marks the node with `seen`, how many of the agent's nodes the prompt had, so a message that arrived during the model call is still unread and the agent takes another step. A node with a result is marked `terminal`; one with neither code, result nor error is marked `idle` (the agent is waiting for runs it launched or for unread results).

`ClaudeCode`, `Pi` and `Person` run one session turn per step; code the session runs through `rlmflow repl` lands as nodes during the turn, and the turn ends with the node `step` returns.

A turn that raises appends a terminal node with the error as its result. Cancellation is not caught.

## Waiting

```python
flow.wake()  # ask the driver to look again; safe from any thread
await flow.wait_for(pred)  # wait until pred() holds, checked whenever the graph changes
```

`await child` in agent code is a `wait_for` on "the child is done". It runs on the Flow's event loop, so code blocked on a child holds no model slot.

## Hooks

Hooks steer and watch one stream. A hook is any object with some of these methods; the rest are skipped:

```python
from rlmflow import Node, Stop


class Policy:
    def start(self, flow, g): ...  # the stream opened
    def before_step(self, flow, run):  # run (a task node) is about to take a step
        return None  # or a Node to append first, or Stop(reason) to end the run

    def before_launch(self, flow, node, task, agent):  # node's code launches a run of agent
        return None  # or a reason to refuse the launch

    async def on_node(self, flow, node): ...  # a node finished; awaited before it is yielded
    def end(self, flow, g): ...  # the stream closed, however it closed


async for node in flow.stream(g, hooks=[Policy()]):
    ...
```

Hooks run in list order. The first `Stop` and the first refusal win; nodes from several hooks are appended in order. A hook that raises fails the stream, and every hook that started still gets `end`. Nodes the caller never saw (it broke out of the loop, or a step failed) still reach `on_node` before `end`. Hooks take the Flow first. `flow.elapsed(g)` is how long the stream has run, and `flow.call_later(g, delay, fn)` runs `fn` on the stream's loop later; `fn` may return an awaitable.

`Budget`, `Checkpoint` and `LiveTree` are hooks.

## Workers versus Runtime

`Flow(workers=n)` bounds how many model calls run at once; by default there is no bound. `Runtime` places code: one REPL per agent, in a worker process, container or sandbox. A parent awaiting its children holds neither.

## Stream boundaries

| `until`            | Stops after                                                                                 |
| ------------------ | ------------------------------------------------------------------------------------------- |
| `"done"` (default) | No turn is in flight and no agent is running.                                               |
| `"next"`           | The next node an agent wrote itself (messages are written by senders, so they don't count). |
| `"idle"`           | The next node that ran code or ended a turn.                                                |
| a callable         | The first node for which `until(node)` is true.                                             |

Stopping cancels the turns still in flight. A node cut off mid-run goes back to ready, keeping what it saved: the agents it launched keep running on the next stream, and the next run executes it again in place, so its launches return the same agents and its finished awaits their saved results. Nothing a child did is paid for twice. Any other turn that was cut off (a model call, a session turn) runs again. A run saved mid-run loads the same way.

## Ordering

Nodes are yielded in the order they finished. Within one agent that is the order of its nodes, which is the order its REPL ran things in; across agents it is completion order, which `seq` records.

## Failure behaviour

Agent code that raises is an observation: the node records the error and the model sees it next turn. A step that raises (a provider error, a harness crash) finishes its agent with a terminal node and an error; for a root it is also re-raised from the stream.

## Invariants

1. One turn per agent in flight, kept by the stream's `Stream.tasks`.
1. Every finished node is yielded exactly once.
1. Results are delivered once, recorded by `return` edges.
1. Edits through `flow.edit` cancel the affected turns before they apply.
