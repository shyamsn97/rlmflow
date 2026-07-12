# Streaming And Scheduling

`Flow.run_streaming(...)` is the low-level way to drive a run. It mutates the
caller-owned `Graph` in place and yields typed graph `Event`s as work is
committed. The same method also controls **where execution stops** through the
`until` argument.

The important model is simple:

- every agent has its own event queue;
- every agent can have at most one active task at a time;
- after an event is yielded, `Flow` decides whether to enqueue that agent's next
  task;
- if `Flow` does not enqueue a next task, that agent stops at the current
  boundary.

There are no parked coroutine frames for step boundaries. Stopping is just "do
not add another task for this agent."

## Basic Use

Run to completion and observe every graph mutation:

```python
flow = Flow(client)
graph = Graph(query="Audit this repository.")

async for event in flow.run_streaming(graph):
    print(event.type, getattr(event, "node_type", ""))

print(graph.result())
```

`flow.run(graph)` is the synchronous convenience wrapper for the same full run.

## Boundaries With `until`

`until` controls whether the scheduler adds more work after each event.

`until="done"` is the default. It keeps scheduling until the whole graph is
finished:

```python
async for event in flow.run_streaming(graph, until="done"):
    ...
```

`until="next"` advances the active frontier by one node-producing task, then
stops:

```python
async for event in flow.run_streaming(graph, until="next"):
    ...
```

Use it when you want the most literal debug step. It surfaces errors immediately:
if an agent emits `ErrorOutput`, the scheduler stops there instead of trying to
heal it.

`until="idle"` advances each active agent until it reaches a rest point:

```python
async for event in flow.run_streaming(graph, until="idle"):
    ...
```

Rest means `ExecOutput` or `DoneOutput`. `ErrorOutput` is not rest, so `idle`
keeps scheduling the agent until it recovers to a clean observation or finishes.

Named event boundaries stop when that event is observed:

```python
async for event in flow.run_streaming(graph, until="supervising"):
    ...

async for event in flow.run_streaming(graph, until="error"):
    ...
```

A callable boundary receives `(event, graph)`:

```python
def child_finished(event, graph):
    return (
        event.type == "append_node"
        and event.agent_id != "root"
        and event.node_type == "done_output"
    )

async for event in flow.run_streaming(graph, until=child_finished):
    ...
```

For non-`done` boundaries, the run is left alive. Continue it by calling
`run_streaming(...)` again, usually without passing the graph:

```python
async for _event in flow.run_streaming(graph, until="idle"):
    pass

graph.inject("Use this controller note before finalizing.")

async for _event in flow.run_streaming(until="done"):
    pass
```

Because the previous streaming call stopped with no active work, the injected
graph edit is read before any new task is scheduled.

## Multiple Steps With `n`

`n` means "number of boundary passes", not "number of raw events".

```python
async for event in flow.run_streaming(graph, until="next", n=3):
    ...
```

With `until="next"`, that advances the active frontier three scheduler steps.
With `until="idle"`, it advances to idle three times. Each step may yield more
than one event because multiple agents can run in parallel.

## What A Task Is

Internally, one scheduled agent task emits one graph-producing unit:

- if the current node is a `UserQuery`, `ExecOutput`, `ErrorOutput`, or
  `ResumeAction`, the task calls the model and commits `LLMOutput`;
- if the current node is `LLMOutput`, the task commits `ExecAction`;
- if the current node is `ExecAction`, the task executes the code and commits
  `ExecOutput`, `ErrorOutput`, `DoneOutput`, or `SupervisingOutput`.

After each committed event, the scheduler decides whether to schedule the next
task for that same agent.

This keeps `next` precise: one scheduler step cannot silently run a full model
turn beyond the event you observed.

## The Task Queue

The queue is intentionally small and graph-free. It tracks:

- `queues[agent_id]`: emitted events waiting to be streamed;
- `tasks[agent_id]`: the currently running task for that agent, if any;
- `queued[agent_id]`: follow-up work remembered for an agent that is still
  running;
- `notify`: an event that wakes the stream when new output or task completion
  happens.

The queue does not know about `until`, graph nodes, child agents, or prompts.
Those are `Flow` concerns.

The core behavior is:

```python
queue.add(agent_id, work)     # schedule next work, or remember it if running
queue.emit(agent_id, event)   # append an event to that agent's stream
queue.stop(agent_id)          # drop queued follow-up work

async for agent_id, event in queue.stream():
    ...
```

When a task finishes, the queue starts any remembered follow-up for the same
agent. If there is no follow-up, the agent simply has no work.

## Parallelism

Parallelism falls out of per-agent scheduling:

- every ready agent can have one active task;
- different agents' tasks run concurrently;
- model/runtime work is still bounded by the configured `Pool`;
- `SequentialPool` serializes work, while the default async pool can fan out.

So `until="next"` is a global step: every ready agent can advance once in the
same streaming call. A parent waiting for children does not consume a step until
the children finish.

## Delegation

`launch_subagents(...)` creates child graphs and emits normal graph events:

1. the parent emits `SupervisingOutput`;
2. child graphs get `UserQuery` nodes;
3. child work is scheduled;
4. the parent waits for child results;
5. the parent emits `ResumeAction`, then continues and eventually emits its next
   observation.

For full runs (`until="done"`), children are gathered directly through the pool
to preserve existing eager behavior and sandbox/runtime compatibility. The stream
still observes their graph events, but the scheduler suppresses duplicate
follow-up scheduling for those child ids.

For bounded runs (`until="next"`, `"idle"`, `"supervising"`, `"error"`, or a
callable), children are scheduled through the same per-agent queue system, so
their events can be observed at boundaries.

## Reactive Control

The safe edit point is between streaming calls:

```python
async for _event in flow.run_streaming(graph, until="idle"):
    pass

graph.inject("Controller instruction: finish now.")

async for _event in flow.run_streaming(until="done"):
    pass
```

At that point no scheduler task is running. The graph is the durable state; the
queue is disposable execution state. Rewinds, forks, and injected nodes should be
represented as graph mutations, then the next streaming call schedules work from
that graph frontier.

## Mental Model

Think of the graph as the source of truth and the queue as a pump:

```text
graph frontier -> schedule ready agent tasks -> emit events -> yield events
              -> decide continue/stop per agent -> schedule next tasks
```

`until` is the "decide continue/stop" policy:

- `done`: keep going until the graph is finished;
- `next`: stop each agent after its next event;
- `idle`: keep going until each agent reaches a clean rest;
- `error`: keep going until an error event is observed;
- `supervising`: keep going until a delegation event is observed;
- callable: keep going until your predicate returns true.

This is why the implementation can stay small: stopping does not require a
special pause primitive. It just means there is no next task in the queue.
