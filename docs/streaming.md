# Streaming and task scheduling

`Flow.run_streaming(...)` is the execution driver. It mutates one or more Node
trees, yields each durable Node as it is created, applies stream boundaries, and
feeds unfinished leaves back into one `TaskQueue`.

```python
from rlmflow import Flow

flow = Flow(client)
root = flow.start("Audit this repository.")

async for node in flow.run_streaming(root):
    print(node.parent_agent.config.path, node.type)

print(root.result())
```

The graph is durable state. The queue contains only live execution state.

```text
durable                                      live
──────────────────────────────────────      ─────────────────────────────
Node trees                                  Flow.run_streaming
  AgentStart                                  TaskQueue
    LLMOutput                                 asyncio tasks
      ExecAction                              Condition waiters
        ExecOutput                            Pool / Runtime resources
```

## Source map

The implementation is deliberately split by responsibility:

- [`rlmflow/flow.py`](https://github.com/shyamsn97/rlmflow/blob/main/rlmflow/flow.py)
  - `Flow.run_streaming`: the driver loop;
  - `Flow.step`: one complete Node transition;
  - `Flow.launch_tool`: child creation, submission, and joining;
  - `Flow.llm_step` / `Flow.exec_step`: model and REPL transitions.
- [`rlmflow/engine/execution.py`](https://github.com/shyamsn97/rlmflow/blob/main/rlmflow/engine/execution.py)
  - `Transition`: submitted Node, created Node, and infrastructure error;
  - `TaskQueue`: active tasks, completed transitions, child-terminal wake-ups;
  - `Pool`, `ThreadPool`, `SequentialPool`: compute placement and capacity.
- [`rlmflow/engine/parallel.py`](https://github.com/shyamsn97/rlmflow/blob/main/rlmflow/engine/parallel.py)
  - `parallel_stream`: public multi-root wrapper;
  - `parallel_run`: drive several roots and return them in argument order.
- [`rlmflow/engine/boundaries.py`](https://github.com/shyamsn97/rlmflow/blob/main/rlmflow/engine/boundaries.py)
  - built-in and custom `until` boundary resolution.
- [`rlmflow/graph/nodes.py`](https://github.com/shyamsn97/rlmflow/blob/main/rlmflow/graph/nodes.py)
  - `Node.append`, `AgentStart.leaves`, `frontier`, `terminal`, and `result`.

## The execution contract

The scheduler has four owners:

```text
Flow.step
  Decides what one Node creates next.

TaskQueue
  Runs submitted leaves and reports completed transitions.

Flow.run_streaming
  Chooses successors, applies boundaries, and yields Nodes.

Pool / Runtime
  Decide where model calls and code execution run.
```

`TaskQueue` does not inspect roots, boundaries, transcripts, or graph topology.
It can run Nodes from several independent graphs in the same queue.

## One step, one Transition

`Flow.step(node)` consumes one frontier Node and creates its next durable Node:

```text
AgentStart ─┐
UserQuery ──┼─> llm_step ─> LLMOutput
ExecOutput ─┤
ErrorOutput ┘

LLMOutput ────────────────> ExecAction
ExecAction ─> exec_step ──> ExecOutput | ErrorOutput | DoneOutput
```

The return value records both sides:

```python
@dataclass(slots=True)
class Transition:
    submitted: Node
    created: Node
    error: BaseException | None = None

    @property
    def is_agent_start(self) -> bool:
        return isinstance(self.created, AgentStart)
```

- `submitted` is the frontier passed to `Flow.step`.
- `created` is the durable Node returned by that step.
- `error` carries an infrastructure exception after the failure has been recorded
  in the graph.

For a normal model turn:

```text
submitted = AgentStart("question")
created   = LLMOutput("...")
```

For a code turn:

```text
submitted = ExecAction("print(...)")
created   = ExecOutput("...")
```

### Why child starts are special

A child `AgentStart` must appear in the stream immediately, while its first model
step is already running. `TaskQueue.submit(..., publish=True)` therefore publishes:

```python
Transition(submitted=child, created=child)
```

This is a child-start notification, not the completion of the child's first step.
`Transition.is_agent_start` identifies it. The queue leaves the child's active task
tracked, and the driver does not submit the child a second time.

Later, the real first-step transition arrives:

```text
submitted = child AgentStart
created   = child LLMOutput
```

## TaskQueue state

The queue owns three small pieces of live state:

```python
self.running: dict[int, tuple[Node, asyncio.Task[None]]]
self.done: asyncio.Queue[Transition]
self.changed = asyncio.Condition()
```

```text
submit(node)
    │
    ├── running[id(node)] = asyncio task
    │
    └── _run(node)
          │
          ├── Pool.run(Flow.step, node)
          ├── done.put(transition)
          └── notify child joiners if terminal

next()
    │
    ├── done.get()
    ├── remove completed task from running
    └── return Transition
```

### `submit`

`submit(node, fn)` creates at most one active task for that exact Node:

```python
key = id(node)
active = self.running.get(key)
if active is not None:
    return active[1]

task = asyncio.create_task(self._run(node, fn))
self.running[key] = (node, task)
```

The identity check prevents the driver from stepping the same frontier twice.
The queue keys work by submitted Node identity, not by root or agent.

### `_run`

The private runner performs orchestration through the Pool and publishes the
result:

```python
transition = await self.pool.run(fn, node)
self.done.put_nowait(transition)
```

If the transition made an agent terminal, `_run` notifies every task waiting on
the queue's condition:

```python
agent = transition.created.parent_agent
if agent.terminal:
    async with self.changed:
        self.changed.notify_all()
```

The notification contains no answer. The answer remains `agent.result()` in the
graph.

### `next`

The driver has one consumer:

```python
transition = await queue.next()
```

For a completed step, `next()` removes `transition.submitted` from `running`.
For the immediate child `AgentStart` publication, it keeps the task tracked
because that child's first step has not completed.

### `join`

A parent REPL block can await several complete children:

```python
values = await asyncio.gather(*(queue.join(child) for child in children))
```

`join` waits for a predicate over durable graph state:

```python
async with self.changed:
    await self.changed.wait_for(lambda: child.terminal)
return child.result()
```

`asyncio.Condition.wait_for`:

1. checks `child.terminal`;
2. registers the task as a waiter if false;
3. releases the Condition lock while sleeping;
4. reacquires it after a notification;
5. checks the predicate again.

One child completing wakes all joiners, but each joiner checks its own child.
Unfinished children go back to sleep. This supports siblings and nested
delegation without a future dictionary or polling loop.

### `cancel`

`TaskQueue.cancel()` cancels every active transition. Passing Nodes selects a
subset:

```python
await queue.cancel(root.walk())
```

Selective cancellation is what lets one parallel root hit a boundary without
stopping the other roots sharing the queue.

## The driver loop

`Flow.run_streaming(root, *roots)` is both the single-root and multi-root driver.
`parallel_stream` is a public convenience wrapper around this function.

The driver performs six phases.

### 1. Normalize and validate roots

String inputs become new `AgentStart` roots carrying the flow's defaults. The
same root object cannot be driven twice, and one Flow permits one active driver
queue:

```python
agents = [
    self.start(item) if isinstance(item, str) else item
    for item in (root, *roots)
]

if self.queue is not None:
    raise RuntimeError("this Flow is already driving a stream")
```

Use `parallel_stream(flow, *roots)` instead of opening competing
`run_streaming` generators on one Flow.

### 2. Restore cold graphs

A graph loaded from disk has durable Nodes but no live REPL namespace.
Before scheduling, the Flow either:

- replays recorded `ExecAction`s when `restore="replay"`; or
- appends a cold-REPL notice when `restore="lazy"`.

Terminal agents are not replayed or submitted.

### 3. Seed current leaves

Every unfinished leaf across every root enters the same queue:

```python
for agent in agents:
    for leaf in agent.leaves():
        owner = leaf.parent_agent
        if owner is not None and not owner.terminal:
            queue.submit(leaf, self.step)
```

The roots are only used to discover initial leaves. Once submitted, `TaskQueue`
treats every item as an independent Node.

### 4. Consume one transition

One driver consumes the shared completion queue:

```python
transition = await queue.next()
node = transition.created
root = node.root
```

The created Node identifies its root. The queue does not route by graph.

```text
                       one TaskQueue
              ┌────────────────────────┐
root A leaves ─>                        │
root B leaves ─> running tasks         │
root C leaves ─>                        │
              │                        │
              │ completed transitions ───> one driver loop
              └────────────────────────┘          │
                                                  ├─> root A boundary/feed
                                                  ├─> root B boundary/feed
                                                  └─> root C boundary/feed
```

There is one queue and one consumer, so transitions cannot be stolen by competing
stream loops.

### 5. Yield, check the boundary, or feed the successor

The created Node is yielded before its boundary is applied:

```python
yield node

stop = boundary is not None and boundary(node, root)
if stop or root.terminal:
    active.discard(id(root))
    await queue.cancel(root.walk())
elif not transition.is_agent_start and not node.parent_agent.terminal:
    queue.submit(node, self.step)
```

The cases are:

- **boundary matched**: stop this root and cancel only its active work;
- **root terminal**: remove this root from the active set;
- **child start publication**: its first step is already running, so do nothing;
- **ordinary nonterminal transition**: submit the created Node as the next leaf.

This is the self-feeding loop:

```text
submit leaf
    │
    v
Flow.step
    │
    v
Transition.created
    │
    ├── boundary / terminal ─> stop this root
    │
    └── unfinished ─────────> submit created Node
```

### 6. Clean up

Every exit path cancels remaining queue tasks and clears `flow.queue`.
`close_repls=True` additionally closes every agent REPL under every driven root.

```python
finally:
    await queue.cancel()
    self.queue = None
```

`await flow.aclose()` also closes the Pool and every REPL owned by the Runtime.

## Delegation timeline

`launch_subagents` runs inside the parent agent's `ExecAction` step.

```text
parent ExecAction task
    │
    ├─ create child A AgentStart
    ├─ create child B AgentStart
    │
    ├─ queue.submit(A, publish=True)
    ├─ queue.submit(B, publish=True)
    │
    └─ await gather(queue.join(A), queue.join(B))
            │                         │
            │                         │
driver: A start, A transitions        │
driver: B start, B transitions        │
            │                         │
            └──── terminal ───────────┘
                         │
                         v
              parent REPL resumes
                         │
                         v
              parent ExecOutput / DoneOutput
```

The parent task remains suspended at an ordinary `await`. It does not block the
event-loop thread and does not occupy a bounded Pool compute slot. Child steps use
the same queue and Pool as every other step.

Child results preserve input order because `asyncio.gather` returns values in the
order of its awaitables, even when children complete in a different order.

## Pool versus Runtime

The scheduler separates lightweight orchestration from bounded compute:

```text
TaskQueue._run
    │
    └─ Pool.run(Flow.step, node)       orchestration; no scarce slot
          │
          ├─ Flow.llm_step
          │    └─ Pool.call(client.chat, ...)
          │                                  bounded model compute
          │
          └─ Flow.exec_step
               └─ Runtime.execute(...)
                                              REPL/runtime placement
```

`ThreadPool.call` sends synchronous functions to a thread executor. Async client
methods are awaited directly. `SequentialPool.call` serializes compute calls with
one lock.

`Pool.run` must not reserve a scarce compute slot while a parent awaits children.
Otherwise enough waiting parents could consume every slot and prevent their
children from running.

```python
flow = Flow(client, workers=16)
```

`workers=16` bounds synchronous model calls to sixteen threads. A custom Pool can
apply a different capacity or placement policy.

## Stream boundaries

`until` is checked after a Node is yielded and before its successor is submitted:

- `"done"` / `"finished"`: run each root until terminal;
- `"next"`: stop each root after its next created Node;
- `"idle"`: stop on `ExecOutput` or `DoneOutput`;
- `"error"`: stop on `ErrorOutput`;
- callable: receives `(node, root)` and returns a boolean.

```python
async for node in flow.run_streaming(root, until="next"):
    ...
```

Custom boundary:

```python
from rlmflow import DoneOutput, ExecOutput

def parent_idle(node, root):
    return (
        node.parent_agent is root
        and isinstance(node, (ExecOutput, DoneOutput))
    )

async for node in flow.run_streaming(root, until=parent_idle):
    ...
```

For several roots, the same boundary is evaluated with the root belonging to each
created Node. A match stops only that root.

Call `run_streaming` again with a stopped root to resume from its current leaves.

## Parallel roots

`run_streaming` can drive multiple roots directly:

```python
async for node in flow.run_streaming(root_a, root_b, root_c):
    print(node.root.content, node.type)
```

The public convenience APIs are:

```python
from rlmflow import parallel_run, parallel_stream

async for node in parallel_stream(flow, root_a, root_b):
    ...

roots = await parallel_run(flow, "query A", "query B")
print([root.result() for root in roots])
```

All roots share:

- one `TaskQueue`;
- one driver consumer;
- the Flow's Pool capacity;
- model clients, tools, and Runtime.

The roots must be distinct objects. Trees that reuse agent ids may also share REPL
identity and should not be driven together.

## Ordering

The stream follows queue completion order, not depth-first tree order.

```text
submitted: A1, B1
completed: B1, A1
yielded:   B1.created, A1.created
```

Within one agent, `Node.append` preserves a single transcript chain and `seq`
preserves that agent's local order. Across agents and roots, completion order is
intentionally nondeterministic.

`parallel_run` returns roots in input order. `launch_subagents` returns child
answers in specification order. Neither promise changes stream completion order.

## Failure behavior

Agent code failure and infrastructure failure are different.

### Agent code failure

An exception raised by code inside the REPL becomes an `ErrorOutput`. The agent
can read the traceback and recover on its next model turn.

```text
ExecAction -> ErrorOutput -> LLMOutput -> ...
```

### Infrastructure failure

An exception from model/runtime orchestration is caught by `Flow.step`. The step:

1. appends a terminal `DoneOutput` containing the failure;
2. returns `Transition(..., error=exception)`.

A child failure is therefore a durable child result that its parent can receive.
A root failure is yielded as a Node and then re-raised to the stream caller.

`asyncio.CancelledError` is not converted into a Node. Cancellation means the
caller stopped live work; the unchanged frontier can be resumed later.

## Cancellation and explicit close

Breaking from an async loop does not communicate a durable boundary by itself.
Close the generator when managing it manually:

```python
stream = flow.run_streaming(root)
try:
    async for node in stream:
        if should_stop(node):
            break
finally:
    await stream.aclose()
```

Prefer `until=` when the stop condition is known. The driver then yields the
matching Node, cancels that root's active tasks, and leaves the graph at a clear
resume point.

Stopping while a parent `ExecAction` awaits children cancels that parent block.
Resuming re-executes the block from its frontier, so boundaries on the parent's
own `ExecOutput` or `DoneOutput` are safer pause points for delegated work.

## Scheduler invariants

The implementation relies on these invariants:

1. An agent has one frontier and one transcript successor chain.
2. Only a frontier is submitted for a step.
3. A submitted Node has at most one active queue task.
4. Every non-cancelled step creates a durable Node.
5. A child `AgentStart` publication does not complete or duplicate its first step.
6. The graph owns terminal status and results; the Condition only wakes waiters.
7. One driver consumes one queue, including for parallel roots.
8. Boundaries are applied by the driver, never by `TaskQueue`.
9. Pool slots are used for compute, not while parents wait for children.

These constraints keep the queue small: it executes leaves and reports
transitions, while the graph and driver retain all domain behavior.
