# Streaming and structured execution

`Flow.run_streaming(...)` mutates a caller-owned Node tree and yields each
durable Node as it is attached or changed.

```python
from rlmflow import Flow, start

flow = Flow(client)
root = start("Audit this repository.")

async for node in flow.run_streaming(root):
    mutation = node.metadata.get("mutation", {})
    print(node.agent_id, mutation.get("type"), node.type)

print(root.agent_result())
```

The tree is the durable state. There is no separate run object, event object, or
worker queue. `TaskQueue` runs structured coroutine batches and retains the
small amount of live stream/control state.

## Execution ownership

One streaming call owns one local `asyncio.Queue[Node]`, registered by
trajectory with `TaskQueue`. Work is structured by the Python call tree:

```text
run_streaming
  -> host round
      -> one step per runnable leaf
          -> exec_turn
              -> child run_leaf tasks
```

Flow submits each local batch to `TaskQueue.run()`, which uses
`asyncio.TaskGroup`. Every function therefore settles the tasks it submits
before returning; top-level rounds are also tracked so `aclose()` can cancel
them.

The host starts a round from every runnable leaf in the entire tree. A
supervisor waiting for unfinished children is not runnable. Delegated children
self-drive until terminal and publish into their parent's stream.

## Stream boundaries

`until` decides whether another host round starts. It never abandons work
already in flight.

- `"done"` / `"finished"`: run until no runnable leaves remain.
- `"next"`: stop after the next appended Node.
- `"idle"`: stop at `ExecOutput` or `DoneOutput`; an `ErrorOutput` is allowed to
  recover first.
- `"error"`: stop when an `ErrorOutput` is published.
- callable: receives `(node, root)` and stops when it returns true.

```python
async for node in flow.run_streaming(root, until="next"):
    ...

def child_done(node, root):
    return node.agent_id != root.agent_id and node.type == "done_output"

async for node in flow.run_streaming(root, until=child_done):
    ...
```

There is no built-in `"supervising"` boundary. Use a callable:

```python
from rlmflow import SupervisingOutput

until_supervising = lambda node, root: isinstance(node, SupervisingOutput)
```

Call `run_streaming` again with the same root to continue after a boundary.
Reactive edits are deterministic between streaming calls because the prior
round has settled.

## Published Nodes

Ordinary transitions stamp:

```python
node.metadata["mutation"] = {
    "type": "append",
    "parent_id": parent.id,
}
```

Other operations use mutation names such as `create`, `update`, `spawn`,
`insert`, `replace`, `remove`, `revert`, `fork`, and `adopt`.

A fresh string run yields its initial `UserQuery`:

```python
async for node in flow.run_streaming("new query"):
    ...
```

Input/schema-only updates publish the updated `UserQuery`. Consumers receive
Nodes directly and derive the live root with `node.root()`:

```python
async for node in flow.run_streaming(root):
    checkpointer.handle(node)
    renderer.handle(node)
```

## Delegation

`launch_subagents(...)` is an awaited REPL tool:

1. append `SupervisingOutput`;
2. attach each accepted child `UserQuery`;
3. drive children concurrently in an `asyncio.TaskGroup`;
4. append `ResumeAction` to the exact supervisor;
5. return results to the suspended parent code;
6. append the parent's final output.

The stream therefore sees supervisor, child, resume, and final parent Nodes in
durable order. A child failure becomes that child's terminal result, so siblings
settle and the parent receives an explicit failure value.

## Parallel roots

`parallel_stream(flow, *roots)` merges independent Node streams. Roots share
the Flow's clients, pool, tools, and Runtime.

Only one active stream per `trajectory_id` is allowed. This rejects both the
same root object and a shared-session clone before query/input updates mutate
the trajectory. Isolated forks have fresh trajectory ids and may run
concurrently.

## Cancellation and cleanup

Closing a stream cancels and settles its current host round. Passing
`close_repls=True` closes that trajectory's REPLs on every exit path: completion,
boundary, error, cancellation, or explicit generator close.

`await flow.aclose()` cancels all tracked top-level rounds, closes every REPL,
and closes the blocking-call pool.

## Parallelism limits

Async transitions and child drivers are ordinary asyncio tasks. Only blocking
calls such as a synchronous `client.chat` consume the Flow's `Pool`:

- `ThreadPool(workers=N)` bounds blocking calls;
- async clients bypass the thread pool;
- `SequentialPool` makes blocking calls deterministic for debugging.

No agent holds a pool slot while awaiting delegated children, so nested
delegation cannot deadlock by exhausting worker threads.
