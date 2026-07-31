# rlmflow internals

rlmflow has one durable type and three resource owners:

```text
Node       recursive durable state
Flow       execution and policy
Runtime    live REPL sessions
TaskQueue  structured task batches and stream routing
```

There is no `Graph`, `Agent`, `Run`, `Driver`, Event hierarchy, detached worker
queue, or merge subsystem. All execution, branching, prompt, model, and tool
methods live directly on `Flow`.

`Pool`, `ThreadPool`, `SequentialPool`, and `TaskQueue` live together in
`rlmflow.execution`; `parallel` and `structured` remain focused top-level APIs.

## Node tree

A root Node is a trajectory. Any Node is also a direct reference to one point
in one agent transcript.

```python
node.parent
node.children
node.agent_id
node.trajectory_id
node.metadata
```

The child with the same `agent_id` is the next transcript step. Children with a
different `agent_id` are spawned agents.

```python
node.step_child
node.spawn_children
node.tail()
node.transcript()
node.latest_query()
node.leaves()
node.finished()
```

`Node.attach(child)` is the only ordinary link primitive. It accepts one
detached, childless Node, sets its parent and inherited identity, and rejects a
second same-agent continuation. It is O(1).

`walk()` is iterative, so deep histories do not consume Python recursion.
`finished()` without an `agent_id` is recursive: every descendant leaf must be
terminal.

Nodes own no clients, REPLs, tasks, callbacks, or sequence allocators.

## Transition path

```text
run_streaming(root)
  -> claim root.trajectory_id
  -> find runnable leaves
  -> TaskQueue.run(step(leaf), ...)
  -> transition attaches and publishes Node
  -> stream yields Node
```

`Flow.steps` is the complete state machine:

```text
UserQuery          -> llm_turn
ExecOutput         -> llm_turn
ErrorOutput        -> llm_turn
ResumeAction       -> llm_turn
SupervisingOutput  -> resume_supervisor
LLMOutput          -> extract_code
ExecAction         -> exec_turn
DoneOutput         -> terminal
```

`step(node)` applies termination and budget policy, dispatches one transition,
and returns the resulting Node.

## Structured concurrency

Each `run_streaming` call owns a local Node queue. `TaskQueue.claim()` registers
that queue by trajectory while the stream is active.

One host round submits one coroutine per runnable leaf to `TaskQueue.run()`.
Delegation submits child drivers the same way. `TaskQueue` implements each batch
with `asyncio.TaskGroup`, so every caller settles its submitted children before
returning.

`TaskQueue` also tracks top-level rounds for `aclose()`, active trajectory
publishers, termination requests, and named halts. A trajectory claim occurs
before an existing root can receive query/input/schema mutations.

## Publication

`Flow.append(parent, child)` is the ordinary mutation point:

1. normalize a string to `UserQuery`;
2. stamp `child.metadata["mutation"]`;
3. call `parent.attach(child)`;
4. call `tasks.publish(child)`;
5. return the child.

There is no Event conversion. `TaskQueue` routes the actual affected Node to the
active queue for its trajectory.

Direct surgery functions return affected Nodes. Live tools publish the returned
Node through `TaskQueue`.

## Model and exec turns

`llm_turn` reads configuration from `node.latest_query()`, lets the prompt
builder append any durable nudge, calls the selected client, and appends
`LLMOutput`.

Synchronous clients run through the shared Pool. Async clients run directly on
the event loop.

An `ExecAction` may append multiple Nodes while its Python frame is suspended:

```text
ExecAction
  -> SupervisingOutput
      -> child agents
  -> ResumeAction
  -> ExecOutput | ErrorOutput | DoneOutput
```

`Flow.continuation(node)` owns the current same-agent cursor for that frame.
Every launch and the final output use the same closure, so no root scan is
needed between appends.

## Delegation and recovery

`launch_subagents` validates safe, unique child identity segments, appends one
supervisor, attaches children, drives them through `TaskQueue.run()`, appends
one resume action to that exact supervisor, and returns results in spec order.

Re-executing a recorded action reuses its recorded child groups. It never
duplicates those subtrees.

A loaded checkpoint restores no suspended Python frame. If it ends on
`SupervisingOutput`, unfinished children run first. Once all children are
terminal, `resume_supervisor` appends `ResumeAction`; the ordinary state machine
then continues with a fresh model turn.

## Runtime identity

Runtime keys live REPLs by:

```python
(node.trajectory_id, node.agent_id)
```

Runtime also remembers the live Node associated with each key so newly
registered factory/scoped tools can be backfilled into open REPLs.

Adopting a prepared branch moves its warm REPL and rebinds:

- `done` and `launch_subagents`;
- dynamic tools;
- `INPUTS`;
- `RLMFLOW_AGENT_ID`;
- depth, parent id, max depth, root status, and replay status.

Loading a saved tree opens no REPL. A graph handed to a `Flow` that has none for it
— loaded, forked, or from another process — is replayed once before the run loop
starts: each unfinished agent's namespace is rebuilt by re-running its recorded
`ExecAction`s in transcript order. Agents that already answered are skipped, since
nothing will run in their namespaces again. `RLMFLOW_REPLAY` is `"1"` during the
rebuild, so
`launch_subagents` hands back the answers its children already gave instead of
launching them again, and agent code can skip work it should not repeat. Pass
`Flow(restore="lazy")` to skip the rebuild and instead tell each unfinished agent, in
its own transcript, that its variables are gone.

## Branches and surgery

Supported branch operations are:

```python
await flow.fork(root)
await flow.rewind(root)
await flow.launch_branches(parent, branches)
flow.discard(branch)

surgery.checkpoint(root)
surgery.revert(root, checkpoint)
surgery.insert(root, ...)
surgery.remove(root, ...)
```

`surgery.fork()` and `surgery.rewind()` own branch selection, copying, and
mutation metadata. The corresponding async Flow methods are thin adapters that
ask Runtime to replay the selected branch or mark it lazy.

Merge is intentionally unsupported. Callers select a branch to continue,
discard alternatives, or attach prepared branches with `Flow.launch_branches`.

Surgery validates before mutating. Internal `surgery.adopt` rebuilds a
single-agent chain through leaf attachment. Checkpoints fingerprint durable
prefix content, not only Node ids, so rewritten history is rejected.

## Persistence

`rlmflow.graph.persistence` writes the v2 recursive JSON shape and derives
`order` values structurally. Loading validates node types and unique ids,
rebuilds parent links, and applies one trajectory id.

Per-agent projection paths accept only ASCII alphanumeric, `_`, and `-`
segments. Resolved paths must stay beneath the run's `agents/` directory.

`latest_summary(root)["finished"]` uses recursive completion.

## Complexity

For local transcript length `L`, tree size `N`, and runnable leaves `F`:

```text
Node.attach                 O(1)
leaf.tail                   O(L remaining)
leaf.transcript             O(L)
leaf.latest_query           O(L)
root.walk / root.leaves     O(N)
host round discovery        O(N)
host round execution        O(F) tasks
serialization               O(N)
fork/adopt                  O(N)
```

## Invariants

1. Nodes contain no live execution objects.
2. Every task has one lexical owner.
3. Every function settles the tasks it creates.
4. One trajectory has at most one active stream per Flow.
5. One Node is never concurrently advanced twice.
6. Connected trees are acyclic and have one trajectory id.
7. Agent ids are unique and child identity segments are filesystem-safe.
8. Ordinary attachment receives one detached leaf.
9. Root completion requires every descendant leaf to be terminal.
10. Every live durable mutation publishes its affected Node once.
11. Mutation provenance lives under `node.metadata["mutation"]`.
12. Save/load is a cold execution boundary.
