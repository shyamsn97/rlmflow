# rlmflow internals

rlmflow has one durable type and three resource owners:

```text
Node       recursive durable state
Flow       execution and policy
Runtime    live REPL sessions
TaskQueue  calls in flight, and when one lands
```

There is no `Graph`, `Run`, `Driver`, Event hierarchy, detached worker queue, or merge subsystem, and no `Agent` class either — an agent is the `AgentStart` node that opened it. `Flow` is the public facade. Restore, builtin factories, child launch, child-completion handoff, and prompt assembly are `(flow, ...)` functions in `engine/restore.py`, `engine/delegation.py`, `engine/handoff.py`, `tools/namespace.py`, and `prompts/messages.py`.

The layout follows that. `rlmflow.graph` is the run itself — the node types and the format they are written to disk in. `rlmflow.flow` is the entry point, and `rlmflow.engine` is what it is built out of: `execution.py` (driver-scoped `TaskQueue`s plus `Pool`, `ThreadPool`, and `SequentialPool`), `boundaries.py` for the `until` vocabulary, `transitions.py` for deterministic resolution plus semantic choice sets, `semantic.py` for selecting once among string guards, plus `steps`, `restore`, `delegation`, and `handoff`. Several roots run through one `flow.run_streaming(root, *roots)` call. Everything public is re-exported from `rlmflow` itself, so which file a name lives in is an implementation detail rather than something a caller imports through.

## Node tree

A root Node is a run. Any Node is also a direct reference to one point in one agent transcript.

```python
node.parent
node.children
node.parent_agent
node.root
node.seq
```

The child with the same `parent_agent` is the next transcript step. Children that are `AgentStart`s are sub-agents.

```python
node.next
node.prev
node.walk()
agent.frontier
agent.transcript()
agent.leaves()
agent.sub_agents
agent.terminal
agent.result()
```

`Node.append(child)` is the only link primitive. It hangs one node off this one, gives it this node's inherited identity, and rejects an append anywhere but the agent's frontier — which is what keeps one agent's transcript a single chain. It is O(1) and returns `None`. The step that called it returns the created node. Appending an `AgentStart` branches instead of advancing: the child joins `agent.sub_agents` and the parent's frontier does not move.

`walk()` is an iterative generator over the subtree; `iter_backwards()` is the single chain back to this agent's `AgentStart`. `agent.terminal` is local: it asks only whether that agent's frontier is a `DoneOutput`.

Nodes own no clients, REPLs, tasks, callbacks, or sequence allocators.

## Transition path

```text
stream driver(roots)
  -> seed one TaskQueue with every root's leaves
  -> queue returns one completed StepResult
  -> stream yields result.created and checks until
  -> stream resubmits that node when its agent is not terminal
```

`Flow.step` calls the pure `transitions.resolve` over the Flow's own functions and then `DEFAULT_TRANSITIONS`, calls the matching producer, appends the result, and returns that node. Each producer in `rlmflow.engine.steps` is the callable object created by one `@transition(...)`; applications add node-only functions with `@flow.transition(...)`. String guards are judged by `engine/semantic.py`.

```text
AgentStart   -> complete (LLMOutput)
UserQuery    -> complete (LLMOutput)
ExecOutput   -> complete (LLMOutput)
ErrorOutput  -> complete (LLMOutput)
LLMOutput    -> to_action (ExecAction), or to_missing_action (empty ExecAction)
ExecAction   -> run_repl (output node)
DoneOutput   -> terminal, never stepped
```

After a child `DoneOutput`, `engine/handoff.py` enqueues the result on the launching parent. If that parent just landed `ExecOutput` (or `DoneOutput` with `resume_on_child_done`), the driver appends a `UserQuery` and submits that frontier instead of the observation it just yielded. Restore rebuilds the mailbox before seeding leaves.

Guards can preempt those rows: `out_of_room` → `FinalQuery`, `needs_truncation` → `TruncationSummary`.

`step(node)` applies budget policy and dispatches one producer. It returns the created `Node`. The queue's internal `StepResult(submitted, created, error)` converts infrastructure failure into a terminal graph node while letting cancellation unwind. The `timed` context manager stamps the created node with how long the step ran.

## Concurrency

Each `run_streaming(root, *roots)` invocation owns one `TaskQueue` and drives all of its roots through the same consumer loop. The created node identifies which root gets the boundary check and successor feed.

Delegation submits each new child to the same queue. A driver-scoped condition wakes parents when a child becomes terminal; its answer remains `child.result()` on the graph. The child `AgentStart` is published when submitted, so child openings stream without a tree diff.

`TaskQueue` executes lightweight transition orchestration directly. Model compute uses `Pool.stream`; the pool holds capacity for the stream's full lifetime. `ThreadPool(workers)` bounds it, while `SequentialPool` runs it one at a time.

## Model and exec turns

`complete` records the system prompt, consumes `llm.stream(...)`, and returns one `LLMOutput`. `AgentStart`, `UserQuery`, `ExecOutput`, and `ErrorOutput` call `complete` after automatic final-turn and truncation guards. Final-answer, repair, and truncation guidance remain specialized `UserQuery` nodes.

`run_repl` calls `WrappedRuntime.execute`, which seeds the agent's REPL with the current tool namespace and direct invocation variables, delegates to the original runtime, and turns the resulting `ReplRun` into exactly one node:

```text
ReplStatus.DONE   -> DoneOutput(result=run.answer)
ReplStatus.OK     -> ExecOutput
ReplStatus.ERROR  -> ErrorOutput(error="exec")
ReplStatus.DEAD   -> ReplDead
```

Synchronous streams run through the shared pool; async streams run directly on the event loop under the same pool policy. `Flow(llm_request_timeout=...)` bounds the stream's full lifetime and also passes `timeout=` to clients that accept one, since cancelling a blocking call frees the caller but not its thread.

## Delegation and replay

`launch_subagent` accepts an optional stable `name=` (otherwise `childN`), validates the child name and arguments, refuses a call past `max_depth` or over `max_query_chars`, opens one `AgentStart` under the running `ExecAction`, records `call_id -> agent.id` on that action, and submits it. Independent calls can run concurrently through `asyncio.gather`, which preserves call order in its result sequence. A handle-valued argument is persisted as an agent-reference marker; the consumer remains blocked until every source is terminal, then receives resolved values as direct REPL variables. Launch ownership never changes. The parent's frontier stays on the action for the whole block, so the step lands a single output when the block returns.

A child failure is a value, not an exception: the crash is recorded as that child's `DoneOutput` so siblings settle and the parent reads an explicit failure string.

Loading a saved tree opens no REPL. A graph handed to a `Flow` that has none for it — loaded, forked, or from another process — reconstructs required worker sessions by re-running recorded `ExecAction`s in execution order. `Flow(restore="replay")` performs that reconstruction before streaming starts; `Flow(restore="lazy")` waits until immediately before an agent's first new `ExecAction`. Sessions containing only terminal agents are skipped. If any shared tenant is unfinished, its worker owner and terminal shared tenants are replayed too, because their session and heap mutations remain dependencies of unfinished work. `RLMFLOW_REPLAY` is `"1"` during reconstruction. A replayed launch first reads its action's launch receipt and returns the existing agent without revalidating inputs or submitting work. Replay appends nothing and discards output.

## Runtime identity

Runtime keys live REPLs by `agent.id`, and hands each one the agent's process env (`RLMFLOW_AGENT_ID`, depth, parent id, max depth) when it opens. Paths and depths follow immutable launch ownership; dependency edges do not alter runtime identity.

`Runtime.execute` is where a dead REPL becomes an observation: it drops the dead entry so the next step opens a fresh one, and returns a `ReplRun` with status `DEAD` rather than raising at whoever asked for the step.

`flow.inject(name, value)` and `flow.add_tool(fn)` write into the flow's tool namespace *and* into every open REPL, so a tool added mid-run reaches agents that are already warm. `flow.remove_tool(name)` does the reverse. Reserved names and invocation argument names cannot be overwritten.

Nothing in the transition path closes a REPL, so a terminal agent's worker stays keyed and warm: its namespace is still readable, and appending a query and running again needs no replay. Closing is the caller's call — `close_repls=True`, `flow.runtime.close_repl(agent)`, or `flow.aclose()`.

## Branches

`node.fork()` is the one branch operation: it copies the whole run, cuts everything after the matching node, moves that agent's frontier back to it, drops the sub-agents that were cut away, and assigns every node a fresh identity. The result is an independent root you can run on any `Flow`.

Merge is intentionally unsupported. Callers select a branch to continue and discard the alternatives.

## Persistence

`rlmflow.graph.persistence` writes the strict v7 flat JSON shape: one shallow record per Node with `parent_id`, in parent-before-child order. Dependencies use stable agent IDs, handle-valued input references are durable, and each `ExecAction` persists its launch receipts. Loading validates the format version, node count, identities, ordering, parent links, receipt targets, dependency cycles, agent ownership, and registered node types, then rebuilds the tree through normal appends. Older formats are rejected rather than guessed.

The format records each agent's complete effective `AgentConfig`, including depth, iteration, budget, truncation, output, and query limits. A resumed run therefore keeps its recorded continuation contract. `Flow(root_config=...)` supplies defaults only to roots the flow creates itself; changing a loaded config is an explicit caller override.

Per-agent projection paths accept only ASCII alphanumeric, `_`, and `-` segments, validated when a child agent is named. A save prunes agent directories left behind by an earlier, larger save.

## Complexity

For local transcript length `L`, tree size `N`, and leaves `F`:

```text
Node.append                 O(1)
agent.frontier              O(1)
agent.transcript            O(L)
node.iter_backwards         O(L)
root.walk / root.leaves     O(N)
root.usage                  O(1)
node.subtree_usage          O(N)
leaf discovery per pass     O(N)
one pass                    O(F) steps
serialization               O(N)
node.fork                   O(N)
```

## Invariants

1. Nodes contain no live execution objects.
1. An append lands on the frontier, or it raises.
1. One agent's transcript is one chain; only sub-agents branch.
1. `seq` counts within one agent, so concurrency changes when nodes are created, never where they land.
1. One Node is never concurrently advanced twice.
1. A step advances its agent's frontier, or raises and leaves it where it was.
1. Agent names are unique among siblings and filesystem-safe.
1. A crashed step is its agent's outcome, held by the queue rather than the tree.
1. Every model call's system prompt is recoverable from the agent's table.
1. Save/load is a cold execution boundary; `restore=` decides how it warms.
