# rlmflow internals

rlmflow has one durable type and three resource owners:

```text
Node       recursive durable state
Flow       execution and policy
Runtime    live REPL sessions
TaskQueue  calls in flight, and when one lands
```

There is no `Graph`, `Run`, `Driver`, Event hierarchy, detached worker queue, or merge subsystem, and no `Agent` class either — an agent is the `AgentStart` node that opened it. All execution, prompt, model, and tool methods live directly on `Flow`.

The layout follows that. `rlmflow.graph` is the run itself — the node types and the format they are written to disk in. `rlmflow.flow` is the entry point, and `rlmflow.engine` is what it is built out of: `execution.py` (driver-scoped `TaskQueue`s plus `Pool`, `ThreadPool`, and `SequentialPool`), `boundaries.py` for the `until` vocabulary, and `parallel.py` for driving several roots through one flow. Everything public is re-exported from `rlmflow` itself, so which file a name lives in is an implementation detail rather than something a caller imports through.

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

`Node.append(child)` is the only link primitive. It hangs one node off this one, gives it this node's inherited identity, and rejects an append anywhere but the agent's frontier — which is what keeps one agent's transcript a single chain. It is O(1). Appending an `AgentStart` branches instead of advancing: the child joins `agent.sub_agents` and the parent's frontier does not move.

`walk()` is an iterative generator over the subtree; `iter_backwards()` is the single chain back to this agent's `AgentStart`. `agent.terminal` is local: it asks only whether that agent's frontier is a `DoneOutput`.

Nodes own no clients, REPLs, tasks, callbacks, or sequence allocators.

## Transition path

```text
stream driver(roots)
  -> seed one TaskQueue with every root's leaves
  -> queue returns one completed Transition
  -> stream yields transition.created and checks until
  -> stream resubmits that node when its agent is not terminal
```

`Flow.step` looks up a `StepFunction` class with `get_step_fn` (MRO over `self.steps`, seeded from `DEFAULT_STEPS`) and constructs it with the same `LLMClient`, `MessageBuilder`, and `WrappedRuntime` every time. The wrapper exposes the original runtime as `.runtime`, so a custom step can use lower-level REPL operations without making the default step ABI depend on `Flow`. Override a type with `update_step_fn`. `InspectQuery` / `PlanQuery` / `FinalQuery` / `ContinueQuery` / `TruncationSummary` inherit the `UserQuery` registration.

```text
AgentStart   -> LLMRequestStep (InspectQuery when the agent has inputs)
UserQuery    -> LLMRequestStep (follow-up InspectQuery after DoneOutput)
ExecOutput   -> LLMRequestStep (PlanQuery via InspectQuery.after)
ErrorOutput  -> LLMRequestStep
LLMOutput    -> LLMOutputStep (ExecAction)
ExecAction   -> ExecActionStep
DoneOutput   -> terminal, never stepped
```

`step(node)` applies budget policy and dispatches one transition. It returns `Transition(submitted, created, error)`, converting infrastructure failure into a terminal graph node while letting cancellation unwind. The `timed` context manager stamps the created node with how long the step ran.

## Concurrency

Each driver invocation owns one `TaskQueue`. `run_streaming` drives one root; `parallel_stream` drives several roots through the same queue and the same consumer loop. The created node identifies which root gets the boundary check and successor feed.

Delegation submits each new child to the same queue. A driver-scoped condition wakes parents when a child becomes terminal; its answer remains `child.result()` on the graph. The child `AgentStart` is published when submitted, so child openings stream without a tree diff.

`TaskQueue` executes lightweight transition orchestration directly. Model compute uses `Pool.stream`; the pool holds capacity for the stream's full lifetime. `ThreadPool(workers)` bounds it, while `SequentialPool` runs it one at a time.

## Model and exec turns

`LLMRequestStep` records the system prompt, consumes `llm.stream(...)`, and appends one `LLMOutput`. Inspect, plan, final-answer, continue, and truncation land as their own nodes before the model call.

`ExecActionStep` calls `WrappedRuntime.execute`, which seeds the agent's REPL with the current tool namespace and `INPUTS`, delegates to the original runtime, and turns the resulting `ReplRun` into exactly one node:

```text
ReplStatus.DONE   -> DoneOutput(result=run.answer)
ReplStatus.OK     -> ExecOutput
ReplStatus.ERROR  -> ErrorOutput(error="exec")
ReplStatus.DEAD   -> ReplDead
```

Synchronous streams run through the shared pool; async streams run directly on the event loop under the same pool policy. `Flow(llm_request_timeout=...)` bounds the stream's full lifetime and also passes `timeout=` to clients that accept one, since cancelling a blocking call frees the caller but not its thread.

## Delegation and replay

`launch_subagent` validates the child name, refuses a call past `max_depth` or over `max_query_chars` by returning a refusal string, opens one `AgentStart` under the running `ExecAction`, submits it, and awaits its answer. Independent calls can run concurrently through `asyncio.gather`, which preserves call order in its result sequence. The parent's frontier stays on the action for the whole block, so the step lands a single output when the block returns.

A child failure is a value, not an exception: the crash is recorded as that child's `DoneOutput` so siblings settle and the parent reads an explicit failure string.

Loading a saved tree opens no REPL. A graph handed to a `Flow` that has none for it — loaded, forked, or from another process — reconstructs each unfinished agent's namespace by re-running recorded `ExecAction`s in execution order. `Flow(restore="replay")` performs that reconstruction before streaming starts; `Flow(restore="lazy")` waits until immediately before an agent's first new `ExecAction`. Agents that already answered are skipped. `RLMFLOW_REPLAY` is `"1"` during reconstruction, so `launch_subagent` hands back an answer already recorded in the graph and agent code can avoid repeating external effects. Replay appends nothing and discards output.

## Runtime identity

Runtime keys live REPLs by `agent.id`, and hands each one the agent's process env (`RLMFLOW_AGENT_ID`, depth, parent id, max depth) when it opens.

`Runtime.execute` is where a dead REPL becomes an observation: it drops the dead entry so the next step opens a fresh one, and returns a `ReplRun` with status `DEAD` rather than raising at whoever asked for the step.

`flow.inject(name, value)` and `flow.add_tool(fn)` write into the flow's tool namespace *and* into every open REPL, so a tool added mid-run reaches agents that are already warm. `flow.remove_tool(name)` does the reverse. Reserved names such as `finish`, `launch_subagent`, and `INPUTS` cannot be overwritten.

Nothing in the transition path closes a REPL, so a terminal agent's worker stays keyed and warm: its namespace is still readable, and appending a query and running again needs no replay. Closing is the caller's call — `close_repls=True`, `flow.runtime.close_repl(agent)`, or `flow.aclose()`.

## Branches

`node.fork()` is the one branch operation: it copies the whole run, cuts everything after the matching node, moves that agent's frontier back to it, drops the sub-agents that were cut away, and assigns every node a fresh identity. The result is an independent root you can run on any `Flow`.

Merge is intentionally unsupported. Callers select a branch to continue and discard the alternatives.

## Persistence

`rlmflow.graph.persistence` writes the strict v3 flat JSON shape: one shallow record per Node with `parent_id`, in parent-before-child order. Loading validates the format version, node count, identities, ordering, parent links, agent ownership, and registered node types, then rebuilds the tree through normal appends. Older formats are rejected rather than guessed.

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
