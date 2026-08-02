# rlmflow internals

rlmflow has one durable type and three resource owners:

```text
Node       recursive durable state
Flow       execution and policy
Runtime    live REPL sessions
TaskQueue  calls in flight, and when one lands
```

There is no `Graph`, `Run`, `Driver`, Event hierarchy, detached worker queue, or
merge subsystem, and no `Agent` class either — an agent is the `AgentStart` node
that opened it. All execution, prompt, model, and tool methods live directly on
`Flow`.

The layout follows that. `rlmflow.graph` is the run itself — the node types and
the format they are written to disk in. `rlmflow.flow` is the entry point, and
`rlmflow.engine` is what it is built out of: `execution.py` (driver-scoped
`TaskQueue`s plus `Pool`, `ThreadPool`, and `SequentialPool`), `boundaries.py` for
the `until` vocabulary, and `parallel.py` for driving several roots through one flow.
Everything public is re-exported from `rlmflow` itself, so which file a name
lives in is an implementation detail rather than something a caller imports
through.

## Node tree

A root Node is a run. Any Node is also a direct reference to one point in one
agent transcript.

```python
node.parent
node.children
node.parent_agent
node.root
node.seq
```

The child with the same `parent_agent` is the next transcript step. Children
that are `AgentStart`s are sub-agents.

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

`Node.append(child)` is the only link primitive. It hangs one node off this one,
gives it this node's inherited identity, and rejects an append anywhere but the
agent's frontier — which is what keeps one agent's transcript a single chain.
It is O(1). Appending an `AgentStart` branches instead of advancing: the child
joins `agent.sub_agents` and the parent's frontier does not move.

`walk()` is a generator over the subtree; `walk(reverse=True)` is the single
chain back to this agent's `AgentStart`. `agent.terminal` is local: it asks only
whether that agent's frontier is a `DoneOutput`.

Nodes own no clients, REPLs, tasks, callbacks, or sequence allocators.

## Transition path

```text
stream driver(roots)
  -> seed one TaskQueue with every root's leaves
  -> queue returns one completed Transition
  -> stream yields transition.created and checks until
  -> stream resubmits that node when its agent is not terminal
```

`Flow.step` is the complete state machine:

```text
AgentStart   -> llm_step
UserQuery    -> llm_step
ExecOutput   -> llm_step
ErrorOutput  -> llm_step
LLMOutput    -> append ExecAction
ExecAction   -> exec_step
DoneOutput   -> terminal, never stepped
```

`step(node)` applies budget policy and dispatches one transition. It returns
`Transition(submitted, created, error)`, converting infrastructure failure into a
terminal graph node while letting cancellation unwind. The `timed` context manager
stamps the created node with how long the step ran.

## Concurrency

Each driver invocation owns one `TaskQueue`. `run_streaming` drives one root;
`parallel_stream` drives several roots through the same queue and the same
consumer loop. The created node identifies which root gets the boundary check and
successor feed.

Delegation submits each new child to the same queue. A driver-scoped condition wakes
parents when a child becomes terminal; its answer remains
`child.result()` on the graph. The child `AgentStart` is published when submitted,
so child openings stream without a tree diff.

`Pool.run` executes lightweight transition orchestration without holding a bounded
compute slot while a parent awaits children. Blocking model work uses `Pool.call`;
`ThreadPool(workers)` bounds it, while `SequentialPool` runs it one at a time.

## Model and exec turns

`llm_step` commits the turn's user content first, so the model always answers a
user: the prompt builder's content, then a nudge if the last turn was not a user
turn, or the final-answer prod on the last allowed iteration. It then renders
messages, records the system prompt in the agent's content-addressed
`system_prompts` table, calls the selected client, and appends one `LLMOutput`
carrying the reply, the extracted code, usage, and the prompt id.

`exec_step` seeds the agent's REPL with the current tool namespace and `INPUTS`,
runs the action's code, and turns the resulting `ReplRun` into exactly one node:

```text
ReplStatus.DONE   -> DoneOutput(result=run.answer)
ReplStatus.OK     -> ExecOutput
ReplStatus.ERROR  -> ErrorOutput(error="exec")
ReplStatus.DEAD   -> ErrorOutput(error="repl"), plus the cold-REPL note
```

Synchronous clients run through the shared pool; async clients run directly on
the event loop. `Flow(llm_request_timeout=...)` wraps the call in `wait_for` and
also passes `timeout=` to clients that accept one, since cancelling a blocking
call frees the caller but not its thread.

## Delegation and replay

`launch_subagents` validates child names, refuses a spec past `max_depth` or
over `max_query_chars` by returning a refusal string in its place, opens one
`AgentStart` per accepted spec under the running `ExecAction`, submits them, and
awaits their answers in spec order. The parent's frontier stays on the action
for the whole block, so the step lands a single output when the block returns.

A child failure is a value, not an exception: the crash is recorded as that
child's `DoneOutput` so siblings settle and the parent reads an explicit
failure string.

Loading a saved tree opens no REPL. A graph handed to a `Flow` that has none for
it — loaded, forked, or from another process — is replayed once before the run
loop starts: each unfinished agent's namespace is rebuilt by re-running its
recorded `ExecAction`s in tree order. Agents that already answered are skipped,
since nothing will run in their namespaces again. `RLMFLOW_REPLAY` is `"1"`
during the rebuild, so `launch_subagents` hands back the answers its children
already gave instead of launching them again, and agent code can skip work it
should not repeat. The replay appends nothing and discards output, because a
block that failed the first time fails the same way here. Pass
`Flow(restore="lazy")` to skip the rebuild and instead tell each unfinished
agent, in its own transcript, that its variables are gone.

## Runtime identity

Runtime keys live REPLs by `agent.id`, and hands each one the agent's process
env (`RLMFLOW_AGENT_ID`, depth, parent id, max depth) when it opens.

`Runtime.execute` is where a dead REPL becomes an observation: it drops the dead
entry so the next step opens a fresh one, and returns a `ReplRun` with status
`DEAD` rather than raising at whoever asked for the step.

`flow.inject(name, value)` and `flow.add_tool(fn)` write into the flow's tool
namespace *and* into every open REPL, so a tool added mid-run reaches agents
that are already warm. `flow.remove_tool(name)` does the reverse. The three
reserved names — `done`, `launch_subagents`, `INPUTS` — are rebuilt per step and
cannot be overwritten.

## Branches

`node.fork()` is the one branch operation: it deep-copies the whole run, cuts
everything after the matching node, moves that agent's frontier back to it,
drops the sub-agents that were cut away, and (by default) re-ids every node. The
result is an independent root you can run on any `Flow`.

Merge is intentionally unsupported. Callers select a branch to continue and
discard the alternatives.

## Persistence

`rlmflow.graph.persistence` writes the v2 recursive JSON shape and derives `order`
from each node's `seq`. Loading rebuilds the tree by replaying its appends, so
the reconstructed run satisfies the same invariants a live one does; a run saved
by an older engine that wrote a node of its own for a delegating turn reads that
node back as an ordinary `ExecOutput`.

The format records what the run *was*, not what it was allowed to be: a loaded
`AgentConfig` carries identity, inputs, model, prompt profile, and schema, while
`max_iters`, `max_depth`, and the budget come back as `AgentConfig`'s own
defaults. A resumed run that needs different limits gets them by setting them on
the loaded config, since `Flow(config=...)` only supplies defaults to roots the
flow creates itself.

Per-agent projection paths accept only ASCII alphanumeric, `_`, and `-`
segments, validated when a child agent is named. A save prunes agent
directories left behind by an earlier, larger save.

## Complexity

For local transcript length `L`, tree size `N`, and leaves `F`:

```text
Node.append                 O(1)
agent.frontier              O(1)
agent.transcript            O(L)
node.walk(reverse=True)     O(L)
root.walk / root.leaves     O(N)
root.tokens                 O(N)
leaf discovery per pass     O(N)
one pass                    O(F) steps
serialization               O(N)
node.fork                   O(N)
```

## Invariants

1. Nodes contain no live execution objects.
2. An append lands on the frontier, or it raises.
3. One agent's transcript is one chain; only sub-agents branch.
4. `seq` counts within one agent, so concurrency changes when nodes are created,
   never where they land.
5. One Node is never concurrently advanced twice.
6. A step advances its agent's frontier, or raises and leaves it where it was.
7. Agent names are unique among siblings and filesystem-safe.
8. A crashed step is its agent's outcome, held by the queue rather than the tree.
9. Every model call's system prompt is recoverable from the agent's table.
10. Save/load is a cold execution boundary; `restore=` decides how it warms.
