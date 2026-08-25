# Persistent Agents and Agent-to-Agent Messaging

> **Status:** Research note. Nothing here is implemented.
>
> **Scope:** whether agents in one run should be able to address each other
> after they have answered, and what the engine would have to change to allow
> it. Companion to [`inter_agent_communication.md`](inter_agent_communication.md),
> which adds a read-only `AGENTS` view, and
> [`delm_vs_rlmflow.md`](delm_vs_rlmflow.md), which adds shared context across
> peer workers. This note is about the third thing neither covers: talking to an
> agent that already finished.
>
> The frontier-safety constraint below is a live corruption bug rather than a
> missing feature, and is scheduled as stage 3 of
> [`repl_plan.md`](repl_plan.md).

## Summary

Agents in rlmflow communicate, but only in one shape: a parent hands a child a
query and `inputs`, the child answers once with `done(...)`, and the parent
reads that answer. The channel is the call stack, and it closes when the child
answers.

The interesting question is not "should agents chat." It is that **a finished
agent still exists**. Its transcript is in the Node tree, and its Python REPL is
still open on the `Runtime`, keyed by `agent.id`, holding whatever it built. The
model has no way to reach either. It threw away a warm namespace and can only
get it back by spawning a new agent that redoes the work.

So the recommendation splits along that line:

- **Build addressable resumption.** Let a supervisor re-open a finished child
  with a follow-up question. It is a small change, it fits the existing model,
  and it is where nearly all of the value is.
- **Do not build a peer message bus.** Free-form agent-to-agent chat costs the
  properties that make this library worth using, and the things people actually
  want from it are better served by `AGENTS` (read) and shared context (write).

Everything below is why, and what the engine gets in the way of.

## What Agents Can Already Say To Each Other

Four channels exist today.

```text
parent -> child    query + inputs, at launch, once
child  -> parent   done(result), at exit, once
any    -> any      AGENTS snapshot (read-only, opt-in, no writes)
host   -> any      append a Node to an agent's frontier
```

The last one is the general one. `docs/injections.md` is built on it: a
controller appends a `UserQuery` to an agent's frontier mid-run and the agent
reads it on its next turn. That is already message delivery — the host is just
the only one allowed to send, and only at moments it has to get right by hand.

What is missing is everything after the child answers. `launch_subagents` is
spawn-and-join, `done()` is exit, and there is no address to send a second
message to.

## Five Constraints The Engine Imposes

These are the things a design has to survive. Each was checked against the
current engine rather than read off the source, because several of them fail in
ways that are quiet.

### 1. A delegating parent is suspended inside its own REPL action

`launch_subagents` ends in
`await asyncio.gather(*(queue.join(child) for child in unfinished))`, and
`TaskQueue.join` waits on `child.terminal`. The parent is not "waiting for a
message" in any recoverable sense — it is blocked partway through executing a
Python block, with an in-flight `ExecAction` task in `TaskQueue.running`. It
cannot take an LLM turn until that block returns.

The consequence is structural, not a bug to patch with a timeout:

```text
A launches B and blocks in launch_subagents
B sends A a question and waits for the answer
A cannot answer: it can only run again once B is terminal
B cannot terminate: it is waiting for A
```

**Any request/response from a descendant up to a blocked ancestor deadlocks.**
It follows directly from delegation being a top-level `await` in the parent's
own code, which is the thing that makes rlmflow's delegation legible. Message
passing has to work around it, not remove it.

### 2. An append only lands on a frontier with no step in flight

`Node.append` raises unless `self is agent.frontier`. A step holds the node it
was submitted with across a long `await` and appends to it afterwards —
`LLMRequestStep` does `turn.append(LLMOutput(...))` after
`await join_chunks(self.llm.stream(messages))`. If
anything else appends to that agent in the meantime, the step's own append
raises.

Appending to the root's frontier during an in-flight LLM call:

```text
ValueError: agent_dc57c68... is not the frontier of agent_dc57c68...
```

raised out of `run_streaming` itself, killing the run. For a sub-agent it is
quieter and worse: `Flow.step` catches it and lands
`DoneOutput(result="[child failed: ValueError: ...]")`, so the child dies of a
delivery race and reports it as its own answer.

The safe window is the moment a node lands and before the next step is
submitted for that agent — the `yield node` in the driver loop, for
`node.parent_agent` only. That is what "as its node lands" means in
`docs/injections.md`. **A messaging feature must enforce that window rather
than document it**, since a sender inside agent code has no idea what the
target is doing.

### 3. A finished agent has no address

`Flow.resolve_child` reuses a child already attached to the *same*
`ExecAction`, which is what makes replay idempotent. Any other reference to an
existing name hits the duplicate check against `parent.sub_agents`. A root that
delegates to `worker`, gets its answer, and tries again on a later turn:

```text
second launch raised: ValueError duplicate child name 'worker'
```

There is no spelling of "the worker I already have." The name is taken by the
agent the model wants to reach.

### 4. The driving scope is the root, and a finished root ends the stream

`run_streaming` tracks `active` roots and stops a root as soon as
`root.terminal`. Two things follow that block the obvious host-side workaround.

Appending a follow-up to a finished child of a finished root and re-driving:

```text
nodes landed   : [('root.worker', 'llm_output')]
worker terminal: False
worker result  : None
```

Exactly one node lands. The driver checks `root.terminal` after the first
transition, cancels everything, and leaves the worker stranded mid-turn on an
`LLMOutput` frontier — a state no organic run produces.

Driving the sub-agent on its own instead, `flow.run_streaming(worker)`:

```text
nodes landed   : []
```

Silence. `active` holds `id(worker)`, but nodes appended inside the worker carry
`node.root` pointing at the top root, so every transition fails the
`id(root) not in active` check and is skipped. Zero nodes, no error.

The only thing that works today is un-terminating both:

```python
worker.config.max_iters += 4
root.config.max_iters += 4
worker.frontier.append(UserQuery(content="Which file was it in?"))
root.frontier.append(UserQuery(content="Wait for the worker."))

async for node in flow.run_streaming(root):
    ...
# worker result: "child answer"
```

which works, and costs a root turn nobody wanted. **A sub-agent cannot be a
unit of execution.** Anything that resumes one has to either run inside a live
root or teach the driver about scopes narrower than a root.

### 5. Iteration budgets are cumulative

`llm_turns()` counts every `LLMOutput` in the agent's transcript, and
`LLMRequestStep.chat` lands `DoneOutput(result="[max_iters exceeded]")` when that reaches
`config.max_iters`. A resumed agent inherits its own history, so an agent that
spent its budget answers a follow-up with `[max_iters exceeded]` before calling
the model. Resumption has to raise the ceiling explicitly; the probe above adds
`max_iters += 4` for exactly this reason.

## Is This Useful?

Partly. The two halves of the question have different answers, and it is worth
keeping them apart.

### Persistent agents: yes, and the reason is specific to rlmflow

In most frameworks an agent is a message list, so "keep it alive" means "keep an
array," and re-spawning costs nothing but tokens. Here an agent owns **a
stateful Python REPL** that lives on the `Runtime`, keyed by `agent.id`, and
survives the stream that created it. By the time a child answers it may hold a
parsed corpus, a fitted model, a warmed Docker container, an open connection, a
half-built directory of files.

`done()` throws away the handle to all of it. The namespace stays in
`runtime.repls` until someone closes it, but nothing can reach it. Asking the
worker one more question means spawning a new agent that reads the corpus
again.

That is a real cost with a small fix, and it gets larger the more expensive the
agent's state is — which is exactly the direction the sandbox runtimes point.
The concrete cases:

- **Follow-up questions.** A supervisor reads three child results, notices two
  disagree, and wants to ask one of them a clarifying question. Today it must
  spawn a fourth agent and re-supply the context.
- **Interactive supervision.** The TUI already does multi-turn against the root.
  "Talk to the stuck worker" is the same gesture one level down, and it is what
  the `shepherd` example approximates by forking.
- **Iterative refinement.** Ask a writer for a draft, review it, ask the same
  writer for a revision. Today the second ask is a new agent that has to be
  handed the first draft.

### Peer messaging: mostly no

Free-form agent-to-agent chat is a well-known failure mode — message volume
grows with the square of the roster, termination stops being well-defined, and
runs get hard to evaluate because no single trajectory explains the outcome. It
is worth being concrete about what rlmflow specifically would lose, because the
tree is load-bearing here in ways it is not elsewhere:

- **Termination.** A run ends when the root answers. With a mesh, "everyone is
  waiting for someone" is a reachable state and constraint 1 makes it reachable
  cheaply.
- **Accounting.** `max_budget` charges `node.root.tokens()`, and
  `AgentConfig.child` derives limits down the tree. Both assume work has an
  owner.
- **Cancellation.** `queue.cancel(root.walk())` assumes the tree is the extent
  of a run.
- **The story a graph tells.** A tree reads as "who asked whom for what." A mesh
  reads as a log with edges, which is the flat-trace problem the README opens by
  rejecting.

And the demand mostly dissolves on inspection. "Agents should share what they
learn" is a blackboard, which is
[`delm_vs_rlmflow.md`](delm_vs_rlmflow.md). "Agents should see what others are
doing" is [`inter_agent_communication.md`](inter_agent_communication.md).
What is left that genuinely needs a message is the follow-up question — and a
follow-up question is resumption, not a mailbox.

### The rule that keeps this rlmflow

If any of this gets built, one constraint should hold over all of it:

> **Every message is a Node.**

A message delivered through a side channel — a queue on the `Flow`, a dict in a
REPL, an out-of-band callback — is a message that does not appear in
`graph.json`, does not survive `save`/`load`, does not replay, does not fork,
does not render in `rlmflow view`, and does not show up in the transcript that
explains why an agent said what it said. The Node tree stops being the run, and
the reason to use this library over a message-passing framework goes with it.

Delivery as a `UserQuery` append satisfies this for free. It is how the host
already talks to agents, it costs no persistence change, and views order by
`created_at` so resumed turns appear in the timeline without any view work.

## Layer 1: Addressable Resumption

The proposal. A supervisor can re-open a child it already launched.

### From agent code

```python
answer = await resume_subagent("worker", "Which file was the constant in?")
```

Named to sit beside `launch_subagents` and to be honest about what it does: it
does not create an agent, and it is not a mailbox. Semantics:

1. Resolve `name` among the caller's own `sub_agents`. No match raises.
2. Require the target to be **terminal**. A `running` or `waiting` target
   raises rather than blocking.
3. Append `UserQuery(content=query)` to the target's frontier, which is its
   `DoneOutput`. The target's transcript continues; nothing is rewritten.
4. Raise the target's `max_iters` by a resumption allowance.
5. `queue.submit(target.frontier, self.step)`.
6. `await queue.join(target)` and return the new `result()`.

Every step is an existing mechanism. Steps 3 and 5 are what
`docs/injections.md` already does by hand; step 6 is what `launch_subagents`
already does.

### Why this cannot deadlock

The terminal requirement does the work, and it is worth spelling out because it
is the reason to prefer this over a mailbox.

A terminal agent has no in-flight step and is not awaiting anything — it
answered. So the target of a `resume_subagent` is never blocked on the caller.
And the caller, while blocked in `resume_subagent`, is inside a REPL action and
therefore not terminal, so it cannot itself be the target of a resume. A cycle
would need a terminal agent that is waiting, which the definition excludes.

It also resolves constraint 2 for free. Terminal means no in-flight step for
that agent, so the append lands on a quiet frontier.

Restricting the target to the caller's own children is deliberate, and it is the
main thing keeping the tree a tree. An agent is only ever resumed by its
supervisor, so token accounting, cancellation scope, and the causal reading of
the graph all stay intact. Sibling-to-sibling resumption would be the first
crack; it should stay closed until something concrete needs it.

### What the graph looks like

Nothing new. The child's transcript gains a second round:

```text
AgentStart worker
├── ... first round ...
├── DoneOutput      "constant is 84721"      <- was the frontier
├── UserQuery       "Which file was it in?"  <- the resume
├── LLMOutput
├── ExecAction
└── DoneOutput      "src/config.py"          <- the new frontier
```

Consequences worth stating, all benign:

- `agent.result()` returns the latest answer; earlier ones stay in the
  transcript.
- `save`/`load` need no format change. `session.jsonl` is a flat chain and
  `graph.json` nests; both already handle a `UserQuery` after a `DoneOutput`,
  because that is how multi-turn works on a root.
- `steps()` and `timeline()` order by `created_at`, so resumed turns land at the
  end of the timeline with no view changes.
- `AGENTS` status moves `completed -> running -> completed`, which the existing
  status rules already produce.
- Replay re-runs the resuming `ExecAction` like any other. `resume_subagent`
  under `RLMFLOW_REPLAY=1` should read the recorded result rather than re-ask,
  matching how `launch_subagents` behaves.

The one real question is prompt projection: the resumed agent's context now
contains its own previous answer. That is correct, and `keep_n_messages` may
need to be re-examined for agents resumed many times.

### From the host

The same thing between streams, for a controller or a TUI:

```python
worker = flow.agent(root, "root.worker")     # by AGENTS selector rules
answer = await flow.resume(worker, "Which file was it in?")
```

This is the part that needs engine work, because of constraint 4. `flow.resume`
has to drive the target without its root being active, and today that is either
a one-node cancellation or a silent no-op.

The narrow fix: give `run_streaming` a notion of a driving scope that is not
necessarily a root. Today it keys `active` by `id(root)` and maps a landed node
back with `node.root`. Instead, key `active` by the agents actually passed in,
and map a landed node to its scope by walking up supervisors
(`agent.parent.parent_agent`) until one is in `active`. Then stop on
`scope.terminal` rather than `root.terminal`.

That is a contained change to the driver loop, it leaves the single-root case
identical, and it also fixes `run_streaming(sub_agent)` silently doing nothing —
which is a bug on its own terms regardless of whether any of this gets built.

## Layer 2: Non-Blocking Delegation

Optional, larger, and only worth it if something needs concurrency that Layer 1
cannot express.

Today a parent that delegates is suspended until every child answers. It cannot
react to the first child, cannot start dependent work early, and cannot be asked
anything. A non-blocking spawn would return handles instead:

```python
handles = spawn([{"name": "a", ...}, {"name": "b", ...}])
# ... the parent's turn ends; children keep running ...
# next turn:
ready = [h for h in handles if h.done]
```

The engine is closer to supporting this than it looks. `run_children` already
submits child leaves to the shared queue and only then awaits; a spawn that
submits and returns leaves children running while the parent's `ExecAction`
completes and its next turn is submitted. Handles persist in the parent's REPL
across turns because the namespace is stateful, and `handle.result()` reads
`child.result()` off the durable graph.

Open questions, none of them small:

- **Orphaning.** If the parent calls `done()` while children run, the driver
  hits `root.terminal` and cancels the subtree. Is that right, or should a
  parent be blocked from finishing with live children?
- **Replay.** Non-blocking spawn under `RLMFLOW_REPLAY=1` has to reconstruct
  handles for children that already exist and are already finished, which is a
  different shape from `launch_subagents` returning recorded results.
- **Prompting.** "You have work in flight, decide what to do this turn" is a
  meaningfully harder thing to ask a model than "here are your results."

Only the third of these is about the model, and it is the one most likely to
decide whether the feature is worth it. Worth a prompt experiment before any
engine work.

## Layer 3: Mailboxes — And Why To Stop Before This

For completeness, since it is where this line of thinking leads.

A mailbox lets any agent post to any other and read what arrived on its next
turn. If it were ever built, the non-negotiables follow directly from the
constraints above:

- Messages are `UserQuery` nodes, recording the sender. No side channel.
- Delivery only when the target has no in-flight step, buffered until then
  (constraint 2).
- **Post only, never await a reply.** Request/response is what deadlocks
  (constraint 1). One-way posting cannot.
- Bounded queue depth, and a bounded number of messages projected into any one
  prompt.

That is a coherent design. The reason to stop before it is that once messages
are one-way, unaddressed in time, and bounded, a mailbox is a worse blackboard —
and the blackboard, with verified admission and evidence refs pointing back at
the Node tree, is already designed in
[`delm_vs_rlmflow.md`](delm_vs_rlmflow.md). Build that instead if the need is
information sharing.

## Non-Goals

Explicitly out of scope, in the style of the `AGENTS` note:

- Synchronous request/response between a descendant and a live ancestor. It
  deadlocks by construction.
- Peer-to-peer resumption across the tree. Only a supervisor resumes its own
  child.
- Broadcast to all agents.
- Reaching an agent in a different root, including peers under
  `parallel_stream`.
- Keeping agents alive after their stream ends beyond what the `Runtime` already
  does. Cross-process persistence stays `save` / `load` / replay.
- Any message that is not a Node.

## Implementation Plan

Ordered so each step is useful alone.

### 1. Fix the driving scope

Key `active` by the agents passed to `run_streaming` rather than by root, and
resolve a landed node to its scope by walking supervisors. Stop on
`scope.terminal`. This is a standalone bug fix: `run_streaming(sub_agent)`
currently lands zero nodes and reports no error.

### 2. Resumption on the host

Add `Flow.resume(agent, query, *, extra_iters=...)`, doing the append, the
budget bump, the submit, and the join. Add `Flow.agent(root, selector)` reusing
the `AGENTS` selector rules so there is one way to name an agent.

### 3. `resume_subagent` in the REPL

Add the reserved tool, restricted to the caller's terminal children. Add it to
`RESERVED_TOOLS` and to the conditional prompt section. Gate it on a Flow flag
(`allow_resume=False` by default) matching the `use_agent_tree` precedent.

### 4. Delivery safety as a guarantee

Make "no in-flight step for this agent" a checked precondition of any framework
append, raising a clear error instead of the current
`ValueError: ... is not the frontier of ...` surfacing from three frames deep.
This is worth doing for `docs/injections.md` regardless.

### 5. Docs and an example

A section in `docs/control.md`, a note in `docs/node_model.md` on a transcript
with two rounds, and `examples/control/resume_subagent.py` — a supervisor that
gets a terse answer, asks the same child to expand it, and shows both rounds in
one child transcript.

Layer 2 is a separate decision, gated on the prompt experiment.

## Test Plan

1. Resuming a terminal child appends a `UserQuery` and returns the new result.
2. The child's earlier `DoneOutput` stays in its transcript.
3. Resuming a `running` or `waiting` child raises rather than blocking.
4. Resuming a name that is not the caller's child raises.
5. Resuming a sibling or an ancestor raises.
6. A resumed child that has spent its budget gets the allowance, not
   `[max_iters exceeded]`.
7. `resume_subagent` does not trip the `duplicate child name` check.
8. Two agents cannot resume each other into a deadlock (assert the cycle is
   unreachable, not that it times out).
9. `run_streaming(sub_agent)` drives that sub-agent and lands its nodes.
10. Driving a scope does not stop early because an unrelated root is terminal.
11. Save, load, and re-save round-trip a twice-answered agent with no format
    change.
12. Replay of a graph containing a resume re-reads the recorded result and
    appends nothing.
13. `steps()` and `timeline()` place resumed turns last.
14. `AGENTS` reports the target `completed` again after the resume.
15. `root.tokens()` includes the resumed turns.
16. A framework append to an agent with a step in flight raises a specific
    error, not a frontier `ValueError` from inside `LLMRequestStep` / `Flow.step`.
17. `allow_resume` defaults to false; disabled runs have no tool and no prompt
    text.

## Recommendation

Build Layer 1. Skip Layer 3. Decide Layer 2 later, on evidence.

The framing that makes the call obvious: **rlmflow agents already talk to each
other, through the graph.** A query is a message down, a result is a message up,
and both are nodes you can read, save, fork, and replay. The gap is not that
there is no channel — it is that the channel closes when a child answers, while
the child itself, transcript and REPL both, is still sitting right there.

Resumption reopens that channel without inventing a second one:

```text
Node tree + open REPL (already durable)
  -> resolve a terminal child of the caller
  -> append UserQuery to its frontier
  -> submit and join on the existing queue
  -> its own next DoneOutput
```

No new node type, no run-format change, no scheduler redesign, no side channel,
and one contained driver fix that is worth making anyway. A mailbox would need
all of those and would give up the tree that makes a run explainable — to solve
a problem that shared context solves better.
