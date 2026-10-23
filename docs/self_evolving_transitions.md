# Agent-Directed Transitions and Graph Evolution

> **Status:** Earlier routing-design analysis. Nothing here is implemented. The named-choice feature it analyzes (`flow.choices(...)`, `transition("name")`, `TransitionSignal`, `InvalidTransitionError`) was removed in 0.7.0. `PlanQuery` and its extra observation-to-plan hop were removed later as well; references below preserve the historical design being evaluated. [`graph_context_evolution.md`](graph_context_evolution.md) supersedes its optional `continue_with(...)` recommendation with a graph-grounded, cross-run context-evolution design requiring no new model-facing control API.
>
> **Question:** should an rlmflow agent be able to change the static transition policy in `rlmflow.engine.steps`, and what would a natural graph-native API look like?

## Decision

Do not let a running agent rewrite Flow's private transition policy, register Python producers, or maintain a mutable state-machine definition.

Keep the existing API precise:

```python
transition("verify")
```

means:

> Select one host-declared behavior that is currently available.

If experiments show value in letting an agent set the objective of its own next turn, add a separate control primitive:

```python
continue_with(
    "Try to falsify the current candidate with focused tests before changing it."
)
```

That call should land one durable `ContinuationQuery` directly after the current `ExecAction`. It records the actual next edge taken; it does not create a reusable behavior, mutate future edges, or pretend to be a workflow optimizer.

Use the existing graph operations according to their real topology:

```text
continue_with(...)       extend this agent's transcript chain
transition("name")       enter a host-declared behavior
launch_subagent(...)     create a concurrent child branch
finish(...)              terminate this agent's chain
```

Genuine workflow evolution should be a separate, evaluated outer loop over saved runs and candidate policies. Research systems that improve workflow structure use repeated execution and an objective; they do not establish that a single solving agent benefits from inventing an untested state machine while it works.

The earlier `PolicyPatch` design remains a technically sound fallback if a future requirement truly needs in-run policy-definition mutation. It is not the recommended first feature.

## Why the Distinction Matters

Three different structures are often called a "dynamic graph."

### 1. Execution graph

This is rlmflow's durable `Node` tree. It records what actually happened: queries, model outputs, actions, observations, child agents, and results.

It is already dynamic. The model grows it whenever it executes code, delegates, or finishes.

### 2. Workflow definition

This is the policy that says what kinds of nodes may follow other kinds of nodes. In rlmflow, it is the trusted producer table registered with:

```python
@flow.transition(...)
```

and the host-declared choice sets registered with:

```python
flow.choices(CurrentQuery, TargetQuery)
```

This definition is static for a `Flow` instance.

### 3. Search or optimization graph

This is a graph of alternative trajectories or alternative workflow definitions. A search algorithm evaluates candidates, assigns scores, and chooses which branch or policy to keep.

Examples include tree search over agent trajectories and MCTS or reinforcement learning over workflow definitions. This graph needs an evaluator and usually spans multiple executions.

These should not be collapsed:

```text
runtime routing          choose the next actual node
definition mutation      change which future routes exist
workflow optimization    evaluate competing definitions and promote one
```

Calling all three "self-evolving transitions" hides the most important design question: where does the evidence come from that one graph is better?

## Current rlmflow Semantics

The default kernel in `rlmflow.engine.steps` is:

```text
AgentStart / observation -> PlanQuery
UserQuery                 -> LLMOutput
LLMOutput + repl          -> ExecAction
LLMOutput + invalid reply -> empty ExecAction -> ErrorOutput
ExecAction                -> ExecOutput | ErrorOutput | DoneOutput | selected UserQuery
```

Budget and truncation guards can preempt ordinary work with `FinalQuery` or `TruncationSummary`.

One agent's transcript is one chain. `Node.append` accepts same-agent sequels only at `agent.frontier`. Child `AgentStart` nodes are the only structural branches, and attaching a child does not move the parent's frontier.

The named transition path is already graph-native:

```text
ExecAction containing transition("verify")
  -> VerifyQuery
  -> LLMOutput
  -> ExecAction
```

There is no intermediate `ExecOutput`. `transition("verify")` raises a `TransitionSignal`; the runtime returns a transition status; `run_repl` resolves the currently offered target and returns that `UserQuery`.

Host-defined behavior persists across utility nodes because choice resolution scans backward for the latest query class participating in a choice set. A `PlanQuery` or `TruncationSummary` does not erase it.

At the time of this review, repository uses of `flow.choices(...)` and `transition("name")` are confined to transition tests; no example or benchmark exercises the feature. The semantics are tested, but there is not yet product evidence that named behaviors improve an rlmflow task.

These properties are valuable and should remain.

## Research Findings

The external systems reviewed largely agree on one architectural split: execution may route dynamically, but the set of executable node kinds remains declared or compiled.

### LangGraph: commands route; they do not author nodes

LangGraph's `Command(update=..., goto=...)` lets a node update state and choose the next runtime destination in one result. `Send(node, state)` dynamically schedules instances of an existing node, commonly for map-reduce fan-out.

The destinations remain part of the graph definition. The official docs ask node functions to annotate valid `Command[Literal[...]]` destinations for rendering and validation. They also warn against mixing static edges and `Command` routing from one node because both can execute.

Transfer to rlmflow:

- a control result should atomically describe the next runtime edge;
- there should be one routing authority for that edge;
- dynamic fan-out and local continuation are different operations;
- "dynamic edge" should not be interpreted as persisted topology mutation.

### Pydantic Graph: return the next typed node

In Pydantic Graph's class API, a node's `run()` method returns an instance of the next node or `End`. Union return annotations declare the possible outgoing edges and permit runtime validation and graph rendering.

This is close to rlmflow's existing internal producer contract:

```python
(flow, current_node) -> next_node
```

Transfer to rlmflow:

- the natural runtime decision is a next-node value;
- termination is a distinct outcome;
- the runtime should interpret only registered node kinds;
- dynamically creating Python node classes is unnecessary.

### AutoGen GraphFlow: use a graph only when strict structure is needed

AutoGen's experimental GraphFlow builds and validates a `DiGraph` of agents before execution. It supports conditions, loops, fan-out, joins, and activation rules. Its own guidance recommends simpler ad-hoc orchestration unless strict ordering, deterministic branching, or explicit cycles are required.

Transfer to rlmflow:

- do not require every autonomous run to declare a workflow;
- structured behavior graphs are primarily host policy;
- conditions and cycles add substantial scheduler semantics;
- a graph of agents is not the same thing as one agent's semantic phases.

### OpenAI Agents: split only when the execution contract changes

OpenAI's orchestration guidance distinguishes:

- handoffs, where a specialist takes ownership;
- agents as tools, where the manager retains ownership.

It recommends introducing specialists only when capability isolation, policy isolation, prompt clarity, or trace legibility materially improves.

Transfer to rlmflow:

- a new local instruction should stay in the same agent;
- work needing fresh context, tools, model, or policy belongs in a child agent;
- delegation is already the graph-native way to create a meaningful branch.

### Temporal: decisions belong in durable history

Temporal requires workflow code to emit the same commands during replay. Nondeterministic operations such as LLM calls belong outside deterministic workflow logic, and their results become event history. Workflow-definition changes require versioning or patching.

rlmflow does not need Temporal's exact replay model, but the transferable rule is strong:

> A model decision that affects future execution must be represented in durable history, not only in a mutable in-memory controller.

A `ContinuationQuery` satisfies this. Mutating Flow's private transition policy from REPL code does not.

### Workflow-optimization research uses an outer evaluation loop

AFlow represents each MCTS search node as a complete code-defined workflow. An LLM modifies prompts and code connections, candidates are executed repeatedly on validation tasks, and scores guide later search.

GPTSwarm models agent systems as computational graphs and optimizes prompts or probabilistic connectivity. Its edge optimization uses sampled graphs and an external utility estimate.

ADAS / Meta Agent Search has a meta-agent generate complete agent implementations, evaluate them, and store them in an archive that informs later candidates.

LATS expands and evaluates alternative reasoning/action trajectories under a tree-search controller.

The common ingredient is not graph mutation. It is:

```text
candidate
  -> execute
  -> evaluate performance/cost
  -> compare
  -> retain or reject
```

Without that loop, a solver changing its own workflow is online planning, not demonstrated evolution.

## What Capability Is Actually Missing?

The current model can already:

- execute arbitrary branching Python inside one REPL action;
- end an action normally and receive its output on the next turn;
- select a host-declared behavior with `transition(name)`;
- launch recursive child branches;
- wait for or observe child results;
- finish its chain.

One narrow semantic operation is missing:

> End this action without treating stdout as an observation, and make this agent-authored instruction the next query node.

That is what `continue_with` should mean.

It adds little raw computational power. Its value is structural:

- the next-turn intent is a typed node rather than stdout;
- the graph distinguishes an observation from a self-directed continuation;
- the instruction survives save/load and appears in views;
- the direct edge avoids an extra `ExecOutput -> PlanQuery` pair;
- forks preserve whether the continuation had happened;
- host-defined behavior identity remains intact.

Because the computational gain may be small, this should be measured before it becomes a permanent default tool.

## Recommended Model-Facing API

Keep four distinct verbs:

```python
transition(name: str) -> NoReturn

continue_with(instruction: str) -> NoReturn

launch_subagent(
    goal: str,
    *,
    model: str,
    ...
) -> AgentHandle

finish(answer: object) -> NoReturn
```

Their meanings are:

```text
transition(name)       choose host policy
continue_with(text)    choose this chain's next query
launch_subagent(...)   add an independently executing branch
finish(answer)         terminate this chain
```

Do not overload `transition` with optional `name`, `query`, `instruction`, or `label` combinations.

The authorities differ:

- `transition(name)` is closed-world and validated against the host's available choices;
- `continue_with(instruction)` is open text bounded by query limits.

Keeping them separate preserves useful errors:

```text
transition "verify" is unavailable; available: ["work", "review"]
```

instead of making a typo look like a newly authored dynamic state.

It also produces simpler model documentation than a union-shaped tool with mutually exclusive arguments.

## `ContinuationQuery`

Use a dedicated built-in query type:

```python
@dataclass
class ContinuationQuery(UserQuery):
    """An agent-authored objective for its own immediate next turn."""

    type: ClassVar[str] = "continuation_query"
```

The existing `content` field is the instruction. No label, policy revision, behavior definition, edge list, or target factory is needed.

A dedicated type is preferable to plain `UserQuery` because it records provenance:

```text
UserQuery                 host or handoff follow-up
PlanQuery                 engine planning boundary
ContinuationQuery         agent-authored next-turn objective
FinalQuery                engine finalization boundary
TruncationSummary         engine context boundary
```

That distinction helps graph views, evaluation, and future policy without introducing a second graph definition.

`ContinuationQuery` should not participate in `Transitions.current_behavior(...)` unless a host explicitly declares it in a choice set. Therefore an ad-hoc continuation does not erase the latest host-defined behavior.

## Exact Runtime Semantics

### Normal action

```text
ExecAction
  -> ExecOutput
  -> PlanQuery
  -> LLMOutput
```

Stdout is observed.

### Named host transition

```text
ExecAction calling transition("verify")
  -> VerifyQuery
  -> LLMOutput
```

Stdout is not observed.

### Agent-directed continuation

```text
ExecAction calling continue_with("Run focused verification.")
  -> ContinuationQuery("Run focused verification.")
  -> LLMOutput
```

Stdout is not observed.

### Delegation plus continuation

```python
review = launch_subagent(
    task="Review the current implementation for correctness.",
)

continue_with(
    "A review is running. Independently run focused tests before waiting for it."
)
```

The graph is:

```text
ExecAction
├── AgentStart("review")
└── ContinuationQuery("Independently run focused tests...")
```

The child branch and parent continuation are different edge kinds. When the child finishes, handoff eventually adds its durable `UserQuery` to the parent at a safe frontier.

### Repair loop

```python
result = run_tests()

if result.failed:
    continue_with(
        "The focused tests failed. Inspect `result`, identify the root cause, "
        "and revise the implementation without unrelated cleanup."
    )

continue_with(
    "The focused tests passed. Inspect the final diff and run regression checks."
)
```

The model writes only the edge it is taking. It does not author hypothetical future nodes it may never visit.

## Prompt Contract

When continuations are enabled, the action footer should describe all exits:

```text
End the REPL block normally to observe stdout and continue.
Or end it immediately; stdout from that block is not observed:
- continue_with("instruction") — make this the objective of your next turn
- transition("verify") — Verify the current candidate.
- finish(answer) — Submit your final answer.
```

When no named choices exist, omit only the named transition rows.

The guidance should say:

- use normal completion when the next turn should react to printed output;
- use `continue_with` when REPL state already contains what is needed and the next turn needs a specific objective;
- do not call it merely to restate the current plan;
- end a later block normally when pending child handoffs need their ordinary `ExecOutput` flush boundary;
- use a child when the next phase needs fresh context, tools, model, or policy.

On `TurnMode.FINAL`, expose only `finish`.

## Validation and Safety

`continue_with` must:

- be opt-in for the first experiment;
- accept exactly one non-empty string;
- apply the agent's `max_query_chars` bound;
- work only during `TurnMode.ACTION`;
- be unavailable on final turns;
- end the REPL action immediately;
- discard stdout from that action, matching `transition`;
- create exactly one `ContinuationQuery`;
- never alter Flow's private transition policy;
- never accept Python callables, node classes, tool names, models, or policy overrides.

Existing hard policy still wins:

```text
ContinuationQuery
  -> FinalQuery if out of room
  -> TruncationSummary if history needs truncation
  -> LLMOutput otherwise
```

An agent cannot use continuations to bypass iteration or token limits.

Repeated continuations can still waste turns. Existing budgets bound that, and evaluation should measure it.

## Internal Implementation

Keep the continuation path parallel to, but distinct from, named transitions.

### Graph

In `rlmflow.graph.nodes`:

- add `ContinuationQuery`;
- inherit rendering and turn mode from `UserQuery`;
- register it in persistence;
- export it only if host code needs to inspect or construct one.

No persistence version bump is required for the data shape, though older package versions will not recognize the new node type.

### Runtime control

In `rlmflow.runtime.repl`:

```python
class ContinueSignal(BaseException):
    def __init__(self, instruction: str) -> None:
        super().__init__(instruction)
        self.instruction = instruction
```

Add a distinct `ReplStatus.CONTINUE` and `ReplRun.continuation`.

Do not encode continuation text inside the existing transition-name string. The wire protocol should preserve the semantic distinction.

### Tool

In `rlmflow.tools.namespace`:

- bind a per-node `continue_with` tool when enabled and not final;
- validate type, emptiness, and length on the host side;
- raise `ContinueSignal` with the normalized instruction;
- reserve the name against host tool injection.

Add it to `Flow.build_tools(node)` only when enabled; the prompt renders its signature from that same mapping.

### Engine

In `rlmflow.engine.steps.run_repl`:

```python
if run.status is ReplStatus.CONTINUE:
    return continuation_target(node, run)
if run.status is ReplStatus.TRANSITION:
    return transition_target(flow, node, run)
return node_for_run(node, run)
```

`continuation_target` checks `TurnMode.ACTION` again and returns:

```python
ContinuationQuery(content=run.continuation)
```

Returning a query directly preserves the current one-step/one-node rule.

### Worker protocol

Transport continuation separately through:

- `ReplResponse`;
- `ProxyResponse`;
- worker proxy signal reconstruction;
- `ReplRun`.

This is mechanical but necessary for local, subprocess, Docker, and Modal runtimes to agree.

### Replay

Recorded `ExecAction` code may call `continue_with` again during REPL replay. Restore already executes actions to rebuild namespace state and discards their outcomes. It must consume the continuation signal without appending another query.

The existing `ContinuationQuery` in the tree remains the durable source of truth.

### Fork

No special policy reconstruction is needed:

```text
fork at ExecAction before continuation -> query not retained
fork at ContinuationQuery              -> query retained
```

This is simpler than a mutable policy overlay because the only state being changed is the actual execution path.

## Why Not Overload `transition(...)`?

An overloaded API initially appears smaller:

```python
transition("verify")
transition(instruction="Verify the candidate.")
```

But it combines two incompatible contracts:

```text
name             host-owned closed set
instruction      agent-owned open text
```

That creates avoidable questions:

- Can both be passed?
- Does instruction augment or replace the target query?
- Does a label create a persistent state?
- Does an unknown name become a dynamic behavior?
- Which transition descriptions appear in the footer?
- Does the inline query retain the current host behavior?

A separate verb makes the model's authority obvious and keeps `InvalidTransitionError` meaningful.

## Why Not a Graph Builder?

An agent-facing builder such as:

```python
g.state("verify", ...)
g.edge("work", "verify")
g.commit()
```

is a mutable object hidden in the REPL until commit. If it is not committed as a node, save/load and forks lose it. If it is committed as a node, it is only a more verbose syntax for `PolicyPatch`.

Graph builders are useful for host-authored workflows. They are not naturally aligned with an agent choosing one next step during a rollout.

## Why Not `PolicyPatch` Yet?

The earlier design proposed:

```text
PolicyPatch
  -> reduce a per-agent behavior policy
  -> AdaptiveQuery
```

That is the correct event-sourced shape for true in-run definition mutation. It preserves fork-before/fork-after semantics and avoids hidden mutable state.

But it also requires:

- an initial policy snapshot on every agent;
- a patch schema and reducer;
- behavior and edge revisions;
- dynamic choice resolution;
- interaction rules with static choices;
- prompt and persistence changes;
- limits on states, edges, text, and churn;
- inheritance semantics across concurrent children;
- model training or prompting for a mini workflow language.

Most importantly, it asks the solving agent to maintain an untested state machine while also solving the task.

Do not pay that cost until simpler continuation experiments demonstrate:

1. explicit self-directed query nodes improve outcomes;
1. the model repeatedly needs reusable named routes rather than one-shot objectives;
1. host-authored static choices cannot express those routes;
1. the graph definition must change within one execution rather than between evaluated executions.

If all four become true, restore the `PolicyPatch` proposal as a second-stage research feature.

## Where True Evolution Belongs

Workflow evolution needs an optimizer outside an individual rollout:

```text
baseline Flow policy
  -> run on training tasks
  -> save Node graphs and metrics
  -> propose candidate policy
  -> validate candidate
  -> run repeated evaluations
  -> compare success, cost, latency, safety
  -> promote or reject
```

The rlmflow graph is valuable input to this process:

- `agent.transcript()` provides each local trajectory;
- child structure reveals decomposition and fan-out;
- `root.stats` and usage expose cost;
- errors and terminal values expose failure modes;
- saved prompts show which policy produced each model turn;
- `node.fork()` can support controlled alternative continuations;
- run views make candidate behavior inspectable.

The candidate workflow definition should be a host artifact—Python code, declarative configuration, or a `Transitions` table—executed in isolation and versioned separately from one run's Node graph.

For online search within one task, a host controller can fork a node, execute several candidate continuations, evaluate them, and select one branch. That is closer to LATS. It is different from allowing one branch to rewrite the policy for itself.

## Experiment Plan

Do not begin with the implementation. Establish the baseline and hypothesis.

### Experiment 0: no new primitive

Measure the current loop:

```text
ExecOutput -> PlanQuery
```

The model can already print a short plan into its next observation. This is the baseline that `continue_with` must beat.

### Experiment 1: opt-in continuations

Enable `continue_with` on a focused task set requiring phase changes:

- investigate, implement, verify;
- hypothesis, falsification, revision;
- parallel delegation followed by local synthesis;
- recovery after a failed action.

Measure:

- task success;
- tokens and wall time;
- model turns and graph nodes;
- continuation count and average instruction length;
- repeated or near-duplicate continuations;
- continuations preempted by finalization;
- invalid calls;
- child count and handoff timing.

Inspect whether continuation text changes subsequent actions or merely restates the previous assistant message.

### Experiment 2: static named choices

Give the same tasks a small host-defined behavior graph using `flow.choices`.

This separates the value of explicit phase structure from the value of agent-authored phase text.

### Decision after evidence

```text
continuation does not beat baseline
  -> do not ship it

continuation helps; reusable phases do not matter
  -> keep continue_with only

static choices help more
  -> improve host workflow ergonomics, not self-mutation

tasks require in-run creation of reusable routes
  -> prototype bounded PolicyPatch

performance improves only across evaluated candidates
  -> build outer-loop workflow optimization
```

## Test Plan for `continue_with`

1. A valid call lands `ExecAction -> ContinuationQuery` directly.
1. There is no intermediate `ExecOutput`.
1. Stdout from the continuing block is not observed.
1. The next step calls the model unless a hard guard preempts it.
1. Empty, non-string, and oversized instructions become repairable execution errors.
1. The call is rejected during `TurnMode.FINAL`.
1. The tool and prompt text are absent when the feature is disabled.
1. The final prompt exposes only `finish`.
1. A continuation does not erase the latest host-defined behavior.
1. Named `transition(...)` availability and errors are unchanged.
1. Save/load round-trips `ContinuationQuery`.
1. Replay appends no duplicate continuation.
1. Forking before and at the continuation gives the expected paths.
1. Local and process-isolated runtimes transport identical continuation text.
1. A continuation after launching children does not corrupt the parent's frontier.
1. A child result arriving during a continuation remains queued and flushes at the next safe handoff boundary.
1. `max_iters`, `max_budget`, `keep_n_messages`, and `max_query_chars` still apply.

## Open Questions

- Is one-shot self-direction measurably better than printing a plan and taking the normal `PlanQuery` path?
- Should continuations be available to every agent or only orchestrators?
- What should the opt-in setting be named?
- Should the footer advertise continuations every turn, or should only selected prompt profiles expose them?
- Does skipping `ExecOutput -> PlanQuery` make planning more focused or merely reduce useful observation structure?
- Should graph views render `ContinuationQuery` differently from external `UserQuery`?
- Should continuation text be included in RAO trajectory exports as an explicit policy action?
- If a child result is already pending, should `continue_with` be rejected, delayed, or allowed to proceed under the existing handoff rules?

## References

- LangGraph, [Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api) — `Command`, conditional routing, and `Send`.
- Pydantic AI, [Graph](https://pydantic.dev/docs/ai/graph/graph/) — typed next-node returns and `End`.
- Microsoft AutoGen, [GraphFlow](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/graph-flow.html) — prebuilt agent graphs, conditions, cycles, and joins.
- OpenAI, [Orchestration and handoffs](https://developers.openai.com/api/docs/guides/agents/orchestration) — manager versus delegated ownership.
- Temporal, [Workflow definition](https://docs.temporal.io/workflow-definition) — durable commands, event history, determinism, and versioning.
- Zhang et al., [AFlow: Automating Agentic Workflow Generation](https://arxiv.org/abs/2410.10762) — evaluated MCTS over code-represented workflows.
- Zhuge et al., [Language Agents as Optimizable Graphs](https://arxiv.org/abs/2402.16823) — computational agent graphs and connectivity optimization.
- Hu et al., [Automated Design of Agentic Systems](https://arxiv.org/abs/2408.08435) — evaluated code generation and an archive of candidate agents.
- Zhou et al., [Language Agent Tree Search](https://arxiv.org/abs/2310.04406) — evaluated search over alternative agent trajectories.

## Final Recommendation

Do not implement live workflow-definition mutation now.

Keep:

```python
transition("name")
```

strictly for host-declared choices.

Run a small opt-in experiment with:

```python
continue_with("the exact objective for my next turn")
```

landing a durable `ContinuationQuery`.

If the experiment does not beat the normal `ExecOutput -> PlanQuery` loop, remove it. If reusable host phases help, improve `flow.choices`. If workflow structure improves only after repeated evaluation, build an outer-loop optimizer over saved rlmflow graphs.

That uses the graph where it is strongest:

```text
the Node tree records actual execution,
the host policy defines legal behavior,
and an evaluator—not wishful self-description—drives evolution.
```
