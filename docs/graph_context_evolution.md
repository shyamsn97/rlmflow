# Graph-Native Context Evolution

> **Status:** Research and architecture proposal. Nothing here is implemented.
>
> **Scope:** use rlmflow's recursive execution graph as the evidence, provenance, and scope system for ACE-style context evolution—without turning the graph into a mutable workflow definition.
>
> **Companion:** [`self_evolving_transitions.md`](self_evolving_transitions.md) separates runtime routing from workflow-definition mutation. This proposal answers the next question: if transitions stay mostly static, what does the graph buy us, and how can learning make stronger use of it?

## Short Answer

rlmflow has a dynamic execution graph, but not a dynamic workflow graph.

More precisely, its durable structure is a dynamically constructed rooted tree:

- one agent's transcript is a chain;
- an `ExecAction` may add child `AgentStart` branches;
- loops create new nodes rather than back-edges;
- handoffs copy results into the parent transcript rather than creating multiple-parent join nodes;
- no node is rewritten, merged, or shared by two parents.

Calling it a "dynamic graph" is therefore defensible when "dynamic" means the runtime topology is discovered during execution. It would be misleading if "dynamic graph" implied that agents rewrite the workflow definition, add arbitrary edge kinds, create cycles, or merge branches.

Even without learning, the graph already provides:

- recursive ownership and fresh-context boundaries;
- durable typed execution state;
- concurrency and cancellation scope;
- replay and warm-REPL restoration;
- safe frontier edits;
- branch-local transcripts;
- checkpointing, visualization, and evaluation;
- `fork()` as an operation over a historical execution point.

That is more than a visualization. The tree is both the run state and its trace. But much of the current value is execution infrastructure and observability. If users never inspect, fork, resume, evaluate, or learn from branches, an event log plus agent IDs could provide many of the same practical benefits.

The best middle ground is to make the graph a learning substrate:

```text
the graph records what happened and where;
reflection extracts lessons bottom-up;
curation decides which lessons propagate;
message building injects the resulting playbook;
future graph construction changes organically.
```

Do not evolve `engine.steps`. Evolve structured context using evidence from the graph.

## What Kind of Graph Exists Today?

There are three structures worth naming separately.

### The storage structure: a rooted tree

Every `Node` has at most one `parent`. The complete run is rooted at one `AgentStart`. A same-agent child advances that agent's transcript; a child `AgentStart` creates a recursive branch.

```text
AgentStart(root)
└── PlanQuery
    └── LLMOutput
        └── ExecAction
            ├── AgentStart(researcher)
            │   └── ...
            ├── AgentStart(reviewer)
            │   └── ...
            └── ExecOutput
                └── PlanQuery
                    └── ...
```

This is mathematically a tree, which is a restricted graph. Runtime references such as retrieved agent IDs add semantic links, but they do not change the single-parent storage structure.

### The execution path: dynamically constructed

The host does not know ahead of time:

- how many model and REPL turns will occur;
- whether an action will succeed, fail, finish, or select a named behavior;
- how many children will be launched;
- which children will recursively delegate;
- when child results will return;
- whether a finished parent will be resumed by a late child handoff.

Those decisions construct topology at runtime. Breadth and depth are constrained by configuration, but the concrete tree is not compiled before execution.

This is the meaningful sense in which rlmflow is dynamic.

### The workflow definition: mostly static

`Flow.transitions` determines which trusted producer handles each frontier node. The default rows in `engine.steps` implement model calls, REPL execution, planning boundaries, truncation, and finalization.

The host may declare named behavior choices. The model may select one with `transition("name")`, but it does not create the behavior or its outgoing choices.

This is static policy driving a dynamic execution tree.

## What the Graph Buys Today

### 1. The root is the complete durable run

There is no separate public `Run`, `Agent`, trace, and branch hierarchy that must be synchronized. Starting from the root, the library can recover:

- every agent;
- each agent's transcript and frontier;
- parent/child ownership;
- token usage;
- terminal results;
- errors;
- creation and execution timing.

The graph is not a derived trace emitted after execution. It is the state being mutated by execution.

### 2. Recursive context boundaries are explicit

Every child `AgentStart` has its own transcript and configuration. A child does not merely appear as another tool span in a flat parent conversation.

This preserves:

- who received which task;
- where fresh context began;
- which model and prompt profile were used;
- which descendants a child created;
- what result crossed back to its supervisor.

That structure matters for debugging delegation and for assigning learning signals at the right level.

### 3. The tree defines execution ownership

The graph scopes:

- budgets over the root tree;
- cancellation over a subtree;
- REPL identity per agent;
- child naming and depth;
- handoff direction;
- restoration of shared worker sessions.

Replacing the graph with a flat log would require rebuilding these ownership relationships elsewhere.

### 4. Append-only frontiers make intervention legible

An append either lands on an agent's current frontier or fails. The host cannot silently rewrite the middle of an in-flight transcript.

Controller feedback, resumed work, and child handoffs therefore become durable new nodes rather than mutations of old messages.

### 5. Forking is a semantic operation

`node.fork()` means:

> Preserve the run through this exact historical point, remove everything below it, assign fresh identities, and continue independently.

That is stronger than copying a chat-message list because the operation also prunes child subtrees and restores the correct agent frontier.

### 6. Typed nodes support consumers

Streaming, checkpointing, views, evaluation, and research code can distinguish:

```text
model output
REPL action
clean observation
error
finalization request
child result
terminal answer
```

without parsing one undifferentiated message log.

## What the Graph Does Not Buy Automatically

The graph does not by itself provide:

- better planning;
- causal credit assignment;
- reusable knowledge;
- workflow optimization;
- branch comparison;
- merge semantics;
- reliable reflection;
- general DAG joins or cycles;
- a learned transition policy.

It preserves enough structure to build those systems, but a consumer, evaluator, or optimizer must use it.

This is the honest boundary:

> Today the graph is primarily execution state, recursive provenance, and a control surface. It becomes a learning architecture only when something consumes its structure.

## Why ACE Is a Better Starting Point Than Transition Mutation

[Agentic Context Engineering](https://arxiv.org/abs/2510.04618) treats context as an evolving playbook rather than repeatedly rewriting one monolithic prompt. Its core loop separates:

```text
Generator -> Reflector -> Curator -> incremental context update
```

ACE's useful ideas for rlmflow are:

- structured, itemized lessons;
- unique identities and usage metadata;
- reflection over concrete execution feedback;
- incremental deltas instead of complete prompt rewrites;
- deterministic merging;
- periodic deduplication and harmful-entry pruning;
- offline and sequential online adaptation.

The paper also reports an important limitation: context adaptation can degrade when feedback is unreliable. A graph supplies provenance, not truth. The quality of tests, graders, environment outcomes, and curation still determines whether learning helps.

rlmflow can contribute something ACE does not get from a flat trajectory: hierarchical reflection over a recursive agent tree.

## Proposed Architecture

Use two durable structures with different jobs.

### Run tree

The existing `AgentStart`/`Node` tree records one execution.

It answers:

- what happened;
- which agent did it;
- which task and context that agent received;
- which branches were created;
- which results returned;
- where failures and costs occurred.

### Playbook

A versioned playbook accumulates reusable guidance across runs.

It answers:

- what has been learned;
- which roles or tasks it applies to;
- what evidence supports it;
- how often it appeared useful or harmful;
- which revision future runs should use.

Do not force the cross-run playbook into the execution tree. It outlives any one root. Instead, pin a playbook revision to each run and retain graph/node references on every learned entry.

```text
Playbook revision 12
  ├── entry coding.verify-004
  │     source: graph A / root.worker / node X
  └── entry delegation-011
        sources: graph B node Y, graph C node Z

Run D
  └── AgentStart(playbook revision 12)
        └── LLMOutput(prompt_id identifies the rendered prompt)
```

## Playbook Data Model

A minimal entry:

```python
@dataclass(frozen=True)
class PlaybookEntry:
    id: str
    section: str
    content: str
    applies_to: tuple[str, ...] = ()
    helpful: int = 0
    harmful: int = 0
    source_graph_ids: tuple[str, ...] = ()
    source_agent_ids: tuple[str, ...] = ()
    source_node_ids: tuple[str, ...] = ()
```

Possible sections:

```text
investigation
delegation
implementation
verification
failure-recovery
synthesis
domain-specific
```

`applies_to` may contain prompt profiles, toolsets, task families, or other host-defined tags. Do not encode arbitrary executable predicates.

An entry should be small and actionable:

```text
[coding.verify-004] helpful=7 harmful=1
Before changing production parsing behavior, inspect whether the failing test
fixture normalizes the value.
```

It should not be a compressed narration of an entire run.

## Graph-Grounded Reflection

### Reflect one agent at a time

Each `AgentStart` defines a local episode:

```text
assigned query and inputs
local transcript
direct child goals and results
local errors
terminal result
usage and timing
external feedback
```

Reflecting locally keeps child implementation detail out of the parent's reflection context and preserves task ownership.

### Traverse bottom-up

Process terminal agents in post-order:

```text
grandchildren
  -> children
  -> root
```

For example:

```text
test-investigator lesson:
  "The fixture normalizes empty strings before parser invocation."

implementer lesson:
  "When parser behavior and tests disagree, inspect fixture normalization
   before editing production parsing."

root lesson:
  "Delegate fixture analysis separately when implementation and tests disagree;
   do not assign two children overlapping repository-wide searches."
```

The leaf retains domain detail. The parent generalizes only what its outcome supports.

### Keep evidence attached

A reflector proposal should identify evidence:

```python
@dataclass(frozen=True)
class LessonCandidate:
    section: str
    content: str
    evidence_node_ids: tuple[str, ...]
    confidence: float
    proposed_scope: tuple[str, ...]
```

Host code verifies that referenced nodes belong to the run and supplies the actual node content to the curator. The model should not be trusted to invent provenance identifiers.

### Structural attribution is not causal attribution

The graph can establish:

- an entry was present in a prompt;
- a child produced a result;
- a handoff reached the parent;
- an error occurred after an action;
- a run with a given playbook revision succeeded.

It cannot establish that one entry caused success merely because both occurred.

Helpful/harmful counters should use conservative evidence:

- hard test or grader outcomes;
- repeated association across runs;
- explicit ablations;
- branch comparisons from the same fork point;
- curator review of contradictory trajectories.

## Generator, Reflector, and Curator

### Generator

The Generator is the ordinary rlmflow execution tree. No new model-facing API is required.

The root and all descendants generate trajectories under a pinned playbook revision.

### Reflector

The Reflector consumes one local agent episode plus reliable feedback.

Its job is to propose:

- reusable strategies;
- recurring failure modes;
- corrections to existing entries;
- scope narrower than the current task when appropriate;
- evidence references.

It does not write the playbook.

The Reflector may be:

- a one-shot LLM call for short episodes;
- a dedicated rlmflow child for complex traces;
- a host model outside the original run;
- omitted when there is no reliable feedback.

Do not automatically spend a recursive agent on every node.

### Curator

The Curator compares candidates with the current playbook and emits a bounded delta:

```python
@dataclass(frozen=True)
class PlaybookDelta:
    additions: tuple[PlaybookEntry, ...] = ()
    revisions: tuple[object, ...] = ()
    helpful_ids: tuple[str, ...] = ()
    harmful_ids: tuple[str, ...] = ()
    rejected_candidates: tuple[str, ...] = ()
```

An LLM may propose the delta. Deterministic host code must:

- validate entry and source IDs;
- enforce token and entry limits;
- apply updates atomically;
- deduplicate;
- preserve unchanged entries;
- write a content-addressed revision;
- reject malformed or unsupported operations.

This separation avoids asking the same model to solve, judge, and permanently rewrite its own instructions in one pass.

## Hierarchical Scope and Promotion

The tree provides a natural scope model.

### Local

A lesson learned by a child initially belongs to that child's role or prompt profile.

```text
root
├── researcher
└── coder
    └── test-investigator learns fixture-specific rule
```

The rule should not silently alter `researcher`.

### Inherited

General parent guidance may be inherited by children launched in later runs or after a pinned context snapshot.

Examples:

- delegation policy applies to spawn-capable orchestrators;
- file-editing policy applies to leaf coding agents;
- synthesis policy applies to roots receiving child handoffs.

### Promoted

A child lesson becomes parent/global guidance only through curation.

Promotion should consider:

- whether the parent received or used the child result;
- whether the root outcome improved;
- whether sibling evidence agrees;
- whether the lesson is role-specific;
- whether the same lesson recurs in other graphs.

### Fork-local

If within-run context updates are added later, a patch on one branch must not affect its sibling. Forking before that patch should recover the old context; forking after it should retain the patch.

This is where graph structure should determine semantics, not merely visualization.

## Message-Building Integration

The playbook should influence behavior through the existing prompt path.

A playbook-aware prompt source:

```python
class PlaybookPrompt(PromptBuilder):
    def __call__(self, flow, node):
        base = super().__call__(flow, node)
        entries = flow.playbooks.retrieve(
            revision=node.parent_agent.config.playbook_revision,
            profile=node.parent_agent.config.prompt_profile,
            query=node.parent_agent.content,
            max_tokens=playbook_budget(node),
        )
        return f"{base}\n\n### Learned playbook\n{render(entries)}"
```

The exact API is illustrative. The important rules are:

- select entries deterministically for a pinned revision;
- bound their token budget;
- include stable entry IDs in rendered text;
- record which entries each model call received;
- preserve the exact rendered system prompt through the existing `prompt_id` table.

### Pinning

A run must not silently switch to the latest playbook when resumed.

Add an immutable revision reference to effective agent configuration or the run artifact:

```text
playbook_name
playbook_revision
playbook_digest
```

Children inherit the pinned revision unless launch configuration explicitly selects another compatible playbook.

### Exposure manifests

For reliable attribution, record the entry IDs selected for each `LLMOutput`.

This can begin as a research sidecar keyed by node ID:

```json
{
  "node_id": "node_...",
  "playbook_revision": "pb_...",
  "entry_ids": ["coding.verify-004", "delegation-011"]
}
```

Do not parse IDs back out of prompt text as the long-term interface.

If the feature graduates into core, a prompt manifest table beside `AgentStart.system_prompts` is cleaner than adding a large list to every Node.

## Versioning and Storage

The shared playbook spans runs, so store it separately:

```text
playbooks/coding/
  revisions/
    pb_<digest>.json
  current.json
```

Each revision is immutable. `current.json` points to the promoted revision.

A completed run records the revision it used. A loaded unfinished run resumes with the pinned revision, not `current`.

Every update produces:

```text
old revision
delta
new revision
source graph IDs
curation metadata
```

This gives rollback and auditability without rewriting historical run graphs.

## Adaptation Timescales

### Stage 1: between completed runs

Build this first:

```text
Run A uses playbook v7
  -> graph completes
  -> external feedback arrives
  -> bottom-up reflection
  -> curation
  -> playbook v8

Run B uses playbook v8
```

This keeps one run internally consistent and matches the strongest evidence from ACE's offline and sequential online settings.

### Stage 2: between terminal child episodes

A long-running root could curate a completed child's lesson before launching a later child.

This requires a pinned per-agent or per-launch context snapshot and explicit promotion. It must not mutate already-running siblings.

### Stage 3: within one agent transcript

Only if earlier stages show value, add a durable context event:

```python
@dataclass
class ContextDelta(Node):
    additions: list[dict[str, object]]
    evidence_node_ids: list[str]
    base_revision: str
    result_revision: str
```

The node path could be:

```text
ExecAction
  -> ContextDelta
  -> PlanQuery
  -> LLMOutput under the new local context
```

Message building folds local deltas before the next model call.

This is event-sourced context adaptation, not transition mutation.

Do not make the solving agent directly and permanently apply arbitrary context. A Reflector proposes; a Curator validates; the normalized delta is what becomes durable.

## Why Not Store Every Lesson as a Graph Branch?

The playbook is shared across runs, while the execution tree belongs to one run. Making every cross-run lesson a child Node would create ambiguous ownership:

- Which run owns a lesson supported by ten runs?
- Is pruning a historical graph rewrite?
- Does loading one run load the current playbook?
- Does forking a run fork global learning history?

Keep the artifacts linked but separate:

```text
run graph          immutable execution evidence
playbook history   immutable learning revisions
provenance refs    links between them
```

Within-run `ContextDelta` nodes are different because their effect is branch-local and belongs to that run's history.

## Why Not Mutate Transitions?

ACE changes what the model knows and which strategies it sees. It does not need to change the engine's legal node transitions.

Transition mutation would introduce:

- a second mutable graph definition;
- replay and versioning problems;
- races between concurrent children;
- a model-authored workflow DSL;
- weakly evaluated state and edge churn;
- overlap with ordinary autoregressive planning.

Context evolution reaches the model through `build_messages`, where behavior is already shaped.

Keep `transition("name")` only for host-defined modes that materially change the execution contract, such as tools, system prompt behavior, finish contract, or allowed routes.

## Relationship to the Research Landscape

### ACE

ACE supplies the central playbook design:

- itemized context;
- Generator/Reflector/Curator separation;
- incremental deltas;
- deterministic merge;
- grow-and-refine maintenance;
- adaptation from execution feedback.

The graph-native addition proposed here is hierarchical reflection and scoped promotion over recursive agent episodes.

### Reflexion and adaptive memory

Reflexion-style systems preserve verbal lessons from prior failures. Dynamic Cheatsheet, A-MEM, and Agent Workflow Memory similarly treat external memory as a way to accumulate reusable strategies or workflows.

The relevant lesson is that memory needs retrieval, scope, and maintenance. Merely appending every reflection to every prompt eventually becomes noise.

### LangGraph and Temporal

LangGraph's command model separates state updates from routing over declared nodes. Temporal's event history requires decisions affecting replay to be durable.

Together they support:

- context changes as explicit state updates;
- runtime routing as a separate concern;
- immutable history plus deterministic reduction.

### LATS, AFlow, GPTSwarm, and ADAS

These systems make improvement meaningful by evaluating alternatives:

- LATS scores alternative execution trajectories;
- AFlow searches over complete workflow candidates;
- GPTSwarm optimizes graph connectivity against utility;
- ADAS evaluates generated agent implementations and retains an archive.

They warn against calling unaudited self-editing "evolution." Graph provenance helps, but promotion still needs outcomes and comparison.

## Evaluation

Compare:

```text
A. static prompt
B. monolithic prompt rewrite
C. flat ACE-style playbook
D. graph-grounded bottom-up playbook
```

Use the same models, task order, and budgets.

Measure:

- task success;
- model tokens and adaptation tokens;
- wall time;
- playbook size and retrieval size;
- lessons proposed, accepted, merged, and rejected;
- helpful/harmful counters;
- unsupported or invalid evidence references;
- repeated lessons across sibling agents;
- transfer across task families and prompt profiles;
- degradation under noisy feedback;
- performance after rollback to earlier revisions.

The graph-native design earns its complexity only if bottom-up scoped reflection outperforms reflecting on a flat root transcript or whole-tree dump.

## Failure Modes

### Reflection pollution

A confident but incorrect lesson enters future prompts.

Mitigations:

- require reliable feedback;
- keep candidates separate from active entries;
- use harmful counters;
- cap promotion scope;
- retain rollback;
- compare against held-out runs.

### Overgeneralization

A leaf-specific workaround becomes global policy.

Mitigations:

- local-first scope;
- explicit promotion;
- task/profile tags;
- recurring evidence from multiple graphs.

### Context growth

The playbook becomes a second unbounded transcript.

Mitigations:

- token budgets;
- retrieval;
- deduplication;
- merge semantically equivalent entries;
- prune repeatedly harmful or obsolete entries;
- preserve archive history outside active context.

### False attribution

An entry is marked helpful because it happened to be present in a successful run.

Mitigations:

- exposure manifests;
- fork-based comparisons;
- repeated outcomes;
- controlled ablations;
- conservative counters.

### Reflection cost

Reflecting on every agent costs more than the improvement.

Mitigations:

- reflect only terminal or anomalous episodes;
- skip agents without feedback;
- use one-shot calls for short traces;
- batch sibling reflections;
- compare adaptation cost with later savings.

### Concurrent context drift

Children in one run see inconsistent live playbook revisions.

Mitigations:

- pin revisions per agent;
- update between runs first;
- never mutate already-running agents;
- make promotion explicit and atomic.

## Implementation Plan

### Phase 0: offline graph exporter

Add a research-layer representation:

```python
@dataclass(frozen=True)
class AgentEpisode:
    graph_id: str
    agent_id: str
    parent_agent_id: str | None
    depth: int
    query: str
    transcript: list[dict[str, object]]
    direct_children: tuple[str, ...]
    result: object
    feedback: object
    usage: object
```

Export episodes from a completed tree in post-order.

No core engine changes.

### Phase 1: playbook and deterministic deltas

Implement outside `Flow`:

```text
Playbook
PlaybookEntry
PlaybookDelta
apply_delta
revision digest
JSON persistence
token-budgeted rendering
```

Unit-test merge, deduplication, revision stability, and rollback without any model calls.

### Phase 2: reflector and curator

Add model adapters that:

- reflect on one `AgentEpisode`;
- cite bounded evidence supplied by the host;
- curate child lessons bottom-up;
- emit schema-validated candidates and deltas.

Start with hard test or benchmark feedback.

### Phase 3: prompt integration

Add a playbook-aware prompt source and pin a revision to each run.

Record prompt exposure manifests beside saved runs.

### Phase 4: sequential adaptation benchmark

Run tasks in a fixed sequence:

```text
predict with current playbook
receive feedback
reflect and curate
advance revision
```

Compare static, monolithic rewrite, flat ACE, and graph-aware variants.

### Phase 5: optional run-local deltas

Only after cross-run value is established, prototype `ContextDelta` and branch-local inheritance.

Do not make `Flow.transitions` mutable. If `ContextDelta` needs a new producer, register that route as ordinary host-owned static policy.

## Test Plan

### Episode extraction

1. Every `AgentStart` produces exactly one local episode.
1. Local transcripts exclude descendant transcript nodes.
1. Direct child identities and results are preserved.
1. Post-order extraction places descendants before supervisors.
1. Incomplete agents are rejected or explicitly marked incomplete.

### Provenance

1. Every source graph, agent, and node ID resolves.
1. A lesson cannot cite a node in an unrelated graph.
1. Saved exposure manifests refer to the correct `LLMOutput`.
1. Prompt IDs still reconstruct the exact rendered system prompt.

### Playbook

1. Applying one normalized delta is deterministic.
1. Unchanged entries retain their IDs and content.
1. Duplicate entries merge without dropping provenance.
1. Token and entry limits are enforced.
1. A prior revision can be loaded after later promotion.

### Scoping

1. Root/profile guidance reaches intended agents.
1. Child-local lessons do not reach siblings.
1. Promotion changes only the declared scope.
1. A pinned run does not switch when `current` advances.
1. A resumed run uses its recorded revision.

### Evaluation safety

1. Missing feedback produces no automatic promotion.
1. Failed or malformed reflection does not change the playbook.
1. Noisy-feedback experiments can roll back.
1. Helpful/harmful counters are updated only for exposed entries.

## Open Questions

- Is the main playbook keyed by prompt profile, task family, repository, user, or some combination?
- What is the minimum reliable feedback for coding runs: test status, user acceptance, diff review, or all three?
- Should leaf reflections be produced by the same model or a dedicated reflector model?
- How much child transcript detail should a parent-level curator receive?
- Can explicit result retrieval and handoff markers support useful attribution, or is controlled ablation required?
- Should playbook revisions live in `rlmflow.rao`, a new research package, or an external project until validated?
- When does a lesson become stale after tools, prompts, or repository code change?
- Is an in-run `ContextDelta` worth the replay and concurrency complexity?

## References

- Zhang et al., [Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models](https://arxiv.org/abs/2510.04618).
- ACE authors, [reference implementation](https://github.com/ace-agent/ace).
- LangGraph, [Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api).
- Temporal, [Workflow definitions and determinism](https://docs.temporal.io/workflow-definition).
- Zhou et al., [Language Agent Tree Search](https://arxiv.org/abs/2310.04406).
- Zhang et al., [AFlow: Automating Agentic Workflow Generation](https://arxiv.org/abs/2410.10762).
- Zhuge et al., [Language Agents as Optimizable Graphs](https://arxiv.org/abs/2402.16823).
- Hu et al., [Automated Design of Agentic Systems](https://arxiv.org/abs/2408.08435).

## Recommendation

Treat rlmflow honestly as a dynamically constructed recursive execution tree, not a mutable workflow DAG.

Its current graph value is real: ownership, durable state, context boundaries, control, restoration, forking, and structured observability. But those capabilities do not automatically produce learning.

The strongest middle ground is:

```text
keep transition policy static;
use the graph as hierarchical experience;
evolve a versioned playbook from reliable feedback;
scope and promote lessons according to tree structure;
inject the playbook through existing message building;
evaluate every claimed improvement.
```

Build cross-run, bottom-up context evolution first. Add branch-local `ContextDelta` nodes only if the evidence shows that same-run adaptation is worth the additional semantics.
