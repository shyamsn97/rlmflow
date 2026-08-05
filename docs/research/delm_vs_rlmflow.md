# DeLM and rlmflow: Research and Adoption Plan

> **Historical terminology:** this research note predates the Node-only API and
> uses older architecture names. Read `Flow` for `RecursiveFlow`,
> `UserQuery.inputs` / `INPUTS` for
> `context` / `CONTEXT` spawn payloads, and `prompt_profile` for per-agent
> prompt selection. See `docs/control.md` / `docs/internals.md` for the
> current surface.

This note compares DeLM, "Decentralized Language Models with shared context",
against the current `rlmflow` architecture and sketches what it would take to
make DeLM-style coordination a first-class approach in this repo.

Sources:

- DeLM repository: <https://github.com/yuzhenmao/DeLM>
- DeLM paper: <https://arxiv.org/abs/2606.10662>
- Local `rlmflow` architecture docs: `README.md`, `docs/internals.md`,
  `docs/control.md`, and the core implementation under `rlmflow/`.

## Short Answer

It is very feasible to add DeLM-style coordination to `rlmflow`, but it should
not replace the RLM graph model.

DeLM is best understood as a coordination layer: many workers operate
asynchronously over a shared verified context and a shared task queue. RecursiveFlow is
best understood as an execution substrate: each worker is an inspectable,
replayable, forkable recursive execution graph with code execution,
subdelegation, structured outputs, graph surgery, and durable workspaces.

The right integration is therefore:

```text
DeLM-style coordinator
  shared task queue
  shared verified context / lessons
  admission-time verifier
  finalizer

RecursiveFlow workers
  each task is solved by one RecursiveFlow graph
  each graph remains inspectable, forkable, replayable, injectable
  completed worker graphs emit candidate lessons into the shared context
```

That gives us DeLM's cross-worker progress sharing without giving up RecursiveFlow's
main advantage: transparent execution graphs.

Difficulty estimate:

- Prototype: moderate. A credible local prototype can be built as an outer
  orchestrator around existing `RecursiveFlow` workers.
- Production-quality research system: high. The hard parts are verification,
  concurrent shared-state admission, context unfolding, benchmark harnesses, and
  evaluation discipline.
- Full DeLM parity on SWE-bench and long-context QA: high to very high. This is
  more benchmark infrastructure and research iteration than core engine work.

## What DeLM Is Actually Doing

The DeLM paper argues that many multi-agent systems are bottlenecked by a
central orchestrator. In a centralized setup, a parent agent assigns work, waits
for children, receives their outputs, merges them, and decides what to broadcast
next. That creates a serial communication point even if the child work itself is
parallel.

DeLM changes the communication substrate:

- There are multiple parallel agents.
- There is a shared task queue.
- There is a shared context containing compact, verified progress.
- Agents claim tasks asynchronously.
- Agents read the shared context before working.
- Agents write compact candidate updates when they finish.
- Candidate updates are verified before admission.
- Later agents build on admitted progress without waiting for a central parent
  to re-summarize and redistribute it.

The paper's key architectural claim is that useful findings, failed hypotheses,
constraints, and partial solutions should become durable shared state rather
than transient messages routed through a main agent.

For long-context settings, DeLM also adds a coarse-to-fine memory hierarchy:
agents normally see compact gists, but can selectively unfold them into more
detailed summaries or raw evidence. That matters because a shared context that
just grows forever becomes unusable.

For SWE-bench-style coding, DeLM scales across attempts: multiple solver
threads explore in parallel, but they are not fully independent. Each thread can
write compact typed notes into shared lessons, and other threads can reuse those
discoveries. The public README describes each solver thread as owning its own
planner, delegated implementer runs, Docker container, and local memory, while
`SharedLessons` holds cross-thread notes.

## Historical rlmflow Baseline

The architecture evaluated when this note was written centered on a recursive
execution graph. The current implementation represents the same tree directly
with `Node`; see `docs/internals.md` for the live design.

One recursive Node tree contains every agent trajectory. An agent is the
`AgentStart` node that opened it, `node.parent_agent` says which trajectory a
node belongs to, and that agent's `config` carries its query, inputs, model,
prompt profile, and output schema.

The node trajectory is typed:

- `AgentStart`
- `UserQuery`
- `LLMOutput`
- `ExecAction`
- `ExecOutput`
- `AppendChild`
- `ErrorOutput`
- `DoneOutput`

The main public delegation surface is:

```python
results = await launch_subagents([
    {"name": "search", "query": "Find evidence", "inputs": {"chunk": chunk_a}},
    {"name": "verify", "query": "Check the answer", "inputs": {"chunk": chunk_b}},
])
done(combine(results))
```

That await is the central supervision point. The children hang off the
`ExecAction` that launched them, the engine runs them, and the delegating turn
lands as a single `ExecOutput` carrying their results.

The engine loop is explicit and inspectable:

- `Flow.run_streaming` selects runnable leaves from the whole Node tree.
- `TaskQueue.run` owns each structured coroutine batch.
- Transitions append and publish the affected Node directly.
- Consumers and `persistence` write durable per-agent logs and `graph.json`.
- Runtime executes REPL code locally, in a subprocess, Docker, or Modal.
- Fork, replay, continuation, and injection operate on the same tree.

This is not just an agent harness. It is an execution trace system.

## The Core Difference

The difference is not "multi-agent vs. single-agent." Both systems are
multi-agent. The difference is the coordination topology.

RecursiveFlow is tree-structured and supervisor-centered. A parent launches children,
children may launch their own children, and results bubble upward. The graph is
a recursive tree, and every delegation has a clear parent supervision point.

DeLM is blackboard-centered. Workers do not need to route every useful update
through a parent. They coordinate through shared state: task queue plus verified
context. The central object is not a parent graph waiting on children; it is a
problem-level memory and work queue.

This affects everything:

- In RecursiveFlow, child results are primarily private to the parent that awaited
  them, unless the parent chooses to pass them onward.
- In DeLM, admitted lessons are visible to all future workers.
- In RecursiveFlow, scheduling follows the recursive graph and supervision state.
- In DeLM, scheduling follows queue availability and worker capacity.
- In RecursiveFlow, correctness is usually enforced by local verification inside a
  graph, structured outputs, tests, or human graph edits.
- In DeLM, correctness of shared memory requires admission-time verification,
  because a bad shared lesson can poison many workers.
- In RecursiveFlow, a run is naturally explainable as a tree of typed node logs.
- In DeLM, the global process is naturally explainable as task/lesson evolution
  across many local trajectories.

## DeLM vs. RLMs As Research Ideas

DeLM and RLMs are not mutually exclusive research ideas. They optimize different
parts of the system.

RLMs make the agent's reasoning loop programmable. The model writes code, uses a
REPL, delegates through a code-level primitive, waits through top-level await,
and produces an inspectable execution graph. The central question is: how do we
make recursive, tool-using model reasoning controllable, resumable, and
inspectable?

DeLM makes the coordination among many attempts state-based. The agents may be
ordinary planner/implementer loops, RecursiveFlow workers, or some other harness. The
central question is: how do we let many parallel workers share useful progress
without routing everything through a single orchestrator?

So the conceptual split is:

```text
RLM:
  unit of reasoning = one recursive code-mediated trajectory
  main artifact     = typed execution graph
  bottleneck solved = opaque, unstructured, hard-to-control agent loops

DeLM:
  unit of reasoning = many peer trajectories over one problem
  main artifact     = verified shared context + task queue
  bottleneck solved = centralized coordination and duplicated exploration
```

That is why the DeLM paper can report that RLM is stronger on some
aggregation-heavy tasks, while RLM+DeLM is stronger than either alone. RLM is
good at exact local computation and traceability; DeLM is good at sharing
discoveries across local computations.

## Where The Systems Are Complementary

The DeLM paper explicitly positions DeLM as complementary to RLM-style
programmatic reasoning. In its OOLONG discussion, vanilla DeLM underperforms
RLM on aggregation-heavy tasks where code-mediated execution is valuable, while
combining RLM with DeLM gives the best result.

That matches the architecture here. RecursiveFlow already gives workers a strong local
reasoning substrate:

- code execution through the REPL
- stateful variables across turns
- structured `done(...)`
- child delegation through `launch_subagents`
- durable graph snapshots
- fork and replay after crashes or branch edits
- injection for controller or human intervention
- workspace artifacts and sandbox runtimes

DeLM gives a missing outer layer:

- many peer workers for the same top-level problem
- shared lessons across workers
- asynchronous task claiming
- admission-time verification of lessons
- compact global state that later workers read by default
- coarse-to-fine unfolding when a lesson needs detail

So the product direction should be "RecursiveFlow workers inside a DeLM-style
coordinator", not "replace recursive language models with DeLM."

## What "Make This Our Approach" Could Mean

There are three plausible interpretations.

### Option 1: Add DeLM As An Example

Build a runnable example under `examples/control/` or at `examples/<name>.py`
showing several RecursiveFlow workers solving subtasks with a shared notes file.

This is easy, but mostly cosmetic. It would demonstrate the idea but not create
a reusable abstraction.

Effort: small.

Best for: validating prompts and user experience quickly.

### Option 2: Add A Reusable Coordinator Library

Add a new module, likely under `rlmflow/control/` or `rlmflow/coordination/`,
that provides:

- `SharedTaskQueue`
- `SharedContext`
- `SharedLesson`
- `EvidenceRef`
- `AdmissionVerifier`
- `DeLMCoordinator`
- `SolverThread` or `WorkerRun`

Each worker run would be a normal `RecursiveFlow` execution in its own branch or
workspace. The coordinator would assign tasks, inject the current shared context
into each worker, collect completed worker graphs, ask a verifier to admit or
reject lessons, and enqueue follow-up tasks when needed.

This is the most natural approach.

Effort: moderate for a prototype; high for a robust version.

Best for: making DeLM-style coordination an actual supported feature.

### Option 3: Rebuild The Core Engine Around A Global Queue

Make the engine itself no longer tree-first. Every agent becomes a peer worker
over a global queue and shared memory.

This is not recommended. It would fight the current design, weaken graph
clarity, and blur the meaning of `Node.children`, `AgentStart`, `ExecAction`,
replay, and injection. It would also force DeLM semantics onto tasks where a
recursive supervisor is the clearer model.

Effort: very high.

Best for: almost nothing in the current repo.

## Effort Estimate

If "make this our approach" means "add a polished research-grade decentralized
coordination mode", the work is substantial but not a rewrite.

### Prototype Scope

Rough effort: one to two focused weeks.

What we can build quickly:

- a local coordinator that runs N RecursiveFlow workers
- a shared JSONL lesson store
- a simple task queue
- structured worker outputs
- a simple LLM verifier for lesson admission
- one deterministic example
- one live LLM example

What this would prove:

- workers can read shared admitted lessons
- workers can propose new lessons
- lessons can cite graph/node evidence
- later workers can avoid duplicate work

What it would not prove:

- benchmark gains
- robust verifier quality
- multi-process reliability
- long-context unfolding quality

### Library Scope

Rough effort: three to six weeks.

Needed additions:

- durable problem-level run manifest
- atomic task claiming and worker leases
- lesson/admission schemas
- graph and artifact evidence refs
- prompt builders for worker/verifier/finalizer roles
- shared context rendering and compaction
- tests for task claiming, admission, worker failure, restart, and replay
- examples under `examples/control/` or as root scripts / task folders under `examples/`

This is where it becomes a real feature rather than a demo.

### Research-Grade Scope

Rough effort: six to twelve weeks, depending on benchmarks.

Needed additions:

- stronger verification prompts and regeneration loops
- duplicate/contradiction detection
- hierarchical gist/detail/raw context
- unfold tools
- SWE-bench or smaller coding benchmark harness
- OOLONG or long-context benchmark harness
- cost accounting and ablations
- manual audit workflow for verifier false accepts/rejects

This is the level required before claiming that `rlmflow` has meaningfully
adopted DeLM as a research approach.

## Recommended Architecture

Add DeLM as an outer orchestration layer that treats RecursiveFlow as the worker
runtime.

```text
DeLMCoordinator
  owns ProblemRun
  owns SharedTaskQueue
  owns SharedContext
  owns verifier/finalizer prompts
  starts N worker loops

Worker loop
  claim task
  read compact shared context
  run RecursiveFlow task in isolated workspace/branch
  produce structured WorkerResult
  propose one or more SharedLesson entries
  submit entries for verification/admission

SharedContext
  compact admitted lessons
  rejected lessons with reasons
  evidence references to graph ids, node ids, files, test logs, raw chunks
  optional summaries and unfoldable backing content

Verifier
  checks lesson against evidence
  accepts, rejects, or requests rewrite
  emits AdmissionRecord

Finalizer
  runs when queue is empty and stop criteria are met
  produces final answer or patch from admitted context
```

The critical design rule: RecursiveFlow graph logs should remain the source of
evidence, while shared lessons are compact derived state. Do not store only the
lesson and throw away the graph. A lesson should always be able to point back to
the worker graph, node ids, command output, file diffs, or source spans that
support it.

## Minimal Data Model

A minimal implementation needs typed records. Pydantic would fit the existing
structured-output direction.

```python
class SharedTask(BaseModel):
    id: str
    query: str
    context_keys: list[str] = []
    priority: int = 0
    status: Literal["pending", "claimed", "done", "failed"] = "pending"
    claimed_by: str | None = None
    depends_on: list[str] = []


class EvidenceRef(BaseModel):
    worker_id: str
    workspace: str
    agent_id: str | None = None
    node_id: str | None = None
    artifact_path: str | None = None
    description: str


class SharedLesson(BaseModel):
    id: str
    kind: Literal["finding", "failure", "constraint", "patch", "test", "question"]
    gist: str
    details: str | None = None
    confidence: Literal["low", "medium", "high"]
    evidence: list[EvidenceRef]
    created_by: str


class AdmissionRecord(BaseModel):
    lesson_id: str
    accepted: bool
    reason: str
    verifier_model: str | None = None
```

For software-engineering tasks, useful lesson types would include:

- "this file/function is relevant"
- "this hypothesis failed and why"
- "this test reproduces the bug"
- "this patch direction is promising"
- "this dependency/version detail matters"
- "this error was infrastructure noise"

For long-context QA, useful lesson types would include:

- "document X supports claim Y"
- "document A contradicts document B on point Z"
- "this entity/date/statistic is likely relevant"
- "this source is irrelevant for the question"
- "unfold this raw span if reasoning about subclaim Q"

## How This Maps Onto Existing rlmflow Pieces

rlmflow already has several pieces we can reuse.

`GraphCheckpointer` and `WorkspaceSync` store per-worker traces. Each worker can
run in a forked working directory or a sibling directory under one problem
directory. The persisted `graph.json` and per-agent logs provide durable
evidence.

The Node tree gives lesson provenance. A lesson can cite a specific `agent_id`,
Node id, `DoneOutput`, `ErrorOutput`, or `ExecOutput`.

`launch_subagents` remains useful inside a worker. DeLM does not eliminate
recursive decomposition; it changes how top-level solver threads share progress.
A worker can still spawn child agents for local subproblems.

Structured output is directly useful. Worker tasks should often return typed
`WorkerResult` or `LessonProposal` objects rather than free-form prose.

`llm_query_batched(...)` can help with verification and compression. Candidate lessons
can be checked or summarized concurrently through the same client pool.

Fork and injection become more valuable. If one worker finds a promising but
incomplete path, another worker can fork that trajectory and continue from the
same Node state. If a verifier rejects a lesson, we can inject feedback or fork
a repair attempt.

Per-agent `UserQuery.inputs` are not enough by themselves. DeLM needs a
problem-level mutable shared context with concurrency control, admission
records, indexes, and perhaps compaction. It should be its own abstraction.

## Hard Parts

### 1. Admission-Time Verification

This is the most important hard part. DeLM works only if shared context is more
reliable than raw chat messages. Bad lessons are worse than no lessons because
they become global state.

For `rlmflow`, a verifier should check each proposed lesson against:

- the worker's Node tree
- relevant `ExecOutput` / `ErrorOutput`
- file diffs or artifacts
- test logs
- source context chunks
- any structured output schema used by the task

The verifier should produce an `AdmissionRecord`, not just a boolean. Rejected
lessons need reasons so workers can repair or avoid repeating the error.

### 2. Shared Context Growth

A shared context that simply appends notes will degrade. DeLM's gist and
unfolding design is important.

We need at least two layers:

- compact always-visible lesson gists
- detailed backing evidence retrievable on demand

Longer term, we may need hierarchical summaries:

- raw graph/output/file evidence
- detailed lesson summary
- compact gist
- problem-level digest

### 3. Concurrency And Locking

Multiple workers will claim tasks and submit lessons concurrently. We need clear
semantics for:

- atomic task claim
- lease timeout and retry
- idempotent worker completion
- concurrent lesson admission
- deterministic replay of coordinator state

The filesystem store may be good enough for local prototypes, but robust
multi-process use needs explicit lock discipline or a database-backed store.

### 4. Prompt Design

Workers need to read shared context without blindly trusting it. Prompts should
say:

- admitted lessons are useful shared state, not proof by themselves
- unfold evidence when a decision depends on a lesson
- write compact candidate lessons with evidence refs
- avoid duplicating existing lessons
- record negative findings and failed attempts

The finalizer needs a different prompt: synthesize from admitted context and
verify the final answer/patch against primary evidence.

### 5. Evaluation

Without benchmarks, this can become a fancy note-sharing demo. We should test
it in two tracks:

- software-engineering tasks, where shared failed attempts and file discoveries
  should improve pass@N and cost
- long-context QA / OOLONG-style tasks, where shared evidence gists and
  unfolding should improve aggregation and reduce repeated reading

Metrics should include:

- success rate / accuracy
- pass@N
- avg@1
- cost per task
- wall-clock time
- lesson acceptance rate
- rejected lesson rate
- duplicate lesson rate
- verifier false accepts / false rejects from manual audits
- context growth over time

## What A First Prototype Should Do

The first version should be deliberately narrow.

Pick one use case: SWE-bench-like local coding tasks or long-context QA. Do not
try both first.

For a coding prototype:

1. Create one problem workspace.
2. Initialize a shared task queue with N solver tasks like "attempt a fix from
   scratch; read shared lessons first".
3. Start N RecursiveFlow workers, each in its own workspace branch/directory.
4. Give each worker a compact rendering of admitted lessons as context.
5. Require each worker to return structured output:
   - final status
   - patch summary
   - tests run
   - lesson proposals
   - evidence refs
6. Verify each lesson against the worker graph and artifacts.
7. Admit accepted lessons.
8. Optionally enqueue follow-up tasks based on admitted lessons.
9. Finalize by selecting or synthesizing a patch.

This prototype can avoid full dynamic task generation at first. Just proving
that later workers benefit from earlier admitted lessons would be enough.

For a long-context prototype:

1. Split source documents into chunks.
2. Queue chunk-reading tasks.
3. Workers produce evidence gists with refs to raw chunks.
4. Verifier checks gists against chunks.
5. Follow-up workers answer subquestions using admitted gists.
6. Finalizer answers from admitted gists and unfolds raw evidence when needed.

## Suggested Implementation Phases

### Phase 0: Research Doc And API Sketch

Output:

- this document
- a short API sketch
- one chosen benchmark/use case

Risk: low.

### Phase 1: Local DeLM Coordinator Prototype

Add a module that can run multiple RecursiveFlow workers over a shared queue in a
single Python process.

Likely files:

- `rlmflow/coordination/__init__.py`
- `rlmflow/coordination/models.py`
- `rlmflow/coordination/store.py`
- `rlmflow/coordination/coordinator.py`
- `examples/control/shared_context/...`

Features:

- in-memory or filesystem-backed task queue
- shared lesson store
- structured worker result schema
- simple verifier prompt
- example with deterministic fake LLMs first

Risk: moderate.

### Phase 2: Durable Store And Provenance

Make every task, lesson, admission decision, and worker graph durable.

Features:

- problem-level manifest
- atomic task claiming
- worker lease/retry
- evidence refs into graph nodes and artifacts
- viewer-friendly exports

Risk: moderate to high.

### Phase 3: Verification Quality

Improve admission from "LLM says yes/no" to evidence-grounded checks.

Features:

- verifier schemas
- regenerate-on-reject loop
- manual audit hooks
- support for test log/file diff/source-span evidence
- lesson deduplication and contradiction detection

Risk: high.

### Phase 4: Context Hierarchy And Unfolding

Add DeLM-style compact global gists with backing detail.

Features:

- gist/detail/raw levels
- context compaction
- explicit unfold tool
- prompt section rendering compact lessons by default
- retrieval over lesson metadata

Risk: high.

### Phase 5: Benchmarks

Build real eval harnesses.

Features:

- SWE-bench or smaller coding benchmark adapter
- OOLONG / long-context adapter
- pass@N and avg@1 reporting
- cost accounting
- ablations:
  - no shared context
  - unverified shared context
  - no unfolding
  - centralized parent-only sharing
  - RecursiveFlow worker alone
  - DeLM coordinator with RecursiveFlow workers

Risk: high to very high.

## API Sketch

The API should make the coordinator feel like a separate mode, not mutate
`Flow.run(...)` into something ambiguous.

```python
from pathlib import Path

import rlmflow
from rlmflow.coordination import DeLMCoordinator, SharedTask

problem_dir = Path("./runs/my-problem")

worker_factory = lambda workdir: rlmflow.Flow(
    llm,
    runtime=runtime_factory(workdir),
    max_depth=2,
    max_iters=12,
)

coordinator = DeLMCoordinator(
    problem_dir=problem_dir,
    worker_factory=worker_factory,
    verifier=llm_verifier,
    max_workers=4,
)

coordinator.enqueue(
    SharedTask(
        id="attempt-1",
        query="Investigate and fix the bug. Read admitted lessons first.",
    )
)

result = coordinator.run()
print(result.final_answer)
print(result.shared_context.render())
```

Worker prompts would receive a compact shared context:

```text
Admitted shared lessons:
1. [finding/high] The failing behavior is in parser.normalize_path.
   Evidence: worker-2 root ExecOutput n_abc, tests/test_paths.py output.
2. [failure/medium] Changing PathResolver.resolve broke Windows paths.
   Evidence: worker-1 patch attempt and pytest failure.

Use these as starting points, but unfold/check evidence before relying on a
lesson for the final patch.
```

Worker output should be structured:

```python
class WorkerResult(BaseModel):
    status: Literal["solved", "partial", "failed"]
    answer: str | None = None
    patch_summary: str | None = None
    tests_run: list[str] = []
    lesson_proposals: list[SharedLesson]
```

## Implications For The Existing Node Model

We should not force the shared task queue into `Node.children`.

A DeLM problem run is not one recursive graph. It is a collection of worker
graphs plus coordinator state. Some worker graphs may have recursive children,
but the top-level relation among workers is peer-like, not parent-child.

A future viewer could show this as two linked views:

- problem-level timeline: tasks claimed, lessons proposed, lessons admitted,
  finalization
- worker tree view: the existing rlmflow Node tree for each task

That preserves conceptual clarity:

- The Node tree remains "one agent and its recursive descendants".
- `DeLMCoordinator` becomes "one problem-level shared-state run".
- `SharedLesson` links the two.

## Naming

I would avoid calling the module `delm` directly unless we intentionally want to
brand it as an implementation of the paper. Safer names:

- `rlmflow.coordination`
- `rlmflow.shared_context`
- `rlmflow.swarm`

User-facing names could be:

- "shared-context mode"
- "decentralized coordination"
- "lesson-sharing workers"

Internally, the design can still reference DeLM.

## Risks

The biggest product risk is making shared context feel magical. Users need to
understand what was admitted, why it was trusted, and which evidence supports
it. If the shared context is opaque, this moves away from RecursiveFlow's strongest
identity.

The biggest research risk is verifier quality. A bad verifier can either admit
false lessons that poison the run or reject useful discoveries that would have
helped. This needs evals and manual audits.

The biggest engineering risk is concurrency. A naive filesystem queue will be
fine in a single-process demo but fragile under multi-process workers or remote
sandboxes.

The biggest UX risk is context bloat. If every worker sees a long wall of
lessons, performance and cost can get worse. Compact rendering and selective
unfolding should be part of the real design, not a later afterthought.

## Recommendation

Make DeLM-style coordination a first-class optional layer in `rlmflow`, but keep
RecursiveFlow's recursive graph semantics intact.

The best near-term milestone is:

1. Build a local `DeLMCoordinator` prototype around existing `RecursiveFlow` workers.
2. Use structured outputs for worker results and lesson proposals.
3. Store every admitted lesson with graph/node evidence refs.
4. Add a deterministic example and one real LLM example.
5. Add an ablation harness comparing:
   - independent parallel RecursiveFlow attempts
   - centralized parent with `launch_subagents`
   - shared-context workers with verified lessons

If that shows real reuse of discoveries across workers, then invest in durable
queue semantics, context hierarchy, and benchmarks.

The strategic framing should be:

```text
RecursiveFlow makes one agent trajectory transparent and controllable.
DeLM-style coordination makes many trajectories share verified progress.
Together: inspectable decentralized test-time reasoning.
```
