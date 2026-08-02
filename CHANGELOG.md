# Changelog

All notable changes to **rlmflow** are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the project is on `0.x`, breaking changes can land on minor bumps —
each one is called out under **Breaking** below.

## [Unreleased]

### Breaking

- **One engine, and it is the package.** The rewrite that lived under
  `rlmflow.minimal` is now `rlmflow` itself: `rlmflow.graph` holds the tree and
  its on-disk format (`nodes.py`, `persistence.py`), `rlmflow.flow` holds `Flow`,
  and `rlmflow.engine` holds what it is built out of (`execution.py`,
  `boundaries.py`, `parallel.py`).
  The engine they replace is deleted along with the old `rlmflow.graph` and
  `rlmflow.minimal`. Every name is still re-exported from `rlmflow` directly,
  which is where callers should import it from. The node API it brings with it: `agent.frontier` for
  `tail()`, `node.append(child)` for `attach`/`submit`/`spawn`, `agent.result()`
  for `agent_result()`, `agent.terminal` for `finished()`, `agent.transcript()`
  with no agent-id argument, and `node.walk(reverse=True)` for reading one
  agent's chain backwards. `Flow` no longer takes `max_iters`, `max_depth`, or
  `child_max_iters`: per-agent limits belong to `AgentConfig`, so pass them to
  `start(query, ...)` or as `config=` for the flow's own defaults.
- **Node streaming.** `run_streaming(root_or_query, until=...)` yields affected
  Nodes, not Event wrappers.
- **Forking is the whole branch API.** `node.fork()` copies the tree and cuts
  everything after that node; `rlmflow.surgery` and its mid-transcript
  `insert`/`remove` are gone, as are `Flow.rewind`, `Flow.launch_branches`, and
  `Flow.discard`. An edit lands as an ordinary node appended to a frontier, which
  `Node.append` enforces — there is no way to rewrite history in place.
- **Deleted with no replacement.** The legacy Graph/Event viewer implementation,
  both CLI entry points, `LLMChannel` (bounding concurrency is
  `Flow(workers=...)` or a `Pool`), and the injection helpers `flow.next_query`,
  `flow.inject_tools`, `flow.register_halt`, and `flow.append` — appending a
  `UserQuery` to a frontier, `flow.add_tool`/`flow.inject`, and an `until`
  callable cover what they did.
- **A delegating step writes one node, like every other step.** Children hang off
  the `ExecAction` that launched them, so `SupervisingOutput` is gone from the node
  model and a delegating turn records a single `ExecOutput` holding everything its
  block printed — including what printed before the launch, which used to be a
  separate node that escaped `max_output_length`. Runs saved with the old node still
  load, reading it back as an `ExecOutput`. With the frontier no longer moving
  mid-step, work in flight is keyed by the node being stepped instead of by its
  agent. `waiting_on` is read off the tree: an `ExecAction`'s `AgentStart` children.

- **`Repl.run` returns a `ReplRun`, not a string.** One value carries how a block
  ended — `output` plus a `ReplStatus` of `OK`, `ERROR`, `DONE`, or `DEAD`, the
  answer from `done(...)`, and the exception that killed the REPL — instead of a
  string plus the `errored` and `done_result` attributes a caller had to reset and
  read around it. `Runtime.execute(node, code)` adds the `DEAD` case: a REPL that
  dies is an observation for the agent and an eviction from the runtime, so no
  caller handles that exception any more. A dead REPL is now told apart from code
  that raised: its `ErrorOutput` carries `error="repl"` and appends the cold-REPL
  note, since the agent's namespace went with it.

### Fixed

- **Node-native visualization and stream consumers.** `LiveTreeRenderer` restores
  compact terminal redraws, `LiveGraphTree` restores the Rich live forest, and
  `FlowTUI` restores the Textual chat/dashboard with overview, tree, agent,
  count, waiting, error, and latest-activity panels. `GraphCheckpointer` restores
  periodic/final saving. They consume Nodes directly through
  `ConsumerGroup.handle(node)` and are imported from `rlmflow.consumers`.
- **`PromptProfile(user=fn)` takes a bare function again**, as `as_user_prompt`
  always claimed it did. The flow normalized its own `user_prompt=` at
  construction but handed a profile's straight to `.build()`, so a plain
  `(flow, node) -> str | None` raised `AttributeError: 'function' object has no
  attribute 'project'` on the first turn. Both call sites now resolve it through
  `Flow.user_builder(agent)`.
- **Two roots on one flow no longer deadlock.** `parallel_run`/`parallel_stream`
  drive several `run_streaming` loops over one shared queue, and each loop took
  whichever step had landed — including the other's, which then waited forever for a
  step nothing was tracking any more. Both loops also queued behind a single wake
  future, so only the last one to ask was ever woken. `TaskQueue.settle` now takes
  the same predicate its caller filters pending work by, and every waiter is woken
  to look.
- **Resuming a saved run rebuilds its live REPL.** `persistence.load` followed by
  a run previously started with an empty namespace, so an agent's own variables
  raised `NameError` while its transcript said it had set them — and it retried
  until it hit an iteration cap. A fork had the same problem. A graph a `Flow` has no
  REPLs for is now replayed once before the run loop starts. Use
  `Flow(restore="lazy")` to keep the old behaviour, which now also appends a note
  to the transcript so the agent knows to re-derive its variables.
- **Resuming a finished run returns its answer.** Whether an agent had answered was
  read from the task queue, which only knows about steps taken in the current run, so
  a loaded terminal agent was submitted for another step and raised `TypeError:
  cannot step DoneOutput`. `submit_leaves` now reads it from the frontier, and
  consults the queue only for the case a graph cannot show: a step that raised.
- **`max_iters` and `max_budget` survive a save.** Both are recorded on the root
  `UserQuery` and preferred over the running `Flow`'s defaults, so resuming a run
  in another process no longer silently changes its limits.
- **A token budget is charged to the agent that spent it.** `budget_exceeded` now
  measures an agent's own subtree against the limit that agent recorded, instead
  of measuring the whole trajectory and terminating whichever agent happened to
  step next — which meant a frugal agent could be killed for a sibling's spend,
  chosen by scheduling order. A root budget still bounds the whole run.

### Added

- **`Flow(restore="replay" | "lazy")`** — how a graph this flow did not run gets its
  live state back. `Flow.replay` re-runs each agent's recorded `ExecAction`s;
  `Flow.note_cold` is the cheap alternative that tells unfinished agents their
  namespace is gone. Both hang off one check in `run_streaming`, so load, fork, and
  cross-process resume share a path.
- **`RLMFLOW_REPLAY`** — `"1"` in an agent's `ENV` while its recorded code is being
  re-run, `"0"` during a live turn. `launch_subagents` reads it and returns the
  answers its children already gave rather than launching again; agent code can read
  it to skip downloads, writes, or anything else it should not repeat.
- **`AgentStart.save(path)` / `AgentStart.load(path)`** — the version 2 run
  directory, unchanged: `graph.json` and `latest.json` beside an `agents/` tree of
  `agent.json` + `session.jsonl`. A re-save prunes agent directories that are no
  longer in the tree, timing is recorded per step, each turn's system prompt is
  kept once per agent and referenced by id, and a structured answer is stored
  parsed rather than as the string the model emitted.
- **Long-running / multi-turn agents.** Append a `UserQuery` to a finished root's
  frontier and stream it again: history and the warm REPL are still there.
- **`node.tokens()`** — what every model call from a node down cost, as an
  `LLMUsage` that adds and reports a `total`. The budget check reads it too.
- **Named `PromptProfile`s** — `Flow(prompt_profiles=..., prompt_router=...)`
  plus per-agent `UserQuery.prompt_profile` / spawn-spec `prompt_profile`.
- **Structured execution module.** `Pool`, `ThreadPool`, `SequentialPool`, and
  the small structured-concurrency `TaskQueue` live in `rlmflow.engine.execution`.
- **One pooled leaf queue per driver.** `run_streaming` seeds one or more roots'
  current leaves, yields each completed `Transition.created`, checks that root's
  `until`, and resubmits the node when its agent is not terminal.
  `parallel_stream` uses this same driver once for all roots rather than merging
  competing stream consumers. Delegation submits children to the active queue and
  uses one run-scoped condition for terminal wake-ups. Tree-wide diffs, `settle`,
  `landing`, cross-root queue predicates, and repeated leaf scans are gone.
- **`Flow.step` is a total transition.** It returns
  `Transition(submitted, created, error)`, converting infrastructure failure into a
  terminal graph node while allowing cancellation to unwind. `step_task` and
  queue-side exception inspection are gone; a failed root is still re-raised to
  its stream caller.
- **Pool orchestration is separate from compute.** `Pool.run` drives lightweight
  transition coroutines without consuming a bounded compute slot while a parent
  awaits children; `Pool.call` continues to place and bound blocking compute.
  `Flow(workers=...)` and custom pools retain their existing public meaning.
- **`Flow(llm_request_timeout=...)`** — bounds every model call. The client's own
  `timeout` is set too, where it takes one: cancelling a blocking call frees the
  caller but not the thread, so only the client can end the request.
- **`Flow(use_llm_query=True)`** — puts `llm_query_batched` in the agent's REPL for
  independent one-shot prompts, which need no trajectory, no REPL, and no
  delegation. Off by default, so an agent is never offered a tool its prompt did
  not describe; `flow.llm_query_batched` calls it from host code either way.

### Changed

- **Prompt selection is explicit.** Spawn specs use `prompt_profile`; children
  inherit their immediate parent's profile when omitted. `prompt_router` is an
  optional callable, and unknown profile names raise `ValueError`.
- **Runtime owns REPL lifecycle.** REPL lookup, replay, closure, and rebinding
  live on `Runtime`; warm branch attachment uses `Runtime.rebind_repl`.
- **Persistence and consumers accept Nodes directly.** Save/load uses
  `rlmflow.graph.persistence`; stream consumers handle one Node at a time.

### Removed

- `rlmflow.pool`, `rlmflow.tasks`, `rlmflow.simple_flow`, the Event hierarchy,
  and the separate `Graph` wrapper.

## [0.4.0] — 2026-06-12

### Added

- **Single delegation surface: `launch_subagents`.** Agents now delegate
  through one `async` launcher installed in the REPL namespace.
  `await launch_subagents(specs)` takes a `list[dict]` only; each spec requires
  `query` and may set `num_steps`, `context`, `name`, and `model`. It returns a
  `list[str]` in spec order, even for a one-item list. Sequential pipelines use
  one-item calls and thread each result into the next spec's `context`;
  parallel fanout uses a multi-item list. The launcher is registered as a real
  core tool and composes over `flow_delegate` / `flow_wait`, so it behaves
  identically on local and remote runtimes.
- **Shared LLM scheduler channel.** All agent LLM turns and
  `llm_query_batched(...)` calls now route through one per-run `LLMChannel`,
  keyed by model/client. `FlowConfig.llm_max_concurrency` controls the global
  LLM request cap; unsafe clients are serialized behind a per-client lock, and
  safe clients can run concurrently within the channel limit. Batched one-shot
  calls now also accept common sampling kwargs: `temperature`, `top_p`,
  `max_tokens`, and `stop`.
- **Thread-safe per-request usage accounting.** `LLMClient.completion(...)`
  returns `(text, LLMUsage)` for each request. `OpenAIClient` and
  `AnthropicClient` implement it directly from provider response usage, so the
  engine no longer depends on racy shared `last_usage` reads under nested
  batching.
- **Example smoke runner.** `python examples/run_examples.py` runs the
  deterministic/offline example suite by default, with opt-in flags for
  optional dependencies, live LLM examples, sandbox providers, and
  manual viewer/interactive checks.
- **Tinker inference client.** `TinkerClient` adapts the Tinker SDK and
  `tinker-cookbook` renderers to the `LLMClient` interface, with an optional
  `rlmflow[tinker]` extra and a live-view example in `examples/providers/tinker_agent.py`.
- **Stricter local install checks.** `make install` now runs `ruff check .`
  through the existing lint target before installing the package.
- **Supervisor injection example.** `examples/control/injection/` generates a
  real word-search run, forks it, replaces supervising nodes with prompt-based
  graph edits, validates structured results, and continues with a live LLM.
- **Simpler iteration defaults.** `FlowConfig.max_iterations` is unbounded by
  default (`None`), while delegated children use the global
  `child_max_iterations=20` engine policy by default.
- **Direct child-writes-file prompt pattern.** The default prompt's multi-file
  fanout example no longer teaches a `PATH: <path>` answer header. It passes
  target paths through child `CONTEXT` and demonstrates plain Python
  `pathlib.Path(...).write_text(...)` writes, either in the parent or directly
  inside child agents.

### Breaking

- **`flow_delegate` / `flow_wait` are now internal primitives, and
  `launch_subagent` has been removed.** Agent code, the default prompt,
  examples, and docs all use `launch_subagents([...])` instead. The AST check
  (`check_wait_syntax`) now permits `await launch_subagents(...)` at
  action-block top level. Update any custom prompts or hand-scripted REPL
  fixtures.
- **`OrphanedDelegatesError` removed.** Because spawn and wait are fused inside
  the launchers, an un-awaited delegate is no longer expressible, so the
  orphaned-delegate detection, its `ErrorOutput(error="orphaned_delegates")`
  node, and the remote exception-injection path are gone.
- **`FlowConfig.async_children` renamed to `eager_children`.** Same semantics
  (work-conserving child drain once a parent is supervising); update config
  literals and any persisted `agent.json` fixtures.
- **`Node.injected` and `Node.injected_reason` removed.** Injected controller
  nodes are now stored with the exact same schema as ordinary graph states.
  `Graph.inject(...)` no longer accepts or persists a reason string; external
  controllers that need provenance should track it outside the node payload.

## [0.3.2] — 2026-05-28

### Added

- **Per-agent LLM transcript log.** Every workspace now writes
  `session/<aid>/transcript.json` — a *single* document per agent
  that grows turn-by-turn. `messages` is the flat conversation as
  the LLM saw it across every turn so far
  (`[{role: system|user|assistant, content: ...}, ...]`);
  `metadata` is a parallel list with one dict per message. Each
  call appends only the new entries (any user nudges plus the
  assistant reply) — no duplicated prefix. Per-assistant metadata:
  `{ts, model, force_final, input_tokens, output_tokens,
  elapsed_s, after_node_id, after_seq}`. Every other message gets
  `{}`. Exact ground truth for "what did the LLM see?" — useful
  for debugging prompt issues, replaying a turn under a different
  model, or auditing context bloat. The `Session` ABC gains
  `read_transcript(agent_id)` and `write_transcript(agent_id,
  transcript)`; `FileSession` round-trips through `transcript.json`,
  `InMemorySession` keeps it in `agent_transcripts`. Transcript
  read/write failures are swallowed so persistence issues never
  break a run.

### Breaking

- **`delegate` / `wait` renamed to `flow_delegate` / `flow_wait`.** The
  two engine-bound REPL tools used names that were too generic and
  shadowed common identifiers in agent code. They are now namespaced as
  `flow_delegate(*, name, query, context, max_iterations=None,
  model="default")` and `flow_wait(*handles)`. The default system prompt,
  built-in examples, error messages (`OrphanedDelegatesError`,
  refusal strings), and the AST check that enforces `yield` before
  `wait` all use the new names. Update agent prompts, custom prompt
  builders, and any test fixtures that script REPL code by hand.
- **Node taxonomy expanded to a 9-leaf, 4-base-class hierarchy with
  strict obs → action alternation.** Every action is now followed by
  exactly one observation; outputs no longer share a node with the
  action that produced them. New leaf classes (and wire-format
  `type` tags): `UserQuery`, `LLMAction`, `LLMOutput`, `ExecAction`,
  `ExecOutput`, `SupervisingOutput`, `ErrorOutput`, `DoneOutput`,
  `ResumeAction`. Base classes (Python-only — not on the wire):
  `Node`, `ObservationNode`, `ActionNode`, `CodeObservation`. The
  old `outcome=` enum on `ExecAction`, the unified `ActionNode`
  (LLM-call) shape, and `SeedAction` / `ResultNode` / `ErrorNode` /
  `QueryNode` / `SupervisingNode` / `ResumeNode` are gone. Predicates
  follow the new taxonomy: `is_user_query`, `is_llm_output`,
  `is_llm_action`, `is_exec_output`, `is_exec_action`,
  `is_supervising`, `is_errored`, `is_done`, `is_resume_action`,
  `is_resumed`, `is_observation`, `is_action`, `is_code_observation`.
  `LLMOutput.code` is the source of truth for executed code;
  `ExecAction` / `ResumeAction` carry an optional echo only. See
  [`docs/node_model.md`](docs/node_model.md).
- **`ErrorOutput` (formerly `ErrorNode`) now distinguishes runtime
  exceptions from normal `ExecOutput`.** The REPL protocol surfaces
  an `errored` flag; `engine/transitions.py` writes `ErrorOutput`
  whenever the runtime reports an exception (including `SyntaxError`
  and the synthetic "no code block" case), instead of mixing
  tracebacks into `ExecOutput`.
- **LLM clients retry transient failures via `tenacity`.** The
  `chat` and `stream` methods on `OpenAIClient` and
  `AnthropicClient` retry on transient HTTP / protocol errors. The
  module-level `_`-prefixed helpers and constants in `rlmflow/llm.py`
  are now public.
- **Workspace step retracing.** `Workspace.load_steps()` returns the
  full history as a list of progressive `Graph` snapshots. The
  retrace simulates **unbounded `max_concurrency`**: every tick
  advances all currently ready agents in lockstep, producing one
  snapshot per tick. The viewer / `save_steps` / `save_gif` /
  `save_html` / `open_viewer` deduplicate consecutive frames that
  collapse to the same visualization (e.g. action nodes hidden by
  their paired observation), so the resulting slider/animation
  shows only visually distinct steps.
- **Viewer renames + node collapsing.** "Yielded" is now
  "supervising" everywhere in display surfaces. By default the
  figure renderer hides bookkeeping action nodes whose paired
  observation has already been written: `llm_action` collapses
  into `llm_output`, `exec_action` / `resume_action` collapse into
  `exec_output` or `supervising_output`. Terminal outcomes
  (`done_output`, `error_output`) are **never** collapsed — the
  preceding action stays visible so `... → exec → done` and
  `... → resume → errored` read explicitly. Action nodes that are
  the latest state on an agent also stay visible so progress is
  observable. The state-detail panel in `open_viewer` renders each
  state as a distinct, color-coded, type-labeled block.
- **Data model is now one recursive class.** `AgentMeta` is gone — its
  fields are flat on `Graph` itself (`graph.query`, `graph.config`,
  `graph.runtime`, `graph.depth`, `graph.model`,
  `graph.system_prompt`, `graph.parent_agent_id`,
  `graph.parent_node_id`). `Graph` is a frozen `dataclass` with
  `states: tuple[Node, ...]` and `children: dict[str, Graph]` for
  sub-agents. Cross-agent navigation is `graph[other_aid]`;
  subtree views are `graph.agents`, `graph.all_nodes`, `graph.edges`.
- `Graph.from_agent_states(...)` is removed. Build `Graph` instances
  directly (frozen dataclass) or rely on `Session.load_graph()`.
- `Edge` no longer ships as a stored object on `Graph` — `graph.edges`
  derives `flows_to` from each agent's state order and `spawns` from
  each child's `parent_node_id`. The class survives as a `NamedTuple`
  for viz consumers.
- `Session.write_agent` now takes a `Graph` (not an `AgentMeta`).
  `Session.record_spawn` is removed; the parent link is captured on
  the child's `parent_node_id` field.
- `Graph.events` is now `Graph.nodes` — every `Node` represents the
  agent's *state* at one step in its trajectory, not a discrete event.
- `latest.json` writes `latest_node_id` instead of `latest_event_id`.

## [0.2.1] — 2026-05-10

### Changed

- Workspace persistence now uses per-call `session/<agent-id>/session.jsonl`
  logs plus a top-level `graph.json` manifest for graph structure and state
  ordering.
- Removed old workspace compatibility paths; `FileSession(path)` and
  `FileContext(path)` now treat `path` as the current workspace root layout.
- Removed the redundant `CONTEXT.fork()` REPL helper; pass `CONTEXT.read()` or
  a slice explicitly to `delegate(...)`.
- Added public prompt customization docs covering `PromptBuilder`,
  `FlowConfig.system_prompt`, and dynamic prompt overrides.

## [0.2.0] — 2026-05-08

### Breaking

- `delegate(name, query, context, *, model=...)` — `context` is now
  **mandatory** and **positional**. The previous `context=None` keyword
  default is gone. Pass `""` for code-only delegations. This eliminates a
  silent footgun where children inherited the parent's payload by
  accident; every delegation now declares its child's input explicitly.
  Migration: `delegate("name", "query")` → `delegate("name", "query", "")`.
- `RecursiveFlow.start(query, *, context=None)` — `context` is keyword-only
  and optional (root agent gets `""` if omitted). No call-site changes
  needed for callers passing only a query.

### Added

- "Inline first" strategy bias in the default prompt: when the parent
  can write a known multi-file artifact end-to-end itself, do not
  delegate per-file. Multi-file delegation example replaced with a
  parent-writes-everything-inline example. Sibling-interface guardrail
  added for the cases where delegation IS the right call (children must
  USE sibling names as-is and PRODUCE their own exports in the exact
  shape the contract declares).
- `tests/test_prompt_capabilities.py` — snapshot-style tests that pin
  the default prompt's required vocabulary so future trims don't drop
  load-bearing phrases.
- `tests/test_session_variable.py` — `SessionVariable` tree-navigation
  methods (`parent`, `ancestors`, `children`, `subtree`, `tree`)
  derived from real cross-agent edges.
- `examples/_data/notebook-coding-agent/` — canonical saved trace shared
  by `coding_agent.ipynb` (generator), `node_basics.ipynb` (querying),
  and `viz_walkthrough.ipynb` (rendering).
- CI workflow (`.github/workflows/ci.yml`): ruff + pytest matrix on
  3.11 / 3.12 / 3.13, runs on every PR and `push: main`. Tag-driven
  publishing remains in `release.yml`.
- Coverage instrumentation: `pytest-cov` in `[dev]`, `--cov=rlmflow` in
  CI, `[tool.coverage.*]` config in `pyproject.toml`.
- Early OOLONG benchmark harness for flat-vs-RLM comparison. This has since
  been superseded by the shared `benchmarks/eval/` harness.
- `rlmflow.utils.save_image(node, path, ...)` — render a node's
  graph to PNG/SVG/PDF. Markers, edges, and fonts auto-scale via
  `element_mult` so the tree stays visually balanced on the larger
  export canvas. Promoted from a one-off notebook helper.
- `rlmflow.utils.save_steps(states, dir, ...)` — multi-snapshot
  variant: writes one image per state under `dir`.
- `rlmflow.utils.render_html(states, ...)` /
  `rlmflow.utils.save_html(states, path, ...)` — single-file
  standalone stepper. Each slide pairs the Plotly graph for one
  snapshot with that snapshot's transcript and a node table; bottom
  nav has arrows + dots, plus keyboard left/right. Drop the file in
  a PR comment, attach to a CI artifact, or commit it next to the
  trace it came from. Promoted from
  `examples/blog_needle_graph.py:render_html_viewer`.
- `rlmflow.utils.save_gif(states, path, ...)` — animate a trace as
  an autoplay GIF. Renders each state to PNG with kaleido, then
  stitches frames with Pillow. Lazy-imports Pillow (raises a clear
  ImportError otherwise) so `[image]` stays focused on still
  exports.
- `Node.save_image(path, ...)` and `Node.save_html(path, ...)`
  shorthands for the helpers above.
- `Node.plot(..., element_mult=)` and
  `node_plot(..., element_mult=)` — scale markers/edges/fonts on
  the returned Plotly figure. Default `1.0` keeps the on-screen
  layout; bump for hi-res rendering.
- Split scaling on `node.plot()` / `save_image` / `save_steps` /
  `save_gif`: `marker_mult` and `text_mult` override
  `element_mult` separately, so labels can stay small (e.g. `2.2`)
  while marker dots get fat (`3.5`). Fixes label collisions on
  dense trees.
- `normalize_labels=` on `node.plot()` and the save helpers —
  forces every node label to `bottom center` so adjacent depths
  can't share the same vertical band. Default off for `node.plot`
  (on-screen alternation still looks fine), default on for
  `save_image` / `save_steps` / `save_gif` / `Node.save_image`.
- CLI: `rlmflow render <trace> -f steps -o frames/` gains
  `--marker-mult`, `--text-mult`, `--normalize-labels` /
  `--no-normalize-labels` flags (also work with `-f image`). One
  invocation now replaces the per-blog one-off scripts.
- `[image]` optional extra (`pip install rlmflow[image]`) — pulls
  `plotly` and `kaleido` for static image export.

### Changed

- Default system prompt rewritten end-to-end. Sections reordered to
  capabilities-first (Role → REPL → Strategy → Tools → Context →
  Recursion → Session → Guardrails → Examples → Status). Per-section
  prose tightened (~20% fewer tokens, zero outbound URLs in the
  shipped prompt). Examples reduced to five canonical patterns: small
  task, chunk-and-aggregate, self-contained multi-file (inline),
  cross-agent recovery, reviewer (`CONTEXT.read()`).
- `[viewer]` extra now declares its `plotly` dependency directly. The
  unused `[viz]` extra was removed (`plotly` was previously declared
  there but only imported by the gated `[viewer]` code path).
- Python support clarified: `requires-python = ">=3.11"` matches the
  shipped classifiers (3.10 dropped — never tested in CI). Ruff target
  bumped to `py311`.
- Project status classifier: `Alpha` → `Beta`.

### Fixed

- Boids notebook regression: cross-file schema drift (`Boid.pos.x`
  vs flat `boid.x`) caused by an over-strict guardrail and an
  over-aggressive multi-file-delegation example. Repaired by adding
  the bidirectional contract guardrail and replacing the example.
- Notebook agent ids reflect filename sanitization (`root.index_html`,
  `root.styles_css`, etc.) — `.` is the agent-tree delimiter, so
  filenames with dots are sanitized to underscores. `node_basics.ipynb`
  and `viz_walkthrough.ipynb` updated.
- Static example payloads no longer use the deprecated 2-arg
  `delegate(...)` form (`view_demo.py`, `showcase.py`, `best_of_n.py`).

## [0.1.3] — 2026-04-29

- Engine refactor: graph-first replay path, deterministic stepping
  semantics tightened, additional integration tests.

## [0.1.2] — 2026-04-29

- Renamed package to `rlmflow`. Session and context layout consolidated
  under `Workspace` with explicit `fork()`. Major engine refactor toward
  the typed-node graph model.

## [0.1.1] — 2026-04-23

- `rlmflow` CLI shipped: `view`, `render`, `version` subcommands;
  `render -f` accepts mermaid / mermaid-flowchart / mermaid-sequence /
  dot / d2 / tree / ascii-boxes / gantt-html / report-md / code-log /
  error-summary / tokens.

## [0.1.0] — 2026-04-23

Initial release.

- Recursive `RecursiveFlow` engine with typed nodes (`QueryNode`,
  `ActionNode`, `ObservationNode`, `SupervisingNode`, `ResumeNode`,
  `ResultNode`, `ErrorNode`).
- Runtimes: `LocalRuntime`, `SubprocessRuntime`, `DockerRuntime`,
  `ModalRuntime`.
- `Workspace` with `Session` (event log) and `Context` (data payload)
  stores, both with `fork()`.
- Visualization: terminal `live` view, mermaid / dot / d2 / sequence
  exports, gantt HTML, code-log, error-summary, token sparkline,
  budget burndown, bench table, Markdown report, Slack/Discord
  webhooks, Gradio viewer.
- Optional extras: `[openai]`, `[anthropic]`, `[viewer]`, `[all]`,
  `[dev]`.

[Unreleased]: https://github.com/shyamsn97/rlmflow/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/shyamsn97/rlmflow/compare/v0.3.2...v0.4.0
[0.3.2]: https://github.com/shyamsn97/rlmflow/compare/v0.2.1...v0.3.2
[0.2.1]: https://github.com/shyamsn97/rlmflow/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/shyamsn97/rlmflow/compare/v0.1.3...v0.2.0
[0.1.3]: https://github.com/shyamsn97/rlmflow/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/shyamsn97/rlmflow/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/shyamsn97/rlmflow/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/shyamsn97/rlmflow/releases/tag/v0.1.0
