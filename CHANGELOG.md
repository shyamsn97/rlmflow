# Changelog

All notable changes to **rlmflow** are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the project is on `0.x`, breaking changes can land on minor bumps —
each one is called out under **Breaking** below.

## [Unreleased]

### Changed

- **LLM compute is an explicit stream-first primitive.** `PooledLLMClient`
  wraps a raw client, pool, timeout, scheduling key, and request kwargs.
  Production step functions receive that client and consume `stream()`; tests
  can pass a small streaming fake with no Flow or pool. `chat()` collects the
  same stream.
- **Step handlers are classes with one stable ABI.** `StepFunction` subclasses
  receive an `LLMClient`, a `MessageBuilder`, and a `WrappedRuntime`, then
  implement `await step(node)`. The wrapper seeds tools and `INPUTS` immediately
  before execution and exposes the underlying runtime as `.runtime`.
  `LLMRequestStep` owns the complete control-node ordering, while
  `LLMOutputStep` and `ExecActionStep` handle their respective transitions.
  `Pool.run` is gone; pools place compute calls only.

### Breaking

- **`AgentConfig.max_iters` defaults to `None`.** That means no iteration cap
  and no final-answer prod. Pass an integer to restore the old 20-turn limit.

- **Chat projection lives on nodes and has one current-frontier override.**
  `Node.render()` returns zero or more messages. `Node.project()` always uses
  canonical node rendering, flattens multi-message nodes, and bounds its
  backward walk by rendered messages. `Flow.build_messages()` applies
  `render_fn(runtime, node)` only to the current frontier, projects history from
  `node.prev`, and preserves every rendered message without coalescing adjacent
  roles. `UserPromptBuilder`, `user_prompt=`, `PromptProfile.user`, the separate
  user-injection path, and `Flow.messages()` are gone.

- **Engine nudges are nodes.** Inspect, plan, final-answer, continue,
  truncation, and a dead REPL are `InspectQuery`, `PlanQuery`, `FinalQuery`,
  `ContinueQuery`, `TruncationSummary`, and `ReplDead`. `Flow.step` looks up
  `StepFunction` classes in `DEFAULT_STEPS` (override with `update_step_fn`).

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
  `flow.start(query, ...)` for one root, or as `Flow(root_config=...)` to set the defaults
  every root from `flow.start(...)` (and every string handed to `flow.run`) picks
  up. A root built by the bare `start(...)` carries `AgentConfig`'s own defaults —
  it has no flow to inherit from.
- **Node streaming.** `run_streaming(root_or_query, until=...)` yields affected
  Nodes, not Event wrappers.
- **Forking is the whole branch API.** `node.fork()` copies the tree and cuts
  everything after that node; `rlmflow.surgery` and its mid-transcript
  `insert`/`remove` are gone, as are `Flow.rewind`, `Flow.launch_branches`, and
  `Flow.discard`. An edit lands as an ordinary node appended to a frontier, which
  `Node.append` enforces — there is no way to rewrite history in place.
- **Deleted with no replacement.** `LLMChannel` (bounding concurrency is
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

- **Explicit subagent model selection.** Every `launch_subagent` call requires a
  registered `model=` key. The system prompt always lists available keys and
  marks the current one, so the agent—not a hidden host fallback—chooses the
  appropriate model for each workstream. Saved action code from older runs that
  omitted `model` must be regenerated before replay.
- **`rlmflow run` / `rlmflow tui`, and a CLI that is more than one command.**
  Every command is a Fire class: flags on the constructor, verbs as methods.
  `rlmflow tui` starts a coding agent over `--workdir` in the Textual dashboard,
  checkpointing to `<workdir>/graph`; `rlmflow run tui "fix the failing test"`
  opens it with that turn already going, and `rlmflow run print` runs the same
  thing headless, streaming the tree and printing the answer. `--model`,
  `--fast-model`, `--reasoning-effort`, `--docker-image`, `--max-depth`,
  `--max-iters`, `--workers`, and `--tools {files,none}` pick the pieces;
  `--resume DIR` continues a saved run and `--agent module:factory` swaps in
  your own `Flow`. Settings resolve flags first, then `RLMFLOW_*`, then
  `./rlmflow.toml`, then `~/.config/rlmflow/config.toml`, and
  `rlmflow config {show,path,init}` reports and writes them. `rlmflow version`
  is back. The command set moved from `rlmflow/cli.py` to a `rlmflow/cli/`
  package on [Fire](https://github.com/google/python-fire), now a runtime
  dependency; `from rlmflow.cli import main` and `python -m rlmflow …` are
  unchanged.
- **`StreamConsumer.init(flow)`.** A no-op by default, forwarded by
  `ConsumerGroup`, and the way an interactive consumer gets hold of the flow it
  drives. `FlowTUI` uses it: `ui.init(flow); ui.run()` replaces
  `ui.run(drive)`, which still works. `FlowTUI(sink=consumer)` fans its nodes
  out to another consumer — a `GraphCheckpointer`, say — and
  `ui.run(query=...)` starts a turn as the dashboard mounts.
- **`rlmflow.llm.client_for(model, *, reasoning_effort=None)`.** The one
  model-name rule (`claude*` → Anthropic, else OpenAI) now lives in the package;
  `examples/common.build_client` is an alias for it.
- **`rlmflow.view` and the `rlmflow view` command, on the new node model.** Every
  view is a render of the graph, and they are all back: `timeline`/`steps` read a
  run in the order it happened, `graph_svg` draws it, `save_html` writes a
  single-file stepper, `replay`/`render_steps` hand back the tree as it stood at
  each step, `open_viewer` opens it in the browser, and `save_frames`/`save_gif`
  rasterise it. From the shell: `rlmflow view show runs/coding/graph` prints the
  agent tree and the numbered timeline, with `--step N`, `--frames-only`, and
  `--tree`; the exports are their own verbs, `rlmflow render {svg,html,gif,
  frames,browser} PATH OUT`. `python -m rlmflow view show …` works the same.
  The figure keeps the old viewer's dark ground, its colour and shape per node
  type, and its tidy-tree layout, but it is hand-written SVG: the figure, the
  timeline, the stepper, and `replay` need nothing outside the standard library.
  Only two views carry dependencies and both say so when reached — `open_viewer`
  wants `rlmflow[viewer]` (Gradio) and the raster exports want `rlmflow[image]`
  (CairoSVG, Pillow) — and neither is imported until it is called, so
  `import rlmflow` stays cheap. The stepper embeds the figure once and reveals it
  a node at a time rather than shipping a drawing per step, which is the
  difference between a 434&nbsp;KB file and an 85&nbsp;MB one on a 526-node run.
  Layout is scaled into a target box instead of a fixed gap per row, so a
  hundred-deep chain is dense rather than thousands of pixels tall.
- **`node.created_at`.** Every node stamps itself when it is built, so the order a
  run happened in is those stamps sorted — across agents, not just within one
  chain. `started_at`/`finished_at` still record execution, which only some nodes
  do. Forking restamps the copies it gives new ids, keeping a copy dated after the
  node it now hangs from. The stamp round-trips through `persistence`; graphs
  saved before it load with their nodes stamped at load time and fall back to tree
  order, which is what they had.
- **Opt-in `AGENTS` tree inspection.** `Flow(use_agent_tree=True)` seeds each
  REPL action with an immutable snapshot of its recursive run. Agents can query
  themselves, parents, siblings, children, ids, paths, statuses, and completed
  results through `AGENTS`, or render the tree with `AGENTS.print_graph()`.
  The snapshot refreshes per action and is disabled by default, so ordinary
  flows pay no prompt or serialization cost.
- **`Flow.start(query, **overrides)`** — a root carrying that flow's defaults, so
  `Flow(root_config=AgentConfig(max_iters=5))` reaches the roots you run on it. The
  module-level `start` still builds the node and remains the way to make one
  without a `Flow` (loading, forking, and tests do); `flow.start` supplies the
  defaults and forwards. It replaces `Flow.new_root`, which only accepted a query
  and existed so `flow.run("a query string")` could coerce a string.
- **`start(query, config=..., **overrides)`** — the base config to build on, plus
  per-field overrides, replacing the eleven keyword arguments that restated
  `AgentConfig`'s defaults in a second place. Existing keyword calls such as
  `start("q", inputs={...}, max_iters=8)` are unchanged, since every one of those
  names is an `AgentConfig` field.
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
- **Pool orchestration is separate from compute.** `TaskQueue` drives transition
  coroutines directly while `Pool.stream` and `Pool.call` place and bound model
  compute. A parent can await children without consuming a compute slot.
  `Flow(workers=...)` and custom pools retain their existing public meaning.
- **`Flow(llm_request_timeout=...)`** — bounds every model call. The client's own
  `timeout` is set too, where it takes one: cancelling a blocking call frees the
  caller but not the thread, so only the client can end the request.
- **`Flow(use_llm_query=True)`** — puts `llm_query_batched` in the agent's REPL for
  independent one-shot prompts, which need no trajectory, no REPL, and no
  delegation. Off by default, so an agent is never offered a tool its prompt did
  not describe; `flow.llm_query_batched` calls it from host code either way.
- **Delegation behavior is measured, not asserted in prose.**
  `examples/behavior/delegation.py` runs scenarios in both directions against a
  live model and grades each from its trajectory: child count, how many children
  a single turn launched, and children that spent a turn returning nothing. One
  scenario must fan out (three independent explainers), two must stay local (a
  single config lookup, and four statistics over one list — separable-looking but
  trivial), and the boids task is available with `--scenario boids`, graded on
  fan-out plus the module contracts it must satisfy. The two
  local scenarios also grade the answer, so "no subagents" cannot be earned by
  answering badly. `--repeat N` reports a pass rate, since the behavior is
  stochastic. `tests/test_delegation_behavior.py` runs the same scenarios under
  the new `live` marker — skipped unless `RLMFLOW_LIVE_TESTS=1` and a key is
  present, so the default suite and CI never call an API. `make test-live` runs
  every scenario including boids and pins the model to keep runs comparable, and
  `make test-live-boids` runs that one by node id when iterating on it. Each live
  test is named for the scenario it runs, the live targets report verbosely, and a
  `pytest_deselected` hook names what a filter dropped and which filter dropped it,
  so a paid run that never reached a scenario no longer looks like one that covered
  it. The graders are covered by unmarked tests in that module against canned
  trajectories.
- **The boids example asks for module contracts, not three files.** The old task
  had three outputs but roughly one turn of work, so solving it inline was the
  correct call and it could not tell us anything about delegation. It now
  specifies `vec.js`, `spatial.js`, `rules.js`, `species.js`, `render.js`,
  `main.js`, `index.html`, and `style.css`, each with the global it owns, and
  asks for 2000 boids at 60fps — a spatial grid instead of all-pairs scanning,
  four species with their own rule weights, and a renderer doing HiDPI scaling,
  fading trails, and an FPS readout. The requirements name interfaces, which is
  what a spec does; they still say nothing about subagents, so the execution
  topology stays the model's decision. `BoidsSimulation` now returns the files
  written and what was verified rather than echoing three file bodies back
  through the transcript.

### Changed

- **Input inspection is now the explicit RLM protocol.** With non-empty
  `INPUTS`, the first block probes the values programmatically, binds useful
  reads/searches to persistent names, prints a bounded task-relevant
  observation, and stops. After that initial inspection, one short user message
  asks for the local-or-delegated decision; there is no repeated integration
  action or reserved plan object.
- **Context and working memory are now an explicit prompt concept.** Adapting
  Prime Agent's applicable RLM doctrine, the prompt describes the REPL as the
  long-lived control environment for named working state and recursive calls.
  The parent protects whole-task context while children receive only the context
  needed for their scope.
- **Inspection uses the actual format instead of sampling it.** `INPUTS` string
  values may encode any format, so agents first inspect size and apparent
  structure, then parse what is actually present and use query-appropriate regex,
  indexing, field selection, filtering, grouping, joins, sorting, or aggregation.
  Parsed data stays bound in the REPL; stdout carries only a bounded confirmation.
  Large raw values and arbitrary previews do not stand in for inspection.
- **Delegation uses concrete work tests.** The orchestrator chooses the smallest
  coherent child scopes that cover independent substantive work instead of
  assigning one child per requested artifact. Related small outputs, cross-scope
  glue, integration, and final verification stay local; each child scope must
  repay a full agent's inspection and coordination cost. Independent launches use
  `asyncio.gather`, and the parent combines results without repeating child work.
- **The input manifest is metadata-only.** It lists caller-defined keys and
  sizes but never copies values into the model prompt, regardless of length.
  Short queries therefore stay short and supporting material remains behind
  the same persistent-REPL boundary used for large contexts.
- **Examples are executable trajectories, not disconnected snippets.** Inspection
  still covers nested task text and query-directed JSONL analysis. The
  report trajectory groups several chapters into coherent child scopes and keeps
  title, contents, and assembly in the parent. The shared-context trajectory
  groups related implementation and migration outputs while preserving an
  independent risk review. The
  unrelated structured-subagent worked example is concise API documentation so
  it does not dilute the central action pattern.
- **The prompt size ceiling is enforced again.** `MAX_STATIC_PROMPT_CHARS`
  documented a regression guard that no test actually asserted, so the statically
  rendered `SYSTEM_PROMPT` had drifted 28% past it unnoticed. A test now enforces
  it, and the ceiling is 12,000 to match the prompt as it stands. Raising it is
  now a deliberate edit rather than something that happens by accretion.
- **Each prompt rule has one owner.** API text describes mechanics, input strategy
  owns extraction policy, delegation strategy owns the direct-versus-delegate
  criteria and synthesis behavior, and turn guidance owns only initial inspection
  plus one grounded orchestration decision.
- **Examples render only where they apply.** The two inspection turns need
  non-empty `INPUTS` and the delegation decisions need spawn budget, so an agent
  no longer carries the walkthrough it cannot act on — worth ~2,000 chars for an
  agent with no inputs. A leaf keeps the inspection examples, since children are
  where filtered bulk text usually lands. The example content is fixed for an
  agent's lifetime; only the inspection and one-time orchestration nudges are
  dynamic, preserving the stable system-prompt prefix for caching.
- **Prompt depth status follows each agent's config.** A root-level
  `max_depth` override now controls delegation sections and status for its whole
  subtree instead of being shadowed by the `Flow` default.
- **File guidance moved onto the file tools.** `write_file` now says it replaces
  the whole file and to `ls`/`read_file` first; `ls` says to check the workspace
  before writing. Because these are `@tool` descriptions they reach the model
  only when `FILE_TOOLS` is registered, so the default prompt stays usable for
  runs with no filesystem — and it no longer mentions one.
- **The prompt names what is not bound.** A line listing invented-looking calls
  (`run_subagent(...)`, `call_tool(...)`, `get_result(...)`) as non-existent.
- **Truncation says the REPL survived it.** `TRUNCATION_SUMMARY` now notes that
  variables and imports from dropped turns are still bound, so an agent past its
  `keep_n_messages` window reuses them instead of redefining them.
- **Prompt selection is explicit.** Spawn specs use `prompt_profile`; children
  inherit their immediate parent's profile when omitted. `prompt_router` is an
  optional callable, and unknown profile names raise `ValueError`.
- **Runtime owns REPL lifecycle.** REPL lookup, replay, closure, and rebinding
  live on `Runtime`; warm branch attachment uses `Runtime.rebind_repl`.
- **Persistence and consumers accept Nodes directly.** Save/load uses
  `rlmflow.graph.persistence`; stream consumers handle one Node at a time.
- **The browser viewer is Gradio only.** It draws `rlmflow.view`'s SVG instead of
  building a Plotly figure, so `rlmflow[viewer]` no longer pulls Plotly or Kaleido
  in, and `rlmflow[image]` is CairoSVG plus Pillow. `open_viewer(source)` takes a
  root, a checkpoint directory, or a path to one; the old `states=`/`session=`
  arguments and the click-a-node-for-its-payload panel are gone — the step slider
  drives the detail panel, and an agent picker drives the transcript beside it.
- **A pre-rewrite checkpoint says so.** `persistence.load` on a graph written by
  the old engine (a flat `root_agent_id`/`agents` table) raises `ValueError` naming
  the format instead of failing on a missing key.

### Removed

- `rlmflow.pool`, `rlmflow.tasks`, `rlmflow.simple_flow`, the Event hierarchy,
  and the separate `Graph` wrapper.
- `Flow.build_system_prompt(node)`. Render a prompt with
  `flow.system_prompt.render(flow, agent)`, or read the whole message list from
  `flow.build_messages(node)`. To own prompt selection at the flow level, assign
  `flow.system_prompt` rather than overriding a method.

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
