# Changelog

All notable changes to **rlmflow** are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). While the project is on `0.x`, breaking changes can land on minor bumps — each one is called out under **Breaking** below.

## [Unreleased]

### Breaking

- **Transitions are declarative and instance-owned.** Built-in producers use module-level `@transition(...)`; applications use `@flow.transition(...)`. Each decorated function is one `TransitionFunction`. Callable guards retain ordered deterministic resolution; eligible string guards are resolved together by one structured multiple-choice request. The public `Transitions` registry, `Flow.transitions`, constructor table injection, and subclass mutation hooks are removed.

- **Model-selected named choices are removed.** `flow.choices(...)`, the REPL `transition("name")` tool, `TransitionSignal`, `ReplStatus.TRANSITION`, `InvalidTransitionError`, and the prompt transition footer are gone. Every edge is now an automatic transition function.

- **Delegation is eager and Flow-owned.** Agent code calls `launch_subagent(task=..., schema=None, verify=None, **json_inputs)` and gets an awaitable handle immediately. Flow selects child model, prompt profile, and placement. The old `goal`, `model`, `name`, `inputs`, `output_schema`, `prompt_profile`, and `reuse_repl` launch arguments are removed.

- **Structured model calls have one tool path.** Host code retains `Flow.structured_query`; REPL code uses `llm_query(..., schema=...)`. The model-facing `structured_query` tool and structured-output compatibility flags are removed.

- **Node messages are durable snapshots and persistence is v6.** Node chat `render()` is removed. A query's `build_messages(user_prompt, flow)` runs once, stores its validated message list on the node, and all later history, save/load, and fork operations reuse that snapshot. Live child status remains a request-only Flow overlay. Persistence v6 records snapshots explicitly and rejects v5 runs.

- **Model turns use one strict REPL protocol.** Every model response must contain exactly one non-empty fenced `repl` block, and only `finish(value)` completes an agent. Plain prose, ordinary code fences, empty blocks, and multiple REPL blocks now enter protocol-error recovery without executing partial content.

- **Flow owns fresh prompt defaults; nodes own specialization.** Use `Flow.default_system_prompt(node)` / `default_user_prompt(node)` for prepared `Prompt` instances, `Node.build_system_prompt(system_prompt, flow)` to customize the system prompt, and `Node.build_messages(user_prompt, flow)` to emit current-turn messages. `PromptBuilder`, `build_user_prompt`, `render_fn`, `RenderFn`, and `default_render` are removed. System/user modifiers remain the additive customization path, and `str(prompt)` renders a `Prompt`.

### Removed

- **`llm_query_batched` is removed.** Use the ordinary `llm_query` tool with
  `await asyncio.gather(*(llm_query(prompt) for prompt in prompts))`. The
  duplicate tool, host method, constructor flag, and reserved name are gone.

- **Dead core surface.** `graph_diff`, `GraphDiff`, `StructuralEdge`, `system_prompt_hint`, `StructuredResult.envelope`, `MissingReplError` (the protocol-error text still starts with `MissingReplError:`), `PromptProfile.description`, `MAX_STATIC_PROMPT_CHARS`, `AgentDirectory.all`, `utils.accepts_kwarg`, `utils.usage_from_client`, the `Flow` tool shims (`finish_tool`, `launch_tool`, `wait_agent_tool`, `observe_agent_tool`, `transition_tool`), `Flow.call_stream`, `Flow.execute_action`, `Flow.tool_namespace_for_prompt`, `Flow.submit_child`, `Flow.new_child`, `Flow.ensure_replayed`, `Flow.register_verifier`, `Flow.verify_candidate`, and the `rlmflow.view` exports `node_shape`, `NODE_COLORS`, `NODE_SHAPES`, and `summarize`. Worker runtimes now implement `Runtime.connect()`; the shared `Runtime.open` wraps it in a session. `Flow.resolve_child` remains the per-child override hook.

- **Unused prompts, runtime, and engine surface.** `SYSTEM_PROMPT` (a static copy of the default prompt that `Flow` never used), `Runtime.namespace_for`, `engine.boundaries.register` (pass a callable `until=` instead), `AppendChild.child`, the unused `key=` placement argument on `Pool.call`/`Pool.stream`, `PooledLLMClient`, and `Flow.call_chat`, the `DockerRuntime(**options)` catch-all, and the worker's `--workdir` flag. Public exports dropped (the code is still used internally): `PromptModifier`, `PromptSectionError`, `SystemPromptFn`, `format_tool_line`, and the duplicate `at_leaf`/`can_spawn` re-exports from `rlmflow.tools` (import them from `rlmflow`).

- **Toolsets.** `@toolset`, `ToolsetMetadata`, `FileTools`, the `BuiltIns` prompt stubs, `@tool(inject=...)`, and `Flow.toolsets` are removed. `@tool` is the only tool decorator; group related tools in a tuple, which `tools=` flattens one level. The `SHOW_VARS` line the stubs advertised is gone too — nothing bound it.

- **Legacy synthetic planning turns.** The old turn-counter `PlanQuery`, `to_plan`, `resume_after_child`, `child_returned`, and `USER_PROMPT` are removed. Each model response plans and executes its next step in one REPL block; no synthetic message is inserted after observations.

### Changed

- **Evidence-first guidance lives in the core protocol.** The pinned system prompt asks each turn to plan the immediate next action from observations already obtained, compute answers from the inputs rather than assumptions, finish in one block when code can derive the answer directly, and look first when the inputs' size, format, or contents are unknown. Before `finish`, agents check the candidate against the inputs. No turn is forced to observe-only, and `finish` is available from the first block.
- **Root and child calls share one argument contract.** `Flow.run(task, **args)`, `Flow.start(task, **args)`, and `launch_subagent(task=..., **args)` all build agents through the same constructor. `task` and named JSON arguments are direct REPL variables; the model-facing `INPUTS` mapping is removed. Runtime controls belong in an explicit `AgentConfig`. Persisted graphs use v7 and store invocation `args` on each `AgentStart`.
- **Worker reuse no longer shares agent globals.** `delegation_reuse_worker=True` colocates child tenants in the parent's process while retaining isolated namespaces. It replaces the misleading `delegation_reuse_repl` name and prevents concurrent children from overwriting one another's direct invocation variables.
- **Benchmark task requirements are runner-neutral and auditable.** Every row fingerprints the same benchmark-owned task, named inputs, and fixture bytes. Framework-specific API guidance is appended separately and receives its own hash. The delegation suite now shares all scored root requirements across PrimeAgent, Official RLM, JAZ, and rlmflow, and scores PrimeAgent's final response directly instead of imposing a unique output-file requirement.
- **Built-ins and application tools share one interface.** `Flow.build_tools(node)` returns the complete map of real decorated callables: Flow-owned built-ins, configured application tools, and subclass-provided environment tools. The prompt renders that map directly, while `Flow.build_namespace(node)` adds non-tool execution bindings. `llm_query` is node-bound like `finish` and `launch_subagent`, so its static description and default model match its calling agent. The old prompt-only stubs and callable descriptions are gone. The local-work example and `AGENTS` notes remain prompt sections (`guidance.local`, `capabilities.agents`).
- **All prompt prose lives in `rlmflow.prompts.text`.** `rlmflow.graph.prompts` is gone; its turn strings, execution policy, and `AGENTS_TEXT` sit with `CORE_PROTOCOL_TEXT` in one import-free module. `rlmflow.prompts` no longer re-exports `build_messages` / `render_tools` (use `rlmflow.prompts.messages`), and argument manifests live in `rlmflow.prompts.prompts`.
- **`FILE_TOOLS` is a tuple of plain functions.** Import individual tools from `rlmflow.tools.filesystem` (`read_file`, `write_file`, …) instead of `FILE_TOOLS.<name>`.
- **Persistence saves node fields generically.** `persistence.to_record(node)` writes every data field of a node, so node classes no longer carry hand-written `to_record` overrides, and loading finds node classes by their `type` name instead of a registry. `persistence.register_node_type` is removed: a custom node is just a dataclass subclass with its own `type`. Records no longer repeat `content` under a separate `output` key, and a base `ActionNode` now keeps its `code` across save and load. `ChatMessage` and `message_snapshot` live in `rlmflow.graph.nodes`.

### Fixed

- **Swarm-fanout child prompts are runner-neutral.** PrimeAgent, Official RLM, JAZ, and rlmflow now invoke every worker with the exact same complete text prompt containing worker name, filename, question, and shard content. Only the framework-owned invocation mechanism differs. The previous adapters used different data-delivery interfaces, confounding child-quality comparisons.

- **Verifier fingerprints match across host and worker.** `rlmflow.engine.verification` owns the one verifier fingerprint and schema hash. Previously a verifier registered in a worker REPL hashed differently from one registered in-process, so replaying a launch on the other side reported a verifier mismatch. Runs saved with launch verifiers before this change carry the old hash and will report that mismatch once on replay.

### Added

- **Durable semantic choices and finish verification.** Source-compatible string transition guards are presented together to the Flow's configured structured engine, which selects one compact transition name or `continue` in a single request. One `TransitionDecision` persists the complete candidate set, selected durable rule ID, and read-set hash for replay. This path works with Chat, Instructor, Jev, retry, and fallback engines. Launch-local finish verifiers reject candidates into `VerificationFailure` and repair the same child.

- **Structured output contracts and producers are modular.** `OutputCodec` owns normalization, prompt guidance, parsing, and local validation. One `StructuredEngine` per Flow powers state-oriented `structured_query` calls and schema-backed `llm_query` calls. Chat is the default; optional Instructor and Jev engines are available, while retry, fallback, and confidence policies remain explicit wrappers. Agent `finish(value)` remains local and deterministic.

## [0.6.0] — 2026-09-21

### Added

- **Three reproducible release showcases.** Shepherd demonstrates graph rewind and recovery, Needle demonstrates recursive parallel search, and Boids demonstrates practical multi-agent software construction. From product-level constraints that prescribe neither files nor graph topology, the recorded Boids root chooses a five-file architecture, launches five independent coding workers concurrently plus one focused repair, validates local references, JavaScript syntax, browser console health, and nonblank rendering, then publishes a browser poster, continuous GIF/MP4 recording, and level-synchronized graph animation. Each showcase has a locked run command and an asset-regeneration Make target.

- **Four focused evaluation suites.** `smoke`, `delegation`, `long-context`, and `interactive` are the complete public surface. Smoke is synthetic needle retrieval, delegation is the four-runtime swarm fan-out matrix, long-context is OOLONG, and interactive combines MiniMaze, Sudoku Search, and MazeBench. A runtime-neutral episode core owns private forks, shared budgets, immutable candidate snapshots, root selection, metrics, and artifacts; all interactive tasks use one normalized environment report contract.

- **Child completion is a parent-owned `UserQuery`.** When a child `finish()`es, the launching agent gets a durable `[handoff {path}]` observation after its next `ExecOutput` (batched if several children completed), unless the result is already routed through another agent's handle-valued `inputs`. A mapping result is listed as fields so the next `PlanQuery` can read the memo; a scalar stays labeled JSON. `ErrorOutput` / `ReplDead` do not flush. `resume_on_child_done=True` by default wakes a finished parent with the same query; pass `False` on scored evals that must stay terminal. Mailboxes are live-only and rebuilt on stream start. See [`docs/control.md`](docs/control.md).

- **Coding children finish a memo.** `AgentConfig.child_output_schema` is the default `output_schema` for launched children (launch `output_schema=` still wins). The coding CLI and `examples/coding/agent.py` set `CODING_HANDOFF_SCHEMA` (`done`, `diff_summary`, `concerns`, `followups`); the root stays untyped. Evals and other `Flow`s are unchanged.

- **Handle-valued `inputs` build flat dependency DAGs.** Passing an `AgentHandle` as an `inputs` value blocks the consumer until that source completes, then injects its result under the chosen key. The encoded input reference is the sole durable dependency record; `after`, `AgentDependency`, and ownership adoption are removed. Launch ownership remains flat, `max_depth` controls recursive spawning only, and persistence v5 keeps input references for replay and graph projection.

### Changed

- **The public release surface is intentionally smaller.** Orphan generated docs and duplicate media, unused delegation corpora and examples, obsolete live-prompt checks, disconnected release-gate wiring, and stale Read the Docs configuration are removed. Research notes, RAO, and autoresearch remain intact.

- **Persistence is now v5.** Dependency references use stable agent IDs, `ExecAction` records persist launch receipts, and handle-valued input references survive replay. Older graph versions are rejected without migration or direct-child replay fallbacks.

- **`UserQuery` after `ExecOutput` now plans.** A `UserQuery` whose previous node is `ExecOutput` (or `DoneOutput`) takes `to_plan` rather than `complete`, so a child-handoff or a host injection after idle REPL output gets a `PlanQuery` before the next model action.

- **`PlanQuery` splits planner vs worker policy.** Spawn-capable agents are told that children provide fresh context, cheaper models, and concurrent execution; substantial outputs with clear interfaces get independent owners when that reduces token cost or wall-clock time. Agents that cannot spawn get a worker addendum: own the slice, no TODOs, `finish` the actual answer.

- **`@tool(..., when=)` gates prompt and REPL together.** A `(flow, node) -> bool` predicate (or a toolset `visible_tools(flow, node)`) hides a host tool from that agent in both the docs and the namespace. `at_leaf` is true when the agent cannot spawn. `FILE_TOOLS` writes (`write_file`, `edit_file`, `append_file`) use it. `launch_subagent` is no longer bound on leaves, matching its prompt. The coding CLI default is now `max_depth=1` so launched children can write.

### Breaking

- **The 0.6 graph, execution, and persistence APIs replace their pre-0.6 counterparts.** Existing integrations and saved runs may require changes; see [`UPGRADING.md`](UPGRADING.md) for the migration guide.

## [0.5.0] — 2026-09-08

### Breaking

- **`Flow.step` returns the created `Node`.** Producers register with `@Flow.transitions.on(...)`. `StepFunction`, `DEFAULT_STEPS`, `update_step_fn`, `DEFAULT_TRANSITIONS`, `Flow.always` / `@flow.when`, and `MessageBuilder` / `FlowMessages` are gone. The queue still uses an internal `Transition`.
- **Named choices transition directly.** Declare model-selectable routes with `Transitions.choices(current, *targets)`. The always-bound `transition("name")` REPL control is documented only when choices exist and lands the selected `UserQuery` directly from `ExecAction`; transitioning ends the action immediately, so stdout from that block is not observed. `.offer(...)`, `ExecAction.requested_transition`, and the intermediate requested-transition `ExecOutput` are removed.
- **`Node.append` returns `None`.** It mutates the tree like `list.append`. Construct the child, append it, and return that child from the step. Fluent `root.append(a).append(b)` is gone. `append_child` / `AppendChild.attach` still return the attached subtree.
- **Query / action / output bases.** `UserQuery.build_system_prompt` is what a turn sends; `Node.build_system_prompt` is removed. `ActionNode` groups executable engine work (`ExecAction` and `AppendChild`), while `OutputNode` groups output produced by model, runtime, and terminal steps (`LLMOutput`, `ExecOutput`, `ErrorOutput`, and `DoneOutput`). `Flow.build_messages` uses the query hook only on `UserQuery`; inheriting `OutputNode` does not make `LLMOutput` prompt-bearing.
- **Official-shaped system prompt.** `SystemPromptBuilder`, `.sections`, `DEFAULT_BUILDER`, `Section`, and `Sections` are removed. Subclass `PromptBuilder` (one `__call__`) and concatenate, or pass a string / `(flow, node) -> str`. `Flow.build_system_prompt(node)` is the inherited protocol; `UserQuery.build_system_prompt(flow)` is what a turn sends (`PlanQuery` appends the orchestrator addendum when the agent can spawn; `FinalQuery` appends last-turn submit guidance). User turns are `Turn N/M:` plus a first-turn INPUTS safeguard.
- **`finish(answer)` accepts strict JSON values only.** Without `output_schema`, strings, finite numbers, booleans, `None`, lists, and dictionaries with string keys remain typed. Unsupported Python values now produce an execution error instead of being stringified. With a schema, JSON values and JSON-encoded strings are parsed and stored as the validated typed value. `Flow(enable_structured_output=True)` remains the default (pass `False` to hide child-schema guidance).
- **Multi-root execution has one API.** `parallel_run`, `parallel_stream`, and `engine/parallel.py` are removed. Pass all roots to one `flow.run_streaming(root, *roots)` call.
- **Children use the same iteration-limit contract as every other agent.** `child_max_iters` is removed; `AgentConfig.child()` carries the parent's `max_iters` forward.

### Added

- **Named benchmark sets.** Eval is organized around questions: `smoke`, `reasoning`, `long-context`, `delegation`, `task-graph`, `research`, and `code`. Pass a set name to `--dataset`; runners default to that set. `python -m benchmarks.eval --list-sets` prints the table. `make eval-delegation` is the eight-case routing set; the frozen 20 is `make eval-task-graph`. See [`docs/benchmarks.md`](docs/benchmarks.md).

- **Tools can own their prompt docs.** `@toolset("File tools")` marks a class as a source of tools, not a tool: `Flow.add_tool` flattens `@tool` methods into the existing flat REPL namespace. Each active toolset renders one `### {title}` subsection containing its member signatures and optional `guidance(flow, node)`; standalone `@tool` callables share a `### Tools` subsection. Empty groups and their guidance are omitted, and toolset members are never duplicated in the standalone list. A toolset is one item in `tools=`, same as a function: `Flow(tools=[FILE_TOOLS, extra])`. `BuiltIns` uses the same renderer and owns `finish`, `launch_subagent`, and the one-shot query signatures (`inject=False` — Flow installs their real per-node closures separately).

- **Benchmark runs can be compared per question and re-graded without re-running.** `python -m benchmarks.eval.matrix RUN` writes a task × (runner, seed) score grid plus a per-question rundown carrying turns, children, depth, tokens, and each failure string; attempts that crashed are reported as unscored rather than averaged as zeros, and answers rejected by the grader are distinguished from answers that never arrived. `python -m benchmarks.eval.regrade RUN` re-scores a finished run from its saved predictions, so a scorer fix costs no tokens.

- **Core stdlib modules are preimported into every REPL.** `json`, `re`, `os`, `sys`, `pathlib`, `csv`, `math`, `asyncio`, `collections`, `datetime`, `itertools`, `statistics`, `textwrap`, and `time` are bound before any agent code runs, so a model reaching for `json.loads` no longer loses a turn to `NameError`. The set is `runtime.DEFAULT_PREIMPORTS`; every runtime takes `preimports=` to override it (an empty sequence binds nothing, and names may include third-party packages), and the worker accepts `--preimport`.

### Fixed

- **A spent token budget finalizes the whole agent tree instead of dropping the run.** Exhausting `max_budget` used to return `[budget exceeded]`, discarding work already paid for. The threshold now creates one `FinalQuery` for every unfinished agent. Descendants finalize first; each waiting parent then receives their results and gets its own synthesis turn. Already generated `LLMOutput` / `ExecAction` work is allowed to land, while new child creation is rejected after exhaustion. Final prompts expose only `finish(...)`, so no new delegation or one-shot query is advertised. `max_budget` is therefore a soft whole-run threshold: overrun is bounded by calls already in flight plus one final turn per unfinished agent.

- **Cold restore rebuilds worker sessions required by unfinished colocated children.** Replay groups agents by physical worker ownership and reconstructs a required terminal owner before an unfinished `reuse_worker=True` child. Agent namespaces remain isolated.

- **Builtin prompt signatures match their runtime closures.** Generated tool documentation now advertises `finish(answer)` and keyword-only `model=` for `launch_subagent`, exactly as the executable tools accept them.

- **Numeric benchmark answers are graded at the answer key's precision.** `exact_or_alias` compared numbers with `Decimal` equality, so `9810.79000000000126` was wrong against a key of `9810.79` and `str(float)` round-tripping was graded as an arithmetic error — six of one runner's thirty attempts in a ten-task comparison, enough to invert the result. Comparison now rounds to the precision the key states and never coarser than two decimals, so an integer key still rejects `3.4`.

### Changed

- **`max_iters` and `max_budget` default to `None`.** Ordinary runs are unbounded unless you set a cap. `max_output_length` and `max_query_chars` are 20,000 characters, matching official RLM's per-block REPL truncation (`format_iteration(..., max_character_length=20000)`). The prompt already said "~20K"; the engine was still clipping at 4,000.

- **Eval CLI defaults to the `reasoning` set.** `--dataset` without a name expands AIME 2025 and Sudoku Extreme. Omit `--runner` and a named set fills in its table (`vanilla` / `rlmflow-local` / `official-rlm` for `reasoning`). `make eval-run` follows. Iteration Make targets (`eval-delegation-five`, `eval-benchmark`, …) are gone; the dataset names still work.

- **The planning turn plans; the turns after it act.** The planning guidance asks the model to fix the target, inventory cheaply, and print the plan; solving the task and calling `finish(...)` belong to later turns. It caps the inventory step (reading a payload in full is branch work, not inventory) and prints the routing decision: what stays local, and every child to launch next turn once the inventory confirms it. Delegation policy states why a child helps — its own context and REPL — and gates children on capacity rather than separability: one oversized source, inputs whose combined volume crowds out integration, or a branch that must read far more than its answer needs.

- **`llm_query_batched` is documented as a strategy, not just a signature.** Its whole presence in the prompt was one generated tool line, while ~1,650 characters of delegation guidance taught capacity triggers — an oversized source, volume crowding out integration, a branch reading far more than its answer needs — and routed every one of them to `launch_subagent`, which carries its own context, REPL, and prompt. Reading at volume now has its own passage, placed ahead of the delegation triggers so those read as an escalation from it: batched one-shot calls for extracting, classifying, or summarizing over chunks, fat prompts rather than many thin ones, and a child agent only when a branch needs its own REPL, several turns, or children of its own. Shown only when the tool is bound.

- **`finish(...)` is print-then-submit.** The contract now forbids finishing from the block that first computes the value: since nothing printed beside that call is ever read, an answer submitted there was never seen. A turn prints the candidate and the results behind it, and the next turn finishes. Every worked example demonstrates the split, because a compute-and-finish example outweighs prose asking for anything else. Submitting unread values is how a regex over a tool report truncated a `10984.869565217392` total to `109`. The value must still be checked against the deliverable — one that came back empty or implausible is a bug to debug rather than an answer to submit — and inputs that fit in context should be reasoned over directly instead of prompting a general solver written for one instance.

- **Graph operations now scale past Python's recursion limit.** Subtree walks, persistence loading, forks, and visualization layout are iterative. Whole-run counts, usage, agent lookup, and errors come from a rebuildable incremental index.

- **Live work is bounded.** The TUI coalesces redraws and `GraphCheckpointer` saves every two seconds or 50 nodes by default, then flushes on terminal events and close.

- **LLM compute is an explicit stream-first primitive.** `PooledLLMClient` wraps a raw client, pool, timeout, scheduling key, and request kwargs. `complete` consumes `stream()` through `flow.llm_for_step(node)`; tests can pass a small streaming fake. `call_chat()` collects the same stream.

- **Step producers are functions on `Flow.transitions`.** `@Flow.transitions.on(...)` registers `(flow, node) -> Node`. `Flow.step` resolves, calls, appends, and returns that node. `WrappedRuntime` still seeds tools and `INPUTS` immediately before execution. `Pool.run` is gone; pools place compute calls only.

### Migration notes

- See [`UPGRADING.md`](UPGRADING.md) for the complete 0.5 migration guide.

- **Persistence v3 is a strict flat format.** `graph.json` stores shallow Node records with `parent_id`; v2 runs are rejected instead of entering the old recursive loader. Use `persistence.to_document` / `from_document`; the old `to_dict`, `from_dict`, and summary helpers are removed.

- **Graph APIs now state their cost and direction.** Use `node.iter_backwards()` instead of `walk(reverse=True)`, `node.subtree_usage()` instead of `node.tokens()`, and `root.usage` / `root.stats` for whole-run aggregates. Public `fork()` always creates fresh identities and no longer accepts `new_ids`.

- **Checkpoint intervals are explicit.** `GraphCheckpointer` accepts `interval_s` and `interval_nodes`; call `flush()` for an immediate save.

- **Compatibility aliases are removed.** Agent code must call `finish(...)`, `AgentInfo` exposes only `result()`, `FlowTUI` must be initialized with `init(flow)` before `run()`, and examples call `rlmflow.llm.client_for` directly.

- **Chat projection lives on nodes and has one current-frontier override.** `Node.render()` returns zero or more messages. `Node.project()` always uses canonical node rendering, flattens multi-message nodes, and bounds its backward walk by rendered messages. `Flow.build_messages()` applies `render_fn(runtime, node)` only to the current frontier, projects history from `node.prev`, and preserves every rendered message without coalescing adjacent roles. `UserPromptBuilder`, `user_prompt=`, `PromptProfile.user`, the separate user-injection path, and `Flow.messages()` are gone.

- **Engine nudges are nodes.** Every ordinary model action starts at a `PlanQuery`, so the graph records a plan-and-act boundary after each observation without a separate inspection phase. Final-answer, truncation, and a dead REPL are `FinalQuery`, `TruncationSummary`, and `ReplDead`. `Flow.step` resolves a `@transitions.on` producer and returns the created node.

- **One engine, and it is the package.** The rewrite that lived under `rlmflow.minimal` is now `rlmflow` itself: `rlmflow.graph` holds the tree and its on-disk format (`nodes.py`, `persistence.py`), `rlmflow.flow` holds `Flow`, and `rlmflow.engine` holds what it is built out of (`execution.py`, `boundaries.py`). The engine they replace is deleted along with the old `rlmflow.graph` and `rlmflow.minimal`. Every name is still re-exported from `rlmflow` directly, which is where callers should import it from. The node API it brings with it: `agent.frontier` for `tail()`, `node.append(child)` for `attach`/`submit`/`spawn`, `agent.result()` for `agent_result()`, `agent.terminal` for `finished()`, `agent.transcript()` with no agent-id argument, and `node.iter_backwards()` for reading one agent's chain backwards. `Flow` no longer takes `max_iters`, `max_depth`, or `child_max_iters`: per-agent limits belong to `AgentConfig`, so pass them to `flow.start(query, ...)` for one root, or as `Flow(root_config=...)` to set the defaults every root from `flow.start(...)` (and every string handed to `flow.run`) picks up. A root built by the bare `start(...)` carries `AgentConfig`'s own defaults — it has no flow to inherit from.

- **Node streaming.** `run_streaming(root_or_query, until=...)` yields affected Nodes, not Event wrappers.

- **Forking is the whole branch API.** `node.fork()` copies the tree and cuts everything after that node; `rlmflow.surgery` and its mid-transcript `insert`/`remove` are gone, as are `Flow.rewind`, `Flow.launch_branches`, and `Flow.discard`. An edit lands as an ordinary node appended to a frontier, which `Node.append` enforces — there is no way to rewrite history in place.

- **Deleted with no replacement.** `LLMChannel` (bounding concurrency is `Flow(workers=...)` or a `Pool`), and the injection helpers `flow.next_query`, `flow.inject_tools`, `flow.register_halt`, and `flow.append` — appending a `UserQuery` to a frontier, `flow.add_tool`/`flow.inject`, and an `until` callable cover what they did.

- **A delegating step writes one node, like every other step.** Children hang off the `ExecAction` that launched them, so `SupervisingOutput` is gone from the node model and a delegating turn records a single `ExecOutput` holding everything its block printed — including what printed before the launch, which used to be a separate node that escaped `max_output_length`. Pre-v3 development runs must be exported or regenerated before upgrading. With the frontier no longer moving mid-step, work in flight is keyed by the node being stepped instead of by its agent. `waiting_on` is read off the tree: an `ExecAction`'s `AgentStart` children.

- **`Repl.run` returns a `ReplRun`, not a string.** One value carries how a block ended — `output` plus a `ReplStatus` of `OK`, `ERROR`, `DONE`, or `DEAD`, the answer from `finish(...)`, and the exception that killed the REPL — instead of a string plus the `errored` and `done_result` attributes a caller had to reset and read around it. `Runtime.execute(node, code)` adds the `DEAD` case: a REPL that dies is an observation for the agent and an eviction from the runtime, so no caller handles that exception any more. A dead REPL is now told apart from code that raised: its `ErrorOutput` carries `error="repl"` and appends the cold-REPL note, since the agent's namespace went with it.

### Reliability and compatibility

- **Node-native visualization and stream consumers.** `LiveTreeRenderer` restores compact terminal redraws, `LiveGraphTree` restores the Rich live forest, and `FlowTUI` restores the Textual chat/dashboard with overview, tree, agent, count, waiting, error, and latest-activity panels. `GraphCheckpointer` restores periodic/final saving. They consume Nodes directly through `ConsumerGroup.handle(node)` and are imported from `rlmflow.consumers`.
- **Two roots on one flow no longer deadlock.** One `run_streaming(root, *roots)` call drives all roots through one `TaskQueue`. Each completed transition is published once, and child joiners wake through one run-scoped condition instead of competing stream loops consuming one another's work.
- **Resuming a saved run rebuilds its live REPL.** `persistence.load` followed by a run previously started with an empty namespace, so an agent's own variables raised `NameError` while its transcript said it had set them — and it retried until it hit an iteration cap. A fork had the same problem. A graph a `Flow` has no REPLs for is now replayed before the run loop starts. `Flow(restore="lazy")` defers the same replay until immediately before each agent's first new action.
- **Resuming a finished run returns its answer.** Whether an agent had answered was read from the task queue, which only knows about steps taken in the current run, so a loaded terminal agent was submitted for another step and raised `TypeError: cannot step DoneOutput`. `submit_leaves` now reads it from the frontier, and consults the queue only for the case a graph cannot show: a step that raised.
- **`max_iters` and `max_budget` survive a save.** The complete effective config is recorded for every agent, so resuming a run in another process no longer silently changes its limits.
- **A token budget observes the complete run.** The exhaustion barrier reads the root's indexed usage, so sibling spend triggers bottom-up finalization across the same tree.

### Additional features

- **Explicit subagent model selection.** Every `launch_subagent` call requires a registered `model=` key. The system prompt always lists available keys and marks the current one, so the agent—not a hidden host fallback—chooses the appropriate model for each workstream. Saved action code from older runs that omitted `model` must be regenerated before replay.
- **`rlmflow run` / `rlmflow tui`, and a CLI that is more than one command.** Every command is a Fire class: flags on the constructor, verbs as methods. `rlmflow tui` starts a coding agent over `--workdir` in the Textual dashboard, checkpointing to `<workdir>/graph`; `rlmflow run tui "fix the failing test"` opens it with that turn already going, and `rlmflow run print` runs the same thing headless, streaming the tree and printing the answer. `--model`, `--fast-model`, `--reasoning-effort`, `--docker-image`, `--max-depth`, `--max-iters`, `--workers`, and `--tools {files,none}` pick the pieces; `--resume DIR` continues a saved run and `--agent module:factory` swaps in your own `Flow`. Settings resolve flags first, then `RLMFLOW_*`, then `./rlmflow.toml`, then `~/.config/rlmflow/config.toml`, and `rlmflow config {show,path,init}` reports and writes them. `rlmflow version` is back. The command set moved from `rlmflow/cli.py` to a `rlmflow/cli/` package on [Fire](https://github.com/google/python-fire), now a runtime dependency; `from rlmflow.cli import main` and `python -m rlmflow …` are unchanged.
- **`StreamConsumer.init(flow)`.** A no-op by default, forwarded by `ConsumerGroup`, and the way an interactive consumer gets hold of the flow it drives. `FlowTUI` uses it: `ui.init(flow); ui.run()` replaces the removed `ui.run(drive)` callback API. `FlowTUI(sink=consumer)` fans its nodes out to another consumer — a `GraphCheckpointer`, say — and `ui.run(query=...)` starts a turn as the dashboard mounts.
- **`rlmflow.llm.client_for(model, *, reasoning_effort=None)`.** The one model-name rule (`claude*` → Anthropic, else OpenAI) now lives in the package and examples call it directly.
- **`rlmflow.view` and the `rlmflow view` command, on the new node model.** Every view is a render of the graph, and they are all back: `timeline`/`steps` read a run in the order it happened, `graph_svg` draws it, `save_html` writes a single-file stepper, `replay`/`render_steps` hand back the tree as it stood at each step, `open_viewer` opens it in the browser, and `save_frames`/`save_gif` rasterise it. From the shell: `rlmflow view show runs/coding/graph` prints the agent tree and the numbered timeline, with `--step N`, `--frames-only`, and `--tree`; the exports are their own verbs, `rlmflow render {svg,html,gif, frames,browser} PATH OUT`. `python -m rlmflow view show …` works the same. The figure keeps the old viewer's dark ground, its colour and shape per node type, and its tidy-tree layout, but it is hand-written SVG: the figure, the timeline, the stepper, and `replay` need nothing outside the standard library. Only two views carry dependencies and both say so when reached — `open_viewer` wants `rlmflow[viewer]` (Gradio) and the raster exports want `rlmflow[image]` (CairoSVG, Pillow) — and neither is imported until it is called, so `import rlmflow` stays cheap. The stepper embeds the figure once and reveals it a node at a time rather than shipping a drawing per step, which is the difference between a 434 KB file and an 85 MB one on a 526-node run. Layout is scaled into a target box instead of a fixed gap per row, so a hundred-deep chain is dense rather than thousands of pixels tall.
- **`node.created_at`.** Every node stamps itself when it is built, so the order a run happened in is those stamps sorted — across agents, not just within one chain. `started_at`/`finished_at` still record execution, which only some nodes do. Forking restamps the copies it gives new ids, keeping a copy dated after the node it now hangs from. The stamp round-trips through `persistence`; graphs saved before it load with their nodes stamped at load time and fall back to tree order, which is what they had.
- **Opt-in `AGENTS` tree inspection.** `Flow(use_agent_tree=True)` seeds each REPL action with an immutable snapshot of its recursive run. Agents can query themselves, parents, siblings, children, ids, paths, statuses, and completed results through `AGENTS`, or render the tree with `AGENTS.print_graph()`. The snapshot refreshes per action and is disabled by default, so ordinary flows pay no prompt or serialization cost.
- **`Flow.start(query, **overrides)`** — a root carrying that flow's defaults, so `Flow(root_config=AgentConfig(max_iters=5))` reaches the roots you run on it. The module-level `start` still builds the node and remains the way to make one without a `Flow` (loading, forking, and tests do); `flow.start` supplies the defaults and forwards. It replaces `Flow.new_root`, which only accepted a query and existed so `flow.run("a query string")` could coerce a string.
- **`start(query, config=..., **overrides)`** — the base config to build on, plus per-field overrides, replacing the eleven keyword arguments that restated `AgentConfig`'s defaults in a second place. Existing keyword calls such as `start("q", inputs={...}, max_iters=8)` are unchanged, since every one of those names is an `AgentConfig` field.
- **`Flow(restore="replay" | "lazy")`** — how a graph this flow did not run gets its live state back. `Flow.replay` re-runs each agent's recorded `ExecAction`s before streaming; lazy restore performs the same reconstruction immediately before the first new action. Load, fork, and cross-process resume share this path.
- **`RLMFLOW_REPLAY`** — `"1"` in an agent's `ENV` while its recorded code is being re-run, `"0"` during a live turn. `launch_subagent` resolves recorded children instead of launching them again; agent code can read the flag to skip downloads, writes, or anything else it should not repeat.
- **`AgentStart.save(path)` / `AgentStart.load(path)`** — the strict version 3 run directory: a flat parent-linked `graph.json` and `latest.json` beside an `agents/` tree of `agent.json` + `session.jsonl`. A re-save prunes agent directories that are no longer in the tree, timing is recorded per step, each turn's system prompt is kept once per agent and referenced by id, and a structured answer is stored parsed rather than as the string the model emitted.
- **Long-running / multi-turn agents.** Append a `UserQuery` to a finished root's frontier and stream it again: history and the warm REPL are still there.
- **Indexed usage.** `root.usage` reads the whole-run total in O(1), while `node.subtree_usage()` explicitly scans one subtree.
- **Named `PromptProfile`s** — `Flow(prompt_profiles=..., prompt_router=...)` plus per-agent `UserQuery.prompt_profile` / spawn-spec `prompt_profile`.
- **Structured execution module.** `Pool`, `ThreadPool`, `SequentialPool`, and the small structured-concurrency `TaskQueue` live in `rlmflow.engine.execution`.
- **One pooled leaf queue per driver.** `run_streaming` seeds one or more roots' current leaves, yields each completed `Transition.created`, checks that root's `until`, and resubmits the node when its agent is not terminal. Delegation submits children to the active queue and uses one run-scoped condition for terminal wake-ups. Tree-wide diffs, `settle`, `landing`, cross-root queue predicates, and repeated leaf scans are gone.
- **`Flow._drive` is a total internal transition.** It returns `Transition(submitted, created, error)`, converting infrastructure failure into a terminal graph node while allowing cancellation to unwind. Public `Flow.step` returns only `Transition.created`; `step_task` and queue-side exception inspection are gone, and a failed root is still re-raised to its stream caller.
- **Pool orchestration is separate from compute.** `TaskQueue` drives transition coroutines directly while `Pool.stream` and `Pool.call` place and bound model compute. A parent can await children without consuming a compute slot. `Flow(workers=...)` and custom pools retain their existing public meaning.
- **`Flow(llm_request_timeout=...)`** — bounds every model call. The client's own `timeout` is set too, where it takes one: cancelling a blocking call frees the caller but not the thread, so only the client can end the request.
- **`Flow(use_llm_query=True)`** — puts `llm_query_batched` in the agent's REPL for independent one-shot prompts, which need no trajectory, no REPL, and no delegation. Off by default, so an agent is never offered a tool its prompt did not describe; `flow.llm_query_batched` calls it from host code either way.
- **The boids example asks for module contracts, not three files.** The old task had three outputs but roughly one turn of work, so solving it inline was the correct call and it could not tell us anything about delegation. It now specifies `vec.js`, `spatial.js`, `rules.js`, `species.js`, `render.js`, `main.js`, `index.html`, and `style.css`, each with the global it owns, and asks for 2000 boids at 60fps — a spatial grid instead of all-pairs scanning, four species with their own rule weights, and a renderer doing HiDPI scaling, fading trails, and an FPS readout. The release showcase explicitly assigns non-overlapping module groups to parallel coding workers, then leaves integration and verification with the root. `BoidsSimulation` returns the files written and what was verified rather than echoing file bodies through the transcript.

### Prompt and runtime changes

- **Planning is an explicit per-action graph boundary.** `AgentStart`, REPL output, execution errors, and returned children automatically enter a compact `PlanQuery`. The model briefly revises its plan from the latest observation and executes the next REPL block in the same response; inspection is part of that iterative work rather than a separate phase.
- **Context and working memory are now an explicit prompt concept.** Adapting Prime Agent's applicable RLM doctrine, the prompt describes the REPL as the long-lived control environment for named working state and recursive calls. The parent protects whole-task context while children receive only the context needed for their scope.
- **Inspection uses the actual format instead of sampling it.** `INPUTS` string values may encode any format, so agents first inspect size and apparent structure, then parse what is actually present and use query-appropriate regex, indexing, field selection, filtering, grouping, joins, sorting, or aggregation. Parsed data stays bound in the REPL; stdout carries only a bounded confirmation. Large raw values and arbitrary previews do not stand in for inspection.
- **Delegation uses concrete work tests.** The orchestrator chooses the smallest coherent child scopes that cover independent substantive work instead of assigning one child per requested artifact. Related small outputs, cross-scope glue, integration, and final verification stay local; each child scope must repay a full agent's inspection and coordination cost. Independent launches use `asyncio.gather`, and the parent combines results without repeating child work.
- **The input manifest is metadata-only.** It lists caller-defined keys and sizes but never copies values into the model prompt, regardless of length. Short queries therefore stay short and supporting material remains behind the same persistent-REPL boundary used for large contexts.
- **Examples are executable trajectories, not disconnected snippets.** Inspection still covers nested task text and query-directed JSONL analysis. The report trajectory groups several chapters into coherent child scopes and keeps title, contents, and assembly in the parent. The shared-context trajectory groups related implementation and migration outputs while preserving an independent risk review. The unrelated structured-subagent worked example is concise API documentation so it does not dilute the central action pattern.
- **The prompt size ceiling is enforced again.** `MAX_STATIC_PROMPT_CHARS` now has a test-enforced 4,000-character ceiling so substantial accidental prompt growth fails CI.
- **Each prompt rule has one owner.** API text describes mechanics, input strategy owns extraction policy, the stable system prompt owns delegation strategy, and each compact `PlanQuery` asks only for the next grounded plan and action.
- **Examples render only where they apply.** Delegation examples and policy require spawn budget, so a leaf no longer carries a walkthrough it cannot act on. Stable strategy stays in the system prompt for caching, while recurring `PlanQuery` nodes remain short enough to retain as auditable history.
- **Prompt depth status follows each agent's config.** A root-level `max_depth` override now controls delegation sections and status for its whole subtree instead of being shadowed by the `Flow` default.
- **File guidance moved onto the file tools.** `write_file` now says it replaces the whole file and to `ls`/`read_file` first; `ls` says to check the workspace before writing. Because these are `@tool` descriptions they reach the model only when `FILE_TOOLS` is registered, so the default prompt stays usable for runs with no filesystem — and it no longer mentions one.
- **The prompt names what is not bound.** A line listing invented-looking calls (`run_subagent(...)`, `call_tool(...)`, `get_result(...)`) as non-existent.
- **Truncation says the REPL survived it.** `TRUNCATION_SUMMARY` now notes that variables and imports from dropped turns are still bound, so an agent past its `keep_n_messages` window reuses them instead of redefining them.
- **Prompt selection is explicit.** Spawn specs use `prompt_profile`; children inherit their immediate parent's profile when omitted. `prompt_router` is an optional callable, and unknown profile names raise `ValueError`.
- **Runtime owns REPL lifecycle.** REPL lookup, replay, closure, and rebinding live on `Runtime`; warm branch attachment uses `Runtime.rebind_repl`.
- **Persistence and consumers accept Nodes directly.** Save/load uses `rlmflow.graph.persistence`; stream consumers handle one Node at a time.
- **The browser viewer is Gradio only.** It draws `rlmflow.view`'s SVG instead of building a Plotly figure, so `rlmflow[viewer]` no longer pulls Plotly or Kaleido in, and `rlmflow[image]` is CairoSVG plus Pillow. `open_viewer(source)` takes a root, a checkpoint directory, or a path to one; the old `states=`/`session=` arguments and the click-a-node-for-its-payload panel are gone — the step slider drives the detail panel, and an agent picker drives the transcript beside it.
- **A pre-rewrite checkpoint says so.** `persistence.load` on a graph written by the old engine (a flat `root_agent_id`/`agents` table) raises `ValueError` naming the format instead of failing on a missing key.

### Removed

- `rlmflow.pool`, `rlmflow.tasks`, `rlmflow.simple_flow`, the Event hierarchy, and the separate `Graph` wrapper.

## [0.4.2] — 2026-06-17

### Added

- **Single delegation surface: `launch_subagents`.** Agents now delegate through one `async` launcher installed in the REPL namespace. `await launch_subagents(specs)` takes a `list[dict]` only; each spec requires `query` and may set `num_steps`, `context`, `name`, and `model`. It returns a `list[str]` in spec order, even for a one-item list. Sequential pipelines use one-item calls and thread each result into the next spec's `context`; parallel fanout uses a multi-item list. The launcher is registered as a real core tool and composes over `flow_delegate` / `flow_wait`, so it behaves identically on local and remote runtimes.
- **Shared LLM scheduler channel.** All agent LLM turns and `llm_query_batched(...)` calls now route through one per-run `LLMChannel`, keyed by model/client. `FlowConfig.llm_max_concurrency` controls the global LLM request cap; unsafe clients are serialized behind a per-client lock, and safe clients can run concurrently within the channel limit. Batched one-shot calls now also accept common sampling kwargs: `temperature`, `top_p`, `max_tokens`, and `stop`.
- **Thread-safe per-request usage accounting.** `LLMClient.completion(...)` returns `(text, LLMUsage)` for each request. `OpenAIClient` and `AnthropicClient` implement it directly from provider response usage, so the engine no longer depends on racy shared `last_usage` reads under nested batching.
- **Example smoke runner.** `python examples/run_examples.py` runs the deterministic/offline example suite by default, with opt-in flags for optional dependencies, live LLM examples, sandbox providers, and manual viewer/interactive checks.
- **Tinker inference client.** `TinkerClient` adapts the Tinker SDK and `tinker-cookbook` renderers to the `LLMClient` interface, with an optional `rlmflow[tinker]` extra and a live-view example in `examples/providers/tinker_agent.py`.
- **Stricter local install checks.** `make install` now runs `ruff check .` through the existing lint target before installing the package.
- **Supervisor injection example.** `examples/control/injection/` generates a real word-search run, forks it, replaces supervising nodes with prompt-based graph edits, validates structured results, and continues with a live LLM.
- **Simpler iteration defaults.** `FlowConfig.max_iterations` is unbounded by default (`None`), while delegated children use the global `child_max_iterations=20` engine policy by default.
- **Direct child-writes-file prompt pattern.** The default prompt's multi-file fanout example no longer teaches a `PATH: <path>` answer header. It passes target paths through child `CONTEXT` and demonstrates plain Python `pathlib.Path(...).write_text(...)` writes, either in the parent or directly inside child agents.

### Breaking

- **`flow_delegate` / `flow_wait` are now internal primitives, and `launch_subagent` has been removed.** Agent code, the default prompt, examples, and docs all use `launch_subagents([...])` instead. The AST check (`check_wait_syntax`) now permits `await launch_subagents(...)` at action-block top level. Update any custom prompts or hand-scripted REPL fixtures.
- **`OrphanedDelegatesError` removed.** Because spawn and wait are fused inside the launchers, an un-awaited delegate is no longer expressible, so the orphaned-delegate detection, its `ErrorOutput(error="orphaned_delegates")` node, and the remote exception-injection path are gone.
- **`FlowConfig.async_children` renamed to `eager_children`.** Same semantics (work-conserving child drain once a parent is supervising); update config literals and any persisted `agent.json` fixtures.
- **`Node.injected` and `Node.injected_reason` removed.** Injected controller nodes are now stored with the exact same schema as ordinary graph states. `Graph.inject(...)` no longer accepts or persists a reason string; external controllers that need provenance should track it outside the node payload.

## [0.3.2] — 2026-05-28

### Added

- **Per-agent LLM transcript log.** Every workspace now writes `session/<aid>/transcript.json` — a *single* document per agent that grows turn-by-turn. `messages` is the flat conversation as the LLM saw it across every turn so far (`[{role: system|user|assistant, content: ...}, ...]`); `metadata` is a parallel list with one dict per message. Each call appends only the new entries (any user nudges plus the assistant reply) — no duplicated prefix. Per-assistant metadata: `{ts, model, force_final, input_tokens, output_tokens, elapsed_s, after_node_id, after_seq}`. Every other message gets `{}`. Exact ground truth for "what did the LLM see?" — useful for debugging prompt issues, replaying a turn under a different model, or auditing context bloat. The `Session` ABC gains `read_transcript(agent_id)` and `write_transcript(agent_id, transcript)`; `FileSession` round-trips through `transcript.json`, `InMemorySession` keeps it in `agent_transcripts`. Transcript read/write failures are swallowed so persistence issues never break a run.

### Breaking

- **`delegate` / `wait` renamed to `flow_delegate` / `flow_wait`.** The two engine-bound REPL tools used names that were too generic and shadowed common identifiers in agent code. They are now namespaced as `flow_delegate(*, name, query, context, max_iterations=None, model="default")` and `flow_wait(*handles)`. The default system prompt, built-in examples, error messages (`OrphanedDelegatesError`, refusal strings), and the AST check that enforces `yield` before `wait` all use the new names. Update agent prompts, custom prompt builders, and any test fixtures that script REPL code by hand.
- **Node taxonomy expanded to a 9-leaf, 4-base-class hierarchy with strict obs → action alternation.** Every action is now followed by exactly one observation; outputs no longer share a node with the action that produced them. New leaf classes (and wire-format `type` tags): `UserQuery`, `LLMAction`, `LLMOutput`, `ExecAction`, `ExecOutput`, `SupervisingOutput`, `ErrorOutput`, `DoneOutput`, `ResumeAction`. Base classes (Python-only — not on the wire): `Node`, `ObservationNode`, `ActionNode`, `CodeObservation`. The old `outcome=` enum on `ExecAction`, the unified `ActionNode` (LLM-call) shape, and `SeedAction` / `ResultNode` / `ErrorNode` / `QueryNode` / `SupervisingNode` / `ResumeNode` are gone. Predicates follow the new taxonomy: `is_user_query`, `is_llm_output`, `is_llm_action`, `is_exec_output`, `is_exec_action`, `is_supervising`, `is_errored`, `is_done`, `is_resume_action`, `is_resumed`, `is_observation`, `is_action`, `is_code_observation`. `LLMOutput.code` is the source of truth for executed code; `ExecAction` / `ResumeAction` carry an optional echo only. See [`docs/node_model.md`](docs/node_model.md).
- **`ErrorOutput` (formerly `ErrorNode`) now distinguishes runtime exceptions from normal `ExecOutput`.** The REPL protocol surfaces an `errored` flag; `engine/transitions.py` writes `ErrorOutput` whenever the runtime reports an exception (including `SyntaxError` and the synthetic "no code block" case), instead of mixing tracebacks into `ExecOutput`.
- **LLM clients retry transient failures via `tenacity`.** The `chat` and `stream` methods on `OpenAIClient` and `AnthropicClient` retry on transient HTTP / protocol errors. The module-level `_`-prefixed helpers and constants in `rlmflow/llm.py` are now public.
- **Workspace step retracing.** `Workspace.load_steps()` returns the full history as a list of progressive `Graph` snapshots. The retrace simulates **unbounded `max_concurrency`**: every tick advances all currently ready agents in lockstep, producing one snapshot per tick. The viewer / `save_steps` / `save_gif` / `save_html` / `open_viewer` deduplicate consecutive frames that collapse to the same visualization (e.g. action nodes hidden by their paired observation), so the resulting slider/animation shows only visually distinct steps.
- **Viewer renames + node collapsing.** "Yielded" is now "supervising" everywhere in display surfaces. By default the figure renderer hides bookkeeping action nodes whose paired observation has already been written: `llm_action` collapses into `llm_output`, `exec_action` / `resume_action` collapse into `exec_output` or `supervising_output`. Terminal outcomes (`done_output`, `error_output`) are **never** collapsed — the preceding action stays visible so `... → exec → done` and `... → resume → errored` read explicitly. Action nodes that are the latest state on an agent also stay visible so progress is observable. The state-detail panel in `open_viewer` renders each state as a distinct, color-coded, type-labeled block.
- **Data model is now one recursive class.** `AgentMeta` is gone — its fields are flat on `Graph` itself (`graph.query`, `graph.config`, `graph.runtime`, `graph.depth`, `graph.model`, `graph.system_prompt`, `graph.parent_agent_id`, `graph.parent_node_id`). `Graph` is a frozen `dataclass` with `states: tuple[Node, ...]` and `children: dict[str, Graph]` for sub-agents. Cross-agent navigation is `graph[other_aid]`; subtree views are `graph.agents`, `graph.all_nodes`, `graph.edges`.
- `Graph.from_agent_states(...)` is removed. Build `Graph` instances directly (frozen dataclass) or rely on `Session.load_graph()`.
- `Edge` no longer ships as a stored object on `Graph` — `graph.edges` derives `flows_to` from each agent's state order and `spawns` from each child's `parent_node_id`. The class survives as a `NamedTuple` for viz consumers.
- `Session.write_agent` now takes a `Graph` (not an `AgentMeta`). `Session.record_spawn` is removed; the parent link is captured on the child's `parent_node_id` field.
- `Graph.events` is now `Graph.nodes` — every `Node` represents the agent's *state* at one step in its trajectory, not a discrete event.
- `latest.json` writes `latest_node_id` instead of `latest_event_id`.

## [0.2.1] — 2026-05-10

### Changed

- Workspace persistence now uses per-call `session/<agent-id>/session.jsonl` logs plus a top-level `graph.json` manifest for graph structure and state ordering.
- Removed old workspace compatibility paths; `FileSession(path)` and `FileContext(path)` now treat `path` as the current workspace root layout.
- Removed the redundant `CONTEXT.fork()` REPL helper; pass `CONTEXT.read()` or a slice explicitly to `delegate(...)`.
- Added public prompt customization docs covering `PromptBuilder`, `FlowConfig.system_prompt`, and dynamic prompt overrides.

## [0.2.0] — 2026-05-08

### Breaking

- `delegate(name, query, context, *, model=...)` — `context` is now **mandatory** and **positional**. The previous `context=None` keyword default is gone. Pass `""` for code-only delegations. This eliminates a silent footgun where children inherited the parent's payload by accident; every delegation now declares its child's input explicitly. Migration: `delegate("name", "query")` → `delegate("name", "query", "")`.
- `RecursiveFlow.start(query, *, context=None)` — `context` is keyword-only and optional (root agent gets `""` if omitted). No call-site changes needed for callers passing only a query.

### Added

- "Inline first" strategy bias in the default prompt: when the parent can write a known multi-file artifact end-to-end itself, do not delegate per-file. Multi-file delegation example replaced with a parent-writes-everything-inline example. Sibling-interface guardrail added for the cases where delegation IS the right call (children must USE sibling names as-is and PRODUCE their own exports in the exact shape the contract declares).
- `tests/test_prompt_capabilities.py` — snapshot-style tests that pin the default prompt's required vocabulary so future trims don't drop load-bearing phrases.
- `tests/test_session_variable.py` — `SessionVariable` tree-navigation methods (`parent`, `ancestors`, `children`, `subtree`, `tree`) derived from real cross-agent edges.
- `examples/_data/notebook-coding-agent/` — canonical saved trace shared by `coding_agent.ipynb` (generator), `node_basics.ipynb` (querying), and `viz_walkthrough.ipynb` (rendering).
- CI workflow (`.github/workflows/ci.yml`): ruff + pytest matrix on 3.11 / 3.12 / 3.13, runs on every PR and `push: main`. Tag-driven publishing remains in `release.yml`.
- Coverage instrumentation: `pytest-cov` in `[dev]`, `--cov=rlmflow` in CI, `[tool.coverage.*]` config in `pyproject.toml`.
- Early OOLONG benchmark harness for flat-vs-RLM comparison. This has since been superseded by the shared `benchmarks/eval/` harness.
- `rlmflow.utils.save_image(node, path, ...)` — render a node's graph to PNG/SVG/PDF. Markers, edges, and fonts auto-scale via `element_mult` so the tree stays visually balanced on the larger export canvas. Promoted from a one-off notebook helper.
- `rlmflow.utils.save_steps(states, dir, ...)` — multi-snapshot variant: writes one image per state under `dir`.
- `rlmflow.utils.render_html(states, ...)` / `rlmflow.utils.save_html(states, path, ...)` — single-file standalone stepper. Each slide pairs the Plotly graph for one snapshot with that snapshot's transcript and a node table; bottom nav has arrows + dots, plus keyboard left/right. Drop the file in a PR comment, attach to a CI artifact, or commit it next to the trace it came from. Promoted from `examples/blog_needle_graph.py:render_html_viewer`.
- `rlmflow.utils.save_gif(states, path, ...)` — animate a trace as an autoplay GIF. Renders each state to PNG with kaleido, then stitches frames with Pillow. Lazy-imports Pillow (raises a clear ImportError otherwise) so `[image]` stays focused on still exports.
- `Node.save_image(path, ...)` and `Node.save_html(path, ...)` shorthands for the helpers above.
- `Node.plot(..., element_mult=)` and `node_plot(..., element_mult=)` — scale markers/edges/fonts on the returned Plotly figure. Default `1.0` keeps the on-screen layout; bump for hi-res rendering.
- Split scaling on `node.plot()` / `save_image` / `save_steps` / `save_gif`: `marker_mult` and `text_mult` override `element_mult` separately, so labels can stay small (e.g. `2.2`) while marker dots get fat (`3.5`). Fixes label collisions on dense trees.
- `normalize_labels=` on `node.plot()` and the save helpers — forces every node label to `bottom center` so adjacent depths can't share the same vertical band. Default off for `node.plot` (on-screen alternation still looks fine), default on for `save_image` / `save_steps` / `save_gif` / `Node.save_image`.
- CLI: `rlmflow render <trace> -f steps -o frames/` gains `--marker-mult`, `--text-mult`, `--normalize-labels` / `--no-normalize-labels` flags (also work with `-f image`). One invocation now replaces the per-blog one-off scripts.
- `[image]` optional extra (`pip install rlmflow[image]`) — pulls `plotly` and `kaleido` for static image export.

### Changed

- Default system prompt rewritten end-to-end. Sections reordered to capabilities-first (Role → REPL → Strategy → Tools → Context → Recursion → Session → Guardrails → Examples → Status). Per-section prose tightened (~20% fewer tokens, zero outbound URLs in the shipped prompt). Examples reduced to five canonical patterns: small task, chunk-and-aggregate, self-contained multi-file (inline), cross-agent recovery, reviewer (`CONTEXT.read()`).
- `[viewer]` extra now declares its `plotly` dependency directly. The unused `[viz]` extra was removed (`plotly` was previously declared there but only imported by the gated `[viewer]` code path).
- Python support clarified: `requires-python = ">=3.11"` matches the shipped classifiers (3.10 dropped — never tested in CI). Ruff target bumped to `py311`.
- Project status classifier: `Alpha` → `Beta`.

### Fixed

- Boids notebook regression: cross-file schema drift (`Boid.pos.x` vs flat `boid.x`) caused by an over-strict guardrail and an over-aggressive multi-file-delegation example. Repaired by adding the bidirectional contract guardrail and replacing the example.
- Notebook agent ids reflect filename sanitization (`root.index_html`, `root.styles_css`, etc.) — `.` is the agent-tree delimiter, so filenames with dots are sanitized to underscores. `node_basics.ipynb` and `viz_walkthrough.ipynb` updated.
- Static example payloads no longer use the deprecated 2-arg `delegate(...)` form (`view_demo.py`, `showcase.py`, `best_of_n.py`).

## [0.1.3] — 2026-04-29

- Engine refactor: graph-first replay path, deterministic stepping semantics tightened, additional integration tests.

## [0.1.2] — 2026-04-29

- Renamed package to `rlmflow`. Session and context layout consolidated under `Workspace` with explicit `fork()`. Major engine refactor toward the typed-node graph model.

## [0.1.1] — 2026-04-23

- `rlmflow` CLI shipped: `view`, `render`, `version` subcommands; `render -f` accepts mermaid / mermaid-flowchart / mermaid-sequence / dot / d2 / tree / ascii-boxes / gantt-html / report-md / code-log / error-summary / tokens.

## [0.1.0] — 2026-04-23

Initial release.

- Recursive `RecursiveFlow` engine with typed nodes (`QueryNode`, `ActionNode`, `ObservationNode`, `SupervisingNode`, `ResumeNode`, `ResultNode`, `ErrorNode`).
- Runtimes: `LocalRuntime`, `SubprocessRuntime`, `DockerRuntime`, `ModalRuntime`.
- `Workspace` with `Session` (event log) and `Context` (data payload) stores, both with `fork()`.
- Visualization: terminal `live` view, mermaid / dot / d2 / sequence exports, gantt HTML, code-log, error-summary, token sparkline, budget burndown, bench table, Markdown report, Slack/Discord webhooks, Gradio viewer.
- Optional extras: `[openai]`, `[anthropic]`, `[viewer]`, `[all]`, `[dev]`.

[0.1.0]: https://github.com/shyamsn97/rlmflow/releases/tag/v0.1.0
[0.1.1]: https://github.com/shyamsn97/rlmflow/compare/v0.1.0...v0.1.1
[0.1.2]: https://github.com/shyamsn97/rlmflow/compare/v0.1.1...v0.1.2
[0.1.3]: https://github.com/shyamsn97/rlmflow/compare/v0.1.2...v0.1.3
[0.2.0]: https://github.com/shyamsn97/rlmflow/compare/v0.1.3...v0.2.0
[0.2.1]: https://github.com/shyamsn97/rlmflow/compare/v0.2.0...v0.2.1
[0.3.2]: https://github.com/shyamsn97/rlmflow/compare/v0.2.1...v0.3.2
[0.4.2]: https://github.com/shyamsn97/rlmflow/compare/v0.3.2...v0.4.2
[0.5.0]: https://github.com/shyamsn97/rlmflow/compare/v0.4.2...v0.5.0
[0.6.0]: https://github.com/shyamsn97/rlmflow/compare/v0.5.0...v0.6.0
[unreleased]: https://github.com/shyamsn97/rlmflow/compare/v0.6.0...HEAD
