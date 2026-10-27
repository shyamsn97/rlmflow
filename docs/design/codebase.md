# rlmflow, file by file

Every Python file under `rlmflow/`: what it is for, what is in it, and who uses
it. Line counts are current; "+ x.py" names the modules a file absorbed in the
flattening pass. The package layout and its import rules are in
[design/file_structure.md](file_structure.md).

## The one-paragraph picture

A run is a `Graph`: an append-only log of nodes (what happened) and edges (how
the nodes relate). `graph/` is that store plus the pure reads and the history
writes over it. `Flow` (`flow.py`) is a driver: it finds agents in the graph that
have work to do, runs one step for each through `engine/`, and appends what the
step produced. A step is one transition: call the model, run the code it wrote
in a REPL, or deliver a message. REPLs live in `runtime/` and are created lazily
the first time an agent needs one, replaying its recorded code if it has
history. The tools an agent can call from its REPL are in `tools/`. Everything
else reads the graph: `prompts.py` turns an agent's history into chat messages,
`consumers.py` and `view/` watch or render it, `rao/` scores it for training.

Imports only point down this stack; `tests/test_layers.py` enforces it.

```text
6  cli/ consumers.py view/ rao/                     apps on top
5  flow.py         the driver       find ready agents, run steps, append results
4  engine/         steps            transitions and defaults, scheduler, handoff
3  tools/          agent's hands    submit, launch_agent, send/ask/fork/connect, AGENTS, files
   prompts.py      agent's eyes     prose, system prompt sections, chat projection
2  llm/            model clients    OpenAI, Anthropic, Tinker, retries, streaming
   runtime/        live state       REPL workers (local, subprocess, Docker, Modal)
   integrations/   third parties    Claude Code, DSPy, Instructor, jev
1  graph/          the data         Graph, nodes, edges, reads, ops, save/load
0  utils.py  structured.py
```

______________________________________________________________________

## Top level

### `__init__.py` — 299 lines

The public API: re-exports `Flow`, `Graph`, every node type, `AgentConfig`,
runtimes, LLM clients, consumers, and structured-output helpers,
so users write `from rlmflow import ...`. `Flow`, `FlowTUI`, `DSPyFlow` and the
view names load lazily through `_LAZY`, so `import rlmflow` doesn't pull in
Textual, DSPy or Gradio.

### `__main__.py` — 6 lines

Lets `python -m rlmflow` run the same CLI as the installed `rlmflow` script.

### `flow.py` — 712 lines (+ `FlowLLM` from `adapters.py`)

`Flow`, the driver. It owns everything that is *not* durable: the LLM clients,
tools, prompt builders, the REPL runtime, the structured-output engine, and the
task queue while a stream is running. It holds no run state of its own.

- **Driving:** `run_streaming(g, until=, close_repls=)` is the core loop. It
  calls `rebuild_unfinished_repls`, then repeatedly asks `g.ready()` which
  agents have work, delivers queued messages (`g.deliver`), submits one step
  per agent to the `TaskQueue`, yields every node and edge that lands, and stops
  at the `until` boundary or when nothing is ready. `run`/`arun` wrap it.
  `step(g, agent)` runs exactly one step and `replay(g, agent)` rebuilds one
  agent's REPL.
- **One step (`_drive`):** picks the transition with `resolve_transition`,
  produces the next node, and appends it with a stale-frontier guard. After it
  lands, `_react` closes REPLs of finished agents (per `ContextPolicy`) and
  calls `handoff.react` so every agent it reports to gets its result.
- **Model calls:** `build_system_prompt`, `build_messages`, `call_chat`,
  `llm_for_step`, `llm_query`, and `structured_query`.
- **Tools:** `build_tools`/`build_namespace` assemble the REPL namespace from
  `tools/`; `inject`, `add_tool` and `remove_tool` change it at runtime.
- **Budget:** `budget_exhausted`/`budget_exceeded` implement `max_budget`.
- `Flow.transition(...)` registers application transitions that run before the
  defaults.
- `FlowLLM` exposes a `Flow` through a synchronous `chat(messages)` call, so one
  agent can be another agent's model; `messages_to_query` flattens chat
  messages into a task string.

### `structured.py` — 451 lines (`structured/codec.py` + `structured/engines.py`)

Schema-constrained answers. Imports nothing from rlmflow, so the graph layer may
use it.

- **Codec:** `OutputCodec` is the replaceable interface; `JsonSchemaCodec`
  (exported as `DEFAULT_OUTPUT_CODEC`) turns Pydantic models into JSON Schema,
  renders the contract for the prompt, parses model text, and validates values.
  `as_finish_value` makes sure a submitted answer is strict JSON.
- **Engines:** how `Flow.structured_query` produces a value that matches a
  schema. `ChatStructuredEngine` (the default) asks the chat model for JSON.
  `RetryingEngine`, `FallbackEngine` and `ConfidenceGate` wrap other engines.

### `consumers.py` — 352 lines (`consumers/{base,checkpoint,sync,live_tree}.py`)

Things that watch a running stream; each has `handle(item, graph)`.
`StreamConsumer` and `ConsumerGroup` (fan one stream out). `GraphCheckpointer`
saves the graph periodically and on close. `WorkspaceSync` mirrors a run
directory while streaming (used by `examples/autoresearch`). `LiveGraphTree`
and `LiveTreeRenderer` draw live terminal trees through `view/text.py`.

### `utils.py` — 129 lines (`utils/{code,helpers,serial}.py`)

Finds fenced code blocks and `repl` blocks (`find_code_blocks`,
`has_repl_fence`), `truncate_output`, `sampling_kwargs`, and the cloudpickle
encoding for host-to-worker values. Imports nothing from rlmflow; the REPL
worker uses it.

______________________________________________________________________

## `graph/` — the data

### `graph/graph.py` — 439 lines (+ `records.py`)

`Graph`, the store, and `Edge` with its kinds. Every durable fact of a run is a
`Node` or an `Edge` in one ordered log.

- **Edges:** `NEXT` links an agent's consecutive nodes. `FROM` links a node to
  what caused it: a child's `AgentStart` to the launching action, or a message to
  its sender. `DEPENDS_ON` means an agent waits for a producer's result. These
  three are `STRUCTURAL`; anything else (`REPORTS_TO`, `peer`, ...) is an
  application edge.
- **History writes:** `append(agent, node, after=)` lands a node, raising
  `StaleFrontier` if the frontier moved; `spawn` creates an agent with optional
  launcher, dependencies and inputs; `send` queues a message and `deliver` moves
  queued messages into history; `fork(at, name=)` creates an agent sharing
  history up to `at`; `attach(parent, *agents)` adopts agents. Each checks its
  preconditions under the graph's lock, then logs its nodes and edges at once.
- **Agent queries:** `history`, `prev`, `frontier`, `terminal`, `status`,
  `result`, `inbox`, `producers`, `inputs`, `ready`, `launcher`, `children`,
  `launched`, `lineage`, `path`, `depth`, `tree`, `usage`, `cost_usd`.
- **Other writes:** `add_edge(edge)` records an
  application edge between existing nodes, and `remove_edge(edge)` deactivates
  one: the edge stays in the log, marked removed. `merge(other)` appends another
  graph's log (ids must not collide). `as_of(seq)` copies the graph as it stood
  at a log position. `_append_unchecked(*items)` is the private primitive under
  all of them: it indexes and logs, checking only that ids are new.
- **Lookups:** `node(id)`, `nodes(Type, agent=)`, `edges(kind=, source=, target=)`, `items(after=seq)`.
- **`check_invariants(g)`:** the whole-graph walk `Graph.load` runs once: every
  node owned by an agent, one predecessor per node, each agent's history one
  chain, forks only at landed nodes, an acyclic launcher tree, and acyclic
  `depends_on` declared before the consumer starts.
- **Errors:** `InvariantError` for a write or a log that would corrupt the
  structure; `StaleFrontier` for an append based on an old frontier.

### `graph/nodes.py` — 699 lines (+ `usage.py`, node payload keys from `records.py`, turn-node text from `prompts/text.py`)

The node types and the step context.

- **Prompt-side nodes:** `AgentStart`, `UserQuery`, `FinalQuery`,
  `RepairQuery`, `TruncationSummary`.
- **Model turns:** `LLMOutput`, with its `LLMUsage` token counts.
- **Actions:** `ExecAction` (code to run, plus saved tool-call receipts),
  `BackendRun`, `AppendChild`, `TransitionDecision`.
- **Outputs:** `ExecOutput`, `ErrorOutput`, `ReplDead`, `VerificationFailure`,
  `DoneOutput`.
- **Step context:** `Step(graph, agent, node)` plus `running_step()` and
  `current_step()`.
- **Payload keys:** `AGENT_REF_KEY`/`agent_ref_id` (the format of launch
  receipts) and `ASK_KEY` (a `UserQuery.metadata` key); `new_node_id` and
  `new_agent_id`.

### `graph/config.py` — 249 lines (+ `arguments.py`)

Per-agent settings. `AgentConfig` holds name, `backend`, model, `max_depth`,
`max_iters`, `max_budget`, `output_schema`, prompt profile, `reuse_worker`,
`resume_on_child_done`, `context_policy`, and `child_config` (field overrides
for launched children). `ContextPolicy`, `Input`, `NATIVE_BACKEND`,
`can_spawn`, `at_leaf`, the name validators, and the argument walkers
`transform_argument`/`visit_argument` that persistence and launch use.

### `graph/persistence.py` — 511 lines

The on-disk format: the graph's item log plus content-addressed `blobs/` for
large arguments. `save_graph` writes atomically; `load_graph` indexes the log
and runs `check_invariants` once. `upgrade_v1` converts runs saved in the old
tree format.

### `graph/__init__.py` — 83 lines

Re-exports the graph package.

______________________________________________________________________

## `engine/` — how a node becomes the next node, and when

### `engine/steps.py` — 679 lines (`steps/{transitions,decide,defaults}.py` + `AgentBackend`)

What one step does.

- **Mechanism:** `@transition(source=..., when=...)` declares which node types a
  producer accepts and under what condition; `resolve_transition` picks the
  first match. A `when` string makes a semantic transition, chosen once by
  `choose_transition` from a bounded read-set and recorded as a
  `TransitionDecision`.
- **Defaults** (`DEFAULT_TRANSITIONS`, first match wins): `to_backend`/
  `run_backend`, `to_final`, `to_summary`, `to_repair`, `complete`, `to_action`,
  `to_text_reply`, `to_missing_action`, `run_repl`. The module docstring has the
  table.
- **Backends:** `AgentBackend`, the protocol for running an agent outside the
  model-and-REPL loop, and `check_backends` for `Flow(backends=...)`.

### `engine/scheduler.py` — 309 lines (`queue.py` + `boundaries.py`)

`TaskQueue` runs at most one step per agent and reports finished steps
(`StepResult`); `join(predicate)` backs waits. `Pool`, `ThreadPool` and
`SequentialPool` decide where blocking model calls run. `resolve_boundary`
maps `run_streaming(until=...)` (`"next"`, `"idle"`, `"error"`, `"done"`, or a
callable `Boundary`) to a stop predicate.

### `engine/handoff.py` — 152 lines

Delivers finished agents' results to the agents they report to: the launcher,
plus any agent with a `reports_to` edge from the finished one. `react(g, agent)` sends one batched message with `from` edges to each `DoneOutput`.

### `engine/__init__.py` — 25 lines

Re-exports the scheduler names.

______________________________________________________________________

## `llm/` — chat clients (was `models/`)

### `llm/__init__.py` — 323 lines (`base.py` + `pooled.py`)

`LLMClient`, the protocol; `LLMChunk`, `as_chunk`, `join_chunks` and
`stream_with_retry` for streaming; `is_retryable`/`retry_transient`;
`PooledLLMClient`, which runs blocking calls on the flow's compute pool; and
`client_for(model)`, which picks Anthropic for `claude*` and OpenAI otherwise.

### `llm/openai.py` — 102, `llm/anthropic.py` — 101, `llm/tinker.py` — 210 lines

`OpenAIClient`, `AnthropicClient`, `TinkerClient`: one file per optional SDK.

______________________________________________________________________

## `runtime/` — REPL workers

### `runtime/runtime.py` — 487 lines (+ `connections.py`, `env.py`, `engine/restore.py`)

`Runtime` owns one logical REPL per agent (`repl_for`, `get`, `execute`,
`close_repl`/`close_repls`). `LocalRuntime` and `SubprocessRuntime` run workers
on this machine; `WrappedRuntime` and `ExecutionGuard` wrap execution.
`PopenConnection` is the lazily started subprocess transport. The `RLMFLOW_*`
constants and `agent_env` are the metadata agent code reads from `ENV`.
`rebuild_repl`, `ensure_repl_rebuilt` and `rebuild_unfinished_repls` rebuild a
namespace by re-running an agent's recorded `ExecAction`s with saved receipts.

### `runtime/repl.py` — 401 lines (+ `references.py`, tool metadata)

The worker-side REPL contract: `Repl`, `ReplRun`, `ReplStatus`, `DoneSignal`,
`VerificationSignal`, per-agent bindings. `ToolMetadata`/`get_tool_metadata`
are what the REPL reads to dispatch proxied calls. `AgentReference` and
`encode_agent_references` carry agent handles across the boundary as data,
with a literal copy of `AGENT_REF_KEY` (a test keeps it equal to the graph's).

### `runtime/repl_client.py` — 573 lines

The host side of a worker process: `WorkerSession` (one connection, several
tenants) and `WorkerRepl`. Proxied tool calls are saved as receipts on the
`ExecAction` and returned on replay.

### `runtime/repl_server.py` — 459 lines (+ `protocol.py`, verifier fingerprints)

The worker process (`python -m rlmflow.runtime.repl_server`) and the typed
JSON-line protocol it speaks. `fingerprint`/`schema_hash` hash verifiers and
schemas; the host imports them so both sides agree. It imports only
`runtime/repl.py`, `structured.py` and `utils.py`, which a test enforces.

### `runtime/docker.py` — 109 lines and `runtime/modal.py` — 159 lines

`DockerRuntime` and `ModalRuntime` run the same worker in a container or a Modal
sandbox.

______________________________________________________________________

## `tools/` — what agents call

### `tools/namespace.py` — 236 lines (+ `decorator.py`)

The `@tool` decorator, `tool_name`, `tool_allowed`, `format_tool_docs` for the
prompt's `### Tools` list, `RESERVED_TOOLS`, and building an agent's REPL
namespace (`build_namespace`, `wait_agent`, `inject`, `add_tool`,
`remove_tool`).

### `tools/builtins.py` — 336 lines (+ `completion.py`, the host-side verifier check)

Tools every agent gets: `submit`, `llm_query`, and (when allowed)
`launch_agent`, `fork_agent` and the messaging tools. `submit_candidate` and
`node_for_candidate` are the one completion contract for REPL submissions and
external backends; `run_verifier` runs a launch's verifier on a candidate.

### `tools/launch.py` — 361 lines (+ `fork_agent`)

`launch_agent`: validates the child's task, name, depth and JSON arguments,
splits handle arguments into `depends_on` inputs, creates the child, and
registers per-launch verifiers. `fork_agent` branches an agent's history.

### `tools/messaging.py` — 211 lines (was `graph_tools.py`)

`send_agent`, `ask_agent`, `connect_agent`/`disconnect_agent` (through
`Graph.add_edge`/`remove_edge`), and `waiting`, which refuses wait cycles.

### `tools/agents_view.py` — 468 lines (was `directory.py`)

The read-only `AGENTS` snapshot (`AgentDirectory`, `AgentInfo`), `AgentHandle`,
and `resolve_agent`/`agent_handle` for turning a selector into an agent.

### `tools/files.py` — 167 lines (`filesystem.py` + `coding.py`)

`FILE_TOOLS` (`read_file`, `write_file`, `edit_file`, `grep`, ...) and
`CODING_HANDOFF_SCHEMA`.

### `tools/__init__.py` — 28 lines

Re-exports.

______________________________________________________________________

## `prompts.py` — what the model sees (was `prompts/`)

### `prompts.py` — 477 lines (`text.py` + `prompts.py` + `sections.py` + `messages.py`)

The system-prompt prose (`CORE_PROTOCOL_TEXT`, `EXECUTION_POLICY`,
`AGENTS_TEXT`, ...), `Prompt` and its sections, `PromptProfile`, the default
system prompt (`prompt_for`, modifiers, argument sections), and message
assembly (`build_messages`, `render_tools`). The text of individual turn nodes
(`FINAL_ANSWER_ACTION`, `TRUNCATION_SUMMARY`, `COLD_REPL_NOTE`) lives with
those nodes in `graph/nodes.py`.

______________________________________________________________________

## `integrations/` — third-party glue

### `integrations/claude_code.py` — 153 lines

`ClaudeCodeBackend`: runs agents with `claude -p` and returns the candidate;
`engine/steps.run_backend` turns it into a node through `node_for_candidate`.
Used by `examples/providers/claude_code_delegation.py`.

### `integrations/dspy.py` — 99 lines

`DSPyFlow`, a DSPy `BaseLM` adapter for anything with `chat(messages)`.

### `integrations/instructor.py` — 129 lines and `integrations/jev.py` — 122 lines

Optional structured engines backed by Instructor and TypeSafe Jev.

______________________________________________________________________

## `view/` — rendering a saved or live graph

### `view/figure.py` — 569 lines (+ `images.py`, `step_svg`)

`graph_svg(g)`: hand-written SVG, one marker per node, in a tidy-tree layout.
`step_svg` draws one step with the rest faded. `save_frames`/`save_gif`
rasterise steps (the `image` extra, imported inside the call).

### `view/html.py` — 352 lines (+ `app.py`)

`save_html`: a single self-contained stepper; `save_svg`. `open_viewer`: the
Gradio browser app (the `viewer` extra, imported inside the call).

### `view/text.py` — 93 lines

`render_tree`/`render_forest`, `top_level`, `launched_by`.

### `view/timeline.py` — 188 lines

`steps`/`timeline`, `replay`, `snapshot` (now `Graph.as_of`), `render_steps`.

### `view/__init__.py` — 69 lines

Re-exports.

______________________________________________________________________

## `cli/` — the `rlmflow` command

### `cli/__init__.py` — 268 lines (+ `app.py`, `config.py`, `options.py`)

The Fire entry point (`RlmflowCLI`, `main`), `rlmflow config` (`ConfigCLI`),
`rlmflow version` (`VersionCLI`), and `resolve_options`, which resolves run settings from flags,
environment, `./rlmflow.toml`, the user config, then defaults.

### `cli/run.py` — 336 lines

`rlmflow run` / `rlmflow tui`.

### `cli/tui.py` — 606 lines (was `consumers/tui.py`)

`FlowTUI`, the Textual dashboard.

### `cli/view.py` — 180 lines

`rlmflow view show` and `rlmflow render svg|html|gif|frames|browser`.

______________________________________________________________________

## `rao/` — Recursive Agent Optimization (training, unchanged)

`rollout.py` (`RolloutFlow`, `Collector`), `env.py` (`Env`, `EnvSession`,
`env_tool`), `credit.py` (per-agent scoring), `export.py` (`Trajectory`, JSONL),
`trainer.py` (`TinkerTrainer`, `JsonlTrainer`, `train`).

______________________________________________________________________

## What could be cut

Nothing below is dead: tests or examples cover all of it. These are the parts
the core graph-and-driver design doesn't need, roughly from most to least
removable.

| Code                                                                                        | Lines | Used by         | Notes                                                  |
| ------------------------------------------------------------------------------------------- | ----- | --------------- | ------------------------------------------------------ |
| semantic transitions in `engine/steps.py` (`choose_transition`, string `when`)              | ~120  | tests only      | Kept for Jev-style judged branching.                   |
| `integrations/instructor.py`, `integrations/jev.py`, the engine wrappers in `structured.py` | ~400  | tests only      | `ChatStructuredEngine` alone covers every in-repo use. |
| `WorkspaceSync` in `consumers.py`                                                           | ~60   | autoresearch    | Could live in that example.                            |
| `integrations/dspy.py`, `FlowLLM`                                                           | ~140  | two examples    | Integration glue, not core.                            |
| `integrations/claude_code.py` plus the backend transitions                                  | ~250  | one example     | Real feature (Claude Code as a child), but optional.   |
| `tools/namespace.py` aliases `finish` / `launch_subagent`                                   | a few | old transcripts | Kept so old REPL blocks still execute on replay.       |
