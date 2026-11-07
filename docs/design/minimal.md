# rlmflow, distilled

Status: proposal, not started
Date: 2026-11-06

rlmflow today is 11,921 lines in 52 modules. This document works out what the
library actually needs, from what it has to do rather than from the current
files, and specifies a version of about 5,200 lines with the same capabilities.
It keeps the semantics settled in [core-v2.md](core-v2.md) (the graph is the
engine, in-cell await, editable history, explicit edges, the plugin) and
replaces its file-by-file plan.

The four audits behind it covered every function in the package with its
callers. Where this document says something is unused, that is from those
searches.

## 1. What rlmflow is

Agents write Python in a persistent REPL. From that code they can launch other
agents, wait for their results in the same cell, and message agents they don't
own. Every model turn, cell, message and result is recorded in one graph, and
that graph is the run: the engine reads it to decide what happens next, you can
edit any part of it, and the next run redoes only what the edit affected. The
agent behind a node can be rlmflow's own REPL loop or a Claude Code or pi
session; they all reach the same REPL.

The pitch against heavier frameworks is that there is little to learn and
little to read. A user learns six things on the host and six names in the REPL,
and the whole engine fits in an afternoon's reading.

## 2. Every concept

**On the host:**

| Name | What it is |
| --- | --- |
| `Graph` | The run: agents, their nodes, and the edges between them. Query it, edit it, save and load it. |
| `Flow` | Runs a graph: steps every ready agent until nothing is left to do, yielding each node as it lands. |
| `AgentConfig` | A named kind of agent: model, harness, prompt, limits, output schema. |
| harness | How one agent takes a turn: `"repl"` (built in), `"claude_code"`, `"pi"`, or your own class. |
| `tool` | A function agents can call from the REPL. A plain function works. |
| runtime | Where REPLs run: in-process, a local subprocess, Docker, or Modal. |

**In the REPL:**

| Name | What it does |
| --- | --- |
| `launch_agent(task, config=, name=, **args)` | Start an agent; returns a handle. |
| `await handle` | Its result, inside the running cell. Only for agents you launched, or their descendants. |
| `handle.send(text)` | Message any agent; `await` the send to wait for the reply, if it's yours. |
| `AGENTS` | Every agent in the run, by name or path: `AGENTS["root.worker-0"]`. |
| `submit(value)` | Finish with a result. |
| `llm_query(prompt, schema=)` | One model call with no history. |

Everything else is internal. Anything not on these two lists that is public
today becomes private or is deleted (§8).

## 3. The graph

### Records

```text
Agent   id, name, task, args, config (a name), nodes: [Node], inbox: [Node],
        base: Node | None (a fork's starting point), metadata: dict, seq
Node    id, type, data: dict, agent_id, seq, metadata: dict, started_at, finished_at
Edge    id, kind, source, target, data: dict, seq
```

### Node types

| Type | Written when | Key data |
| --- | --- | --- |
| `llm` | A model turn ends | `text`, `usage` |
| `code` | A cell starts; filled in when it ends | `code`, `output`, `error`, `calls` |
| `message` | Someone sends to the agent; sits in `inbox` until a turn boundary | `text`, `sender` |
| `idle` | A session ends its turn with nothing left to do but wait | `text`, `session_id` |
| `done` | The agent finishes | `result` |

What changes from today:
- **A cell that calls `submit` stays a `code` node**, followed by a separate
  `done`. Today the cell itself becomes the `done` node, so every "which nodes
  ran code" check has to look at two types plus whether `code` is set.
- **A cell is in the graph from the moment it starts** (§4, rule 5).

### Edges

| Kind | From → to | Meaning |
| --- | --- | --- |
| `spawn` | The cell (or agent, for host launches) → the child agent | Who launched whom. The parent is the owner of the source. |
| `message` | The sending cell → the `message` node | Who said what to whom. |
| `return` | The child's `done` → the cell that awaited it, or the message that delivered it | Where a result went. |

Applications add their own kinds with `g.add_edge(kind, a, b)`; the engine
never reads them.

What changes from today:
- **The `fork` edge goes.** A fork is `agent.base`; the viewer draws it from
  that.
- **`spawn` edges stop carrying `data["agent"]`**, a copy of the parent's id
  that is already the owner of the edge's source.

### Graph API (complete)

```text
reads    agent, node, owner, agents, top_level, nodes, edges, history, frontier,
         status, result, parent, children, subtree, path, usage
writes   create_agent, spawn, fork, append, update, send, deliver,
         add_edge, remove_edge, subscribe
edits    edit, delete, remove, replace            -> Invalidation; all take dry_run
files    save, load, copy
```

Removed from today's API, with what replaces each:

| Removed | Instead |
| --- | --- |
| `provenance` | One caller; it's a short loop over `history` and `edges`. |
| `cost_usd` | Becomes a field of `usage`. |
| `lineage`, `depth` | Private, behind `path`. |
| `is_ready`, `ready`, `finished` | `status`. |
| `unseen_results` | Moves to `flow.py`, its only user. |
| `calls`, the `Call` class | `cell.data["calls"]` (§5). |
| `prune_pending` | Not needed (§4, rule 5). |
| `import_subtree` | `g.add_agent(parent=...)`, or load the other graph. |
| `inbox()` | `agent.inbox`. |
| The module-level `cells()` | Private to `flow.py`. |

## 4. The engine in ten rules

This is the whole engine. Today these rules are spread across `is_ready`,
`status`, `finished`, `result`, `deliver`, `deliver_results`, `_schedule`,
`run_step`, `prepare_repl`, `rerun`, `_keep_stopped_cell` and `prune_pending`.
In the distilled version, rules 1–3 are one function, `status`, and the rest
live in `flow.py`.

1. **Status.** Look at the agent's last node:
   - no nodes yet → `running`;
   - a cell marked `rerun` → `running`;
   - `done` with an empty inbox → `done`;
   - `idle` with an empty inbox → `idle`;
   - anything else → `running`.

   An agent is *ready* when it is `running` and not already stepping. That one
   function is all of readiness.
2. **Results.** An agent's result is the `result` of its last node if that node
   is a `done` not marked `rerun`; otherwise `None`.
3. **Turn boundaries.** Queued messages move from `inbox` into history only when
   the last node isn't an `llm` node (a cell is about to run) and isn't a cell
   waiting to re-run.
4. **Stepping.** For each ready agent, take one step at a time: the harness
   returns the next node, or the Flow re-runs a cell marked `rerun`. A step that
   raises lands `done` with the error as its result.
5. **Cells land when they start.** A cell is appended to the graph before its
   code runs, and updated when it ends. So everything a running cell does
   (launches, messages) points at a node that exists. If a run stops or crashes
   mid-cell, that cell is simply marked `rerun`. `prune_pending`, the Flow's
   in-flight cell table and "never landed" go away.
6. **Awaiting.** `await handle` blocks the cell until that agent is `done`, and
   records a `return` edge from its `done` to the cell. Only descendants can be
   awaited; anyone else raises `NotYourAgent`.
7. **Delivering results.** At a turn boundary, children that finished and whose
   result reached nobody (no `return` edge from their `done`) are reported to
   the parent in one `message`, with a `return` edge from each `done`.
8. **Re-running.** An edit marks the cells that used changed data with `rerun`
   and cuts everything after them. On the next run, a marked cell runs again in
   place; its saved calls (§5) return what they returned before, or what
   `replace` put there.
9. **Rebuilding a REPL.** A REPL is rebuilt when it dies, after an edit, or
   after loading, by running the agent's finished cells again in order, with
   every saved call answered from its record. Nothing runs twice.
10. **Limits.**
    - `max_iters` caps model turns per agent; `max_budget` caps tokens for the
      whole run.
    - When either is about to be hit, the agent's next model turn is told it is
      its last.
    - If it still doesn't finish, it lands `done` with `[limit reached]`.

    This is one function, replacing today's four mechanisms: `budget_exhausted`,
    `budget_nearly_spent` with its final-turn message, `final_turn_spent`, and
    `MAX_BUDGET_EXCEEDED`.

## 5. Saved calls (receipts)

A cell records what each host tool call returned, so re-running or rebuilding
doesn't launch agents twice or pay for model calls again. Today this concept
takes eight names across two files:
- **In `nodes.py`:** `record_call`, `Node.calls`, `Node.launched`,
  `LAUNCH_TOOLS`, `AGENT_REF_KEY`, `agent_ref_id`.
- **In `graph.py`:** `AWAIT_CALL`, `Call`.

It is stored as `data["calls"][tool]["0"]`, and the tool decorator has
`restore=` and `receipt=` options just to shape it.

Distilled, it is one flat list in call order:

```json
"calls": [
  {"tool": "launch_agent", "agent": "agent_9f..."},
  {"tool": "await", "agent": "agent_9f...", "done": "node_41..."},
  {"tool": "llm_query", "value": "Paris"},
  {"tool": "read_issue", "value": {"title": "..."}}
]
```

- **Recording.** Every host tool call is recorded if its result is JSON, so
  there are no options. A launch records the agent, an await records the agent
  and the `done` it got, and anything else records its value.
- **Matching.** On replay, the n-th call to a tool gets the n-th record for that
  tool. Counting per tool keeps concurrent calls in one cell, such as a
  `gather` over a launch and an `llm_query`, stable across runs.
- **Missing records.** A call with no record runs live. That's how a re-run cell
  reaches new code, and how a non-JSON result is handled.
- **Replace.** `g.replace(cell, i, value=... | agent=...)` rewrites entry `i`
  and marks the cell `rerun`.
- **Where it lives.** One module, `graph/calls.py` (~60 lines), with `record`,
  `saved` and `rebind`.

## 6. Components

For each component: its job, what it keeps, what goes, and the size now →
target. Targets are ±15%.

### 6.1 `graph/`: records, graph, files (now 1,905 → 870)

| File | Now | Target | Contents |
| --- | --- | --- | --- |
| `nodes.py` | 461 | 200 | `Agent`, `Node`, `Edge`, `Step`, node/edge kind constants, the running-step context |
| `graph.py` | 932 | 500 | The API in §3, rules 1–3 as `status`/`result`/`deliver`, the edits with `Invalidation` |
| `calls.py` | — | 60 | §5 |
| `persistence.py` | 286 | 110 | `graph.json` (`agents`, `nodes`, `edges`, `edits`) and blobs for large string args |
| `config.py` | 226 | — | Moves to `rlmflow/config.py` (§6.9) |

**Removed:**
- **The metadata mixin.** `_Annotated` and `annotate()` go; `metadata` is a
  plain dict on `Agent` and `Node`, checked as JSON when saving. `record_metadata`
  keeps its append option.
- **Context variables.** Today there are three (`_active_step`, `_active_code`,
  `_replaying`); one, "the current cell", is enough. A replaying cell is a cell
  whose calls are answered from records, so it doesn't need its own flag.
- **The `Input` walkers.** `transform_argument`, `visit_argument` and `RECURSE`
  exist only for `Input` (§9) and nested blob encoding. Blobs apply to
  top-level string arguments only.
- **Leftovers.** `IMAGES_KEY` (a constant on the cell is enough), `isoformat`,
  `Node.timing()` and `strict_json` (one private helper replaces them).

### 6.2 `flow.py`: the driver (now 656 + 291 scheduler → 420)

**What `Flow` holds:** configs, model clients, tools, runtime, workspaces.

**What it does:**
- **`run_streaming(g, until=)`** runs rules 4–10 and yields `Step(graph, agent, node)`.
  - `until` is `"done"`, `"next"`, `"idle"` or a callable.
  - Stopping cancels in-flight steps, and their cells are marked `rerun`
    (rule 5).
- **`run`** and **`arun`** are thin wrappers around it.
- **`call_tool(cell, name, args)`** is the single place a host tool runs. It
  checks the saved calls first (§5), then calls the function. Both runtimes
  and the session server go through it.
- **The REPL lifecycle** is `prepare_repl` (rule 9), `run_cell` and `rerun`.

**Removed:**
- **The pools.** `Pool`, `ThreadPool` and `SequentialPool` go, with
  `Flow(workers=)`. A sync model client or tool runs in `asyncio.to_thread`;
  `Flow(max_concurrent=N)` is one semaphore around model calls.
- **The queue's extra layers.** `TaskQueue` becomes `asyncio.Task`s in a dict
  inside `run_streaming`, plus one `asyncio.Event` for "something changed".
  `StepResult`, `notify`/`join`/`changed`, `Boundary` and `resolve_boundary` go.
  `until` is resolved by a three-line dict.
- **Methods with no real caller:** `Flow.step`, `land` (cells land in
  `run_cell`), `namespace_for` (shepherd uses a tool), `replay` (the public
  rebuild; a stale agent is rebuilt automatically), `inject`/`remove_tool` (use
  `add_tool`), `resolve_config` and `config_for` (one `config_of(agent)`).
- **The stopped-cell bookkeeping:** `_cells`, `_keep_stopped_cell`, the
  `prune_pending` call and `g.stale`.

### 6.3 Harnesses (now 1,066 → 760)

A harness is `async def step(flow, agent) -> Node`. There are three kinds.

**`ReplHarness` (`harnesses/repl.py`, 278 → 200):**
- **The loop.** If the last node is `llm`, run its code block; otherwise call
  the model. A reply with no block is the final answer, and a malformed block
  becomes a cell that raises with the reason.
- **Removed:** `keep_n_messages` (§9) and the duplicate history helper.
  `final_turn` and `final_turn_spent` fold into the limits function (rule 10).

**`SessionHarness` (`harnesses/sessions.py`, 284 → 200):**
- **The turn.** One turn of an external session. Its cells arrive through the
  session server (§6.6) and land as `code` nodes. The turn ends with `done` if
  the session submitted, otherwise `idle`.
- **Waking.** `idle` wakes on any message. Today it also waits for every child
  to finish, which is a special case; rule 7 already batches results.

**`claude_code.py` (143 → 120) and `pi.py` (103 → 90):** thin adapters, kept as
they are.

**`prompts.py` (138 → 150):** the prose both prompt families share. It's
product copy, not logic. It could move to `.md` files next to the orchestrate
skill.

**Removed:**
- **`interactive.py` (63)**, if the plugin simplification in §9 is accepted.
- **The `Options` dataclass machinery** in `base.py`; `harness_options` is
  passed through as a dict.

### 6.4 Runtime: the REPL worker (now 2,124 → 780)

**The job:**
- run cells with top-level `await` in a persistent namespace, per agent;
- capture output;
- turn host tools into stubs that call back to the host;
- report the namespace's variables for session resume prompts.

**One interface with two transports.** A REPL is
`run(code, globals) -> (output, error)`. It runs either in this process (no
isolation; good for tests and notebooks) or in a worker process (subprocess,
Docker or Modal). Host tool calls from either transport go through
`flow.call_tool`, so receipts work identically. Today, in-process exists only
in `tests/helpers.py` and duplicates the proxy logic.

**The wire protocol shrinks to five messages**, one JSON object per line:

```text
host -> worker   run {id, code, globals}      define {name, async}
worker -> host   call {id, tool, args, kwargs}   result {id, output, error}
host -> worker   reply {id, value | error, error_type}
```

**What goes:**
- **Multiplexing.** Tenants, `run_id`, the binding blob and `_RunState`
  exist so one worker can serve several agents and overlapping runs. In
  practice it is one worker per agent and one cell at a time, so per-cell
  context lives on the host keyed by that worker.
- **The live-binding proxies.** `CurrentMapping` and `CurrentObject` go: `ENV`
  and `AGENTS` are plain values sent in `globals` with each cell.
- **The other message types.** `inject`, `inject_proxy`, `retrieve`, `remove`,
  `describe` and `ping` collapse into `run` and `define`.
  - **Variables:** reading one is `run` of an expression.
  - **Describing the namespace:** the resume prompt runs a fixed snippet.
  - **Removing a tool:** `run` of `del name`.
- **Replay signals.** There are three today (the `RLMFLOW_REPLAY` env var, the
  `restoring` wire flag and the `replaying()` context); the host knows when it
  is rebuilding.
- **Dead code:** `current_rpc_call_id`/`current_rpc_name`,
  `Runtime.describe_namespace`, the `agent_process_env` export,
  `parse_host_message`/`parse_client_message` as two functions, and the
  `ReplStatus` enum (a run either errors or doesn't; a dead worker raises).
- **`SubprocessRuntime`**, which differs from `LocalRuntime` by about 20 lines,
  becomes `LocalRuntime(python=, env=)`.

| File | Target | Contents |
| --- | --- | --- |
| `runtime/repl.py` | 180 | The in-process REPL: compile with top-level await, capture output, namespace |
| `runtime/worker.py` | 180 | The worker entry point: read messages, run cells, tool stubs |
| `runtime/host.py` | 220 | The worker client, process connection, `LocalRuntime` |
| `runtime/docker.py`, `modal.py` | 200 | Kept, smaller, sharing one connection class |

### 6.5 Tools and the REPL namespace (now 1,154 → 570)

**`@tool`** (60 lines) takes `name`, `description` and `where` (`"host"` by
default, or `"repl"` to copy a pure function into the worker). Async is
detected. A plain function is a tool, and `tool_from_schema` builds one from an
MCP or function-calling schema.

**The built-ins** are the six REPL names in §2, implemented once in
`tools/builtins.py` (~220).
- **Handles.** `AgentHandle` keeps `id`, `name`, `path`, `status`,
  `result()`, `parent`, `children`, `__await__` and `send`.
- **`AGENTS`** is a mapping over a snapshot of handles taken when the cell
  starts.

**What goes:**
- **Tool options.** `when=` and `at_leaf` (§9), `restore=`, `receipt=`,
  `proxy`, and `ToolMetadata` as a separate class go. The metadata is three
  attributes on the function.
- **Exports.** `RESERVED_TOOLS` becomes a private check, and `as_tool_items` is
  inlined into `Flow`.
- **`AGENTS` extras.** `AGENTS.edges()`, `AGENTS.render_graph()`/`print_graph()`
  (use `print(AGENTS)`, which renders the tree with `view/text.py`),
  `AgentRecord`, `AgentEdge` and `AgentDirectory` as a public class go. The
  duplicate name lookup goes too: `resolve_agent` and `AgentDirectory._find`
  become one function.
- **`Input`** goes (§9).
- **Separate packaging.** `agent_tools/tools.py` and `agent_tools/view.py` merge
  into `tools/builtins.py`.

**`tools/files.py` (173 → 170):** kept.

### 6.6 Session bridge: how Claude Code and pi reach the REPL (now 414 → 280)

- **`server.py` (240 → 150).** One HTTP route, `POST /repl {token, code}`. It
  runs a cell for the agent whose turn the token belongs to, through the same
  `run_cell` as the REPL harness.
- **`cli.py` (124 → 80).** `rlmflow repl`, which pi calls from its shell.
- **`mcp.py` (50).** One MCP tool, `python`, which Claude Code calls.

### 6.7 The Claude Code plugin (now 513 → 200 or 450)

The current flow:
- The user types `/rlmflow:run` in Claude Code.
- Hooks start a serve process, which records the Claude session as an
  interactive root agent with a turn per prompt, wakes it when children finish,
  and supports `rlmflow fork` of that session.
- The code for this is `cli/serve.py`, `agent_tools/hooks.py`,
  `harnesses/interactive.py`, `cli/fork.py` and the plugin's hook config.

The simpler alternative (recommended, §9):
- The Claude session is a client, not a recorded agent.
- `/rlmflow:run` starts `rlmflow serve`, which holds a graph with one root
  agent whose cells come from that session through MCP `python` calls.
- Workers it launches run under the Flow as normal, and their results show up
  in the next cell's output.
- No hooks, no interactive harness, no fork CLI: about 200 lines.
- **What is lost:** Claude's reasoning text and turn boundaries aren't in the
  graph, and the Claude session itself can't be forked from it. Its cells and
  everything it launched still are.

### 6.8 Models (now 546 → 250)

**The client protocol** is async-only: a client has
`async def stream(messages, **kw)` yielding text chunks and usage.
`client_for(model)` picks OpenAI or Anthropic from the name. Errors are retried
before the first token. The two providers stay at about 70 and 90 lines.

**What goes:**
- **The pool wrapper.** `PooledLLMClient` goes; `Flow` calls the client
  directly, with a timeout.
- **Sync support.** The sync `chat`/`completion` defaults on `LLMClient` exist
  only for test doubles. A test double defines `async def stream`, or is
  wrapped by a 10-line `sync_client(fn)` helper.
- **Dead code:** `retry_transient` (no callers), `thread_safe` (never read),
  `last_usage` shared state, and `sampling_kwargs` (tests only).

### 6.9 Config and schema (now 226 + 105 + 129 → 280)

**`AgentConfig`** keeps `name`, `model`, `harness`, `prompt` (section
overrides), `description`, `output_schema`, `max_iters`, `max_depth`,
`max_budget` and `harness_options`. It drops `keep_n_messages` (§9) and turns
`max_output_length` and `max_query_chars` into constants.

**`schema.py`** is the output-schema check, kept.

**`utils.py`** keeps fence parsing and output truncation. `code_block`,
`find_code_blocks` and `repl_code_block` go (no callers).

### 6.10 Views (now 1,286 + consumers 351 → 510)

| Keep | Target | Why |
| --- | --- | --- |
| `view/text.py` `render_tree` | 70 | Terminal status, `print(AGENTS)`, the CLI's live view |
| `view/svg.py` `graph_svg` | 280 | One picture of the run. Today's `figure.py` is 614 lines, mostly a node-level tidy-tree layout; an agent-per-column layout with nodes stacked is enough. |
| `view/html.py` `save_html` | 160 | A self-contained stepper: the SVG plus a slider over `timeline`. |

**What goes:**
- **Raster exports.** `open_viewer` (the Gradio app and the `viewer` extra) and
  `save_gif`/`save_frames`/`step_svg` (the `image` extra) leave the library.
  The doc GIFs are rendered by a script in `docs/static/`.
- **Timeline helpers.** `timeline.replay`/`snapshot`/`render_steps` go;
  `timeline` (the seq order) stays private to the HTML view.
- **`consumers.py` (351):**
  - **Live views:** `LiveGraphTree` and `LiveTreeRenderer` become a 30-line
    redraw in the CLI.
  - **Checkpointing:** `GraphCheckpointer` becomes `g.save(path)` in your loop.
  - **The rest:** `StreamConsumer` has no implementations, and `WorkspaceSync`
    moves to the autoresearch example, its only user.

### 6.11 CLI (now 1,262 → 200, or 400 with today's plugin)

**argparse instead of Fire.** Fire mishandles a positional query, which is why
the docs say `rlmflow run print "..."`.

| Command | Does |
| --- | --- |
| `rlmflow run "task" [--model --workdir --docker-image --flow module:factory --resume DIR]` | A coding run with a live tree |
| `rlmflow view DIR [--html out.html \| --svg out.svg]` | Print the tree, or write a file |
| `rlmflow serve` | The plugin's server (§6.7) |
| `rlmflow repl` / `rlmflow mcp` | The session bridge (§6.6) |

**What goes:**
- **Commands:** `config show/path/init`, `version` (use `--version`), `plugin
  path`, `render` with five subcommands, `fork` (it's a library call:
  `g.fork(node)`, then `flow.run(g)`), and `hook` (with the plugin
  simplification).
- **Layered config.** Four layers (user TOML, project TOML, env, flags) become
  flags plus `RLMFLOW_*` env vars.
- **Duplicated runners:** the two graph loaders and the two headless runners.

### 6.12 Workspaces and integrations

- **`engine/workspaces.py` (120 → 110).** Kept: `LocalWorkspace` and
  `GitWorktrees` snapshot files at cell ends, so forks and re-runs of coding
  agents see the right files.
- **`integrations/mcp.py` (117 → 110).** Kept: `MCPTools` serves an MCP
  server's tools as rlmflow tools.

### 6.13 Public API (now ~110 names → 24)

```python
from rlmflow import (
    Graph,
    Agent,
    Node,
    Edge,
    Step,
    Invalidation,
    GraphError,
    Flow,
    AgentConfig,
    tool,
    tool_from_schema,
    Toolkit,
    FILE_TOOLS,
    LocalRuntime,
    InProcessRuntime,
    DockerRuntime,
    ModalRuntime,
    client_for,
    OpenAIClient,
    AnthropicClient,
    render_tree,
    graph_svg,
    save_html,
    NotYourAgent,
)
```

Node and edge kinds are strings (`"llm"`, `"spawn"`); constants exist in
`rlmflow.graph.nodes` for those who want them.

## 7. Size

| Area | Now | Target |
| --- | --- | --- |
| Graph (records, graph, calls, files) | 1,905 | 870 |
| Flow and scheduling | 947 | 420 |
| Harnesses and prompts | 1,066 | 760 |
| Runtime | 2,124 | 780 |
| Tools, built-ins, `AGENTS` | 1,138 | 570 |
| Session bridge | 414 | 280 |
| Plugin (serve, hooks, interactive, fork) | 689 | 200 |
| Models | 546 | 250 |
| Config, schema, utils | 460 | 280 |
| Views and consumers | 1,637 | 510 |
| CLI (without serve and fork) | 785 | 200 |
| Workspaces, MCP tools, `__init__` | 437 | 300 |
| **Total** | **11,921** | **~5,400** |

The prompt prose (about 400 lines of the harness total) is product copy. Moved
to `.md` files, the Python is about 5,000 lines.

## 8. Everything removed

| Removed | Replacement |
| --- | --- |
| `_Annotated`, `annotate`, `strict_json` checks on every write | A plain `metadata` dict, checked on save |
| `Call`, `g.calls`, `Node.record_call/calls/launched`, `LAUNCH_TOOLS`, `AWAIT_CALL`, `agent_ref_id` | `data["calls"]` as a list, and `graph/calls.py` |
| `prune_pending`, `Flow._cells`, `_keep_stopped_cell` | Cells land when they start (rule 5) |
| `fork` edges | `agent.base` |
| `spawn` edge `data["agent"]` | `owner(edge.source)` |
| `is_ready`, `ready`, `finished`, `unseen_results` on `Graph` | `status`; results delivery in `flow.py` |
| `provenance`, `cost_usd`, `lineage`, `depth`, `import_subtree`, `Graph.inbox` | A loop, `usage`, private helpers, `spawn(parent=)`, `agent.inbox` |
| `Pool`, `ThreadPool`, `SequentialPool`, `PooledLLMClient`, `workers=` | `asyncio.to_thread`, `max_concurrent=` |
| `TaskQueue`, `StepResult`, `Boundary`, `resolve_boundary` | Tasks plus an event in `run_streaming` |
| `Flow.step/land/replay/namespace_for/inject/remove_tool/resolve_config/config_for` | `run_streaming`, `add_tool`, `config_of` |
| `budget_exhausted`, `budget_nearly_spent`, `final_turn`, `final_turn_spent`, `MAX_BUDGET_EXCEEDED` | One limits function (rule 10) |
| Worker tenants, `run_id`, the binding blob, `CurrentMapping`, `CurrentObject`, six message types, `ReplStatus`, `RLMFLOW_REPLAY` | One worker per agent and five messages (§6.4) |
| `SubprocessRuntime` | `LocalRuntime(python=, env=)` |
| `ToolMetadata`, `restore=`, `receipt=`, `when=`, `at_leaf`, `RESERVED_TOOLS` export | Three function attributes; every JSON result is recorded |
| `Input`, `transform_argument`, `visit_argument`, `RECURSE`, `input_parts` | Plain JSON arguments; describe them in the task |
| `AgentDirectory`, `AgentRecord`, `AgentEdge`, `AGENTS.edges/render_graph/print_graph` | `AGENTS` as a mapping of handles; `print(AGENTS)` |
| `keep_n_messages`, `max_output_length`, `max_query_chars` config fields | Constants |
| `retry_transient`, `thread_safe`, `last_usage`, sync client defaults, `sampling_kwargs` | — |
| `consumers.py` (`LiveGraphTree`, `LiveTreeRenderer`, `GraphCheckpointer`, `StreamConsumer`, `WorkspaceSync`) | A CLI redraw; `g.save` in the loop; `WorkspaceSync` moves to the example |
| `open_viewer`, `save_gif`, `save_frames`, `step_svg`, `replay`, `snapshot`, `render_steps` | `save_html`; a docs script for GIFs |
| Fire, `config`/`version`/`plugin`/`render`/`fork`/`hook` commands, the four-layer config | argparse; five commands; flags plus env |
| `interactive.py`, `hooks.py`, most of `serve.py` (if §9.1 is accepted) | The session is an MCP client of a served root |
| `code_block`, `find_code_blocks`, `repl_code_block` | — |
| Extras `viewer`, `image` | — |

## 9. Decisions for you

Each comes with a recommendation; the targets above assume all of them.

1. **The plugin's interactive root.** Keep the Claude session as a recorded,
   forkable agent with hooks (about 450 lines), or make it a client of a served
   root (about 200)? *Recommend the client.* The graph still holds every cell
   and everything launched; what's lost is Claude's own reasoning text and
   forking the Claude session itself.
2. **`Input` (argument descriptions).** It's used only in tests. *Recommend
   dropping it*; put descriptions in the task text.
3. **`when=` and `at_leaf`.** These hide file-writing tools from agents that can
   launch others. *Recommend dropping them.* It's a policy the prompt and
   `max_depth` already express, and it's the only reason tools need a
   visibility hook.
4. **`keep_n_messages`.** This trims old turns from the REPL harness's context.
   *Recommend dropping it.* The REPL is the agent's memory, and long outputs are
   already truncated. If long runs need trimming later, it comes back as one
   harness option.
5. **Viewers.** *Recommend keeping* the text tree, the SVG and the HTML
   stepper, and moving the Gradio app and GIF/frame export out of the library.
6. **The edit log (`g.edits`).** It's small and makes edits auditable.
   *Recommend keeping it.*
7. **Saved runs.** The format changes again: `submit` cells, the flat `calls`
   list, no `fork` edges. *Recommend format 5 with no migration*, as with v4.

## 10. Order of work

Each step ends with `uv run pytest -q`, ruff, and the offline examples green.
Tests are rewritten alongside: they assert behaviour (the ten rules, receipts,
edits, harness turns, the wire protocol), not internals.

1. **Graph:** records, `calls.py`, `status`, cells-land-at-start support,
   edits, format 5.
2. **Runtime:** the in-process REPL, the five-message worker, `LocalRuntime`;
   delete `TestRuntime` from the test helpers.
3. **Flow:** `run_streaming` with tasks and an event, `call_tool`, the REPL
   lifecycle, results, limits; delete the scheduler and pools.
4. **Tools and harnesses:** the built-ins and `AGENTS`, `ReplHarness`,
   `SessionHarness`, Claude Code, pi, the session bridge.
5. **Plugin, CLI, views:** `serve`, argparse CLI, text, SVG and HTML.
6. **Models, config, public API:** the async-only client, trimmed
   `AgentConfig`, the 24 exports.
7. **Docs, README, examples, CHANGELOG/UPGRADING.**
