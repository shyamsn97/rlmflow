# rlmflow core v2: a smaller library with an editable graph

Status: implemented in place in `rlmflow/` (no branch, no tag); see §14 for where the code differs from this plan
Date: 2026-11-06

This document is the plan for rewriting rlmflow's core. It says what the
library is for, what the new graph and engine look like, how behaviour changes
for users, and what happens to every file in `rlmflow/`, the tests, the
examples and the docs, with line estimates. It supersedes
[`minimal_dynamic_graph.md`](minimal_dynamic_graph.md), the decision sections
of [`examples-cleanup.md`](examples-cleanup.md), and the "Three things" and
"Shapes" sections of [`scope.md`](scope.md).

Line counts for current code are from the working tree on 2026-11-06. Estimates
for new code are ±20%.

## 1. The story

> **rlmflow lets coding agents build multi-agent systems in code, and records
> every run as a graph you can edit.**
>
> Every agent has a Python REPL. In it, an agent launches other agents with
> real values as inputs (not summaries), awaits their results inside the same
> cell, and messages any agent in the run. Every launch, result and message is
> an explicit edge in one graph. You can stream the graph live, save it, fork
> it, or edit any node, and rlmflow re-runs only what depended on the edit. Any
> coding harness (rlmflow's own REPL loop, Claude Code, pi) can be an agent in
> the graph.

RLM trees, DeLM boards, peer teams, pipelines, review loops and Shepherd-style
recovery are programs written against this API. None of them is an engine
feature.

What rlmflow is not: a workflow DSL, a catalogue of role teams, a coding CLI
competing with Claude Code or Codex, a distributed scheduler, or a benchmark
suite.

## 2. Why rewrite

The library is 14,639 lines in `rlmflow/`. The repository adds 13,061 lines of
benchmark harness, 12,748 lines of tracked example code, and about 8,100 lines
of tests. Two things drive the size of the library itself:

1. **Coordination policies built into the engine.** `depends_on` with `inputs`,
   `reports_to` with `engine/handoff.py`, the shared store inside `Graph`,
   `attach`, `merge`, `waiting` nodes that list what they wait `on`, `blocked`
   status, and wait-cycle detection. Each one encodes one way of organizing
   agents. Together they make the engine prescribe structure.
2. **A generic edge store doing the work of a few fixed relations.** Agents are
   `agent_start` nodes, every relation is an `Edge` in a log with in/out
   indexes, and recovering simple facts (whose launches does a fork inherit?)
   needs code like `Graph.launched` and `_skipped_launches`.

Several earlier passes reshaped this code (92 modules down to 56, transitions
removed, configs moved to `Flow`) without removing requirements, so the size
stayed. This rewrite removes requirements.

## 3. Decisions

These were settled in review and are not reopened by this document.

1. **The graph stays the engine.** `Flow` reads the graph, steps every ready
   agent with its harness, and writes the result back. Pausing, streaming,
   steering, resuming and forking work the same way for every agent.
2. **Agents await other agents inside a cell**, RLM style:
   `results = await asyncio.gather(*workers)`. This is the core of the idea and
   is kept.
3. **You can only await agents you launched** (directly or further down).
   Everyone else gets messages. Waits always point down a tree, so cycles are
   impossible and wait-cycle detection is deleted.
4. **`send` never blocks on a peer.** A message lands in the recipient's inbox
   and is seen at its next turn. A message reopens a finished agent.
5. **No forking in the REPL.** Agents get `launch_agent`, `await`, `send`,
   `submit`, `llm_query`, `AGENTS` and the application's tools. Forking,
   editing and replacing are graph operations for host code.
6. **Explicit edges for every cross-agent relation**: `spawn`, `return`,
   `message`, `fork`. Communication is part of the graph, not inferred.
7. **The graph is mutable.** Host code can edit, delete, replace and splice
   nodes. Edits invalidate exactly what depended on them, found by following
   edges, and the next run re-runs only that.
8. **The host interface stays**: `Flow(configs=..., tools=..., runtime=...)`,
   `flow.run(...)`, `g.add_agent(...)`, `flow.run_streaming(g, until=...)`
   yielding one `Step` per node, `g.save(path)`, `Graph.load(path)`.
9. **Harnesses reach the REPL through one server.** External harnesses call it
   with `rlmflow repl` or an MCP tool. Their cells, launches and messages are
   the same nodes and edges as the native harness's.

## 4. The agent API (inside the REPL)

| Call | Behaviour |
| --- | --- |
| `a = launch_agent(task, *, config="default", name=None, **vars)` | Create a child agent. `vars` become variables in its REPL and carry real data. Returns an `Agent` handle immediately. Raises if `max_depth` or `max_budget` forbids it. |
| `await a`, `await asyncio.gather(*agents)` | The child's result, inside the running cell. Raises `NotYourAgent` for anything that is not a descendant. |
| `AGENTS[x].send(message)` | Queue a message for any agent and return its handle. Never blocks. Reopens a finished agent. `await agent.send(...)` waits for the reply, and is allowed only for descendants. |
| `submit(value)` | Finish, checked against the config's `output_schema` when set. A plain-text reply with no `repl` block is still a final answer. |
| `llm_query(prompt, schema=None, model=None)` | One model call with no history. Unchanged. |
| `AGENTS` | Lookup by name, path or id; `AGENTS.self`; iteration; `AGENTS.children`. |
| application tools | Plain functions, `@tool`, `tool_from_schema`, toolkits such as `MCPTools`. Unchanged. |

The `Agent` handle in the REPL is an id plus calls to the server. Its reads
(`name`, `task`, `status`, `result`, `parent`, `children`) have the same names
as the host-side `Agent` record. Its actions are `send` and `__await__`. It
serializes as `{"agent": id}`, which replaces `AGENT_REF_KEY` and
`encode_agent_references`.

Removed from the REPL namespace: `Input` (see §5.4), handle-valued launch
arguments, the hidden `wait_agent` and `send_agent` bindings, the `send_agent`
alias, `AGENTS.get()`, `get_parent()`, `get_children()`, `get_siblings()`, and
the store tools (they become a recipe, see §9).

## 5. The graph

### 5.1 Data model

```python
@dataclass
class Agent:
    id: str
    name: str
    task: str
    args: dict[str, Any]  # REPL variables (large values stored as blobs)
    config: str | None  # config name; None for a root (runs with root_config)
    nodes: list[Node]  # its own history, in order
    inbox: list[Node]  # message nodes not yet delivered
    metadata: dict[str, Any]  # the config JSON it last ran with, timing, etc.


@dataclass
class Node:
    id: str
    agent: str  # owning agent id
    type: str  # "llm" | "code" | "message" | "idle" | "done"
    data: dict[str, Any]
    seq: int  # global creation order
    metadata: dict[str, Any]


@dataclass(frozen=True)
class Edge:
    kind: str  # "spawn" | "return" | "message" | "fork" | application kinds
    source: str  # node or agent id
    target: str  # node or agent id
    data: dict[str, Any]
```

Node types:

| Type | Written by | `data` |
| --- | --- | --- |
| `llm` | native harness | `text`, `usage`, `model` |
| `code` | REPL server | `code`, `output`, `error`, `calls`, `images`, `workspace`, `session` |
| `message` | `send`, result delivery, host | `text`, `sender`, `final` |
| `idle` | session harnesses | `text`, `usage`, `cost_usd`, `session` (turn ended, waiting on children) |
| `done` | `submit`, final reply, failures | `result`, `error`, `usage`, `session` |

`agent_start` disappears (it is the `Agent` record). `waiting` becomes `idle`
and loses its `on` list.

Edges:

| Kind | Source → target | Created by |
| --- | --- | --- |
| `spawn` | launching `code` node → child `Agent` | `launch_agent`, `g.add_agent(..., parent=cell)` |
| `return` | child's `done` → the `code` node that awaited it, or the `message` node that delivered it | engine |
| `message` | sending `code` node (or `None` for host messages) → `message` node | `send`, `g.send` |
| `fork` | forked `Agent` → base node it continues from | `g.fork` |
| any other kind | any node or agent → any node or agent | `g.add_edge`; stored, never interpreted |

Order within an agent is the order of `agent.nodes`; there is no `next` edge.
A fork's history is the base agent's history up to the base node, followed by
the fork's own nodes.

`code.data["calls"]` is a list, one entry per tool or agent call the cell made,
in call order: `{"tool": "llm_query", "result": ...}` for tools, and
`{"tool": "launch_agent", "agent": id}` or `{"tool": "await", "edge": id}` for
agent calls. Replay returns these instead of re-running them. Agent calls point
at the agent or edge rather than copying values, which is what lets
`g.replace` swap them.

### 5.2 API

Kept, with the same names:

```python
g = Graph()
root = g.add_agent(task, name=None, **args)        # a top-level agent
child = g.add_agent(task, parent=root, parent=cell, config="worker", name=None, **args)
g.append(agent, node)
g.send(agent, text, sender=None, by=cell)             # host or agent message
g.fork(node, name=None, config=None)                   # a new agent continuing from node
g.add_edge(Edge(...)); g.remove_edge(edge)             # application edges only
g.node(id); g.nodes(type=None, agent=None); g.agents(); g.edges(kind=, source=, target=)
g.history(agent_or_node); g.status(agent); g.result(agent)
g.parent(agent); g.children(agent); g.lineage(agent); g.path(agent); (len(g.lineage(agent)) - 1)
g.usage(agents=None); g.usage(agents=None).cost_usd
g.save(path); Graph.load(path)
```

New:

```python
g.calls(cell)  # [Call(index, tool, agent, edge, result)]
g.provenance(node)  # every node upstream along return/message/spawn edges
g.subtree(agent)  # the agent and everything it spawned, transitively
g.copy()  # a full copy, for editing without losing the original
g.subscribe(fn)  # called after every write; the engine uses it to wake

g.edit(node, **data)  # change a node's data
g.delete(node)  # remove a node from its agent's history
g.remove(agent)  # remove an agent and its subtree
g.replace(call, value | agent)  # bind one call in a cell to a value or another agent
g.splice(at, nodes)  # insert nodes after `at` (e.g. a summary replacing old turns)
g.import_subtree(other_graph, agent)  # copy an agent and its subtree from another run
```

Every editing method returns an `Invalidation` listing the agents it rewound
and the nodes it dropped, and has a `dry_run=True` form that only computes it.

Removed: `deliver` (internal to the engine), `attach`, `merge`, `as_of` (see
below), `store`, `store_write`, `depends_on`, `inputs`, `producers`,
`launcher`, `launched`, `own_history` (use `agent.nodes`), `frontier` (use
`agent.nodes[-1]`), `terminal`, `ready` (the engine owns readiness), `tree`
(renamed `subtree`), `items` (replaced by `subscribe` and the save format),
`check_invariants` (replaced by a short `validate` on load), `StaleFrontier`.

`as_of(seq)` was used only by `view.timeline.snapshot`. With global `seq` on
every node, a viewer filters nodes and edges by `seq` instead.

### 5.3 Invalidation

An edit or deletion at node `n` of agent `A`:

1. **Rewind `A`.** Drop every node of `A` after `n` (for `delete`, from `n`
   on). Mark `A`'s REPL for rebuild.
2. **Follow edges out of every dropped node:**
   - `spawn` from a dropped cell: remove the child agent and its subtree.
   - `message` from a dropped cell: if the message is still in the recipient's
     inbox, remove it; if it was delivered, rewind the recipient to just
     before it.
   - `return` from a dropped `done`: rewind the agent that owns the target to
     just before the target node (the awaiting cell re-runs, or the delivering
     message is removed).
   - `fork` edges pointing at a dropped node: remove the fork.
3. **Shared prefixes.** Every fork whose base is at or after `n` in `A`'s
   history saw the changed node, so it is rewound to its base (all its own
   nodes are dropped).
4. **Repeat** for every agent rewound in steps 2–3 until nothing changes.
5. **Record the edit** in `g.edits` with the invalidation it caused.

`replace(call, x)` on call `k` of cell `c` in agent `A`:

1. Rewind `A` to just before `c`, keeping `c`'s saved calls as pending receipts
   for the re-run.
2. Override receipt `k`: a value is returned as-is; an agent becomes the
   awaited agent (a new `return` edge is created when it finishes, and the
   engine runs it first if it is unfinished).
3. Apply step 2 of the invalidation rules to everything `c` and later nodes
   produced.
4. The next run re-executes the same cell code. Other calls replay from their
   receipts, call `k` gets the substitute, and everything after it runs live.

For the native harness the re-run is exact, because the `llm` node before `c`
holds the code. For session harnesses the agent is rewound to the last turn
boundary before `c` and the turn runs again from the session pointer there; see
§13.

Editing an agent while it is stepping waits for its step to finish, or cancels
it with `cancel=True`.

### 5.4 Arguments

`create_agent` and `launch_agent` take `**args` as REPL variables, stored as
strict JSON with large values in blobs, as today. `Input(value,
description=...)` stays for describing arguments to the model; it no longer
accepts agents. Passing an `Agent` handle as an argument raises
`TypeError("await the agent first and pass its result")`.

### 5.5 Persistence

`g.save(path)` writes a directory:

```text
path/
  graph.json    {"format": 4, "id": ..., "agents": [...], "nodes": [...], "edges": [...], "edits": [...]}
  blobs/        content-addressed large arguments, as today
```

`Graph.load(path)` reads it back into the same agents, nodes and edges and runs
`validate` (ids exist, edge endpoints exist, each node's agent exists, the
`spawn` tree is acyclic). Formats 1–3 are not loaded; there is no converter.

`GraphCheckpointer` keeps its triggers (every finished agent, every 50 nodes,
every 2 seconds) but writes in a worker thread instead of on the event loop.

### 5.6 What a run looks like

```text
root            task "Find the magic number", args: haystack
  llm           "split into 10 chunks and launch a searcher for each"
  code#2        workers = [launch_agent(...)]; print(await asyncio.gather(*workers))
                calls: launch_agent → searcher-0 … searcher-9; await → 10 return edges
  │ spawn
  ├─ searcher-0   llm → code → done: None ───────────── return ─▶ code#2
  ├─ searcher-7   llm → code → done: 4721 ───────────── return ─▶ code#2
  │  ...
  llm           "searcher-7 found it"
  done          4721
```

## 6. The engine

### 6.1 Readiness

An agent is **stepping** while the engine has a step in flight for it. An agent
that is not stepping is **ready** when:

| Its last node | Ready when |
| --- | --- |
| none (new agent) | always |
| `llm` (native: a reply whose cell has not run) | always |
| `code` | always |
| `message` | always |
| `idle` | its inbox is non-empty and none of its children are unfinished |
| `done` | its inbox is non-empty (a message reopens it) |

`g.status(agent)` is `"running"`, `"idle"` or `"done"` from the same table.

### 6.2 The loop

```text
run_streaming(g, until, root_config):
    bind configs; open toolkits; subscribe to g so writes wake the loop
    rebuild REPLs of agents rewound by edits since the last run
    loop:
        for each ready agent not stepping:
            apply the budget rule (§6.5)
            move its inbox into its history
            start step(agent)
        if nothing is stepping: return
        wait for a finished step or a wake-up
        yield one Step per node written since the last yield, with its edges
        stop if the `until` boundary matched
    finally: cancel steps in flight, close toolkits, unsubscribe
```

`step(agent)` calls the agent's harness, appends the node it returns, and
applies the result rule. A harness exception becomes
`done(result="[run failed: ...]" | "[child failed: ...]", error=...)`, as
today.

### 6.3 The result rule (replaces `engine/handoff.py`)

When an agent with a parent lands `done`:

- If a running cell of the parent is awaiting it, the engine resolves that
  await and adds a `return` edge from the `done` to that cell's `code` node
  when the cell lands.
- Otherwise the engine sends the parent a `message` containing the result and
  adds a `return` edge from the `done` to it.

Results that reach the parent in a cell are never also sent as messages.
Messages that arrive together are delivered together at the next turn
boundary, so a parent that ended a turn with five children running sees all
results that are ready when it next steps.

This is about 40 lines in `flow.py` plus result formatting moved from
`handoff.render`.

### 6.4 Awaiting inside a cell

1. `launch_agent` writes the child and its `spawn` edge immediately (the
   pending cell's id is known before it lands) and wakes the loop. The engine
   starts stepping the child on its next pass.
2. `await child` checks the child is a descendant of the caller, then waits on
   a future the engine resolves when the child's `done` lands.
3. The parent's step stays in flight until its cell finishes. Its `code` node
   lands with `calls` pointing at the launches and the `return` edges.

There is no `waiting` registration, no cycle check, and no `flow._waits`.

### 6.5 Limits

- `max_iters` (native harness): unchanged.
- `max_depth`: unchanged, checked in `launch_agent`.
- `max_budget`: unchanged semantics (a soft whole-run threshold that gives
  every unfinished agent one final turn, then `[max_budget exceeded]`).
  Implemented on `g.subtree(root)` instead of `g.tree(g.lineage(agent)[0])`.
- `keep_n_messages`, `max_output_length`, `max_query_chars`: unchanged.

### 6.6 Concurrency and host writes

Today a step records an anchor and appends with `after=anchor`, raising
`StaleFrontier` if the host wrote to that agent mid-step. In v2, host writes go
through the graph's lock, `g.send` only touches the inbox, and edits to a
stepping agent wait for or cancel the step (§5.3), so the append guard and
`StaleFrontier` are deleted. `g.subscribe` wakes the loop on any write, which
fixes the known limit in `scope.md` about host messages during a run (the
thread-safe wake is part of `subscribe`).

### 6.7 Boundaries

`until=` keeps `"next"` (stop after the first step), `"done"` (run to quiet),
`"error"` and callables over `Step`. `"idle"` now means "stop when the step's
node is `code`, `idle` or `done`", as today with `waiting` renamed.

### 6.8 Replay

Unchanged in mechanism: `prepare_repl` re-runs an agent's cells, in history
order including a fork's prefix, with saved calls returned instead of
executed. What changes:

- Saved calls are one list per cell (§5.1) instead of per-tool dictionaries
  plus `AWAITED_KEY` plus store records.
- The relaunch special case in `Flow.execute` (re-binding agents launched by a
  cell that never landed) becomes the general "pending receipts" mechanism
  that `replace` also uses.
- Agents rewound by an edit are rebuilt before the run continues.

### 6.9 Streaming

`run_streaming` still yields one `Step(graph, agent, node)` per node.
`step.agent` is an `Agent` record instead of an `agent_start` node, and
`step.edges` lists the edges written with that node. Consumers keep the
`handle(item, graph)` interface.

## 7. Harnesses and the REPL server

### 7.1 Native harness (`harnesses/repl.py`)

Unchanged protocol: one `repl` block per reply, a reply with no block is the
final answer, `submit` in a block finishes. Changes:

- `node_messages` drops the `agent_start` and `waiting` cases (the task comes
  from the `Agent` record; `idle` nodes are not produced by this harness).
- The "Background agents" note uses `g.status(child)`.
- `AGENTS_TEXT` is rewritten for the new rules: await only agents you
  launched, send to anyone, `await agent.send(...)` only for agents you
  launched.

### 7.2 Session harnesses (`harnesses/sessions.py`, `claude_code.py`, `pi.py`)

One session turn per step, as today. During the turn, `rlmflow repl` cells land
`code` nodes through the server. When the turn ends, `outcome` returns:

- `done` if the agent submitted, or if it has no unfinished children (its
  reply text, checked against `output_schema`, is the result);
- `idle` otherwise. The agent wakes when its inbox has messages and its
  children are finished (§6.1), which is today's `waiting` behaviour without
  the `on` list or `handoff.pending`.

Each `code`, `idle` and `done` node records `session` (`{"id": ..., "message":
...}` where the harness exposes a message id), which forking and editing use
to resume the session at that point.

`Session.for_step`, `next_prompt`, `resume_prompt`, `describe_variables` and
`run_command` are kept. `SessionHarness.prompt_sections` drops
`g.inputs(agent)`.

The Claude Code harness keeps `SendMessage` and `ListAgents` disallowed.

### 7.3 The server

`agent_tools/server.py` (the endpoint external harnesses call), `agent_tools/
tools.py` (spawn, send, done, repl as functions of `(flow, step)`) and the
in-REPL tool builders in `tools/__init__.py` are one code path: every agent
call, from the native REPL or from `rlmflow repl`, goes through
`rlmflow/agents/api.py`. The server only adds the transport (a local socket)
and turn bookkeeping. `rlmflow repl` (the client CLI) and the MCP server
(`python` tool) stay as the two ways a harness reaches it.

### 7.4 Workers

One worker process per agent, as today. The worker protocol keeps its
messages (run, inject, retrieve, remove, describe, proxy calls). Removed:
`tenant_id` on every request and the multi-tenant `ReplServer._tenants` (each
worker already serves one agent), `CurrentMapping`/`CurrentObject`/
`current_binding` (bindings are plain namespace entries in a one-agent
worker), and `AgentReference`/`encode_agent_references` (handles are ids).

## 8. Behaviour changes for users

| Area | Today | v2 | Migration |
| --- | --- | --- | --- |
| Await | Any agent can await any agent; cycles refused at runtime | Only descendants; `NotYourAgent` otherwise | Peers use `send` and end their turn |
| `await peer.send(msg)` | Waits for the peer's reply | Allowed only for descendants | `send`, then read the reply as a message next turn |
| Handle as launch argument | Becomes a `depends_on` input; the child waits for it | `TypeError` | `x = await a; launch_agent(..., x=x)` |
| `g.add_agent(..., depends_on=, inputs=)` | Pipeline declared up front | Removed | A root that awaits stages in order, or host code that runs stages in order |
| `reports_to` edges | Results also delivered to other agents | Removed | The supervisor launches the workers, or workers `send` to it |
| Shared store (`STORE_TOOLS`, `g.store`) | Built into the graph | A `board` tool in `examples/collaboration/` | Copy the recipe |
| `g.attach`, `g.merge` | Adopt agents or import a graph | Removed; `g.import_subtree` covers import | |
| Status values | `done`, `blocked`, `waiting`, `running` | `done`, `idle`, `running` | |
| Node types | `agent_start`, `message`, `llm_output`, `code`, `waiting`, `done` | `llm`, `code`, `message`, `idle`, `done`; agents are `Agent` records | Rename in consumers |
| Edges | `next`, `from`, `depends_on`, `reports_to` | `spawn`, `return`, `message`, `fork`; order is `agent.nodes` | Rewrite graph walks |
| Results of children not awaited | Batched `message` via handoff | Same, plus a `return` edge | None |
| Editing a run | Fork and run the branch | Fork, or edit/replace in place with invalidation | New capability |
| Host messages mid-run | Delivered at the next pass; needed `flow.queue.wake()` | `g.send` wakes the loop | None |
| Streaming | `Step.agent` is an `agent_start` node | `Step.agent` is an `Agent`; `Step.edges` added | Use `step.agent.name` |
| Saved runs | `graph.json` format 3, formats 1–2 converted | Format 4; older formats not loaded | Re-run |
| Checkpoint writes | On the event loop | In a thread | None |
| `StaleFrontier` | Raised to host writers | Removed | None |
| Exports | 80+ names, including TUI, DSPy, consumers | About 45 (§10.1) | Import extras from their modules |
| CLI | `run`, `tui`, `serve`, `view`, `render`, `fork`, `repl`, `config`, `plugin` | First release: `run print`, `serve`, `plugin`, `view`, `render`, `fork`, `repl`, `config`; `tui` returns later (§12) | |
| Claude Code plugin (`/rlmflow:run`, your session as the root) | Supported | Supported; its root agent ends turns with `idle` instead of `waiting` | None |
| RAO | `rlmflow.rao` in the package | Not in the first release; returns as an extra with credit along `return` edges | |

Unchanged: `Flow(...)`, `AgentConfig` fields, `flow.run`, `arun`,
`run_streaming(g, until=, root_config=)`, tools and toolkits, `FILE_TOOLS`,
`MCPTools`, runtimes (`LocalRuntime`, `SubprocessRuntime`, `DockerRuntime`,
`ModalRuntime`), the native prompt protocol, `llm_query`, `submit`, structured
output, `max_*` limits, Claude Code and pi as harnesses, `g.fork`.

## 9. What each structure looks like in v2

| Structure | Code |
| --- | --- |
| RLM tree | `results = await asyncio.gather(*[launch_agent("find X", chunk=c) for c in chunks])` |
| Recursive tree | the same, inside children, bounded by `max_depth` |
| Pipeline | `spec = await launch_agent("plan", config="planner")`, then `code = await launch_agent("build", spec=spec)` |
| Review loop | `draft = await writer`; `fixed = await writer.send("fix these issues")` |
| Peers | top-level agents from host code; `AGENTS["bob"].send(...)`; `submit` when idle; messages reopen |
| DeLM board | top-level workers plus a `board` tool (`claim`, `post`, `read`) whose results are saved like any tool's; about 40 lines |
| Supervisor | the supervisor launches workers and awaits them, or workers `send` to it |
| Recovery (Shepherd) | host code, or a meta-agent given a small `fork` tool built on `g.fork` |
| Swap one call | `g.replace(g.calls(cell)[0], "fixed value")`, then `flow.run(g)` |

## 10. File by file

Actions: **keep** (minor edits), **adapt** (same role, real changes),
**rewrite**, **merge** (into another file), **move** (out of the core, to an
extra or an example), **defer** (not in the first release), **delete**.

### 10.1 Top level

| File | Lines | Action | New lines | Notes |
| --- | ---: | --- | ---: | --- |
| `__init__.py` | 180 | rewrite | 90 | Exports: `Flow`, `Graph`, `Agent`, `Node`, `Edge`, `Step`, node and edge kind constants, `AgentConfig`, `Input`, runtimes, `OpenAIClient`, `AnthropicClient`, `LLMClient`, `client_for`, `tool`, `tool_from_schema`, `Toolkit`, `FILE_TOOLS`, `MCPTools`, `GraphCheckpointer`, `StreamConsumer`, `ConsumerGroup`, `attach_image`, `record_metadata`, `SchemaError`, harness classes. Dropped: `WAITING`, `AGENT_START`, `DEPENDS_ON`, `REPORTS_TO`, `FROM`, `NEXT`, `InvariantError`/`StaleFrontier` (one `GraphError` remains), `TaskQueue`, pools, `STORE_TOOLS`, `WorkspaceSync`, `FlowTUI`, `DSPyFlow`, `TinkerClient`, `FlowLLM`, `AgentToolsServer`, `code_block`, `find_code_blocks`, `persistence`, the view names (import from `rlmflow.view`). |
| `__main__.py` | 6 | keep | 6 | |
| `flow.py` | 594 | rewrite | 380 | See §10.3. |
| `consumers.py` | 352 | adapt + move | 90 | Keeps `StreamConsumer`, `ConsumerGroup`, `GraphCheckpointer` (threaded writes). `LiveGraphTree`/`LiveTreeRenderer` move to `view/live.py`. `WorkspaceSync` moves to `examples/autoresearch/`. |
| `schema.py` | 105 | keep | 105 | |
| `utils.py` | 129 | keep | 110 | Drop helpers only the deleted code used. |

### 10.2 `graph/` (1,837 → about 1,160)

| File | Lines | Action | New lines | Notes |
| --- | ---: | --- | ---: | --- |
| `graph.py` | 920 | rewrite | 560 | Method plan below. |
| `nodes.py` | 391 | adapt | 230 | Keeps `Node`, `LLMUsage`, `Step`, ids, `strict_json`, `annotate`, `timing`, the context variables (`running_step`, `running_cell`, `replaying`), `attach_image`, `record_metadata`. Moves the `agent_start` accessors (`config_name`, `config`, `bind_config`, `name`, `task`, `args`, `resolved_args`, `argument_descriptions`) to `Agent`. Replaces `record_call`/`calls`/`launched` (per-tool dicts, launch validation) with one list of calls. Drops `AGENT_REF_KEY`, `agent_ref_id`, `LAUNCH_TOOLS`. Adds `Agent` (≈50 lines). |
| `config.py` | 226 | adapt | 170 | Keeps `AgentConfig` (all fields), harness loading, name validation, `Input`, `transform_argument`/`visit_argument`. `Input` no longer wraps agents. `can_spawn` uses `g.depth`. |
| `persistence.py` | 243 | rewrite | 170 | Format 4 (§5.5), blobs kept. Drops `_convert_format_2`, `upgrade_v1` and the `tests/fixtures/v1_tree_run` path. |
| `__init__.py` | 57 | adapt | 30 | |

`graph.py` method by method:

| Today | v2 |
| --- | --- |
| `Edge` (frozen, `seq`, `removed_seq`, `active`, `structural`) | `Edge` without `seq`/`removed_seq`/`active`; removal deletes |
| `InvariantError`, `StaleFrontier` | one `GraphError` |
| `locked`, `write_under_lock` | kept |
| `agent_node` | becomes `Agent(...)` construction |
| `Graph.__init__` (11 indexes, store state) | `agents`, `nodes`, `edges`, `_out`, `_in`, lock, subscribers, edits |
| `append(agent, node, after=)` | `append(agent, node)`; no frontier guard |
| `spawn(..., depends_on=, inputs=)` | `spawn(task, parent=, by=, config=, name=, **args)`; writes the `spawn` edge |
| `send` | kept; writes into `agent.inbox` and the `message` edge |
| `deliver` | moves to the engine (5 lines) |
| `fork` | kept; writes the `fork` edge |
| `attach` | deleted |
| `add_edge`, `remove_edge` | kept for application kinds; structural kinds refused |
| `merge` | deleted; `import_subtree` added |
| `as_of` | deleted (viewers filter by `seq`) |
| `store`, `store_write`, `_apply_store_write` | deleted |
| `node`, `nodes`, `agents`, `edges` | kept |
| `items` | replaced by `subscribe` |
| `prev` | deleted (list order) |
| `history` | kept; walks `fork` edges for the prefix |
| `own_history` | `agent.nodes` |
| `frontier` | `agent.nodes[-1]` (or the base node for an empty fork) |
| `terminal`, `ready` | engine readiness (§6.1) |
| `status` | kept: `running`/`idle`/`done` |
| `result`, `inbox` | kept (`inbox` is `agent.inbox`) |
| `depends_on`, `inputs`, `producers`, `launcher` | deleted |
| `parent`, `children` | kept, from `spawn` edges |
| `launched`, `_skipped_launches` | deleted; a fork's inherited children are the `spawn` edges from cells in its prefix |
| `lineage`, `path`, `depth` | kept |
| `tree` | renamed `subtree` |
| `usage`, `cost_usd` | kept |
| `create_agent(..., depends_on=, inputs=)` | `create_agent(task, name=, **args)` |
| `save`, `load` | kept |
| `_require`, `_require_agent`, `_default_name`, `_last` | kept as small helpers |
| `_append_unchecked` | replaced by direct writes (no generic log) |
| `check_invariants`, `_check_depends_acyclic` | replaced by `validate` (≈30 lines) |
| — | new: `calls`, `provenance`, `subtree`, `copy`, `subscribe`, `edit`, `delete`, `remove`, `replace`, `splice`, `import_subtree`, `Invalidation` (≈200 lines together) |

### 10.3 `flow.py` and `engine/` (1,184 → about 720)

| File | Lines | Action | New lines | Notes |
| --- | ---: | --- | ---: | --- |
| `flow.py` | 594 | rewrite | 380 | Method plan below. |
| `engine/scheduler.py` | 291 | adapt | 220 | Keeps `Pool`, `ThreadPool`, `SequentialPool`, `TaskQueue`, `StepResult`, `resolve_boundary`. `"idle"` boundary uses `IDLE`. `TaskQueue.join` is no longer needed by waits (kept for host `flow.wait`). |
| `engine/handoff.py` | 154 | delete | 0 | The result rule is §6.3, in `flow.py`. `render`/`_format_result` move to `harnesses/prompts.py` (≈40 lines). |
| `engine/workspaces.py` | 120 | keep | 110 | `LocalWorkspace`, `GitWorktrees`. Snapshots at turn ends and cell ends, checkout on rebuild. Needed for editing and forking coding agents whose files changed. |
| `engine/__init__.py` | 25 | adapt | 10 | |

`flow.py` method by method:

| Today | v2 |
| --- | --- |
| `__init__(llm_clients, configs, runtime, tools, workspaces, workers)` | unchanged |
| `llm`, `complete`, `llm_query` | unchanged |
| `add_config`, `config`, `resolve_config`, `config_for` | unchanged; `config_for` reads `agent.config` |
| `inject`, `add_tool`, `_open_toolkits`, `_close_toolkits`, `remove_tool`, `namespace_for` | unchanged |
| `spawn(step, task, inputs=...)` | no `inputs`; writes the `spawn` edge from the pending cell |
| `agent_tools` | becomes `server` (§7.3) |
| `_repl_args` | no `g.inputs` |
| `_run_cell`, `prepare_repl`, `execute`, `replay` | kept; pending receipts generalize the relaunch block; rebuild agents rewound by edits |
| `anchor`, `land` | `land` kept for cells written by session turns; anchors deleted |
| `wake` | replaced by the graph subscription |
| `step`, `run_step` | kept; no `StaleFrontier` branch; applies the result rule |
| `budget_exhausted`, `budget_nearly_spent` | kept, on `g.subtree` |
| `run_streaming` | loop of §6.2; no `handoff.react` passes |
| `_react` | replaced by the result rule |
| `_schedule` | kept, with §6.1 readiness |
| `arun`, `run`, `aclose` | unchanged |
| `recorded_cells`, `launched_from` | `recorded_cells` kept; `launched_from` deleted |
| `_final_sent`, `_stamp` | kept |
| `messages_to_query`, `FlowLLM` | move to `integrations/dspy.py` (deferred, §12) |

### 10.4 Agent API: `tools/` and `agent_tools/` (1,966 → about 1,290)

| File | Lines | Action | New lines | Notes |
| --- | ---: | --- | ---: | --- |
| `tools/__init__.py` | 581 | split | 250 | Keeps `tool`, `as_tool`, `tool_from_schema`, `Toolkit`, `tool_allowed`, `format_tool_line`, `format_tool_docs`, `tool_name`, `as_tool_items`, `RESERVED_TOOLS`. The agent calls move to `agents/api.py`. |
| `tools/files.py` | 173 | keep | 173 | |
| `tools/store.py` | 51 | move | 0 | Becomes the `board` tool in `examples/collaboration/shared_board.py`. |
| `agents/api.py` (new) | — | new | 260 | From `tools/__init__.py`: `launch_tool` (minus `split_args`, `_producer`, `_check_json` handle cases), `query`, `llm_query_tool`, `submit_tool`, `send_tool`, `repl_tools`, `build_namespace`. From `agent_tools/tools.py`: `spawn`, `send`, `done`, `repl`. New: the descendant check and await futures. Deleted: `waiting` (cycle check), `wait_agent_tool`, `restore_launch`/`restore_send` (generic receipts), `_on_flow_loop` stays. |
| `agent_tools/view.py` | 508 | rewrite | 130 | Becomes `agents/handle.py`: the REPL `Agent` handle (id, live reads, `send`, `__await__`) and `AGENTS`. Drops `AgentFrontier`, `_snapshot`, `AgentRecord`, `AgentEdge`, `AgentDirectory.edges`, `render_graph`/`print_graph` (use `view.text`), `get_parent`/`get_children`/`get_siblings`, `depends_on`, `siblings`. |
| `agent_tools/tools.py` | 90 | merge | 0 | Into `agents/api.py`. |
| `agent_tools/server.py` | 240 | adapt | 190 | Becomes `agents/server.py`: transport and turn bookkeeping for external harnesses; routes every call through `agents/api.py`. Keeps the `Host` prompt/stop hooks that `serve` uses. |
| `agent_tools/cli.py` | 124 | keep | 100 | `rlmflow repl`, the client harnesses call. |
| `agent_tools/mcp.py` | 50 | keep | 50 | MCP `python` tool. |
| `agent_tools/hooks.py` | 149 | keep | 140 | Claude Code hooks for the plugin: start `serve` with the session, forward prompts, stop it at session end. |

### 10.5 `harnesses/` (1,070 → about 900)

| File | Lines | Action | New lines | Notes |
| --- | ---: | --- | ---: | --- |
| `base.py` | 57 | keep | 50 | |
| `repl.py` | 269 | adapt | 230 | §7.1. |
| `sessions.py` | 288 | adapt | 200 | §7.2: `outcome` returns `done` or `idle`; drops `handoff.pending`, `g.launched`, `g.inputs`; records `session` on nodes. |
| `claude_code.py` | 143 | keep | 140 | Records the session message id where available. |
| `pi.py` | 103 | keep | 100 | |
| `prompts.py` | 146 | adapt | 120 | Argument manifest without inputs; gains result formatting from `handoff`. |
| `interactive.py` | 63 | keep | 60 | The harness for the plugin's root agent (the user's own Claude Code session). |

### 10.6 `runtime/` (2,097 → about 1,760)

| File | Lines | Action | New lines | Notes |
| --- | ---: | --- | ---: | --- |
| `runtime.py` | 383 | adapt | 330 | `agent_env` takes the `Agent` record. |
| `repl.py` | 392 | adapt | 300 | Drops `CurrentMapping`, `CurrentObject`, `current_binding`, `_ContextStream` per-tenant routing if unused, `AgentReference`, `encode_agent_references`. |
| `repl_client.py` | 558 | adapt | 450 | Drops tenants from `WorkerSession`; receipts written as the call list. |
| `repl_server.py` | 437 | adapt | 360 | Drops `tenant_id` from every request model and `_tenants`. |
| `docker.py` | 121 | keep | 121 | |
| `modal.py` | 165 | keep | 165 | |
| `__init__.py` | 41 | keep | 35 | |

The worker protocol, proxied host tools, receipts and replay are the hard,
necessary part of the library; this rewrite trims them but does not remove
them.

### 10.7 `llm/` and `integrations/` (973 → about 640)

| File | Lines | Action | New lines | Notes |
| --- | ---: | --- | ---: | --- |
| `llm/__init__.py` | 323 | keep | 300 | |
| `llm/openai.py` | 102 | keep | 102 | |
| `llm/anthropic.py` | 121 | keep | 121 | |
| `llm/tinker.py` | 210 | defer | 0 | Returns with RAO. |
| `integrations/mcp.py` | 117 | keep | 117 | `MCPTools`. |
| `integrations/dspy.py` | 99 | defer | 0 | Returns as an extra with `FlowLLM`. |

### 10.8 Apps: `view/`, `cli/`, `rao/`, plugin (4,740 → about 2,390 in the first release)

| File | Lines | Action | New lines | Notes |
| --- | ---: | --- | ---: | --- |
| `view/figure.py` | 619 | adapt | 560 | Layout reads `spawn` edges for the tree and draws `return` and `message` edges directly; `handoffs()` and `parent_ids()` simplify; peer arcs come for free. |
| `view/html.py` | 351 | adapt | 330 | |
| `view/text.py` | 75 | adapt | 70 | `waiting` → `idle`. |
| `view/timeline.py` | 162 | adapt | 120 | `snapshot` filters by `seq`. |
| `view/live.py` (new) | — | move | 170 | `LiveGraphTree`, `LiveTreeRenderer` from `consumers.py`. |
| `view/__init__.py` | 69 | adapt | 40 | |
| `cli/__init__.py` | 367 | adapt | 300 | Keeps `serve`, `plugin`, `fork`, `config`, `version`. |
| `cli/run.py` | 339 | adapt | 200 | `run print` only; `tui` deferred. |
| `cli/view.py` | 182 | keep | 170 | |
| `cli/fork.py` | 113 | keep | 100 | |
| `cli/tui.py` | 592 | defer | 0 | |
| `cli/serve.py` | 363 | adapt | 330 | "Your Claude Code session as the root." `WAITING` checks become `IDLE`; turn ends read `session` from nodes. |
| `rao/*` | 1,508 | defer | 0 | Returns as an extra; credit along `return` edges fixes the "launch tree only" limit. |
| `plugin/skills/orchestrate/SKILL.md` | 226 | rewrite | 60 | The four calls and the await rule. |
| `plugin/commands/run.md`, hooks, MCP config | 36 | keep | 36 | |

### 10.9 Totals

| Part | Today | First release | Change |
| --- | ---: | ---: | ---: |
| Core: §10.1–10.7 (graph, engine, agent API, harnesses, runtime, LLM, misc) | 9,899 | about 6,880 | −30% |
| Apps: §10.8 (view, CLI including `serve`, RAO) | 4,740 | about 2,390 | −50% |
| **`rlmflow/` total** | **14,639** | **about 9,270** | **−37%** |

The Claude Code plugin path (`cli/serve.py`, `agent_tools/hooks.py`,
`harnesses/interactive.py`, the plugin files) is in the first release; it
accounts for about 600 of these lines.

Deferred code that could return later as extras: `cli/tui.py` (592),
`rao/` (1,508), `llm/tinker.py` (210), `integrations/dspy.py` plus `FlowLLM`
(about 130). About 2,400 lines if all of it returns.

The core estimate is higher than the 3,000 lines guessed in review. The
difference is the parts that are necessary and not policy: the worker protocol
and replay (about 1,760), the LLM clients (about 520), and Docker and Modal
(about 290). The graph, engine and agent API together are about 2,700.

## 11. Tests, examples, docs, repository

### 11.1 Tests (about 8,100 lines today)

| File | Lines | Action | Notes |
| --- | ---: | --- | --- |
| `test_graph_core.py` | 649 | rewrite | New model, `validate`, forks, `calls`, `provenance`, every invalidation rule in §5.3. About 700. |
| `test_graph.py` | 115 | merge | Into `test_graph_core.py`. |
| `test_graph_scalability.py` | 182 | adapt | |
| `test_flow.py` | 1,859 | rewrite | Keeps scheduling, replay, limits, failures, streaming, boundaries; drops handoff/`depends_on`/`waiting` cases; adds await rule, result rule, edit-then-run. About 1,100. |
| `test_handoff.py` | 438 | rewrite | The result rule only. About 120. |
| `test_flow_configs.py` | 129 | keep | |
| `test_agent_tools.py` | 287 | rewrite | Agent API and handle. About 250. |
| `test_agent_tools_server.py` | 175 | adapt | |
| `test_external_harnesses.py` | 371 | adapt | `idle` instead of `waiting`. |
| `test_modes_e2e.py` | 264 | adapt | |
| `test_concurrency.py` | 191 | adapt | Host `send` mid-run wakes the loop. |
| `test_runtime.py` | 378 | adapt | No tenants. |
| `test_worker_transport.py` | 380 | adapt | No tenants. |
| `test_llm.py` | 443 | keep | |
| `test_metadata.py` | 105 | adapt | |
| `test_submission.py` | 62 | keep | |
| `test_filesystem_tools.py` | 45 | keep | |
| `test_workspaces.py` | 136 | keep | |
| `test_figure.py` | 55 | adapt | |
| `test_layers.py` | 125 | rewrite | New package layout and import rules. |
| `test_minimal_interface_conformance.py` | 87 | rewrite | Public API surface of §10.1. |
| `test_store.py` | 106 | move | Becomes a test of the board recipe, or is deleted. |
| `test_interactive.py` | 350 | adapt | The plugin path: `serve`, the interactive root, hooks. |
| `test_rao_textcraft.py` | 553 | defer | With RAO. |
| `test_benchmark_fanout.py`, `test_benchmark_metrics.py` | 122 | delete | |
| `helpers.py` | 159 | adapt | |
| `fixtures/fake_claude.py` | 199 | keep | |
| `fixtures/v1_tree_run/` | — | delete | No format converters. |
| `live/test_live_modes.py` | 207 | adapt | |

New: `test_edit.py` (about 300) for edit, delete, remove, replace and splice
end to end with a fake model, including a session-harness replace.

### 11.2 Examples (12,748 tracked lines of Python today)

| Example | Action |
| --- | --- |
| `architectures/tree.py`, `review.py` | keep, adapt |
| `architectures/pipeline.py` | rewrite: a root that awaits stages |
| `needle/haystack.py`, `needle/harnesses/run.py`, `needle/render_parallel.py` | keep, adapt (flagship: RLM and mixed harnesses) |
| `needle/filesystem.py` | keep |
| `needle/fork_halfway.py` | move into the graph tour |
| `collaboration/shared_board.py` | rewrite with a `board` tool (DeLM flagship) |
| `collaboration/decentralized_peers.py` | keep, adapt |
| `coding/boids/*` | keep, adapt (peers flagship) |
| `coding/agent.py` | delete (the CLI covers it) |
| `shepherd/*` | keep, port to host-side `g.fork` plus a `fork` tool for the meta-agent (recovery flagship) |
| `autoresearch/*` | keep; gains `WorkspaceSync` |
| `graph/01_query.py` … `06_fork.py`, `deep_tree.py` | rewrite as a four-script tour: query, save/load, fork, edit/replace |
| `graph/07_attach_subtree.py` | delete |
| `control/delegation/*`, `control/streaming_until.py`, `control/controller_injection.py`, `control/injection/*` | merge into `control/steering.py` (host `send`, `until=`) |
| `control/branching/best_of_n.py`, `fork_repair.py` | merge into `control/edit_and_fork.py` |
| `providers/mcp_weather*.py` | keep |
| `providers/claude_code_delegation.py` | delete (duplicates the needle harness matrix) |
| `providers/dspy_drop_in.py`, `drop_in_llm.py` | defer with DSPy |
| `providers/tinker_agent.py`, `research/*` | defer with RAO |
| `sandboxes/*` | keep |
| `showcase.py`, `structured_output.py` | keep |
| `skills.py`, `dynamic_skills.py` | merge into one |
| `llm_query_fanout.py`, `summarizer.py` | delete |
| `behavior/`, `graph_queue/` (empty) | delete |
| `notebooks/` | delete unless it runs on v2 |
| `run_examples.py` | rewrite the manifest (about 250 lines) |
| `common.py` | keep |

Expected result: about 6,000 lines of example Python.

### 11.3 Docs

| Doc | Action |
| --- | --- |
| `README.md` | Rewrite around §1: the story, the API table of §4, one RLM example, one edit/replace example, the six structure GIFs, the four flagships. |
| `docs/usage.md` | Rewrite for v2 host API, including editing. |
| `docs/agents.md` | Rewrite for §4. |
| `docs/control.md`, `docs/injections.md`, `docs/streaming.md` | Merge into one "Running and steering" page. |
| `docs/node_model.md`, `docs/internals.md`, `docs/observability.md` | Merge into "The graph" (§5) and "Internals" (§6–7). They describe the 0.7 architecture today. |
| `docs/positioning.md` | Rewrite: §1 plus the comparison with LangGraph, PydanticAI, ADK, CrewAI, Claude Code subagents, RLM. |
| `docs/design/scope.md` | Rewrite: the three parts, the await rule, the rule for what enters the core. |
| `docs/design/codebase.md`, `docs/design/file_structure.md` | Rewrite after the code lands. |
| `docs/design/minimal_dynamic_graph.md`, `docs/design/examples-cleanup.md` | Mark superseded by this document. |
| `docs/design/showcase-tasks.md`, `docs/benchmarks.md` | Delete with the benchmark harness. |
| `docs/cli.md`, `docs/runtimes.md`, `docs/security.md`, `docs/shepherd.md`, `docs/skills.md`, `docs/prompt_customization.md` | Adapt. |
| `docs/history/*` | Keep as archive. |
| `CHANGELOG.md`, `UPGRADING.md` | One v2 entry listing §8. |

### 11.4 Repository

- Remove the `benchmark-contract` jobs in `ci.yml` and `release.yml`. Done.
  `benchmarks/` itself is kept for now (it holds uncommitted work), with its
  Make targets and `docs/benchmarks.md`; it targets the old API.
- `pyproject.toml`: drop the `eval`, `eval-environments`, `official-rlm`,
  `jev`, `instructor`, `textcraft`, `openenv`, `sandbox` (duplicate of
  `modal`) extras; keep `rao`, `tinker` and `dspy` only once those return.
  Do not regenerate `uv.lock` until the extras change is agreed.
- Version: 0.8.0 (breaking, pre-1.0).

## 12. Order of work

Each phase ends with the listed checks green. The work was done in place in
`rlmflow/`, in the working tree: no `rlmflow2/` package, no branch, no tag.

| Phase | Work | Done when |
| --- | --- | --- |
| 0 | Remove the benchmark CI jobs | CI green without them |
| 1 | `graph/` (model, API, invalidation, persistence) | `test_graph_core.py`, `test_edit.py` (graph-only parts) pass with no model |
| 2 | `flow.py`, scheduler, native harness, agent API, runtime trims | With a fake model: tree, pipeline, review, RLM needle, result rule, await rule, budget, replay, edit-then-run. Live: `needle/haystack.py` |
| 3 | Server, `rlmflow repl`, MCP, session harnesses, `serve`, hooks, interactive harness, plugin | `needle/harnesses/run.py --all` live (REPL × Claude Code × pi), Boids with Claude Code peers, `/rlmflow:run` from a Claude Code session launches workers into the graph, `test_interactive.py` |
| 4 | Host editing in practice | Shepherd ported; edit/replace example; DeLM board example |
| 5 | View, CLI (first-release commands), docs, README; delete the old package | All examples in the manifest run; docs build; release 0.8.0 |
| 6 (later) | TUI, RAO, DSPy, Tinker | Each returns only if a real user needs it, as an extra |

## 13. Risks and open questions

1. **Long awaits from Claude Code.** A cell that awaits a long child is a
   blocking `rlmflow repl` call, which can hit Claude Code's Bash timeout.
   Mitigation: a configurable server-side timeout that returns "still
   running" to the cell, and prompt guidance to await in a later cell.
2. **Replace and edit on session harnesses are coarser.** The native harness
   re-runs a cell exactly. Session harnesses rewind to the last turn boundary
   (or to a session message, where the harness supports resuming at one), so
   the model may write different code on the re-run.
3. **Effects outside the REPL are not undone.** Files are restored from
   workspace snapshots when workspaces are configured. External API calls are
   the application's responsibility.
4. **Invalidation can cascade.** Editing an early node of a large run can
   rewind most of it. `dry_run=True` shows the cost first; `g.copy()` keeps
   the original.
5. **Peers cannot block on each other.** That is deliberate. If a real
   application needs a blocking request to a peer, the answer is a host-side
   fork of the peer awaited by the caller, added later as a recipe, not a
   primitive.
6. **No old saved runs.** Format 4 only. Acceptable at 0.x; stated in
   `UPGRADING.md`.
7. **Core size.** About 6,900 lines, not 3,000 (§10.9). Further cuts would
   have to come from the worker protocol, which needs its own design.

## 14. Where the implementation differs

1. **Receipts are per tool.** A cell's `data["calls"]` maps each tool name to
   `{call_index: saved}`; launches and awaits save `{"$rlmflow_agent": id}`,
   and a replaced await saves `{"value": v}`. `g.calls(cell)` flattens them into
   `Call(cell, tool, index, saved)`.
2. **No splice.** Edits are `edit`, `delete`, `remove` and `replace`; inserting
   nodes mid-history is not an operation.
3. **Edits raise on busy agents** (`GraphError`), rather than queueing until the
   agent's step lands. Stop the stream (`until="idle"`) first.
4. **Naming.** The REPL handle class is `AgentHandle` (its repr is still
   `Agent(...)`); `rlmflow.Agent` is the graph record.
5. **Kept from the old graph API:** `Graph.ready()`, `Graph.frontier()` and
   `Graph.deliver()`, which the scheduler and session harnesses use directly.
6. **A replaced await keeps a `return` edge** from the old child's `done`, with
   `data["replaced"] = True`, so the old result is not delivered again as a
   message.
7. **A stopped cell is kept, not discarded.** When `run_streaming` stops with a
   cell mid-run, the cell lands marked to re-run with the receipts it saved, so
   its children survive and the re-run gets them back. Steps cut off outside a
   cell, and runs loaded from a save taken mid-step, are cleaned up by
   `g.prune_pending()`, which takes back agents and messages from steps that
   never landed. When a stream stops, results that finished children haven't
   delivered yet are delivered.
8. **Extras and benchmarks.** The extras listed in §11.4 are dropped and
   `uv.lock` regenerated; a `bench` extra carries `datasets`. The old harness
   under `benchmarks/eval` is deleted without an archive tag (it is in git
   history); `benchmarks/table`, which compares Claude Code, Claude Code with
   rlmflow, and rlmflow on OOLONG and ALE-Bench, replaces it.

