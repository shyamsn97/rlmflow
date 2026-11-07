# The bare minimum

This replaces `minimal.md`. It is built from an audit of every part of rlmflow
that reads or writes the graph: the Flow, the harnesses, the REPL tools, the
agent-tools server, `AGENTS`, the CLI, the views, the plugin skill and every
example. Section 1 lists what they need. The design after it is the smallest
thing that provides all of it.

---

## 1. What has to keep working

### Running

| Need | Who needs it |
| --- | --- |
| Step every agent that has work, concurrently | `Flow.run_streaming` |
| Model turn: history → chat messages → `llm` node | `ReplHarness` |
| Run the reply's cell; `submit` ends the agent | `ReplHarness`, `Flow.execute` |
| Session turns (Claude Code, pi) that run cells through `rlmflow repl` and end `idle` or `done` | `SessionHarness`, `AgentToolsServer` |
| `launch_agent`, `await agent`, `agent.send`, `submit`, `llm_query`, host tools | `tools/__init__.py` |
| `await` only for descendants (`NotYourAgent`) | `await_tool` uses `lineage` |
| Unawaited results reach the parent as one message, once | `deliver_results`, `unseen_results` |
| A message wakes a `done` or `idle` agent | status rule |
| `max_iters`, `max_budget` (final-turn message, sent once), `max_depth` | harness, `_schedule`, `can_spawn` |
| A failed step becomes a `done` with the error; a root's error is raised | `run_step` |
| Stop at a boundary: `"done"`, `"next"`, `"idle"`, or a predicate | `resolve_boundary` |
| Workspaces: branch on launch, snapshot after a cell, check out after replay | `workspaces` |
| A dead REPL is rebuilt; the cell that killed it isn't replayed | `execute`, `recorded_cells` |
| Images attached by a tool reach the model | `attach_image` |
| Tool metadata on the calling cell | `record_metadata` |

### Host-side manipulation (examples, CLI, plugin)

| Need | Example |
| --- | --- |
| Build agents and history by hand | `graph/01_query.py`, `deep_tree.py`, `07_host_spawn.py` |
| Launch a child from host code | `g.add_agent(parent=...)` |
| Message an agent between runs **and between yields of a running stream** | `controller_injection.py`, `streaming_until.py`, `fork_halfway.py` |
| Script a turn: append an `llm` reply, then run its cell | `shepherd.py` (`play_jam`) |
| Step one agent by hand, outside a run | `flow.step(g, agent)` in shepherd |
| Rebuild one agent's REPL on demand and read its env | `flow.replay`, `runtime.repl_for(a).env` |
| Fork at any node, or a whole agent; as a child of a cell; with another config | `06_fork.py`, `recovery.py`, `cli/fork.py` |
| Edit, delete, remove, replace, with a preview first | `03_edit.py`, orchestrate skill |
| Custom relationships between agents (`"peer"`) visible to agents | README, orchestrate skill, `AGENTS.edges` |
| History up to a node (`g.history(node)`) | `recovery.py`, harness `history(step)` |

### Visuals

| Need | Who |
| --- | --- |
| A global order of every agent and node | `timeline(g)` |
| The graph as it stood after step *k* | `snapshot`, `replay`, GIF/PNG frames, HTML stepper |
| Each item's drawn parent: launching cell or fork base for an agent, the previous node for a node | `figure.parent_ids` |
| Cross-agent arrows: results to the cell or message that got them, messages from their sending cell | `figure.handoffs` |
| Token size per node, elapsed and run time | figure, HTML |
| The agent tree with status per agent | `render_tree`, `AGENTS.print_graph` |

### Saving

| Need | Who |
| --- | --- |
| Save and load the whole run, atomically | everything |
| Load a run saved mid-cell and resume without redoing children | `test_edits`, streaming |
| Large arguments not duplicated in every checkpoint | blobs |
| A cheap "landed since" check | `serve.py` (uses `_seq` today) |

---

## 2. Three layers

| Layer | Answers | Contains |
| --- | --- | --- |
| **Graph** | What happened? | Records and reads. No policy. |
| **Flow** | What happens next? | Reads the graph, runs steps, appends. |
| **Edits** | What if it had gone differently? | Rewrite history and redo what depended on it. |

Views, persistence and queries are functions that read the graph.

---

## 3. The data

```python
@dataclass
class Node:
    id: str
    agent: str            # owner
    type: str             # "llm" | "code" | "message" | "idle" | "done"
    data: dict            # JSON
    metadata: dict        # JSON, the application's
    seq: int              # global order: when it (last) started
    pending: bool         # its step started and hasn't finished
    started_at: float | None
    finished_at: float | None


@dataclass
class Agent:
    id: str
    name: str
    task: str
    args: dict            # JSON, Input wrappers allowed
    config: str | None    # a config name; None for a root
    parent: str | None    # the cell (or agent) that launched it
    base: str | None      # the node its history continues from
    nodes: list[Node]
    metadata: dict
    seq: int


@dataclass
class Link:               # application relationships only ("peer", ...)
    kind: str
    source: str
    target: str
    data: dict
```

**Why `seq` stays.** Every visual is built on a global order of agents and
nodes: the timeline, snapshots at step *k*, frames, GIFs, the HTML stepper.
Timestamps can tie and come from a clock. `seq` is one integer per record,
assigned in exactly two places (`add_agent`, `append`) and when a pending
cell is rerun after an edit, so "when it last started" stays true.

**What used to be an edge is a field:**

| Relationship | Field |
| --- | --- |
| Launched by | `agent.parent` (cell id, or agent id from host code) |
| Forked from | `agent.base` |
| Message sent by | `message.data["by"]` (the sending cell or agent id; absent for host) |
| Result awaited by a cell | the cell's `{"tool": "await", "agent", "done"}` saved call |
| Result delivered by message | the results message's `data["results"]`: list of `done` ids |
| Application relationship | `g.links` |

`queries.edges(g)` turns these into `(kind, source, target)` triples for views
and `AGENTS`. Nothing is stored twice.

---

## 4. The graph

```python
class Graph:
    agents: dict[str, Agent]
    links: list[Link]

    # writes
    def add_agent(self, task, *, name=None, parent=None, base=None, config=None, **args) -> Agent
    def append(self, agent, node) -> Node

    # reads
    def node(self, id) -> Node
    def history(self, agent_or_node) -> list[Node]   # through a node: the history up to it
    def status(self, agent) -> str                   # "running" | "idle" | "done"
    def result(self, agent)
    def parent(self, agent) -> Agent | None
    def children(self, agent) -> list[Agent]
    def roots(self) -> list[Agent]
```

`links` is a plain list; adding one is `g.links.append(Link(...))`. No other
mutation lives on `Graph`.

No lock, no subscribers, no `busy`, no `revision`, no edit log, no indexes
beyond `agents`. `node(id)` and `children` scan. Anything that needs fast
reverse lookups (edits, views) builds them once per call (section 7).

---

## 5. Unread messages: the one real decision

A message can arrive while its recipient is mid-step. Today it waits in
`agent.inbox` and is delivered between turns. Dropping the inbox is only safe
with an explicit rule, because otherwise this happens:

```text
llm, code (pending), message, done   <- done; the message was never read
```

**Option A, keep the inbox.** `agent.inbox: list[Node]`. Sending appends there;
the Flow moves it into `nodes` between turns. Status: running if the inbox is
non-empty. Simple to state, but it's a second list per agent that views,
persistence, edits and snapshots all have to handle.

**Option B, no inbox, a read rule.** Messages are appended to `nodes` right
away. A message is *read* once a model turn starts after it. Status: running
if any message comes after the agent's last `llm` node. This needs every
turn to land its `llm` node when it starts, session turns included, so the
order stays true:

```text
llm (pending), message, llm (filled), code, done   <- message after the llm: unread, agent wakes
```

Recommendation: **B**. One list per agent, the history is exactly what
happened in the order it happened, and the rule is one line. The cost is that
session harnesses must land an `llm` node at the start of each turn, which
also makes them uniform with the REPL harness.

Status under B:

```python
def status(self, agent):
    history = self.history(agent)
    if not history or any(n.pending for n in agent.nodes):
        return "running"
    last_llm = max((i for i, n in enumerate(history) if n.type == "llm"), default=-1)
    if any(n.type == "message" for n in history[last_llm + 1:]):
        return "running"
    last = history[-1]
    return last.type if last.type in ("done", "idle") else "running"
```

---

## 6. The Flow

### State it keeps

```python
tasks: dict[str, asyncio.Task]      # agent id -> its step in flight (was g.busy)
repls: dict[str, str]               # agent id -> id of the last cell its live REPL ran
changed: asyncio.Condition          # notified on every write the Flow makes
landed: list[Node]                  # finished nodes to yield
```

### Rules it owns

- **Scheduling.** Every agent whose status is running and has no task gets
  one. Before the step, the budget's final-turn message (once; marked
  `data["final"]`).
- **A step.** The harness returns one node. Before the model call or cell, the
  node lands pending; it's filled in when done. A cell that submits is
  followed by a `done`. A pending node left from a cut-off step is rerun.
- **Results.** When an agent has no step in flight, its finished children
  whose `done` no saved `await` or results message references get one
  results message.
- **`await child`.** Checks lineage, then waits on `changed` until the child
  is done; saves `{"tool": "await", "agent", "done"}` on the cell.
- **Writes from REPL threads** (launch, send, await) hop onto the loop. The
  Flow is the only writer during a run. Host code inside `async for` runs on
  the loop between yields, so it writes safely too.
- **REPLs.** Reused if the agent's last finished cell is the one the REPL ran;
  otherwise rebuilt by replay (section 8).
- **Configs.** The resolved `AgentConfig` travels in `Step(graph, agent, node,
  config)`, not on the `Agent` record. Its JSON is recorded in
  `agent.metadata["config"]` for viewers.
- **Failures.** A raising step lands `done` with `[run failed: ...]` or
  `[child failed: ...]`; a root's error is re-raised.

### Entry points kept

`run_streaming(g, until=)`, `run`, `arun`, `step(g, agent)` (one step by
hand), `replay(g, agent)` (rebuild a REPL), `deliver_results(g, agent)`,
`llm_query`, `add_tool`, `inject`, `add_config`.

### Removed

`TaskQueue`, `Pool`/`ThreadPool`/`SequentialPool` (model calls use async
clients or `asyncio.to_thread`, with an optional semaphore), `subscribe`, the
context variables `current_step`/`current_code` (the Flow passes the cell to
tool calls).

---

## 7. Edits

Functions in `graph/edits.py`. Every one checks that the agents it touches
have no step in flight (the caller passes the Flow's `tasks`, or edits between
runs), works on a copy first so a failure changes nothing, and returns an
`Invalidation(rewound, removed, dropped, rerun)`. A preview is the same call
with `dry_run=True`, which returns the copy's result.

```python
edits.fork(g, node_or_agent, *, name=None, parent=None, config=None)
edits.edit(g, node, **data)
edits.delete(g, node)
edits.remove(g, agent)
edits.replace(g, cell, index, value=... | agent=...)
```

What each keeps from today, exactly:

| Edit | Effect |
| --- | --- |
| `edit(node)` | Update data; cut the owner's history after it; results it delivered go stale; forks based at it start over |
| `delete(node)` | Cut from the node on |
| `remove(agent)` | Cut everything; remove agents it launched (recursively) and its forks |
| `replace` a launch | Point the call at another agent; the old child is removed |
| `replace` an await | Return a value or another agent's result; the old `done` stays referenced so it isn't delivered again as a message |
| `replace` a tool call | Return a value |

Dropping a node undoes what it caused:

- agents it launched are removed;
- messages it sent are taken back (their recipient is cut back to before them);
- results it delivered go stale: a results message is taken back, a cell that
  awaited it is marked pending and its history after it is cut;
- forks based at it are removed.

**Reverse lookups are built per edit, not maintained.** At the start of an
edit, one pass over the graph builds four dicts: launched-by, sent-by,
consumers-of-a-done, forks-at. The cascade uses them and they are thrown away.
That keeps a large `remove` linear without any bookkeeping on `Graph`.

---

## 8. Replay

A REPL is rebuilt by running the agent's finished cells in chain order, with
saved calls answering:

```python
for cell in g.history(agent):
    if cell.type == "code" and not cell.pending and not cell.data.get("dead"):
        repl.run(cell.data["code"], saved=cell.data["calls"])
```

then the latest workspace snapshot is checked out.

- The n-th call to a tool in a cell gets the n-th saved entry for that tool
  (Temporal's model).
- A fork's history starts with its base's chain, so those cells replay first.
- `record_metadata` and `attach_image` do nothing during replay.

**Limits, disclosed:** non-deterministic code (`random()`, time, changed
files) can rebuild different state; results that weren't saved (not JSON, or
over 64KB) run again with their side effects. Calls are matched by position;
a mismatch doesn't raise.

---

## 9. Visuals

Everything the views need, from the data above:

| View need | From |
| --- | --- |
| Global order | sort agents and nodes by `seq` |
| Snapshot at step *k* | keep records with `seq` below the cutoff; links whose ends survive |
| Agent's drawn parent | `agent.base`, else `agent.parent` |
| Node's drawn parent | the previous node in its agent's chain |
| Cross-agent arrows | `queries.edges(g)`: return and message triples |
| Tree with status | `roots`, `children`, `status`, `result` |
| Size, timing | `data["usage"]`, `started_at`, `finished_at` |

Two consequences of nodes landing when they start:

- A node sits in the timeline where it started, with its final content. The
  HTML stepper shows "ran 3.2s" for it, so the duration is still visible.
- A message that arrived mid-turn sits after that turn's `llm` node, which is
  what the model actually saw.

---

## 10. Persistence

`persistence.save(g, path)` and `persistence.load(path)`:

```json
{"format": 6, "agents": [{"id": "...", "nodes": [...], ...}], "links": [...]}
```

- Written to a temp file and renamed, so a crash leaves the previous save.
- Large argument strings go to `blobs/` by content hash, written once. This
  is persistence-only; the graph never sees it.
- Loading validates that every reference (`parent`, `base`, `by`, saved
  `agent`/`done` ids, links) points at something that exists.
- A cell saved mid-run loads pending and reruns, reusing what it launched.
- `copy(g)` is save-and-load in memory, used by edits and snapshots.

The checkpointer saves on its interval and always on close. `serve.py`
checks "landed since" with `len(root.nodes)` instead of `_seq`.

---

## 11. Coverage check

| Today | Bare minimum |
| --- | --- |
| `g.send(agent, text)` | `messages.send(g, agent, text)`: append a message node (a 3-line helper, since everyone sends) |
| `g.deliver(agent)` | gone (option B) |
| `g.fork(...)` | `edits.fork(...)` |
| `g.edit/delete/remove/replace(...)` | `edits.*` |
| `g.add_edge("peer", a, b)` | `g.links.append(Link("peer", a.id, b.id, {}))` |
| `g.edges(kind=...)` | `queries.edges(g, kind=...)` |
| `g.subtree/lineage/path/usage` | `queries.*` |
| `g.subscribe`, `g.busy`, `g.revision`, `g.edits` | Flow state, or gone |
| `agent.inbox` | gone (option B) |
| `agent.config` (live object) | `step.config` |
| `Edge`, `_out`, `_in`, `_parent`, `_children`, `_forks` | fields, plus per-call reverse lookups |
| `flow.step`, `flow.replay`, `flow.deliver_results` | kept |
| `timeline`, `snapshot`, `replay`, figure, HTML | kept, reading `seq` and `queries.edges` |
| `AGENTS` (status, result, parent, children, edges) | kept, built from the reads and `queries.edges` |

---

## 12. Decisions still open

1. **Inbox (section 5).** Recommend option B.
2. **Edit methods on `Graph` or functions in `edits`.** Functions keep `Graph`
   at two writes; methods read better in the README (`g.fork(cell)`).
   Recommend functions, re-exported at the top level.
3. **The edit log.** Nothing reads it today. Recommend dropping it.
4. **`send` as a helper.** Every harness, tool and example sends; a helper that
   builds the message node keeps the shape (`by`, `sender`) in one place.
   Recommend keeping it, as a function.

---

## 13. Order of work

1. Records, `add_agent`, `append`, reads, `links`, `queries.edges`; option B.
2. `edits.py` and `persistence.py`.
3. Flow: tasks dict, pending-at-start for every step, config on `Step`, no
   queue or pool, no context variables.
4. Tools, harnesses (session turns land `llm`), agent-tools server.
5. Views, CLI, consumers, docs, examples.

Each step ends with `pytest`, `ruff check` and `ruff format` green.
