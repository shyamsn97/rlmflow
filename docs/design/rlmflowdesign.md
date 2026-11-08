# rlmflow redesign: nodes, edges, and a REPL

rlmflow has two primitives:

1. **A graph** of `Node` and `Edge` records.
2. **A REPL** that runs nodes.

Every feature is built from those two. A node is one chat message: text the
model reads, inputs that become variables, and code the REPL runs. An agent
is a root node with an ordered history. Spawn, fork and results are edges.
Talking to another agent is attaching a node to its history. Replay runs a
root's history again, with saved tool results standing in for anything that
already happened.

This version folds in the audit of the previous draft: history is a list, not
a chain of `next` edges; agents never choose code that runs elsewhere; session
turns run many cells; edits cut suffixes; and the argument-reseeding bug ships
first.

---

# 1. A node is text, inputs and code

```text
Node
├── role       "assistant" (this root's model) or "user" (task, attached nodes, results, host)
├── text       markdown the model reads
├── input      JSON values, bound as variables before the code runs
├── code       Python the REPL runs ("pass" if there is nothing to run)
├── calls      saved results of tool calls, returned on replay
├── output     what the run printed
├── result / error
├── state      "ready" | "running" | "done"
└── data       outcome flags (terminal, idle), usage, seen, producer, ...
```

Running a node is always the same two steps:

1. bind each `input` entry as a variable;
2. run `code`.

Every kind of history entry is the same record:

| Node | role | text | input | code |
|---|---|---|---|---|
| Root | user | the task, plus argument descriptions | `task` and the arguments | `pass` |
| Model turn | assistant | the model's reply | | the fenced code blocks from the reply |
| Session cell | assistant | | | the code sent through `rlmflow repl` |
| Session turn end | assistant | the session's last message | | `pass` |
| Attached by another agent | user | the sender's text | the sender's values | `pass` |
| Unawaited result | user | "root.worker-0 finished: ..." | the result | `pass` |
| Budget warning | user | "Final turn: submit now." | | `pass` |
| Host | user | anything | anything | anything (host code is trusted) |

Only the model's own turns and the host carry code. Everything else is text
plus inputs.

There are no `LLM`, `CODE`, `MESSAGE`, `IDLE` or `DONE` node types. Submit,
idle and failure are flags on the node that ended the turn.

## 1.1 Binding inputs

Input names follow the rule arguments already follow: valid identifiers, not
keywords, not reserved bindings (`submit`, `launch_agent`, `llm_query`,
`AGENTS`, tool names). Binding happens once, when that node runs, and again
on replay. Overwriting one of the agent's own variables is allowed; that is
what passing a value means. The node's text tells the model what was set.

## 1.2 Where code comes from

`code` is a separate field, filled once when the node is created:

- **Model turn:** the fenced Python blocks extracted from `text`, in order. A
  later parser change can't change what replays.
- **Host:** whatever the host wrote.
- **Anyone else:** `pass`. Their text is shown verbatim; a
  ```` ```python ```` block inside it is never run.

Editing a model turn edits `text`; `code` is re-extracted and the node reruns.

## 1.3 Turning history into chat

One loop, no per-type branches:

```python
def chat(history: list[Node]) -> list[dict]:
    messages = []
    for node in history:
        if node.text:
            messages.append({"role": node.role, "content": node.text})
        if node.output or node.error:
            messages.append({"role": "user", "content": render_output(node)})
    return messages
```

The system prompt is not a node. It comes from the root's config and prompt
profile, as today.

## 1.4 Data versus live objects

- **Data passed to an agent** (arguments, attached values, results) is a
  node's `input`: recorded, saved, replayed.
- **Live objects** (tools, clients, functions) are bound by the Flow before
  every run (`flow.inject`, `add_tool`). They aren't JSON and aren't history.

---

# 2. Agents are root nodes

```text
RootData                      on a root node only
├── name, task
├── arg_descriptions          from Input(...) wrappers
├── config_name
├── history: list[node_id]    history[0] is the root itself; O(1) append
└── metadata                  recorded config JSON and application data
```

`RootData` is the agent's identity, typed, stored on the node that starts its
history. Every other node has `owner = root.id`.

The argument values live in the root's `input`, so running the root binds
`task`, `needle` and `haystack` once. Nothing rebinds them before later cells,
so `haystack = haystack[:100]` sticks (section 9).

A fork is a root node too. Its text says where it came from ("Forked from
root.solver at node 12"), its code is `pass`, and a `fork` edge from the base
node gives its replay prefix, which includes the original root and its
inputs. It copies the source's task and config name unless told otherwise.

---

# 3. Edges

```text
Edge(relation, source, target, data)
```

| Relation | From | To |
|---|---|---|
| `spawn` | the cell (or host) that launched | the child root |
| `fork` | the base node | the fork root |
| `return` | the child's terminal node | the cell that awaited it, or the result node |
| `write` | the cell that attached a node | the attached node |
| anything else | application-defined (`"peer"`, ...) | |

History order is not an edge; it is `RootData.history`.

---

# 4. Ordering: history is what the REPL ran

One rule: **a root's history order is the order its REPL ran things in.**

- Every node is appended at the tail, ready.
- One executor per root runs ready nodes in history order. It is a lock
  around the root's REPL; model and session turns are separate tasks.
- A model or session turn records `seen`: how many history entries were in
  its prompt. The Flow captures it when the turn starts and writes it on the
  node that turn produces.

### A node attached during a model call

```text
model request starts with [A]          seen = 1
W attached, runs                       [A] [W]
reply arrives, appended as M, runs     [A] [W] [M seen=1]
```

M did not see W. W is past M's `seen`, so the root is still running and the
next turn reads W.

### A node attached during a session turn

```text
turn starts with [A]                   seen = 1
cell C1 runs                           [A] [C1]
W attached, runs                       [A] [C1] [W]
cell C2 runs                           [A] [C1] [W] [C2]
turn ends                              [A] [C1] [W] [C2] [end seen=1, idle]
```

Nothing is inserted mid-history. `seq` order and history order agree, so the
timeline, snapshots and frames show what really happened.

### A node attached during a blocked `await`

The awaiting cell holds the REPL, so W waits for the executor and runs after
the cell finishes. Same rule.

## 4.1 Status

```python
def status(g, root) -> str:
    history = g.own_history(root)
    if history[-1].state != "done":
        return "running"                      # something left to run
    turn = last_turn(history)                 # last node with data["seen"]
    if turn is None:
        return "running"                      # never had a turn
    unread = any(n.data["producer"] != "self" for n in history[turn.data["seen"]:])
    if unread:
        return "running"                      # someone wrote since it last looked
    if turn.data.get("terminal"):
        return "done"
    if turn.data.get("idle"):
        return "idle"
    return "running"                          # a ReplHarness turn that didn't submit
```

Nodes from this root's own turns (replies, session cells, turn ends) have
`producer="self"`. Everything else (attached nodes, results, host guidance)
is from someone else. A node attached after a terminal turn is unread, so it
wakes a done root. Unfinished nodes are always at the tail (finished prefix,
suffix cuts on edit), so checking the tail, or scanning back a step or two,
finds the next one to run. There is no stored cursor.

There is no inbox, delivery pass, reserved draft node or session cursor.

---

# 5. Attaching a node to another agent

`send` is the agent-facing name for attaching a node:

```python
AGENTS["root.reviewer"].send("Escaped quotes break around line 40.", lines=[40, 41, 63])
```

is exactly:

```python
node = g.append(
    reviewer,
    role="user",
    text="From root.lexer: Escaped quotes break around line 40.\nSet: `lines`",
    input={"lines": [40, 41, 63]},
    producer="root.lexer",
)
g.link("write", sending_cell, node)
```

When it runs, the reviewer has a `lines` variable, exactly as if `lines` had
been an argument at launch. The model reads the text.

| Who attaches | What they control |
|---|---|
| Host code | text, inputs and code (`g.append(...)`) |
| Another agent | text and inputs; the code is `pass` |

An agent never chooses code that runs in another agent's REPL. This matters
because a child in Docker or Modal can attach a node to a parent running
locally.

---

# 6. Results: `await` or a result node

A child's result reaches its parent one of two ways. Both are a `return` edge
from the child's terminal node to the node that read it.

**`await` in a cell**, when the parent needs the values to continue:

```python
a, b = await asyncio.gather(launch_agent("check parser"), launch_agent("check lexer"))
submit(merge(a, b))
```

The cell pauses; the result is saved in the cell's `calls`. Only descendants
can be awaited (`NotYourAgent`), which rules out two agents waiting on each
other.

**A result node**, for children nobody awaited: when a child finishes, the
Flow appends a user node to the parent with text "root.worker-0 finished:
..." and the result as an input (`worker_0`, from the child's name). It is
unread, so it wakes the parent.

A result is delivered once: a child that was awaited never also arrives as a
result node.

---

# 7. Replay

A REPL is rebuilt by running the root's history again: the fork prefix first,
then its own nodes, skipping nodes that aren't done and nodes whose run killed
the worker.

```python
for node in g.history(root):
    if node.state == "done" and not node.data.get("dead"):
        await repl.execute(node.code, input=node.input, saved=node.calls)
```

Then the latest workspace snapshot is checked out.

Saved calls (section 7.1) return their results, so:

| Call | On replay |
|---|---|
| `launch_agent(...)` | the same agent; nothing launches |
| `await agent` | the saved result; no waiting |
| `llm_query(...)` | the saved answer; no model call |
| `@tool(cache=True)` host tools | the saved value |
| other host tools | run again |
| `agent.send(...)` | nothing is attached again |
| `submit(...)` | stops at the same point |

Node outputs can't replace replay: replay exists to rebuild live Python
variables, which can't be saved.

**Limits, disclosed:** plain Python that isn't a saved call (`random`, the
clock, network, files outside the workspace) can rebuild different state.
Results over 64KB aren't saved and run again.

## 7.1 Saved calls

A saved call is keyed by the call itself:

```text
key = sha256(tool name + canonical JSON of its arguments)
entry = {"key": "a1f3...", "n": 0, "tool": "launch_agent", "result": ...}
```

Canonical JSON has sorted keys and no whitespace. `n` counts repeated
identical calls in one cell, so a loop that makes the same call twice gets two
entries.

Which calls are saved:

| Tool | Key covers |
|---|---|
| `launch_agent` | task, config, name, args, and the resolved config |
| `await agent` | the awaited agent's id |
| `agent.send` | the target's id, the text and the inputs |
| `llm_query` | prompt, model, the schema's JSON schema, sampling |
| `@tool(cache=True)` | the tool's arguments; a non-JSON argument raises |
| `submit` | not saved; it ends the cell |
| other host tools | not saved; they run again |

The built-in tools take only JSON or ids, so their keys are exact. Host tools
opt in with `cache=True`, which declares JSON arguments and a JSON result.

When a cell runs (first run, replay or rerun after an edit):

- **hit:** the saved result comes back; nothing runs;
- **miss:** the call runs for real and is saved;
- **saved but not made again:** once the cell finishes, its effect is undone.
  An unclaimed launch removes that child; an unclaimed send takes the attached
  node back.

Replay and edits share this one mechanism. An unchanged cell hits on every
call. An edited cell reuses whatever it still calls the same way: fixing a
`print` keeps a ten-minute child; changing the child's arguments launches a
new one and removes the old.

During replay every call should hit. A miss means the code did something
different (randomness, time, a changed file), and replay logs a divergence
warning naming the root, node and call.

---

# 8. Edits

## 8.1 In-flight work

Edits go through the Flow when a run is live:

1. Preview the edit on a copy to find the affected roots.
2. Cancel those roots' turn and execution tasks.
3. Apply the edit.

Cancelling first matters: a cell still running after an edit could otherwise
keep launching children and attaching nodes, all linked to a node that no
longer means what it did.

Nodes also carry a `version`. A completion whose node changed or vanished is
discarded and that root's REPL is rebuilt. This is the backstop; cancellation
is the rule.

## 8.2 What an edit does

Editing or rerunning a node:

1. applies the "source changed" rule to its outgoing edges;
2. cuts every later node in its root's history;
3. clears its output, marks it ready, bumps its version.

Its saved calls stay. The rerun claims the ones it makes again; the rest are
undone when it finishes (section 7.1).

If the rerun submits early, no stale future turns are left after it.

## 8.3 Rules are constants

```python
RULES = {
    # relation: (source deleted, source changed)
    "spawn":  ("remove_root", "match_calls"),
    "fork":   ("remove_root", "restart_root"),
    "return": ("rerun_target", "rerun_target"),
    "write":  ("retract_target", "match_calls"),
}
```

- `match_calls`: a changed cell keeps the children it launches the same way
  and the nodes it attaches the same way; the others are removed or taken
  back once the rerun finishes.
- Retracting an attached node cuts its recipient back to before it.
- Application relations have no rule.

Edits: `edit(node, text=... | input=... | code=...)`, `delete(node)` (cut from
it on), `remove(root)` (and everything it caused), `replace(cell, i, value |
agent=)` (swap one saved call; an awaited result that's swapped out stays
marked as read). Every edit returns what it removed, rewound and reran, and
accepts `dry_run=True`.

---

# 9. The argument bug

Today, before every cell:

```python
repl.seed(build_namespace(self, step), {"task": agent.task, **agent.resolved_args()})
```

so `x = 999` on an argument is undone before the next cell, and large
arguments cross to the worker on every execution. The fix: bind tools before
each run, bind arguments only when the root runs. This ships on its own,
before anything else.

---

# 10. Records

```python
NodeState = Literal["ready", "running", "done"]


@dataclass(slots=True)
class RootData:
    name: str
    task: str
    config_name: str | None
    arg_descriptions: dict[str, str] = field(default_factory=dict)
    history: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Node:
    role: Literal["assistant", "user"]
    text: str = ""
    input: dict[str, Any] = field(default_factory=dict)
    code: str = "pass"
    calls: list[dict[str, Any]] = field(default_factory=list)
    output: str = ""
    result: Any = None
    error: str | None = None
    state: NodeState = "ready"
    data: dict[str, Any] = field(default_factory=dict)
    root_data: RootData | None = None
    owner: str = ""
    id: str = field(default_factory=lambda: new_id("node"))
    seq: int = -1
    version: int = 0
    started_at: float | None = None
    finished_at: float | None = None


@dataclass(slots=True)
class Edge:
    relation: str
    source: str
    target: str
    data: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: new_id("edge"))
    seq: int = -1
```

---

# 11. Graph

```python
class Graph:
    # writes
    def add_root(self, task, *, name, config=None, parent=None, **args) -> Node
    def append(self, root, *, role, text="", input=None, code="pass", producer) -> Node
    def fork(self, at, *, name, parent=None, config=None) -> Node
    def link(self, relation, source, target, **data) -> Edge

    # reads
    def node(self, id) -> Node
    def root(self, node) -> Node
    def roots(self) -> list[Node]
    def own_history(self, root) -> list[Node]
    def history(self, root_or_node) -> list[Node]     # with fork prefix; through a node
    def tail(self, root) -> Node
    def status(self, root) -> str
    def result(self, root)
    def edges(self, *, relation=None, source=None, target=None) -> list[Edge]
    def parent(self, root) -> Node | None             # source root of its spawn edge
    def children(self, root) -> list[Node]
```

- `append`, `tail` and `status` are O(1). `history` is O(length) because
  returning it is.
- `append` validates input names (section 1.1).
- `seq` is one counter, assigned when a node or edge is added.
- `fork(parent=...)` adds a `spawn` edge too, so a supervisor can adopt a
  branch (Shepherd recovery). A finished fork for previews is `fork(...)`
  followed by appending a terminal node.
- No lock, subscribers, `busy`, revision, edit log or cursor. The Flow is the
  only writer during a run; host code inside `async for` runs on the same loop.

---

# 12. Flow

```python
class Flow:
    tasks: dict[str, asyncio.Task]      # root id -> its turn in flight
    executors: dict[str, asyncio.Lock]  # root id -> its REPL, one run at a time
    repls: dict[str, str]               # root id -> last node its live REPL ran
```

- **Schedule:** for every root, run any ready nodes; if its status is running
  and it has no turn in flight, start a turn.
- **Model turn:** capture `seen`, call the model, append the reply as one
  assistant node with `seen`, run it. A failure appends a terminal node with
  the error; a root's failure is raised.
- **Session turn:** capture `seen`; each `rlmflow repl` call appends and runs
  a cell; the turn ends with an assistant node carrying `seen` and `terminal`
  or `idle`.
- **Results:** when a child finishes and nothing awaited it, append a result
  node to its parent.
- **Budget:** before the last affordable turn, append the warning node once.
- **Configs:** the resolved `AgentConfig` travels in
  `Step(graph, root, node, config)`; its JSON is recorded in
  `root_data.metadata["config"]`.
- **Workspaces:** branch on launch, snapshot after a cell, check out after
  replay.
- **Host control kept:** `run_streaming(g, until=)`, `run`, `arun`,
  `step(g, root)`, `replay(g, root)`, `edit(...)` (cancel then apply).

Removed: `TaskQueue`, the pools, `PooledLLMClient` (one semaphore in the
Flow), the three context variables (the running node is passed to tools).

---

# 13. Visuals

| Need | From |
|---|---|
| Global order | `seq` |
| Snapshot at step *k* | nodes and edges below the cutoff; each history filtered |
| A node's drawn parent | the previous node in its root's history |
| A root's drawn parent | its `fork` source, else its `spawn` source |
| Cross-root arrows | `return` and `write` edges |
| Tree with status | `roots`, `children`, `status`, `result` |
| Node card | `text`, `input` names, `code`, `output`; size from `data["usage"]` |

Every node renders the same way: its text, what it set, its code, its output.

---

# 14. Persistence

```json
{"format": 6, "id": "graph_...", "nodes": [...], "edges": [...]}
```

Written to a temp file and renamed. Large values in `input`, `output` and
`result` use the blob store. A node saved `running` loads `ready` and runs
again, reusing its saved calls.

Format 5 can load through a one-way upgrade (open decision 1): agents become
root nodes with the same ids and their arguments as inputs; `llm` nodes become
assistant nodes with their text and `code="pass"`, followed by their existing
code nodes; messages become attached user nodes with their text; inbox entries
are appended; fork bases become `fork` edges.

---

# 15. Order of work

Each step passes `pytest`, `ruff check` and `ruff format` before the next.

1. Fix argument reseeding, with a regression test.
2. Add node versions and cancel-then-edit.
3. History-equals-execution-order with `seen`; remove the inbox and delivery,
   keeping today's records.
4. One binding rule: every node's inputs become variables; `send` attaches
   text plus inputs; results arrive as inputs.
5. Root nodes with `RootData`; one node record with `role`, `text`, `input`,
   `code`.
6. Edits on constant rules.
7. Format 6 persistence.
8. Delete the scheduler classes and old state.
9. Views, CLI, docs, plugin, examples, evals.

---

# 16. Costs

1. `RootData` is the agent concept, typed. That is the point: one graph
   representation, without pretending an agent has no data.
2. Model turns merge reply and code into one node; a migrated run keeps them
   as two. Both render as text, code, output.
3. Another agent can overwrite a variable by attaching an input with its name.
   That is the feature; the text always says what was set.
4. Cancelling or version checks can't undo side effects outside rlmflow's
   tools.
5. Replay is only as deterministic as the code.

---

# 17. Open decisions

1. **Format 5:** start fresh at format 6 (loading a format-5 run raises), or
   ship an upgrade loader.

---

> If a feature can't be expressed as nodes with text, inputs and code, edges
> between them, and running a root's history in a REPL, it doesn't belong in
> the core.
