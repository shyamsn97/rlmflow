# The graph

A `Graph` holds runs, the nodes each run owns, and edges between them. Everything the engine does is a write to it, everything it decides comes from reading it, and you can edit it between runs.

## Nodes

There is one record, `Node`, a dataclass. A node is one chat message: text the model reads, inputs that become REPL variables, and code the REPL runs.

```python
Node(role, text, *, input=, code=, output=, result=, error=, state=, calls=,
     data=, metadata=, producer=, ...)
Node("assistant", "hi", data={"seen": 3, "idle": True})
```

| Field                              | Meaning                                                                                    |
| ---------------------------------- | ------------------------------------------------------------------------------------------ |
| `role`                             | `"assistant"` for the agent's own turns, `"user"` for everything else.                     |
| `text`                             | What the model reads: a reply, a task, a message.                                          |
| `input`                            | JSON values, bound as variables when the node runs.                                        |
| `code`                             | Python the REPL runs; `"pass"` when there is nothing to run (`node.runs_code` is `False`). |
| `calls`                            | What its tool calls returned, so running it again repeats nothing.                         |
| `output`, `result`, `error`        | What running it printed, what it submitted, and whether it failed.                         |
| `state`                            | `"ready"`, `"running"` or `"done"`.                                                        |
| `data`                             | Flags the engine reads: `seen`, `terminal`, `idle`, `usage`, `model`, ...                  |
| `producer`                         | Who wrote it: `"self"` for the agent, else the sender (`"user"`, an agent id, ...).        |
| `root_id`                          | The id of the `AgentRoot` it belongs to.                                                   |
| `id`, `seq`, `version`, `metadata` | Stable id, creation order, edit count, application JSON.                                   |
| `graph`                            | The graph it was added to; set by the graph.                                               |

Running a node is always the same two steps: bind each `input` entry as a variable, then run `code`. What an agent's nodes are:

| Node                  | role      | text                            | input                    | code                                 |
| --------------------- | --------- | ------------------------------- | ------------------------ | ------------------------------------ |
| AgentRoot             | user      | the task                        | `task` and the arguments | `pass`                               |
| Model turn            | assistant | the model's reply               |                          | its `repl(code)` tool call           |
| Session code          | assistant |                                 |                          | the code sent through `rlmflow repl` |
| Session turn end      | assistant | the session's last message      |                          | `pass`                               |
| Message from an agent | user      | `From root.lexer: ...`          | the values it sent       | `pass`                               |
| Results               | user      | the finished children's results | one variable per child   | `pass`                               |
| Host                  | user      | anything                        | anything                 | anything                             |

Only an agent's own turns and the host carry code; code written inside a message is never run. A turn's `text` is what the model wrote, and its `code` is what its `repl` call passed.

The node that ends a turn carries `seen` (how many of the agent's nodes its prompt had). It also carries `terminal` and `result` when the agent finished, either by `submit(value)` in its code or by a reply that doesn't call `repl`, or `idle` when a session ended its turn waiting. A failed turn is a terminal node with `[run failed: ...]` or `[child failed: ...]` as its result. `node.producer` is `"self"` on an agent's own nodes, and the sender otherwise. `rlmflow.view.text.kind(node)` names a node for display: `agent` (a task node), `message`, `done`, `idle`, `model`, `code` or `reply`.

`rlmflow.repl.record_metadata(key, value, node=...)` adds metadata; a host tool can leave out `node=` to annotate the node whose code called it. Metadata survives save, load and fork.

## Agent roots

One invocation of an agent starts with an `AgentRoot`, a `Node` subclass with fields shared by its whole chain:

| Field              | Meaning                                                                                  |
| ------------------ | ---------------------------------------------------------------------------------------- |
| `name`             | Unique among its siblings. `run.path()` is `root.worker-0`.                              |
| `task`             | What it was asked; `run.input` is `{"task": ..., **args}`.                               |
| `agent`            | The name of the Flow agent it runs with (`"default"` unless given).                      |
| `arg_descriptions` | From `Input(value, description=...)` arguments.                                          |
| `repl_id`          | The REPL, workspace and session it runs in: its own id, unless a fork gave it a new one. |

Every node has `root_id` (its `AgentRoot`'s id) and `seq`; the graph keeps an unsaved index from them, so there is no stored list of the root's nodes. `node.root` is its `AgentRoot`, and `node.transcript()` is the chain up to and including `node`: what the agent had seen there. Chain- and tree-wide methods live on `AgentRoot`: `root.nodes()`, `status()`, `final_result()`, `parent()`, `children()`, `path()` and the rest.

## Edges

| Kind     | From                          | To                                             | Meaning                |
| -------- | ----------------------------- | ---------------------------------------------- | ---------------------- |
| `spawn`  | The node that launched it.    | The child's root.                              | Who launched whom.     |
| `return` | A child's terminal node.      | The node that awaited it, or the results node. | Where a result went.   |
| `write`  | The node that sent a message. | The message node.                              | Who said what to whom. |

These three are written by the engine. Any other kind is yours: `g.link("peer", a, b, note="...")` and `g.unlink(edge)`. The engine never reads them; views draw them, and `AGENTS.edges(kind=...)` shows them to agents.

Reading the graph:

```python
g.agent_roots(), g.top_level_roots(), g.agent_root(id), g.node(id), node.root
g.nodes(), r.nodes(), g.edges(kind="return", target=node.id)
r.parent(), r.children(), r.lineage(), r.path(), r.subtree()
r.nodes()[-1], r.last_turn()
r.status()  # "done", "idle" or "running"
r.final_result()  # the last turn's result once done, else None
g.usage(), g.usage().cost_usd
```

Less common reads are plain functions in `rlmflow.graph.queries`, such as `unseen_results(g, run)` (finished children whose result reached no node yet).

## Status

A run is **running** when its last node hasn't finished, when it has never had a turn, or when a node from someone else arrived after its last turn started (`seen`). Otherwise its last turn decides: **done** if `terminal`, **idle** if `idle`, and **running** otherwise. The Flow starts a step for every running run that has none in flight. A message wakes a done or idle run, and the results of the children it launched arrive as a message.

## How results move

Inside the code, `await child` returns the child's result and adds a `return` edge from its terminal node to the awaiting node. Only descendants can be awaited; anyone else answers by message.

A child that finishes without being awaited reaches its parent in one results node before the parent's next turn, which binds each result as a variable named after the child, with a `return` edge from each child's terminal node. A child is delivered once: the `return` edge marks it as seen.

## Editing

The graph is mutable. Each edit follows the edges to what depended on what it changed, and returns an `Invalidation`:

| Field     | Meaning                                 |
| --------- | --------------------------------------- |
| `rewound` | Runs whose nodes were cut back.         |
| `removed` | Runs removed with their subtrees.       |
| `dropped` | Nodes removed.                          |
| `rerun`   | Nodes set ready, to run again in place. |

```python
g.edit(node, text="...")  # or input=, code=, result=; its run rewinds to it
g.delete(node)  # remove a node and everything after it
g.remove(run)  # remove a run, its subtree, and what they caused
g.replace(node, 2, value)  # bind the node's third saved call to a value
g.replace(node, 0, run=other)  # ... or to another run
g.edit(node, text="...", dry_run=True)  # report, change nothing
```

What follows from an edit is a constant table, `rlmflow.graph.RULES`, keyed by edge kind:

- Dropping a node removes the agents it launched (`spawn`) and takes back the messages it sent (`write`).
- Every edit cuts the nodes after the edited one. Changing a node's `code` or `input` also reruns it. Its saved calls stay: the rerun reuses the ones it makes the same way, and launches or messages it no longer makes are undone.
- Changing or removing a child's terminal node reruns the node that awaited it (`return`). On the next `flow.run(g)` it runs again in place: its other calls return what they saved, so only the changed value is new. A results message built from a changed result is taken back, and the results are delivered again.

While a Flow is running the graph, edit through it: `await flow.edit(g, "edit", node, text=...)` stops the affected runs first, then applies the edit.

## Forking

`g.fork(node)` leaves `g` alone and returns a new graph: a copy with the same ids, cut right after `node` by the same rules; every run that didn't depend on what was cut is copied as it is. If `node`'s code launched runs or sent messages, those are dropped and `node` runs again. `branch.metadata["forked_from"]` records the graph and node it came from. `repl="copy"` (the default) gives every run in the copy a new `repl_id`, so its REPL is rebuilt and its session forked; `repl="same"` keeps them. See [Control](control.md#fork).

## Persistence

`g.save(path)` writes `path/graph.json`:

```json
{"format": 8, "id": "...", "metadata": {...}, "nodes": [...], "edges": [...]}
```

Each node record keeps `root_id` under the format-8 JSON key `"owner"`, plus `seq` and `producer`. An `AgentRoot` record also has a `"run"` object: `name`, `task`, `agent` (its agent's name), `arg_descriptions` and `repl`. Large input strings go to `path/blobs/` by content hash. `Graph.load(path)` rebuilds the same graph with the same ids. A node saved while it ran loads ready and runs again; roots it already launched are reused through its saved calls rather than launched twice. Format 7 runs still load: the `"agent"` object is read as the root record, its `config` as the agent name, and a node's `session_id` as `data["session"]`. Older formats are not read.

## Streaming

`flow.stream(g)` yields each node as it finishes; `g.edges(touching=node.id)` are the edges touching it. Host code inside the `async for` runs on the Flow's loop, so it can read and write the graph between steps.
