# The graph

A run is a `Graph`: agents, the nodes in each agent's history, and edges between
them. Everything the engine does is a write to it, everything it decides comes
from reading it, and you can edit it between runs.

## Agents

An `Agent` is a record, not a node:

| Field | Meaning |
| --- | --- |
| `id`, `name` | Stable id; a name unique among its siblings. `g.path(agent)` is `root.worker-0`. |
| `task`, `args` | What it was asked, and the variables its REPL starts with. |
| `config_name` | The `AgentConfig` it runs with (`None` for a root, which uses the run's `root_config`). |
| `nodes` | Its own history, oldest first. |
| `inbox` | Messages queued for its next turn. |
| `base` | For a fork, the node it branched at; its history continues from there. |
| `metadata`, `seq`, `created_at` | Annotations, creation order, timestamp. |

`g.history(agent)` is the fork base's history followed by `agent.nodes`; a fresh
root's history is empty. `Step.latest(g, agent)` is the agent at the last node
of that history, or at itself before it has any.

## Nodes

| `type` | Lands when | Data |
| --- | --- | --- |
| `llm` | A model call returns. | `text`, `usage`, `model` |
| `code` | A cell starts (`pending`), and is filled in when it finishes. | `code`, `output`, `error`, `calls` (what each tool call returned) |
| `message` | Something arrives from outside: a batch of results, a peer's or host's message. | `text`, `sender` |
| `idle` | A session agent ends a turn waiting for messages or for agents it launched. | |
| `done` | After a cell that called `submit(value)`, a reply with no `repl` block, or a failure. | `result`, `error` |

Every node has `id`, `type`, `agent_id`, `data`, `metadata`, `seq`,
`created_at`, and the `started_at` / `finished_at` of the step that produced it.
`rlmflow.record_metadata(key, value, node=...)` adds namespaced metadata; a host
tool can leave out `node=` to annotate the cell that called it. Metadata
survives save, load and fork.

A `code` node's `calls` make replay deterministic. They are a list in call
order: `{"tool": "llm_query", "value": ...}` for a tool's JSON result,
`{"tool": "launch_agent", "agent": id}` for a launch, and
`{"tool": "await", "agent": id, "done": id}` for an await. The n-th call to a
tool in the cell gets the n-th entry for that tool. Results that aren't JSON,
or are over 64KB, aren't saved, and those calls run again on replay.

## Edges

| Kind | From | To | Meaning |
| --- | --- | --- | --- |
| `spawn` | The cell that launched it (or the parent agent, from host code). | The child agent. | Who launched whom. |
| `return` | A child's `done`. | The cell that awaited it, or the `message` that delivered it. | Where a result went. |
| `message` | The cell (or agent) that sent it. | The `message` node. | Who said what to whom. |

These three are written by the engine. A fork isn't an edge: it is the
`agent.base` node it branched at. Any other kind is yours:
`g.add_edge("peer", a, b, note="...")` and `g.remove_edge(edge)`. The engine
never reads them; views draw them, and `AGENTS.edges(kind=...)` shows them to
agents.

Reading the graph:

```python
g.agents(), g.roots(), g.agent(id), g.node(id), g.owner(node)
g.nodes("code", agent=a), g.edges(kind="return", target=cell.id)
g.parent(a), g.children(a), g.lineage(a), g.path(a), g.subtree(a)
g.status(a)  # "done", "idle" or "running"
g.result(a)  # the latest done's result, or None
g.usage(), g.usage().cost_usd
```

Less common reads are plain functions in `rlmflow.graph.queries`, such as
`unseen_results(g, agent)` (finished children whose result hasn't been
delivered).

## Status

An agent is **running** when it has no nodes yet, its last node is a pending
cell, or messages are waiting in its inbox. Otherwise its last node decides:
**done** after `done`, **idle** after `idle`, and **running** after anything
else. The engine steps every running agent that isn't already stepping. A
message wakes an idle agent, and the results of the children it launched arrive
as messages.

## How results move

Inside a cell, `await child` returns the child's result and adds a `return` edge
from its `done` to the cell. Only descendants can be awaited; anyone else
answers by message.

A child that finishes without being awaited reaches its parent in one batched
`message` at the parent's next turn, with a `return` edge from each child's
`done`. A child is delivered once: the `return` edge marks it as seen.

## Editing

The graph is mutable. Each edit follows the edges to what depended on what it
changed, and returns an `Invalidation`:

| Field | Meaning |
| --- | --- |
| `rewound` | Agents whose history was cut back. |
| `removed` | Agents removed with their subtrees. |
| `dropped` | Nodes removed. |
| `rerun` | Cells marked pending, to run again in place. |

```python
g.edit(node, **data)  # change a node; its agent rewinds to it
g.delete(node)  # remove a node and everything after it
g.remove(agent)  # remove an agent, its subtree, and what they caused
g.replace(cell, 2, value)  # bind the cell's third saved call to a value
g.replace(cell, 0, agent=other)  # ... or to another agent
g.edit(node, text="...", dry_run=True)  # report, change nothing
```

What follows from an edit:

- Cutting a cell removes the agents it launched and the messages it sent.
- Changing or removing a child's `done` marks the cell that awaited it to re-run.
  On the next `flow.run(g)` that cell runs again in place: its other calls
  return what they saved, so only the changed value is new, and the agent
  continues from the new output.
- A results `message` built from a changed result is removed, and the results
  are delivered again.

Edits are refused while an affected agent is stepping (`g.busy`). Every edit is
appended to `g.edits`, which is saved with the graph.

## Persistence

`g.save(path)` writes `path/graph.json`:

```json
{"format": 5, "id": "...", "agents": [...], "nodes": [...], "edges": [...], "edits": [...]}
```

Each agent lists its `nodes` and `inbox` by id; large string arguments go to
`path/blobs/` by content hash. `Graph.load(path)` rebuilds the same graph with
the same ids. A run saved in the middle of a cell loads with that cell still
pending, and the cell runs again; agents it already launched are reused through
its saved calls rather than launched twice. A stream that stops mid-cell leaves
the same state.

## Streaming

`flow.run_streaming(g)` yields a `Step(graph, agent, node)` for each node as it
lands; `step.edges` are the edges touching it. For everything else, subscribe:

```python
unsubscribe = g.subscribe(lambda event, item: print(event, item))
# events: "agent", "node", "edge", "edit", "deliver"; g.revision counts them
```
