# rlmflow internals

rlmflow has one durable type and three resource owners:

```text
Graph      the run: agents, nodes, edges, edits
Flow       execution and policy
Runtime    live REPL sessions
TaskQueue  steps in flight, and when one lands
```

`Flow` is the public facade. Everything public is re-exported from `rlmflow`
itself, so which file a name lives in is an implementation detail.

The layout follows that. `rlmflow.graph` is the run: `nodes.py` (the `Agent`
and `Node` records, `Edge`, `Step`, the context variables a step runs under),
`graph.py` (reads, writes, edits and invalidation) and `persistence.py` (the
saved format). `rlmflow.flow` steps agents and delivers results.
`rlmflow.engine.scheduler` holds the `TaskQueue`, `Pool`, `ThreadPool`,
`SequentialPool` and `resolve_boundary` for the `until` vocabulary.
`rlmflow.harnesses` turn a step into a node (`repl`, `claude_code`, `pi`,
`interactive`). `rlmflow.tools` builds the REPL namespace: `launch_agent`,
`await`, `send`, `submit`, `llm_query`. `rlmflow.agent_tools` is the same agent
API over HTTP, for session harnesses and the Claude Code plugin.
`rlmflow.runtime` runs REPLs in worker processes, containers or sandboxes.

## The graph

```python
g.agent(id), g.node(id), g.owner(node)
agent.nodes, agent.inbox, agent.base
g.history(agent), g.history(agent)[-1]
g.edges(kind=..., source=..., target=...)
```

An agent's history is a list; there is no linked structure to keep consistent.
Edges are kept in one table with per-endpoint indexes, so "which edges touch
this node" is a lookup. `spawn` edges also maintain a parent-to-children index.
A fork records its base node and shares the base's history without copying it.

Every write takes the graph's lock, stamps a `seq`, bumps `g.revision` and
notifies subscribers, so a stream and a host thread can both write.

## The loop

```text
run_streaming(g)
  -> deliver unseen results to every agent between turns
  -> schedule every ready agent: deliver its inbox, mark it busy, submit one step
  -> queue returns one StepResult
  -> yield a Step for each node that landed since the last one
  -> deliver results to that agent and its parent, then schedule again
```

`g.status(agent) == "running"` is the scheduling rule: an agent runs when it
has no nodes, its last node is a pending cell, or messages are waiting in its
inbox; otherwise its last node decides (`done` and `idle` stop it, anything else
runs). An agent with a step in flight is in `g.busy` and is never stepped twice
at once.

`run_step` calls the agent's harness, which returns one node, and appends it.
The native `repl` harness turns `llm`/`code`/`message` history into chat
messages, calls the model (landing `llm`), or runs the `llm` node's single
`repl` block. A step that raises lands `done` with `[run failed: ...]` or
`[child failed: ...]`, so a crashed child is a value its parent reads.

A cell lands when it starts: the `code` node is appended with
`data["pending"] = True` before its code runs, and filled in place with its
output when it finishes. A cell that submitted is followed by a `done` node with
the result. A cell stopped mid-run stays pending, and its agent's next step runs
it again.

## Results

`deliver_results(g, agent)` runs when the agent is between turns (not busy, its
last node not `llm`, `done` or a pending cell). It collects children that are
done and whose `done` has no `return` edge (`queries.unseen_results`), sends one
results message, and adds a `return` edge from each (`g.link_return`). Those edges are the delivery record, so a
resumed stream delivers nothing twice.

## Awaiting inside a cell

`await child` runs on the Flow's event loop, waking whenever the graph changes,
until the child is done. It checks first that the child is a descendant
(`NotYourAgent` otherwise), so a cycle of awaits can't form. When the child
finishes, the await draws a `return` edge from its `done` to the cell and saves
`{"tool": "await", "agent": ..., "done": ...}` in the cell's calls.

## Replay

A Flow keeps a REPL per agent it has stepped. When it reaches an agent whose
REPL it doesn't hold (a loaded run, a fork, a different process), it replays the
agent's recorded cells, the fork base's included, before the step. Each tool
call returns what the cell saved instead of running: a launch resolves to the
existing child, an await to the result of the `done` it saved, `llm_query` and
other host tools to their saved value. The n-th call to a tool in a cell gets
the n-th saved entry for that tool. Nothing is appended, and `RLMFLOW_REPLAY` is
`"1"` while it happens.

The Flow remembers the last cell each live REPL ran. When an agent's last
finished cell is no longer that one (an edit cut or changed its history), the
REPL is dropped and rebuilt by replay from the new history.

## Re-running a cell

`g.edit`, `g.delete`, `g.remove` and `g.replace` mark the cells that consumed a
changed result pending again: the history after the cell is cut, its output is
cleared, and its `return` edges are removed. Its agent then runs, and its next
step re-executes the cell in place (`Flow.rerun`) on a REPL replayed up to the
cell before it: unchanged calls return what they saved, the changed one returns
the new value or the new agent's result. The cell's output is replaced, its
`return` edges are linked again, and the agent continues from there.

## Runtime identity

The runtime keys live REPLs by `agent.id`, and hands each one the agent's process
env (`RLMFLOW_AGENT_ID`, depth, parent id, max depth) when it opens.
`Runtime.execute` is where a dead REPL becomes an observation: it drops the dead
entry so the next step opens a fresh one.

`flow.inject(name, value)` and `flow.add_tool(fn)` write into the flow's tool
namespace and into every open REPL. Nothing in the step path closes a REPL, so a
finished agent's worker stays warm for a message that reopens it. Closing is
the caller's: `flow.runtime.close_repl(agent)`, `close_repls()`, or
`flow.aclose()`.

## Persistence

`g.save(path)` writes one JSON document (format 5): agents with their node and
inbox ids, nodes, edges and the edit log, plus content-addressed blobs for large
arguments. Loading validates that every node is in exactly one history or inbox
and that every edge endpoint exists. A cell saved mid-run loads pending, and
runs again with what it launched still in place. An agent records only the name of the config it runs with;
the config a Flow bound is in `metadata["config"]` for viewers and is never read
back.

## Invariants

1. Records hold no live execution objects.
1. Every node is in exactly one agent's history or inbox.
1. Structural edges (`spawn`, `return`, `message`) are written by the engine
   and by edits only. A fork is `agent.base`, not an edge.
1. An agent with a step in flight is never stepped again or edited.
1. A crashed step is its agent's outcome: a `done` with the error.
1. Agent names are unique among siblings and filesystem-safe.
1. Save and load are a cold boundary: REPLs are rebuilt by replay.
