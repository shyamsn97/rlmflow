# rlmflow internals

rlmflow has one durable type and two resource owners:

```text
Graph      the run: nodes in agents' histories, and edges
Flow       execution and policy: turns, results, edits
Runtime    live REPL sessions
```

`Flow` is the public facade. Everything public is re-exported from `rlmflow`
itself, so which file a name lives in is an implementation detail.

The layout follows that. `rlmflow.graph` is the run: `nodes.py` (`Node`,
`RootData`, `Edge`, `Step`, and the context variable a running node's tools
read), `graph.py` (reads and writes), `edits.py` (edits and `RULES`),
`calls.py` (saved tool calls) and `persistence.py` (the saved format).
`rlmflow.flow` takes turns, runs nodes and delivers results.
`rlmflow.harnesses` take one turn for an agent (`repl`, `claude_code`, `pi`,
`interactive`). `rlmflow.tools` builds the REPL namespace: `launch_agent`,
`await`, `send`, `submit`, `llm_query`. `rlmflow.agent_tools` is the same agent
API over HTTP, for session harnesses and the Claude Code plugin.
`rlmflow.runtime` runs REPLs in worker processes, containers or sandboxes.
`rlmflow.workspaces` gives agents their working directories.

## The graph

```python
g.agent(id), g.node(id), g.owner(node)
root.root_data.history, g.base(root)
g.history(root), g.tail(root), g.last_turn(root)
g.edges(kind=..., source=..., target=..., touching=...)
```

An agent's history is a list on its root; history order is the order its REPL
ran things in, so there is no linked structure to keep consistent. Edges are
kept in one table with per-endpoint indexes, so "which edges touch this node"
is a lookup. `spawn` edges also maintain a parent-to-children index. A fork is a
root with a `fork` edge from its base node, and shares the base's history
without copying it.

There is no lock. The Flow is the only writer during a run, and host code inside
`async for step in flow.run_streaming(g)` runs on the same loop between steps.

## The loop

```text
run_streaming(g)
  -> for every agent with no turn in flight:
       attach unseen results; if status is "running", start a turn task
  -> yield a Step for each node that finished since the last pass
  -> wait for the graph to change, then schedule again
```

`g.status(agent) == "running"` is the scheduling rule (see
[the graph](node_model.md#status)). `flow.tasks` maps an agent id to its turn in
flight, so an agent never takes two turns at once.

A turn runs a node an edit left ready, if there is one, and otherwise calls the
agent's harness. The native `repl` harness turns the history into chat
messages, calls the model, appends the reply as one assistant node with its
`repl` block as `code`, and runs it in place. A turn that raises appends a
terminal node with `[run failed: ...]` or `[child failed: ...]`, so a crashed
child is a value its parent reads.

`flow.run_node(g, agent, node)` holds the agent's REPL lock, brings the REPL up
to the node before it, binds the node's `input`, and runs its `code`. Each run
gets a `CallLog` on its `Step`, which the tools save to. A node stopped mid-run
goes back to ready, and its agent's next turn runs it again.

## Results

`deliver_results(g, agent)` runs when the agent is between turns (no turn in
flight, not done, nothing left ready). It collects children that are done and
whose terminal node has no `return` edge (`queries.unseen_results`), appends one
user node binding each result as a variable, and links a `return` edge from
each. Those edges are the delivery record, so a resumed stream delivers nothing
twice.

## Awaiting

`await child` runs on the Flow's event loop, waking whenever the graph changes,
until the child is done. It checks first that the child is a descendant
(`NotYourAgent` otherwise), so a cycle of awaits can't form. When the child
finishes, the await links a `return` edge from its terminal node to the
awaiting node and saves `{"tool": "await", "agent": ..., "done": ...}` in that
node's calls.

## Replay

A Flow keeps a REPL per agent it has run. When it reaches an agent whose REPL
it doesn't hold (a loaded run, a fork, a different process), it runs the
agent's history again, the fork base's included, before the next node. Each
tool call claims what the node saved instead of running: a launch resolves to
the existing child, an await to the saved result, `llm_query` and
`@tool(cache=True)` tools to their saved value. Calls are keyed by tool name and
arguments, so repeated identical calls get their entries in order. Nothing is
appended, and `RLMFLOW_REPLAY` is `"1"` while it happens.

The Flow remembers the last node (and its version) each live REPL ran. When
that node is no longer in the agent's history, the REPL is dropped and rebuilt
from the new history.

## Running a node again

`g.edit`, `g.delete`, `g.remove` and `g.replace` set the nodes that consumed a
changed result ready again: the history after the node is cut, its output is
cleared, its `return` edges are removed, and its version is bumped. The
agent's next turn runs it in place on a REPL rebuilt up to the node before it:
calls it makes the same way return what they saved, the changed one returns
the new value or the new agent's result, and saved calls it no longer makes are
undone (their agents removed, their messages taken back).

`flow.edit(g, op, target, ...)` previews the edit, cancels the affected agents'
turns, then applies it. A run that started on an older version of its node is
discarded.

## Runtime identity

The runtime keys live REPLs by agent id, and hands each one the agent's process
env (`RLMFLOW_AGENT_ID`, depth, parent id, max depth) when it opens.
`Runtime.execute` is where a dead REPL becomes an observation: the node is
marked `dead`, skipped on replay, and the next run opens a fresh REPL.

`flow.inject(name, value)` and `flow.add_tool(fn)` write into the flow's tool
namespace and into every open REPL; `Repl.seed(tools)` binds them before each
run. Arguments are bound once, when the root runs. Nothing in the turn path
closes a REPL, so a finished agent's worker stays warm for a message that
reopens it. Closing is the caller's: `flow.runtime.close_repl(agent)`,
`close_repls()`, or `flow.aclose()`.

## Persistence

`g.save(path)` writes one JSON document (format 6): nodes, with each root's
history by id, and edges, plus content-addressed blobs for large inputs.
Loading validates that every node is in exactly one history and that every
edge endpoint exists. A node saved mid-run loads ready, and runs again with
what it launched still in place. A root records only the name of the config it
runs with; the config a Flow bound is in `root_data.metadata["config"]` for
viewers and is never read back.

## Invariants

1. Records hold no live execution objects.
1. Every node is in exactly one agent's history.
1. Engine edges (`spawn`, `fork`, `return`, `write`) are written by the Flow,
   its tools and edits only.
1. An agent with a turn in flight is never given another; edits through
   `flow.edit` cancel it first.
1. A crashed turn is its agent's outcome: a terminal node with the error.
1. Agent names are unique among siblings and filesystem-safe.
1. Save and load are a cold boundary: REPLs are rebuilt by replay.
