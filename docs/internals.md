# rlmflow internals

rlmflow has one durable type and three live ones:

```text
Graph      the run: task nodes, the nodes after them, and edges
Flow       execution and policy: steps, results, edits
Agent      how one kind of run takes a step (its harness)
Runtime    live REPL sessions
```

`Flow` is the public facade. The names most code needs are re-exported from `rlmflow` itself; the rest live in their subpackages.

The layout follows that. `rlmflow.graph` is the run: `nodes.py` (`Node`, `AgentRoot`, `Edge`), `graph.py` (reads and writes), `edits.py` (edits and `RULES`), `calls.py` (saved tool calls), `queries.py` (who has work, unseen results, usage) and `persistence.py` (the saved format). `rlmflow.agents` holds the harnesses: `agent.py` implements `Agent` (the model-and-REPL loop), while `claude_code.py`, `pi.py` and `person.py` implement the session harnesses; `prompts.py` holds their prompt text. `rlmflow.llm.base` defines the provider-neutral client protocol, replies and completion path, while the provider adapters live in `openai.py` and `anthropic.py`. `rlmflow.cli.app` defines the command-line application and configuration API. Their package `__init__.py` files only expose public names. `rlmflow.repl` is the `Repl` handle a step and a tool get for their root. `rlmflow.flow` takes steps, runs nodes and delivers results. `rlmflow.tools` is split by responsibility: `declare.py` defines host tools and prompt docs, `agents.py` holds `AgentHandle` and `AGENTS`, `builtins.py` implements orchestration calls and their replay receipts, and `namespace.py` assembles a REPL's bindings. `rlmflow.agent_tools` is the HTTP/CLI/MCP bridge for CLI agents and the Claude Code plugin; `agent_tools.view` remains a compatibility import for handles now implemented in `tools.agents`. `rlmflow.runtime` runs REPLs in worker processes, containers or sandboxes. `rlmflow.workspaces` gives roots their working directories.

## The graph

```python
g.agent_root(id), g.node(id), node.root
root.nodes(), root.repl_id, node.transcript()
root.nodes()[-1], root.last_turn()
g.edges(kind=..., source=..., target=..., touching=...)
```

An agent invocation starts at an `AgentRoot`, which names the configured Flow agent (`root.agent`). Every node records its `root_id` and `seq`, and the graph keeps a private index from them that is not saved. The order of a root's nodes is the order its REPL ran things in, so there is no linked structure to keep consistent. Edges are kept in one table with per-endpoint indexes, so "which edges touch this node" is a lookup. `spawn` edges also maintain a parent-to-children index. A fork is another graph: `g.fork(node)` copies the graph (same ids) and cuts the copy with the edit rules (`rlmflow.graph.edits.fork`).

There is no lock. The Flow is the only writer during a run, and host code inside `async for node in flow.stream(g)` runs on the same loop between steps.

## The loop

```text
stream(g)
  -> hand each node that finished since the last pass to the hooks, then yield it
  -> for every run with work and no step in flight:
       attach unseen results; ask the hooks; start a step task
  -> wait for the graph to change, then look again
```

`queries.has_work` is the scheduling rule: the run is running (see [the graph](node_model.md#status)), or it is idle with every node done and a result it hasn't read. The stream's `Stream` record maps a run id to its step in flight, so a run never takes two steps at once.

A step runs a node an edit left ready, if there is one, and otherwise calls `agent.step(node, llm, repl)`, which returns one node. The Flow appends it: a node with code runs in place (`repl.exec(node)`); otherwise it is marked `terminal` if it has a result and `idle` if not. `Agent.step` turns the run's nodes into chat messages (`messages`), calls the model with one `repl(code)` function tool, and returns the reply as one assistant node. The call's code becomes the node's `code`; its printed output becomes the following tool result in later model messages. A step that raises appends a terminal node with `[run failed: ...]` or `[child failed: ...]`, so a crashed child is a value its parent reads.

`repl.exec(node)` holds the run's REPL lock, brings the REPL up to the node before it, binds the node's `input`, and runs its `code`. Tools reach the running node through `current_repl()`, whose `calls` they save to. A node stopped mid-run goes back to ready, and its run's next step runs it again.

## Results

`deliver_results(g, run)` runs when the run is between steps (no step in flight, not done, nothing left ready). It collects children that are done and whose terminal node has no `return` edge (`queries.unseen_results`), appends one user node binding each result as a variable, and links a `return` edge from each. Those edges are the delivery record, so a resumed stream delivers nothing twice.

## Awaiting

`await child` runs on the Flow's event loop, waking whenever the graph changes, until the child is done. It checks first that the child is a descendant (`NotYourAgent` otherwise), so a cycle of awaits can't form. When the child finishes, the await links a `return` edge from its terminal node to the awaiting node and saves `{"tool": "await", "agent": ..., "done": ...}` in that node's calls.

## Replay

A Flow keeps a REPL per `repl_id` it has run: a run's own id, unless a fork gave it a new one. Workspaces and Claude Code or pi session forking are keyed the same way. When it reaches a run whose REPL it doesn't hold (a loaded run, a fork, a different process), it runs the run's nodes again before the next node; `flow.repl(run).catch_up()` does the same from host code. Each tool call claims what the node saved instead of running: a launch resolves to the existing child, an await to the saved result, `llm_query` and `@tool(cache=True)` tools to their saved value. Calls are keyed by tool name and arguments, so repeated identical calls get their entries in order. Nothing is appended, and `RLMFLOW_REPLAY` is `"1"` while it happens.

Each live REPL remembers the last node (and its version) it ran, in `repl.last_ran`. When that node is no longer among the run's nodes, the REPL is dropped and rebuilt from the ones that remain.

## Running a node again

`g.edit`, `g.delete`, `g.remove` and `g.replace` set the nodes that consumed a changed result ready again: the run's nodes after it are cut, its output is cleared, its `return` edges are removed, and its version is bumped. The run's next step runs it in place on a REPL rebuilt up to the node before it: calls it makes the same way return what they saved, the changed one returns the new value or the new run's result, and saved calls it no longer makes are undone (their runs removed, their messages taken back).

`flow.edit(g, op, target, ...)` previews the edit, cancels the affected runs' steps, then applies it. A step that started on an older version of its node is discarded.

## Runtime identity

The runtime keys live REPLs by `repl_id`, and hands each one the run's process env (`RLMFLOW_AGENT_ID`, depth, parent, max depth) when it opens. `Runtime.execute` is where a dead REPL becomes an observation: the node is marked `dead`, skipped on replay, and the next run opens a fresh REPL.

`flow.add_tool(fn)` (or `add_tool(value, name=...)`) writes into the flow's tool namespace, and `flow.repl(run).set(...)` into one run's. Before each run of code, the REPL is seeded with exactly the flow's tools, those of the stream's toolkits and that run's bindings: new names are added, removed names deleted, and unchanged ones left alone. Arguments are bound once, when the run starts. Nothing in the step path closes a REPL, so a finished run's worker stays warm for a message that reopens it. Closing is the caller's: `flow.runtime.close_repl(run)`, `close_repls()`, or `flow.close()`.

## Persistence

`g.save(path)` writes one JSON document (format 8; format 7 still loads): nodes, each with its `owner`, `seq` and `producer`, a task node also with its `run` record (name, task, agent name, argument descriptions, `repl`), and edges, plus content-addressed blobs for large inputs. Loading validates that every node's owner is a run in the graph, that no node comes before its run, and that every edge endpoint exists. A node saved mid-run loads ready, and runs again with what it launched still in place. A run records only the name of the agent it runs with; the `Agent` itself is never saved.

## Invariants

1. Records hold no live execution objects.
1. Every node belongs to exactly one run (its `owner`).
1. Engine edges (`spawn`, `return`, `write`) are written by the Flow, its tools and edits only.
1. A run with a step in flight is never given another; edits through `flow.edit` cancel it first.
1. A crashed step is its run's outcome: a terminal node with the error.
1. Run names are unique among siblings and filesystem-safe.
1. Save and load are a cold boundary: REPLs are rebuilt by replay.
