# Scope: what rlmflow is, and what it leaves to you

rlmflow runs **swarms of coding agents on one live graph**. This document
says what the library provides, what it deliberately does not, and the rule
that decides between the two. When a feature request comes in, check it
against [the rule](#the-rule) before adding anything.

## Three things

The library is three things. Everything else is built from them.

| Thing   | What it is                                                                                                                              | Where                                                           |
| ------- | --------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| `Graph` | The state: every agent, turn, message and relationship, as an append-only log of typed nodes and edges.                                 | `rlmflow/graph/`                                                |
| `Flow`  | The executor: it finds the agents that are ready, steps each one, and appends what the step produced. It holds no run state of its own. | `rlmflow/flow.py`, `rlmflow/engine/`                            |
| Agents  | Coding agents: each one thinks and acts in its own Python REPL, finishes with `submit(value)`, and coordinates by calling functions.    | `rlmflow/engine/steps.py`, `rlmflow/tools/`, `rlmflow/runtime/` |

### Graph

- **Nodes.** `AgentStart` opens an agent. Each turn appends nodes such as `LLMOutput`, `ExecAction`, `ExecOutput` and `DoneOutput`. A `UserQuery` is a message.
- **Structural edges**, which are never removed:
  - `next`: history order. A branch in history is a fork.
  - `from`: who sent a message. On an `AgentStart`, it is the action that launched the agent.
  - `depends_on`: a consumer waits for its producer, and `data["as"]` binds the result to a name.
- **Application edges** can have any kind, are added with `connect_agent` or `Edge(...)`, and can be removed. One kind has engine meaning: `reports_to`. A finished agent's result is delivered to its launcher *and* to every agent it reports to. Other kinds, such as `peer` or `reviews`, mean whatever your prompts and code say they mean.
- **Editing.** `Graph`'s history writes (`spawn`, `append`, `send`, `deliver`, `fork`, `attach`) edit a graph with no `Flow` around, and its queries (`ready`, `frontier`, `result`, `children`, `producers` and so on) are pure reads.
- **Persistence.** `Graph.save` and `Graph.load` round-trip the whole run. A saved graph can be resumed by any `Flow`.

### Flow

- `run_streaming(g)` (async) or `run(g)` drives every ready agent and yields each node and edge as it lands.
- **When it stops.** It returns when nothing is ready: every agent is finished, blocked, or waiting on a dependency.
- **Configuration** is about *how* agents act, not *which* agents exist:
  - models (`llm`, `llm_clients`);
  - tools (`tools=[...]`, `@tool`);
  - the sandbox (`runtime=`);
  - external agents (`backends=`);
  - prompts (`system_prompt`, `prompt_profiles`);
  - step rules (`@flow.transition`);
  - concurrency (`workers=`).
- **Per-agent policy** lives on `AgentConfig`: model, backend, limits, schema, and `child_config` for what that agent launches.

### Agents

Every native agent is a coding agent. Code is its action format, and coordination is functions in its REPL:

| Call                                                        | What it writes to the graph                                                           |
| ----------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| `launch_agent(task=..., **args)`                            | A new agent, with a `from` edge to this action. The call returns an awaitable handle. |
| `fork_agent(handle, message=...)`                           | A new agent that shares the forked agent's history up to the fork point.              |
| `send_agent(target, message)`                               | A `UserQuery` in the target's inbox, which reopens it if it had finished.             |
| `ask_agent(target, message)`                                | The same as `send_agent`, then a wait for the target's next result.                   |
| `connect_agent(target, kind)` / `disconnect_agent(edge_id)` | Adds or removes an application edge.                                                  |
| `submit(value)`                                             | `DoneOutput`, after the schema and verifier checks.                                   |
| `AGENTS`                                                    | Read-only: every agent's status and result.                                           |

An agent can also finish by replying in plain text with no `repl` block; the text is checked exactly like `submit(text)`. A malformed reply becomes an action that raises `MissingReplError`, so it recovers through the same loop as failing code.

Agents that are not native (for example Claude Code) plug in as a backend chosen by `AgentConfig.backend`. To the graph they are the same kind of agent.

## Shapes are just graphs

There is no topology API. RLM, DeLM and multi-agent teams are different
graphs built from the same calls:

| Shape             | How it is built                                                                                     | Example                                         |
| ----------------- | --------------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| RLM tree          | A top-level agent calls `launch_agent` and awaits the handles; children can launch again.           | `examples/needle/haystack.py`                   |
| Task board (DeLM) | The host creates several top-level agents; each reads `AGENTS` and may launch follow-up tasks.      | `examples/collaboration/shared_board.py`        |
| Peer team         | Top-level agents call `ask_agent` and `send_agent` and draw edges with `connect_agent`.             | `examples/collaboration/decentralized_peers.py` |
| Pipeline          | The host creates agents with `depends_on` and `inputs=`; each one starts when its producers finish. | `g.create_agent(..., depends_on=[spec])`        |
| Supervisor        | Workers have `reports_to` edges to a reviewer; the reviewer `send_agent`s fixes back.               | Hive (planned)                                  |
| Recovery          | A meta-agent reads another run's graph, forks it at earlier nodes and picks a winner.               | `examples/shepherd/`                            |

Mixing shapes needs nothing extra. A board task can launch an RLM subtree, and
a peer can report to a supervisor. Topology can come from the host, which
creates agents and edges up front, or from the agents themselves during the
run. It's usually both.

## The rule

A feature belongs in the library only if at least one of these is true:

1. **It needs engine internals.** Scheduling, turn boundaries, replay, or the step loop.
1. **It has to survive save, load and replay.** It must be a node, an edge, or a config field, or replaying the run would give a different result.
1. **Every agent needs it to coordinate.** It's a REPL function with graph semantics, such as `launch_agent` or `send_agent`.

Otherwise it's a pattern for user code, and at most an example. When in
doubt, leave it out until a real run needs it.

How the rule has played out so far:

| Request                            | Decision  | Why                                                                                                                                                   |
| ---------------------------------- | --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| Delegation settings on `Flow`      | Removed   | A child is just an agent; its policy is `AgentConfig.child_config` (rule 2).                                                                          |
| A shared task-board object         | Removed   | Its claims lived outside the graph and were lost on replay. Tasks are agents now (rule 2).                                                            |
| `reports_to` delivery              | In        | The handoff is engine work at a turn boundary (rule 1) and has to replay (rule 2).                                                                    |
| Backends picked per agent          | In        | The choice must be saved with each agent (rule 2).                                                                                                    |
| Plain-text answers                 | In        | A reply with no `repl` block finishes the agent through the same check as `submit`. It adds no node type, so the protocol RAO trains on is unchanged. |
| Chat or tool-calling actions       | Not now   | Code is a strictly more capable action format, one protocol keeps training simple, and external backends cover other agent types.                     |
| Standing agents, broadcast, topics | User code | A host loop plus `g.send` and `connect_agent` covers them; see below.                                                                                 |
| Workspace layout and isolation     | User code | That's runtime and tool configuration, which you already control.                                                                                     |

## Patterns that stay in user code

**A long-lived team.** `run_streaming` returns when the graph goes quiet. A
finished agent reopens when it gets a message, and `RESIDENT` agents keep
their REPL between runs:

```python
while True:
    event = await next_event()                 # CI result, webhook, human, timer
    g.send(reviewer, event.text)
    async for item in flow.run_streaming(g):
        checkpointer.handle(item, g)
```

**Broadcast and topics.**

```python
for peer in [a for a in g.nodes(AgentStart) if a.config.name.startswith("worker")]:
    g.send(peer, "API changed: see SPEC.md v2")
```

A topic is an application edge kind, for example `connect_agent(channel, "subscribes")`, plus a filter over `g.edges(kind="subscribes", target=channel.id)`.

**Workspaces.** Give each agent its own files or directory through its task
and arguments. Use a merge agent that `depends_on` the workers when outputs
have to come together. Use a separate `Runtime` working directory when agents
must not see each other's files.

**Humans.** A person is a message: `g.send(agent, text)` between runs, or
a controller node appended at a stream boundary (`docs/injections.md`).

## Known limits

These are real and deliberately left alone for now. Each one says what would make it worth fixing.

- **Messages from the host during a run.** `g.send` from host code while `run_streaming` is active is delivered at the next scheduling pass. If every agent is blocked, that pass waits until `flow.queue.wake()` is called, and `g.send` is not thread-safe. *Fix when* a real integration needs to push events mid-run: add a thread-safe `flow.send(g, agent, message)` that writes and wakes.
- **Checkpoint cost.** `GraphCheckpointer` rewrites all of `graph.json` on each flush (every finished agent, every 50 nodes, or every 2 seconds), on the event loop. Total I/O therefore grows with the square of the run size. That's fine at tens of agents but stalls at hundreds to thousands. *Fix when* a run gets that big: flush in a thread first, then switch to append-only JSONL. Mutated nodes (tool receipts, timing, metadata) would be re-appended, and the last record per id would win.
- **RAO credit follows only the launch tree.** `score_tree` credits an agent from its launched children only. Peers, `reports_to` recipients and `depends_on` producers get no team credit. *Fix when* training a non-tree swarm: credit along edges where one agent's output feeds another.
- **The viewer draws only the launch tree.** Application and `depends_on` edges are not drawn, so peers look disconnected. *Fix*: planned, as overlay arcs in `view/figure.py`.
- **One scheduler process.** Sandboxes can be remote (Docker, Modal), but scheduling is a single asyncio process. This is a non-goal.

## Why coding agents

- **Code is the better action format.** One action can call several tools, loop, branch, keep intermediate values in variables, and launch or wait on other agents. JSON tool calls do one call per turn.
- **Coordination is just more functions.** Every agent can join any shape (tree, board or peers) with no extra protocol.
- **One protocol is easier to train.** Every trajectory has the same node vocabulary, which is what RAO learns from.
- **Big inputs stay out of context.** An agent can hold a million-line value in a variable and inspect it with code instead of reading it into its prompt.

## Non-goals

- A topology DSL or a hand-authored state machine; for that, use LangGraph.
- Prebuilt role teams or a catalogue of integrations.
- JSON tool-calling as a native action protocol. (Plain-text final answers are supported; acting is always code.)
- Distributed scheduling.
