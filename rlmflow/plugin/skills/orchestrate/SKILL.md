---
name: orchestrate
description: Ways to split work across agents in an rlmflow run — subagents, fan-out over chunks, recursive trees, pipelines, review loops, peers, and a DeLM-style task board — with the code for each and when each pays off, inside a run or from host code. Use when you are an rlmflow agent with launch_agent, when deciding whether to launch agents at all, or when writing code that builds an rlmflow graph.
---

# Orchestrating agents in rlmflow

Every agent you launch is an agent in the run's graph, with its own REPL,
history and result, joined to you by an edge. The patterns below are shapes of
that graph. Pick the smallest one that fits; most tasks need no agents at all.

## The tools

| Call | Does |
| --- | --- |
| `a = launch_agent(task, config=None, name=None, **args)` | Start an agent and get it back as an `Agent`. `args` become its REPL variables; pass values, not agents. |
| `await a` | The result of an agent you launched (or one they launched). |
| `await asyncio.gather(*agents)` | Several results, in order. |
| `a.send(message)` | Message any agent. Reopens it if it finished. For an agent you launched, `await a.send(...)` waits for its reply. |
| `a.status`, `a.result()` | `"done"`, `"idle"` (a session waiting for messages) or `"running"`; the result once done. |
| `a.parent`, `a.children` | Where it sits in the run. |
| `await llm_query(prompt, model=None, schema=None)` | One model call with no REPL or tools. Much cheaper than an agent. |
| `AGENTS.print_graph(show_results=True)` | The run's agents, their status and results. |
| `AGENTS["name"]`, `AGENTS.self`, `AGENTS.children` | Any agent by name, path or id; you; the ones you started. Each is an `Agent`. |
| `Input(value, description="...")` | Wrap an arg to tell the agent what it is. |
| `submit(value)` | Your final result. Once. |

**Waiting.** In a Python REPL turn, `await` blocks until the result is in. In a
Claude Code or pi session, launch in one cell, then end your turn without
submitting; you're resumed with the results of the agents you launched. Don't
poll in either.

**Who you can await.** Only agents you launched, and the agents they launched.
Anyone else (a peer, your parent) answers with a message at your next turn: send,
then finish the block or end the turn.

**Snapshots.** `AGENTS` and an `Agent`'s status, result and relatives are as of
when your cell started, so an agent you launch shows up in `AGENTS` from your
next cell. `await` is live.

**Depth.** An agent can launch others only if its config's `max_depth` allows
it. The configs and their models are listed in your prompt; use a cheaper one
for mechanical work.

## Decide first

Launch agents when the work splits into pieces that are **independent**, **too
big for one context**, or **need a fresh view**. Otherwise do it yourself.

- One transform per item (classify, extract, summarize)? Use
  `await asyncio.gather(*[llm_query(p) for p in prompts])`, not agents.
- A few files or a few thousand lines? Read them yourself.
- More agents is not better. Past 3 or 4 working on one codebase, coordination
  costs more than it saves.

## Patterns

### Subagents: delegate distinct pieces

Different pieces of work, each owned end to end by one agent.

```python
api = launch_agent("Design the public API for the cache. Submit it as a Python stub.", name="api")
tests = launch_agent(
    "Write pytest cases for an LRU cache with TTL. Submit the test file.", name="tests"
)
api_stub, test_file = await asyncio.gather(api, tests)
```

Give each one a deliverable and a format. "Look into X" comes back as an essay.

### Fan-out: the same job over chunks

One input too big for a context: split it, run one agent per chunk, combine.

```python
chunks = [doc[i : i + 200_000] for i in range(0, len(doc), 200_000)]
hits = await asyncio.gather(
    *[
        launch_agent("List every passage in `chunk` about the merger, with its offset.", chunk=c)
        for c in chunks
    ]
)
answer = await llm_query(f"Using only these passages, answer: {question}\n\n{hits}")
```

Pass the data as an arg (it lands in the agent's REPL), not in the task text.
If one call per chunk is enough, use `llm_query` per chunk instead.

### Recursive tree: agents that split further

Workers that can launch their own workers, for work whose size you can't see up
front (a big directory tree, a survey). Needs a config with `max_depth` of 2 or
more.

```python
parts = [
    launch_agent(f"Audit `{d}`. If it's too big for you, split it across agents.", config="deep")
    for d in top_level_dirs
]
reports = await asyncio.gather(*parts)
```

### Pipeline: one agent's result feeds the next

Await each stage and pass its result on as an arg.

```python
spec = await launch_agent("Write a spec for a slugify() function: inputs, outputs, edge cases.")
code = await launch_agent(
    "Implement `spec` in Python and test it in your REPL. Submit the source.", spec=spec
)
defects = await launch_agent(
    "Check `code` against `spec`. Submit a list of defects.", spec=spec, code=code
)
```

### Review loop: send a finished agent back to work

`.send` reopens a finished agent with its REPL and history intact, so it
revises instead of starting over.

```python
writer = launch_agent("Write the migration script.", name="writer")
draft = await writer
for _ in range(3):
    issues = await llm_query(f"List concrete bugs in this script, or reply NONE:\n{draft}")
    if issues.strip() == "NONE":
        break
    draft = await writer.send(f"Fix these issues and resubmit:\n{issues}")
```

Cap the rounds. Use a separate reviewer agent when the review itself needs tools
(running tests, reading files).

### Peers: agents that talk to each other

Agents split shared work and negotiate over `.send` instead of reporting
only to you. Name them and tell each who the others are.

```python
team = 'Your teammates are frontend and backend; message them with AGENTS["backend"].send(...).'
fe = launch_agent(f"Build the UI in web/. {team}", name="frontend")
be = launch_agent(f"Build the API in server/. {team}", name="backend")
await asyncio.gather(fe, be)
```

Inside a peer, `AGENTS["backend"].send("I need GET /items to return ids.")`
queues the message; the reply arrives as a message at the peer's next turn.
Peers can't await each other. Peers rarely message unprompted: say in the task
when they should.

### Task board (DeLM-style): workers that claim and post tasks

No manager. A few workers share a board: tasks, claims and results. A worker
claims a task atomically, so no task runs twice, and posts follow-up tasks as it
finds them. Fits open-ended work where the next tasks are only found by doing
the first ones. The board is a host tool you give the Flow (see host code
below); inside a worker it looks like this:

```python
WORKER = """You are one worker on a shared task board; others run in parallel.
Repeat until board_open() is empty:
1. Pick a task id from board_open() and claim it with board_claim(id). False means
   another worker has it: pick again.
2. Read board_results() for what helps, do the task, and board_finish(id, result)
   with a short, reusable result.
3. Post follow-up work no task covers: board_post(new_id, task).
Then submit the ids you finished."""

await asyncio.gather(*[launch_agent(WORKER, config="worker", name=f"w{i}") for i in range(3)])
```

## The same shapes from host code

Outside a run, in Python that drives one, the graph calls build the same
shapes. A `Flow` steps whatever the graph says is running, so you can edit the
graph and call `flow.run(g)` again.

```python
g = Graph()
fe = g.add_agent("Build the UI. Your teammate is backend.", name="frontend")  # peers
be = g.add_agent("Build the API. Your teammate is frontend.", name="backend")
g.add_edge("peer", fe, be)  # your own edge kinds; the engine ignores them
flow.run(g)

g.send(fe, "Use the v2 endpoint instead.")  # steer an agent
cell = next(n for n in g.history(be) if n.type == "code")
g.send(g.fork(cell, name="retry"), "Try another approach.")  # branch from a cell
flow.run(g)

g.edit(g.history(be)[-1], result="...")  # change a result; whatever used it re-runs
g.replace(cell, 0, "cached answer")  # bind the cell's first saved call; the cell re-runs
g.remove(fe, dry_run=True)  # what removing an agent would take with it
flow.run(g)
```

The task board is a few host functions over one dict and a lock. Plain
functions passed as `tools` run in the host process, so every worker sees the
same board:

```python
import threading
from rlmflow import Flow

tasks, claims, results, lock = {}, set(), {}, threading.Lock()


def board_post(id: str, task: str) -> bool:
    """Post a task unless the id is taken."""
    with lock:
        return tasks.setdefault(id, task) == task


def board_claim(id: str) -> bool:
    """Claim an open task; False if another worker has it."""
    with lock:
        if id in claims or id not in tasks:
            return False
        claims.add(id)
        return True


def board_open() -> dict:
    """Tasks nobody has claimed."""
    with lock:
        return {k: v for k, v in tasks.items() if k not in claims}


def board_finish(id: str, result: str) -> None:
    """Record a claimed task's result."""
    with lock:
        results[id] = result


def board_results() -> dict:
    """Every finished task's result."""
    with lock:
        return dict(results)


board = [board_post, board_claim, board_open, board_finish, board_results]
flow = Flow(workers=3, tools=board)
```

## Rules that save runs

- **Check what comes back.** An empty or `None` result, or an error message, is
  a failure: retry it, or report it. Never turn it into a plausible answer.
- **Keep big values in variables.** Print sizes and samples, not whole results.
- **Write down the plan.** Keep a `notes` variable with what's launched, what's
  back and what's ruled out.
- **Budget.** Each agent costs about as much as you do. Launch the fewest that
  cover the work, and stop when the answer is supported.
