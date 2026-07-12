# Observability

Everything you need to debug a run lives in the `Graph`. `Flow.start(...)` seeds
it, `run`/`run_streaming` mutate it in place, and `Graph.save`/`Graph.load`
persist it.

## Data model

A `Graph` is a recursive structure: it represents **one agent**, and
`graph[other_aid]` returns the `Graph` rooted at any descendant agent. Per-agent
fields live on `Graph` itself; sub-agents live in `graph.children`; the
trajectory lives in `graph.nodes`.

```python
graph.agent_id           # str — this agent's id
graph.graph_id           # str — session id (changes on an isolated fork)
graph.depth              # int — recursion depth
graph.query              # str — original task
graph.inputs             # dict[str, str] — the agent's INPUTS
graph.model              # str — model label for this agent
graph.parent_agent_id    # str | None — id of the spawning agent
graph.output_schema      # required output schema, if any

graph.nodes              # list[Node] — this agent's trajectory (seq order)
graph.children           # dict[str, Graph] — direct sub-agents

# subtree views
graph.agents             # dict[agent_id, Graph] — every agent in the tree
graph.walk()             # iterator over every Graph (this agent, then descendants)
```

`Node` carries only what changes per turn — the payload (`content`, `code`,
`output`, `reply`, `result`, `error`, and token deltas in `metadata`) — tagged by
its `type`. Trajectories strictly alternate **observation** and **action** nodes;
every action is followed by exactly one observation. Eight concrete node classes
live under three base classes. See [`node_model.md`](node_model.md) for the full
flow and wait/resume semantics:

| `type`               | Class               | Base              | Carries                                                                     |
|----------------------|---------------------|-------------------|-----------------------------------------------------------------------------|
| `user_query`         | `UserQuery`         | `ObservationNode` | initial task content (root user query / spawn prompt for a child)           |
| `llm_output`         | `LLMOutput`         | `ObservationNode` | model reply, extracted REPL code, token deltas in `metadata["usage"]`       |
| `exec_action`        | `ExecAction`        | `ActionNode`      | "ran fresh code" — optional code echo                                       |
| `exec_output`        | `ExecOutput`        | `ObservationNode` | runtime stdout/stderr                                                       |
| `supervising_output` | `SupervisingOutput` | `ObservationNode` | code suspended at an awaited launcher; `waiting_on` lists pending children  |
| `resume_action`      | `ResumeAction`      | `ActionNode`      | "supervisor resumed paused code" — produces the next observation            |
| `error_output`       | `ErrorOutput`       | `ObservationNode` | failure observation                                                         |
| `done_output`        | `DoneOutput`        | `ObservationNode` | terminal answer from `done(...)`                                            |

## Querying the graph

```python
from rflow import render_tree

render_tree(graph)                             # ASCII tree render
graph.current()                                # latest node on the root agent
graph.result()                                 # terminal answer (DoneOutput.result)
graph.finished                                 # root node terminal and all descendants finished
graph.tokens()                                 # (in, out) — recursive by default
graph.tokens(recursive=False)                  # (in, out) — just this agent
graph.total_tokens()                           # in + out

graph["root.scanner_api"]                      # sub-Graph rooted at that agent
graph.agents["root.scanner_api"]               # same, but explicit
graph.children                                 # dict[str, Graph] of direct children
graph.agents["root.scanner_api"].nodes         # ordered list[Node] for one agent
graph.agents["root.scanner_api"].result()      # that agent's latest DoneOutput payload
```

There is no separate node index — iterate the tree with `walk()` and filter on
`node.type` (or `isinstance`):

```python
errors = [n for a in graph.walk() for n in a.nodes if n.type == "error_output"]
dones  = [n for a in graph.walk() for n in a.nodes if n.type == "done_output"]
```

## Run persistence

`Graph.save(path)` writes a self-contained run directory. The manifest is
`graph.json`; per-agent logs live under `agents/`; ordinary files produced by
agent tools live beside the saved graph when your runtime working directory is
the same directory.

```text
run/
  graph.json
  agents/
    root/
      agent.json
      session.jsonl
      latest.json
      child_a/
        agent.json
        session.jsonl
        latest.json
```

`Graph.load(path)` rehydrates the same recursive `Graph` shape the engine emits:

```python
run_dir = graph.save("runs/deep_research")
latest = rflow.Graph.load(run_dir)
```

For live checkpointing, save inside the stream loop:

```python
graph = agent.start(query)

async def drive():
    async for _event in agent.run_streaming(graph):
        graph.save("runs/deep_research")

asyncio.run(drive())
```

## Live terminal tree

`render_tree(graph)` renders the whole agent tree from any snapshot. Reprint it
each tick, or hand events to `LiveTreeRenderer`:

```python
from rflow import render_tree

async def drive():
    async for _event in agent.run_streaming(graph):
        print("\033[2J\033[H" + render_tree(graph))

asyncio.run(drive())
```

## Gradio viewer

`open_viewer(source)` launches a browser app with a step slider, an agent
selector, and a per-agent transcript over a swimlane of the run. `source` can be
a saved run directory, a `Graph`, or a list of snapshots:

```python
from rflow import open_viewer

open_viewer("runs/deep_research")   # saved run directory
open_viewer(graph)                  # single snapshot
```

Requires `pip install rlmflow[viewer]`.

## Static image / step exports

For blog posts, PR comments, papers, or CI artifacts, render the graph to a PNG
(or SVG/PDF), or write one frame per execution step. `replay(graph)` gives the
per-step snapshots these build on.

```python
from rflow import save_image, save_steps, replay

save_image("runs/deep_research", "final.png")
save_steps("runs/deep_research", "frames/",     # step_000.png, step_001.png, ...
           width=1600, height=1200, scale=2, marker_mult=3.5, text_mult=2.2)

snapshots = replay(graph)                        # list[Graph], one per step
```

Image export needs `kaleido`:

```
pip install rlmflow[image]
```
