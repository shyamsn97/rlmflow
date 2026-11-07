# Observability

Everything needed to inspect a run lives in its graph, live or saved.

## Query the graph

```python
g.result(root)  # the root's answer, or None
g.status(root)  # "running" | "idle" | "done"
g.history(root)  # its nodes, oldest first (a fork's base history included)
g.history(root)[-1]  # its latest node
g.children(root)  # the agents it launched
g.subtree(root)  # it and every agent below it
g.usage(), g.usage().cost_usd  # tokens and cost, for the run or g.usage(g.subtree(a))

g.agents(), g.nodes("code"), g.edges(kind="return")
[n for n in g.nodes("code") if n.data.get("error")]  # every failed cell
```

To ask about a worker, hold the worker:

```python
worker = g.children(root)[0]
g.path(worker)  # "root.worker-0"
g.history(worker)
g.result(worker)
g.usage(g.subtree(worker))
```

Each node carries `type`, `id`, `agent_id`, `data`, `metadata` and `seq`. Model
usage is on `llm` nodes' `data["usage"]`. Model turns and cells also carry
host-measured timing, as epoch seconds:

```python
node.started_at, node.finished_at  # None for nodes no step timed
```

A node written from inside a step rather than by one, such as a message, has no
timing. Timing is measured on the Flow host, so local, subprocess, Docker and
Modal runs share one clock. Saved graphs store it as ISO timestamps plus a
`duration_ms`.

Edges answer where things came from: `g.edges(kind="spawn", target=agent.id)` is
the cell that launched an agent, and `g.edges(kind="return", target=cell.id)`
the results that reached a cell. See [The graph](node_model.md).

## Persistence

```python
g.save("runs/shepherd")
loaded = Graph.load("runs/shepherd")
```

The directory holds `graph.json` (agents, nodes, edges, edits; see
[The graph](node_model.md#persistence)) and `blobs/` for large arguments. The
file is written to a temporary name, flushed and atomically replaced, so an
interrupted save leaves the previous complete graph. Loading restores the graph
but no REPL; see [Control](control.md#save-and-load).

## Live consumers

`run_streaming` hands you each landed node as a `Step`, which is all a consumer
needs. `rlmflow.consumers` is the small protocol for fanning that out:

```python
from rlmflow.consumers import ConsumerGroup, GraphCheckpointer, LiveTreeRenderer, StreamConsumer


class Progress(StreamConsumer):
    def handle(self, step, graph):
        print(graph.path(step.agent), step.node.type, step.node.finished_at)

    def close(self):
        print("done")


consumers = ConsumerGroup([LiveTreeRenderer(), Progress(), GraphCheckpointer("runs/shepherd")])
try:
    async for step in flow.run_streaming(g):
        consumers.handle(step, g)
finally:
    consumers.close()
```

`handle(step, graph)` is the whole contract; `close()` is optional.
`ConsumerGroup` fans a step out to each consumer in turn and closes them in
reverse, suppressing errors from a close so one failing consumer cannot strand
the others.

The shipped consumers are:

- `LiveTreeRenderer`: clears and redraws a compact agent tree;
- `LiveGraphTree`: a Rich live forest with status colors and spinners (falls back
  to the plain renderer without Rich or a terminal);
- `GraphCheckpointer`: saves after two seconds or 50 nodes by default, and on
  every `done` and on close; configure `interval_s` and `interval_nodes`, or call
  `flush()`;
- `WorkspaceSync`: mirrors a sandbox working directory into a local copy.

For events that aren't landed nodes (new agents, edges, edits, deliveries),
subscribe to the graph: `g.subscribe(lambda event, item: ...)`.

```text
Find the needle across 500 files.
└── root: children running 2 (1 turns)
    ├── root.batch-0: running code (1 turns)
    └── root.batch-1: planning (1 turns)
```

## Stepping through a run

Every agent and node is stamped with a `seq` when it is added, so the order a
run happened in is those stamps sorted, across agents. That is what `timeline`
and `steps` hand back:

```python
from rlmflow.view import steps, timeline

for item in timeline(loaded):  # every agent and node, in the order they appeared
    print(item.type)

for step in steps(loaded):  # the same order, with content and offsets
    print(f"{step.index:>3} +{step.elapsed:6.2f}s {step.title:<32} {step.summary}")
```

The same run can be drawn. `graph_svg` is one marker per agent and node,
coloured and shaped by type, with `return` and `message` edges dashed, and
`render_html` is a single self-contained file that walks the run: the graph with
the step you are on ringed, everything after it faded back, and its content
beside it.

```python
from rlmflow.view import save_html, save_svg

save_svg(loaded, "run.svg")
save_html(loaded, "run.html")  # steppable, no server, no network
```

From the shell:

```bash
RUN=examples/_runs/shepherd/shepherd
rlmflow view show $RUN               # the agent tree, then the timeline
rlmflow view show $RUN --step 12     # one step, with its content
rlmflow render svg $RUN run.svg
rlmflow render html $RUN run.html
```

In the stepper, the arrow keys move a step, `Home` and `End` jump to the ends,
and the address bar carries the step you are on, so `run.html#12` opens on step 12.

## Replaying a run

`replay` hands back the graph as it stood after each step, so anything that
reads a graph works on a replay:

```python
from rlmflow import render_tree
from rlmflow.view import render_steps, replay

for snap in replay("examples/_runs/shepherd/shepherd"):
    print(render_tree(snap))

for frame in render_steps("examples/_runs/shepherd/shepherd"):
    print(frame)
```

Each snapshot is rebuilt from the saved records, so they can be collected or
rendered out of order. Edits are applied in place, so a replay of an edited run
shows the edited graph cut at each step, not what it held before the edit.

## The browser viewer

`open_viewer` puts a step slider over the figure, an agent picker beside it, and
that agent's history underneath, all as of the step you are on:

```python
from rlmflow import open_viewer

open_viewer("examples/_runs/shepherd/shepherd")  # needs: pip install rlmflow[viewer]
```

## Frames and GIFs

For a strip of images or an animation, `rlmflow[image]` adds a rasteriser:

```python
from rlmflow.view import save_frames, save_gif

save_frames(g, "out/", every=5)  # a PNG per fifth step
save_gif(g, "run.gif", every=5, ms_per_frame=120)
```

`rlmflow render frames $RUN out/ --every 5` and
`rlmflow render gif $RUN run.gif --every 5` do the same from the shell.
