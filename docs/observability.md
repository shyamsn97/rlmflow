# Observability

Everything needed to inspect a run lives in its Node tree.

## Query the tree

```python
from rlmflow import AgentStart, ErrorOutput

root.result()                 # the root agent's answer, or None
root.terminal                 # has it answered?
root.transcript()             # this agent's own nodes, start to frontier
root.frontier                 # where it is now
root.sub_agents               # the agents it launched
root.leaves()                 # the frontier of every agent in the tree
root.llm_turns()              # how many model turns this agent has taken
root.tokens()                 # usage summed over the whole subtree

list(root.walk())             # every node, preorder
[a for a in root.walk() if isinstance(a, AgentStart)]   # every agent
errors = [n for n in root.walk() if isinstance(n, ErrorOutput)]
```

Those accessors belong to an agent rather than to the run, and every
`AgentStart` is an agent — so to ask about a worker, hold the worker:

```python
worker = root.sub_agents[0]
worker.config.path            # "root.worker"
worker.transcript()
worker.result()
worker.tokens()
```

Each Node carries:

```python
node.type
node.id
node.content
node.seq                      # position in its own agent's transcript
node.parent / node.children
node.parent_agent             # the AgentStart it belongs to
node.root                     # the top of the run
node.next / node.prev         # the same agent's neighbours
```

`node.next` is the same-agent continuation; the children of an `ExecAction` that
delegated are the agents it launched. LLM usage is on `LLMOutput.usage`, and the
system prompt a turn ran under is `agent.system_prompt_for(node)` — the text is
stored once per distinct prompt in `agent.system_prompts` and each turn keeps
only its id.

Nodes produced by a step also carry host-measured timing:

```python
node.timing()
# {"started_at": "...+00:00", "finished_at": "...+00:00", "duration_ms": 123.4}
```

A node written from inside a step rather than by one — a nudge, a child's
`AgentStart` — has no timing and returns `{}`. Overlapping child intervals
demonstrate wall-clock concurrency. Timing is measured on the Flow host, so
local, subprocess, Docker, and Modal runs share one clock.

See [`node_model.md`](node_model.md) for the concrete Node classes.

## Persistence

```python
from rlmflow import AgentStart

run_dir = root.save("runs/deep_research")
loaded = AgentStart.load(run_dir)
```

The directory contains:

```text
run/
  graph.json
  latest.json
  agents/
    root/
      agent.json
      session.jsonl
      latest.json
      worker/
        ...
```

`graph.json` is the complete recursive tree, and is what `load` reads. The
per-agent files are stable projections for tools and external readers:
`agent.json` is that agent's identity, query, inputs, and system prompts,
`session.jsonl` is its own transcript one node per line, and `latest.json` is
its frontier. `latest.json` at the top is the run summary — agent ids, node
count, whether it finished, and its result.

Loading restores nodes and parent links but no REPL; see the restore modes in
[`control.md`](control.md).

## Live consumers

`run_streaming` hands you each node as it lands, which is all a consumer needs.
`rlmflow.consumers` is the small protocol for fanning that out:

```python
from rlmflow.consumers import (
    ConsumerGroup,
    FlowTUI,
    GraphCheckpointer,
    LiveGraphTree,
    LiveTreeRenderer,
    StreamConsumer,
)


class Progress(StreamConsumer):
    def handle(self, node):
        print(node.parent_agent.config.path, node.type, node.timing().get("duration_ms"))

    def close(self):
        print("done")


consumers = ConsumerGroup([
    LiveTreeRenderer(),
    Progress(),
    GraphCheckpointer("runs/deep_research"),
])

try:
    async for node in flow.run_streaming(root):
        consumers.handle(node)
finally:
    consumers.close()
```

`handle(node)` is the whole contract; `close()` is optional and defaults to
doing nothing. `ConsumerGroup` fans a node out to each consumer in turn and
closes them in reverse, suppressing errors from a close so one failing consumer
cannot strand the others.

Every consumer can recover the whole run from `node.root`.

The shipped consumers are:

- `LiveTreeRenderer`: clears and redraws a compact recursive agent tree;
- `LiveGraphTree`: Rich live forest with status colors and active spinners;
- `FlowTUI`: Textual chat/dashboard with overview, tree, agents, counts,
  waiting, errors, and latest-activity panels;
- `GraphCheckpointer`: periodically saves `node.root`, then flushes on close;
- `WorkspaceSync`: mirrors a sandbox working directory into a local copy.

## Watching a run go by

`LiveTreeRenderer` displays agent status and child progress:

```python
viewer = LiveTreeRenderer()
try:
    async for node in flow.run_streaming(root):
        viewer.handle(node)
finally:
    viewer.close()
```

For a Rich live display:

```python
viewer = LiveGraphTree(title="research run")
try:
    async for node in flow.run_streaming(root):
        viewer.handle(node)
finally:
    viewer.close()
```

Install `rlmflow[viewer]` for Rich output. `LiveGraphTree` falls back to the
plain renderer when Rich is unavailable or stdout is not a terminal.

`FlowTUI` is a `StreamConsumer` plus an interactive Textual shell. Install
`rlmflow[tui]`, then pass it a drive callback that streams Nodes through
`ui.handle(node)`:

```python
from rlmflow import UserQuery
from rlmflow.consumers import FlowTUI

ui = FlowTUI()

async def drive(root, *, query=None, inputs=None, until="done"):
    root = root or flow.start(query, inputs=inputs)
    if query is not None and root.content != query:
        root.frontier.append(UserQuery(content=query))
    async for node in flow.run_streaming(root, until=until):
        ui.handle(node)
    return root

ui.run(drive)
```

```text
Find the needle across 500 files.
└── root: children running 2/2 (1 turns)
    ├── root.batch_0: running code (1 turns)
    └── root.batch_1: planning (1 turns)
```

...and once the children answer and the root composes them:

```text
root: audited 2 modules (2 turns)
  root.scanner_auth: found SQL injection in login.py (1 turns)
  root.scanner_db: no issues found (1 turns)
```

## Reading a saved run

A saved run reloads into the same tree, so every query above works on it
offline:

```python
loaded = AgentStart.load("runs/deep_research")

print(loaded.result(), loaded.tokens())
for node in loaded.walk():
    print(node.seq, node.type, node.timing().get("duration_ms", ""))
```

For anything else — a diff between two runs, an export — `persistence.to_dict(root)`
is the whole run as JSON-serializable data, and `persistence.summary(root)` is the
one-line version.

## Stepping through a run

Every node stamps itself when it is created, so the order a run happened in is
those stamps sorted — across agents, not just down one chain. That is what
`timeline` and `steps` hand back:

```python
from rlmflow.view import steps, timeline

for node in timeline(loaded):          # every node, in the order it appeared
    print(node.type)

for step in steps(loaded):             # the same order, with content and offsets
    print(f"{step.index:>3} +{step.elapsed:6.2f}s {step.title:<32} {step.summary}")
```

The same run can be drawn. `graph_svg` is one marker per node — coloured and
shaped by type, agents named — and `render_html` is a single self-contained file
that walks the run: the graph with the node you are on ringed, everything after it
faded back, and its content beside it.

```python
from rlmflow.view import save_html, save_svg

save_svg(loaded, "run.svg")            # the whole graph
save_svg(loaded, "step12.svg", step=12)  # as it stood at step 12
save_html(loaded, "run.html")          # steppable, no server, no network
```

Neither needs a plotting library, a browser, or a running flow. From the shell,
the same thing without writing a script:

```bash
rlmflow view runs/deep_research               # the agent tree, then the timeline
rlmflow view runs/deep_research --tree        # just the tree
rlmflow view runs/deep_research --step 12     # one step, with its content
rlmflow view runs/deep_research --svg run.svg --html run.html
```

In the stepper, the arrow keys move a step, `Home` and `End` jump to the ends, and
the address bar carries the step you are on, so `run.html#12` opens on step 12.

Runs saved before nodes were stamped still load; their nodes are stamped at load
time and fall back to tree order. A run written by the engine before the node-tree
rewrite does not load at all — `persistence.load` raises `ValueError` naming the
format.

## Replaying a run

`replay` hands back the tree as it stood at each step, so anything that reads a
tree works on a replay without knowing it is one:

```python
from rlmflow import render_tree
from rlmflow.view import render_steps, replay

for snap in replay("runs/deep_research"):
    print(render_tree(snap))        # the live terminal tree, after the fact

for frame in render_steps("runs/deep_research"):
    print(frame)                    # the same thing, already rendered
```

Each snapshot is rebuilt rather than mutated, so they can be collected, compared,
or rendered out of order, and the run they came from is left alone.

## The browser viewer

`open_viewer` puts a step slider over the figure, an agent picker beside it, and
that agent's transcript underneath — all as of the step you are on:

```python
from rlmflow import open_viewer

open_viewer("runs/deep_research")            # needs: pip install rlmflow[viewer]
open_viewer(root, server_port=7861)          # extra keywords go to launch()
```

## Frames and GIFs

For a strip of images or an animation, `rlmflow[image]` adds a rasteriser:

```python
from rlmflow.view import save_frames, save_gif

save_frames(root, "out/", every=5)           # a PNG per fifth step
save_gif(root, "run.gif", every=5, ms_per_frame=120)
```

`--frames`, `--gif`, and `--every` do the same from the shell.
