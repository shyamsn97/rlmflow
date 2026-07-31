# Observability

Everything needed to inspect a run lives in its Node tree.

## Query the tree

```python
from rlmflow import ErrorOutput, render_tree

print(render_tree(root))

root.agent_ids()
root.child_agents("root")
root.tail("root.worker")
root.transcript("root.worker")
root.latest_query("root.worker")
root.agent_result("root.worker")
root.tokens("root.worker")
root.finished()

errors = [node for node in root.walk() if isinstance(node, ErrorOutput)]
```

Each Node carries:

```python
node.type
node.id
node.agent_id
node.trajectory_id
node.parent
node.children
node.metadata
```

`step_child` is the same-agent continuation. `spawn_children` are child-agent
roots. Mutation provenance is under `metadata["mutation"]`; LLM usage and the
actual system prompt are stored on `LLMOutput.metadata`.

Nodes produced by `Flow.step()` also carry host-measured transition timing:

```python
node.metadata["timing"]
# {"started_at": "...+00:00", "finished_at": "...+00:00", "duration_ms": 123.4}
```

Overlapping child intervals demonstrate wall-clock concurrency. Timing is
measured on the Flow host, so local, subprocess, Docker, and Modal runs share one
clock.

See [`node_model.md`](node_model.md) for the concrete Node classes.

## Persistence

```python
from rlmflow import persistence

run_dir = persistence.save(root, "runs/deep_research")
loaded = persistence.load(run_dir)
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

`graph.json` is the complete recursive tree. Per-agent files are stable
projections for tools and external readers. Loading restores Nodes and parent
links but no REPL or suspended task.

## Live consumers

Consumers receive published Nodes directly:

```python
from rlmflow import ConsumerGroup, GraphCheckpointer, LiveGraphTree

consumers = ConsumerGroup([
    GraphCheckpointer("runs/deep_research"),
    LiveGraphTree(title="run"),
])

try:
    async for node in flow.run_streaming(root):
        consumers.handle(node)
finally:
    consumers.close()
```

Every consumer can recover the current root with `node.root()`.

`LiveTreeRenderer` is a simple redraw-on-each-Node renderer. `FlowTUI` provides
the full-screen dashboard.

## Saved-run viewers

```python
from rlmflow import open_viewer, render_steps, replay, save_image, save_steps

open_viewer("runs/deep_research")

for snapshot in replay("runs/deep_research"):
    print(render_tree(snapshot))

for frame in render_steps("runs/deep_research"):
    print(frame)

save_image("runs/deep_research", "final.png")
save_steps("runs/deep_research", "frames/")
```

Viewer functions accept a saved run directory or a root Node. Browser viewing
uses the optional viewer dependencies; static image export uses the image
dependencies.
