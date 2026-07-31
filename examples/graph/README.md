# Node features

Tiny self-contained scripts that show what a `rlmflow.Node` can do. No
LLM keys needed — every example builds its own graph by hand, so they finish in
milliseconds and can be read top-to-bottom.

| script | what it shows |
|---|---|
| `01_query.py` | `walk()`, `agent_ids()`, node types, errors, token usage, and final result |
| `02_navigate.py` | child lookup, traversal, and `render_tree()` |
| `03_mutate.py` | graph edits with `Node.attach()` and `surgery` |
| `04_save_load.py` | `persistence.save()` / `persistence.load()` JSON round-trip |
| `05_timeline.py` | explicit snapshots as graph state changes |
| `06_fork.py` | `surgery.fork()` for divergent branches |
| `07_render.py` | `render_tree()` and the lightweight minimal viewer adapter |

Run any of them directly:

```bash
python examples/graph/01_query.py
python examples/graph/05_timeline.py
python examples/graph/06_fork.py
```

Most scripts just print to stdout. `07_render.py` also saves a minimal run
directory.
