# Graph manipulation

Tiny self-contained scripts that show what an `rlmflow.Graph` can do. No
LLM keys needed — every example builds its own graph by hand with `ops`, reads
it back with `reads`, so they finish in milliseconds and can be read top-to-bottom.

The graph is an append-only log. `ops.append` continues an agent, `ops.fork`
branches from an earlier node to revise history, and `ops.attach` gives agents
prepared elsewhere a launch action under a parent. There is intentionally no
in-place delete or reparent operation.

| script | what it shows |
|---|---|
| `01_query.py` | `g.nodes()`, node types, errors, token usage, and the final result |
| `02_navigate.py` | each agent's history, its children, and an indented agent tree |
| `03_edit_frontier.py` | message a finished agent, reject stale edits, and revise with `ops.fork` |
| `04_save_load.py` | `g.save()` / `Graph.load()` run-directory round-trip |
| `05_timeline.py` | every prefix of the log is an earlier graph |
| `06_fork.py` | `ops.fork` for divergent branches that share a prefix |
| `07_attach_subtree.py` | import agents from another graph and adopt them with `ops.attach` |
| `deep_tree.py` | build a deep agent tree with `ops.spawn`, print it, plot it |

Run any of them directly:

```bash
python examples/graph/01_query.py
python examples/graph/03_edit_frontier.py
python examples/graph/05_timeline.py
python examples/graph/07_attach_subtree.py
python examples/graph/deep_tree.py --depth 3 --branch 2
```

Most scripts just print to stdout. `04_save_load.py` also saves a run directory,
and `deep_tree.py` writes a PNG (or an SVG, without matplotlib).
