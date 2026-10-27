# Graph manipulation

Tiny self-contained scripts that show what an `rlmflow.Graph` can do. No
LLM keys needed — every example builds its own graph by hand and reads it back
with `Graph`'s methods, so they finish in milliseconds and can be read top-to-bottom.

The graph is an append-only log. `g.append` continues an agent, `g.fork`
branches from an earlier node to revise history, and `g.attach` gives agents
prepared elsewhere a launch action under a parent. There is intentionally no
in-place delete or reparent operation.

| script | what it shows |
|---|---|
| `01_query.py` | `g.nodes()`, node types, errors, token usage, and the final result |
| `02_navigate.py` | each agent's history, its children, and an indented agent tree |
| `03_edit_frontier.py` | message a finished agent, reject stale edits, and revise with `g.fork` |
| `04_save_load.py` | `g.save()` / `Graph.load()` run-directory round-trip |
| `05_timeline.py` | every prefix of the log is an earlier graph |
| `06_fork.py` | `g.fork` for divergent branches that share a prefix |
| `07_attach_subtree.py` | import agents from another graph and adopt them with `g.attach` |
| `deep_tree.py` | build a deep agent tree with `g.spawn`, print it, plot it |

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
