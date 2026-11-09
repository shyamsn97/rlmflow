# Graph manipulation

Tiny self-contained scripts that show what an `rlmflow.Graph` can do. No
LLM keys needed: they build their graph by hand, or run a scripted model that
stands in for the LLM, so they finish in seconds and can be read top-to-bottom.

A graph holds agents, the nodes of each agent's history, and the edges between
them (`spawn`, `fork`, `return`, `write`). An agent is its root node. `g.append`
continues an agent, `g.start(parent=node)` launches one from a node,
`g.send` messages one, `g.fork` branches from an earlier node, and `g.edit`,
`g.delete`, `g.remove` and `g.replace` change the past: each returns the
`Invalidation` it caused (pass `dry_run=True` to only see it), and the next
`flow.run(g)` redoes just that.

| script | what it shows |
|---|---|
| `01_query.py` | `g.agents()`, `g.nodes()`, edges, errors, statuses, token usage, and the final result |
| `02_navigate.py` | each agent's history, its children, and an indented agent tree |
| `03_edit.py` | `g.edit` a child's result and `g.replace` one saved call (with `dry_run`), then message a finished agent |
| `04_save_load.py` | `g.save()` / `Graph.load()` run-directory round-trip |
| `05_timeline.py` | `replay(g)`: the graph after each agent and node, as real earlier graphs |
| `06_fork.py` | `g.fork` for divergent branches that share a prefix |
| `07_host_spawn.py` | finished agents written from host code, and `g.start(parent=...)` to launch a child |
| `deep_tree.py` | build a deep agent tree with `g.start(parent=...)`, print it, plot it |

Run any of them directly:

```bash
python examples/graph/01_query.py
python examples/graph/03_edit.py
python examples/graph/05_timeline.py
python examples/graph/07_host_spawn.py
python examples/graph/deep_tree.py --depth 3 --branch 2
```

Most scripts just print to stdout. `04_save_load.py` also saves a run directory,
and `deep_tree.py` writes a PNG (or an SVG, without matplotlib).
