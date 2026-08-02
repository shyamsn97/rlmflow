# Node features

Tiny self-contained scripts that show what a `rlmflow.Node` can do. No
LLM keys needed — every example builds its own graph by hand, so they finish in
milliseconds and can be read top-to-bottom.

| script | what it shows |
|---|---|
| `01_query.py` | `walk()`, node types, errors, token usage, and the final result |
| `02_navigate.py` | child lookup, traversal, and an indented walk of the tree |
| `04_save_load.py` | `save()` / `load()` run-directory round-trip |
| `05_timeline.py` | explicit snapshots as graph state changes |
| `06_fork.py` | `Node.fork()` for divergent branches |
| `deep_tree.py` | build a deep agent tree with `append`, print it, plot it |

Run any of them directly:

```bash
python examples/graph/01_query.py
python examples/graph/05_timeline.py
python examples/graph/deep_tree.py --depth 3 --branch 2
```

Most scripts just print to stdout. `04_save_load.py` also saves a run directory,
and `deep_tree.py` writes a PNG (or an SVG, without matplotlib).
