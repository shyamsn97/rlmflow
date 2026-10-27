# Supervisor Injection Example

This directory demonstrates steering a real saved `Flow` run down a different route. The point is injection: fork the run at the turn you disagree with, hand the model a new instruction there, and continue every fork in the same graph.

## Files

- `word_search.py` generates the baseline delegated run for finding `AGENT` and saves its graph to `examples/_runs/word-search/baseline/`.
- `inject_variants.py` loads that graph, forks the root twice just before the turn where it decided to delegate, sends each fork a different message, and runs both forks in parallel with one `flow.run_streaming(g)`. The baseline and both forks live in one graph, saved beside the baseline at `examples/_runs/word-search/variants/`. The baseline and variants use the same structured `WordSearchResult` shape, so validation checks the typed result instead of scraping a prose answer.

Inspect the baseline with:

```python
from rlmflow import Graph

g = Graph.load("examples/_runs/word-search/baseline")
for node in g.nodes():
    print(g.path(g.node(node.agent_id)), node.type)
```

## Flow

1. Generate the baseline trace:

   ```bash
   python examples/control/injection/word_search.py
   ```

1. Steer it two ways:

   ```bash
   python examples/control/injection/inject_variants.py
   ```

`inject_variants.py` adds two forks to the loaded graph:

- **Variation A** replaces the delegated column route with one direct column helper function.
- **Variation B** replaces the delegation with one direct all-direction scanner.

Both edits are prompts. The example does not inject precomputed answers; it changes the route and lets the model continue to a structured `submit({"found": ..., "missing": ...})` result.

The baseline prompt explicitly forbids a root-level all-direction scan so the saved trace is structurally different from the direct-scan variant: the original route is child-driven direction analysis, while the injected variant swaps in a direct deterministic scanner.

## What To Look For

- `g.fork(node)` copies nothing: the fork shares the history up to `node` through one `next` edge, so the children the later turn launched are not part of it — and the baseline is untouched.
- The new turn is an ordinary message sent with `g.send`; it is delivered at the fork's next turn boundary, so an edit cannot land mid-transcript.
- Both forks are driven by the same run, so you can watch them diverge.
- `g.result(fork)` returns the structured word-search payload from the `done` node, and the example validates it with `WordSearchResult.model_validate(...)`.
