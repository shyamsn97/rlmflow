# Supervisor Injection Example

This directory demonstrates steering a real saved `Flow` run down a different
route. The point is injection: fork the run at the turn you disagree with, hand
the model a new instruction there, and continue each copy on a fresh `Flow`.

## Files

- `word_search.py` generates the baseline delegated run for finding `AGENT` and
  saves it to `examples/_runs/word-search/baseline/` (manifest + `agents/` logs).
- `inject_variants.py` loads that run, forks it twice at the turn where the root
  decided to delegate, appends a different `UserQuery` to each fork, and runs
  both variants in parallel on one `Flow`. It saves the finished variants beside
  the baseline at `examples/_runs/word-search/variant-cols/` and
  `.../variant-root/`. The baseline and variants use the same structured
  `WordSearchResult` shape, so validation checks the typed result instead of
  scraping a prose answer.

Inspect the baseline with:

```python
from rlmflow import AgentStart

root = AgentStart.load("examples/_runs/word-search/baseline")
for node in root.walk():
    print(node.parent_agent.config.path, node.type)
```

## Flow

1. Generate the baseline trace:

   ```bash
   python examples/control/injection/word_search.py
   ```

2. Steer it two ways:

   ```bash
   python examples/control/injection/inject_variants.py
   ```

`inject_variants.py` creates two edited Node trees in memory:

- **Variation A** replaces the delegated column route with one direct column
  helper function.
- **Variation B** replaces the delegation with one direct all-direction scanner.

Both edits are prompts. The example does not inject precomputed answers; it
changes the route and lets the model continue to a structured
`finish({"found": ..., "missing": ...})` result.

The baseline prompt explicitly forbids a root-level all-direction scan so the
saved trace is structurally different from the direct-scan variant: the original
route is child-driven direction analysis, while the injected variant swaps in a
direct deterministic scanner.

## What To Look For

- `node.fork()` copies the whole tree and cuts everything after that node, so
  the children the forked turn had launched are gone from the variant — and the
  baseline it was forked from is untouched.
- The new turn is an ordinary `UserQuery` appended to the fork's frontier.
  `Node.append` refuses anywhere else, so an edit cannot land mid-transcript.
- Both variants are streamed in the same loop, so you can watch them diverge.
- `root.result()` returns the structured word-search payload from `DoneOutput`,
  and the example validates it with `WordSearchResult.model_validate(...)`.
