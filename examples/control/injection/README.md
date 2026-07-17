# Supervisor Injection Example

This directory demonstrates prompt-based graph surgery on a real saved
`Flow` run. The point is injection: replace a real `SupervisingOutput` node,
truncate the now-obsolete children, adopt the edited graph on a fresh
`Flow`, and continue stepping.

## Files

- `word_search.py` generates the baseline delegated run for finding `AGENT` and
  saves it to `examples/_runs/word-search/baseline/` (manifest + `agents/` logs).
- `inject_variants.py` loads that run, edits copies with
  `graph.replace(..., truncate="descendants")`, and continues both variants in parallel with
  separate `Flow` instances. It saves the finished variants beside the baseline
  at `examples/_runs/word-search/variant-cols/` and `.../variant-root/`. The
  baseline and variants use the same structured `WordSearchResult` shape, so
  validation checks the typed graph result instead of scraping a prose answer.

Inspect the baseline with:

```python
from rflow import open_viewer

open_viewer("examples/_runs/word-search/baseline")
```

## Flow

1. Generate the baseline trace:

   ```bash
   python examples/control/injection/word_search.py
   ```

2. Inject alternate supervisor outcomes:

   ```bash
   python examples/control/injection/inject_variants.py
   ```

`inject_variants.py` creates two edited graphs in memory:

- **Variation A** replaces the `root.cols` delegated route with one direct
  column helper function.
- **Variation B** replaces the root supervisor with one direct all-direction
  scanner.

Both edits are prompt-based. The example does not inject precomputed answers; it
changes the supervisor route and lets the model continue to a structured
`done({"found": ..., "missing": ...})` result.

The baseline prompt explicitly forbids a root-level all-direction scan so the
saved trace is structurally different from the direct-scan variant: the original
route is child-driven direction analysis, while the injected root variant swaps
in a direct deterministic scanner.

## What To Look For

- The edited node is a real `SupervisingOutput` from the baseline trace.
- `truncate="descendants"` removes obsolete waited-on child routes.
- Each variant gets its own `Flow`; `flow.run_streaming(graph=edited, until=...)`
  continues the edited copy (no `flow.graph = ...`).
- Both variants are streamed in the same loop so you can watch them diverge.
- `graph.result()` returns the structured word-search payload from `DoneOutput`,
  and the example validates it with `WordSearchResult.model_validate(...)`.
