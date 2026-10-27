# Shepherd recovery

Shepherd is the flagship demonstration of why `rlmflow` keeps agent execution as a durable graph. The example first records a reproducible failed worker trace by scripting eight myopic Sokoban pushes. GPT-5 then inspects earlier graph states and proposes eight rewind/instruction pairs. The host forks the worker at those nodes, runs the GPT-5-mini recoveries as child agents, and selects the highest-scoring result.

This is a live model recovery run, not a canned result or replay-only simulation. The deliberate jam is fixed so attempts start from the same failure. Shepherd planning and recovery play come from the configured model APIs; the Sokoban environment is local and deterministic so retained actions can be replayed exactly.

## Run

From the repository root:

```bash
export OPENAI_API_KEY=...
uv run --locked --python 3.12 --extra openai --extra image \
  python examples/shepherd/shepherd.py \
  --model gpt-5 \
  --fast-model gpt-5-mini \
  --branches 8 \
  --time-budget 1800 \
  --out-dir examples/_runs/shepherd
```

The 30-minute budget stops new branch turns and preserves the best result reached so far. A model outcome is evidence, not a fixture: do not rerun a completed attempt merely to improve the score.

The checked-in September 2026 artifact records:

- models: GPT-5 shepherd, GPT-5-mini recovery workers;
- outcome: 6/8 branches solved; `branch2` selected at rewind 7 with 11 total pushes (score 989);
- wall time: 11m 12s;
- graph: 9 agents, 585 nodes;
- usage: 439,855 input, 105,216 output, and 86,784 reasoning tokens.

The graph retains one recoverable error cell: the shepherd wrote `await branch(...)` after the synchronous proxy had already recorded all eight plans. The host observed the plans and launched the recoveries normally; the error is preserved rather than edited out of the trace.

Add `--gradio` and the `viewer` extra for the live browser board, or `--trace` for one terminal line per landed graph node:

```bash
uv run --locked --python 3.12 --extra openai --extra image --extra viewer \
  python examples/shepherd/shepherd.py --gradio
```

## Render and publish

The run writes:

- `worker/`: the standalone failed worker graph;
- `shepherd/`: the complete v7 recovery tree, including the selected branch;
- `traces/worker.{json,gif}`: the deliberate jam;
- `traces/branch0..branch7.{json,gif}`: each recovery trajectory.

Regenerate the graph summary, every-node figure, rewind animation, and checked-in documentation assets from that one run:

```bash
make showcase-shepherd-assets
```

The source renderers can also be called directly:

```bash
uv run --locked --python 3.12 --extra image \
  python examples/shepherd/render_graph.py --run-dir examples/_runs/shepherd
uv run --locked --python 3.12 --extra image \
  python examples/shepherd/render_rewind.py --run-dir examples/_runs/shepherd
```

Generated run artifacts stay under `examples/_runs/shepherd/`. The make target publishes the selected recovery animation and branch traces under `docs/static/shepherd/`, plus `docs/static/shepherd_graph.svg` and `docs/static/shepherd_nodes.svg`.

See [`../../docs/shepherd.md`](../../docs/shepherd.md) for the run walkthrough and the exact graph operations behind observe, rewind, fork, re-instruct, and select.
