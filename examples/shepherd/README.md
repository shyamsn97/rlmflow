# Shepherd recovery

Shepherd is the flagship demonstration of why `rlmflow` keeps agent execution as a durable graph. The example first records a reproducible failed worker trace by scripting eight myopic Sokoban pushes. GPT-5 then inspects earlier graph states and proposes eight rewind/instruction pairs. The host forks the run at those nodes, injects each plan into its copy, and runs the GPT-5 recoveries in parallel. When they finish, the host sends the shepherd every branch's outcome and final board, and the shepherd picks the one to keep.

This is a live model recovery run, not a canned result or replay-only simulation. The deliberate jam is fixed so attempts start from the same failure. Shepherd planning and recovery play come from the configured model APIs; the Sokoban environment is local and deterministic so retained actions can be replayed exactly.

## Run

From the repository root:

```bash
export OPENAI_API_KEY=...
uv run --locked --python 3.12 --extra openai --extra image \
  python examples/shepherd/shepherd.py \
  --model gpt-5 \
  --fast-model gpt-5 \
  --branches 8 \
  --time-budget 1800 \
  --out-dir examples/_runs/shepherd
```

The 30-minute budget stops new branch turns and preserves the best result reached so far. A model outcome is evidence, not a fixture: do not rerun a completed attempt merely to improve the score.

The checked-in November 2026 artifact records:

- models: GPT-5 shepherd, GPT-5 recovery workers;
- outcome: 8/8 branches solved; the shepherd picked `branch6`, rewound 8 to the fresh board, with 12 total pushes;
- wall time: 9m 12s;
- graph: 10 agents, 157 nodes;
- usage: 360,109 input, 170,490 output, and 161,152 reasoning tokens.

Add `--gradio` and the `viewer` extra for the live browser board, or `--trace` for one terminal line per landed graph node:

```bash
uv run --locked --python 3.12 --extra openai --extra image --extra viewer \
  python examples/shepherd/shepherd.py --gradio
```

## Render and publish

The run writes:

- `graph/`: the worker's run, ending in its jam;
- `shepherd/`: the shepherd's own graph, its planning turns and the pick;
- `branches/branchN/`: each recovery branch, a fork of the run cut at its rewind point, with the worker's copy and its plan;
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
uv run --locked --python 3.12 --extra image \
  python examples/shepherd/render_fanout.py --run-dir examples/_runs/shepherd
uv run python examples/shepherd/combine_graph.py examples/_runs/shepherd
uv run --extra image rlmflow render gif examples/_runs/shepherd/combined shepherd.gif
```

`render_rewind.py` follows the picked branch alone; `render_fanout.py` shrinks the jam into the root of a tree, spawns the branches below it, rewinds each to its depth, and plays them all at once. Each fork is saved as its own graph, so `combine_graph.py` merges the worker, the shepherd and the nodes each fork added into one graph for `rlmflow render`.

Generated run artifacts stay under `examples/_runs/shepherd/`. The make target publishes the picked recovery animation and branch traces under `docs/static/shepherd/`, plus `docs/static/shepherd_graph.svg` and `docs/static/shepherd_nodes.svg`.

See [`../../docs/shepherd.md`](../../docs/shepherd.md) for the run walkthrough and the exact graph operations behind observe, rewind, fork, re-instruct, and pick.
