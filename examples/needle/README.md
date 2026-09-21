# Needle: recursive decomposition

Two variants of the same search task:

- [`haystack.py`](haystack.py) — one massive in-memory `INPUTS["haystack"]` string; the agent must chunk and delegate.
- [`filesystem.py`](filesystem.py) — hundreds of files under `haystack/`; uses `FILE_TOOLS`.

The canonical showcase is the million-line in-memory variant. It gives the root the value through its REPL rather than placing a million lines in the model prompt; the useful behavior to inspect is how the root partitions that value, launches fresh-context workers, and combines their answers.

```bash
export OPENAI_API_KEY=...
uv run --locked --python 3.12 --extra openai \
  python examples/needle/haystack.py \
  --num-lines 1000000 \
  --model gpt-5-mini \
  --fast-model gpt-5-nano \
  --out-dir examples/_runs/needle-haystack
```

The generated magic number is printed after the run, so the reported answer can be checked without trusting the model. The saved persistence-v5 graph is `examples/_runs/needle-haystack/`.

The checked-in September 2026 artifact generated `5335239` at line 466,831 and returned the same value:

- models: GPT-5-mini root, GPT-5-nano workers;
- outcome: correct, using 13 child workers;
- wall time: 3m 26s;
- worker timing: launches from +23.5s to +81.2s, with peak concurrency of 10;
- graph: 14 agents, 207 nodes;
- usage: 91,036 input, 58,273 output, and 27,392 reasoning tokens.

The graph also retains ten recoverable error nodes from the live search rather than presenting a cleaned-up synthetic trajectory.

Render the same saved graph as a static figure, a steppable trace, and a level-synchronized parallel animation:

```bash
make showcase-needle-assets
```

That writes `docs/static/needle-haystack.{svg,gif}` and keeps the steppable `run.html` beside the ignored run graph. The SVG and HTML expose graph structure and the exact event order. The GIF is deliberately structural: it reveals the same local sequence level across every worker in one frame, so parallel branches light together instead of being serialized by event arrival time. The measured wall-clock concurrency remains recorded separately above.

The underlying commands are:

```bash
RUN=examples/_runs/needle-haystack
uv run --locked --python 3.12 rlmflow render svg \
  "$RUN" docs/static/needle-haystack.svg
uv run --locked --python 3.12 rlmflow render html "$RUN" "$RUN/run.html"
uv run --locked --python 3.12 --extra image \
  python examples/needle/render_parallel.py \
  --run-dir "$RUN" --out docs/static/needle-haystack.gif
```

Both Needle variants render the live agent tree and checkpoint through a `ConsumerGroup`; pass `--no-viz` to disable terminal redraws. The filesystem variant remains useful when the point of the demonstration is tool-mediated search:

```bash
uv run --locked --python 3.12 --extra openai \
  python examples/needle/filesystem.py --num-files 500
```

It saves to `examples/_runs/needle-filesystem/` by default. Pass `--docker-image rlmflow:local` to either variant to run agent code in the repository's Docker sandbox.
