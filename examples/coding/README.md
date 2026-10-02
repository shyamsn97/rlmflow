# Coding agent

Interactive coding agent as a plain REPL. The example owns the `flow.run_streaming(...)` loop: it prints one line per node the run lands (agent path and node type), checkpoints the tree under `<workdir>/graph` as it goes, and prints the final answer when the root is done.

```bash
pip install -e ".[openai]"
python examples/coding/agent.py --workdir ./myproject
python examples/coding/agent.py --workdir ./myproject --docker-image rlmflow:local
```

Default working directory: `examples/_runs/coding/`.

## Boids showcase

[`boids/boids.py`](boids/boids.py) is the release coding showcase. Its prompt gives the model product constraints for a dependency-free browser simulation, but no filenames, module boundaries, or delegation instructions. The model chooses the application architecture and execution graph.

Run rlmflow:

```bash
uv run --locked --python 3.12 --extra openai --extra image \
  python examples/coding/boids/boids.py \
  --runner rlmflow --model gpt-5 --fast-model gpt-5-mini --force
```

Optionally run the same task with both supported runners:

```bash
uv run --locked --python 3.12 --extra openai --extra official-rlm \
  python examples/coding/boids/boids.py \
  --runner both --model gpt-5 --fast-model gpt-5-mini --force
```

The checked-in [`boids/recorded-run.json`](boids/recorded-run.json) describes one rlmflow showcase run, not a benchmark. In that run, the root chose a five-file architecture, launched five GPT-5-mini file owners concurrently, and used one focused child for an integration repair. The prompt prescribed neither the files nor the graph.

Each workspace includes `run-metadata.json` and `validation.json`; the rlmflow workspace also contains the generated application and retained graph. Validate and smoke-load it directly:

```bash
uv run --locked --python 3.12 \
  python examples/coding/boids/validate.py \
  examples/_runs/coding/boids-rlmflow \
  --browser --screenshot docs/static/boids/simulation.png
```

After a successful rlmflow run, `make showcase-boids-assets` repeats validation, captures a poster plus a continuous browser GIF/MP4, and renders the retained execution graph as a level-synchronized GIF. Chrome or Chromium and ffmpeg must be available for the video assets.
