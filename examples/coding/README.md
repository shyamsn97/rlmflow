# Coding agent

Interactive coding agent as a plain REPL. The example owns the `flow.run_streaming(...)` loop: it prints one line per node the run lands (agent path and node type), checkpoints the tree under `<workdir>/graph` as it goes, and prints the final answer when the root is done.

```bash
pip install -e ".[openai]"
python examples/coding/agent.py --workdir ./myproject
python examples/coding/agent.py --workdir ./myproject --docker-image rlmflow:local
```

Default working directory: `examples/_runs/coding/`.

## Boids: peers build components

[`boids/boids.py`](boids/boids.py) gives three peer agents (`peer1`, `peer2`, `peer3`) the same product spec and one instruction: there's no manager, so split the work, agree on interfaces by messaging each other with `AGENTS[name].send(...)`, and write only the files you took on. Their tools are the file tools, their own REPL, and `open_page`, which returns a page's headless-Chrome console output. The app is validated only after the run.

```bash
uv run --locked --python 3.12 --extra claude \
  python examples/coding/boids/boids.py --harness claude_code
uv run --locked --python 3.12 --extra openai \
  python examples/coding/boids/boids.py
```

The run directory has the generated app, its graph, `run-metadata.json` and `validation.json`. Validate and smoke-load it directly:

```bash
uv run --locked --python 3.12 \
  python examples/coding/boids/validate.py \
  examples/_runs/coding/boids \
  --browser --screenshot docs/static/boids/simulation.png
```

After a successful run, `make showcase-boids-assets` repeats validation, captures a poster plus a continuous browser GIF/MP4, and animates the graph. Chrome or Chromium and ffmpeg must be available for the video assets.
