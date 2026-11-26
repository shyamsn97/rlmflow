# Coding

For a coding agent over your own project, use `rlmflow run` (see [Command line](../../docs/cli.md)).

## Boids: peers on a shared task board

[`boids/boids.py`](boids/boids.py) gives three peer agents (`peer1`, `peer2`, `peer3`) the same product spec and a shared `TaskBoard`, with no manager. One of them plans: each posts and claims a `plan` task first, and only the winner publishes the interface and posts the file tasks. They claim tasks (the first claim wins), publish the interfaces they define, finish what they claimed, and write only the files of tasks they claimed; they can message each other with `AGENTS[name].send(...)` about bugs. Their other tools are the file tools, their own REPL, and `open_page`, which shows them screenshots of a page in headless Chrome and returns its console output. The app is validated only after the run, and the board is written to `board.json`.

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
