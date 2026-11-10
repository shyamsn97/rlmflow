# Needle: recursive decomposition

Two variants of the same search task:

- [`haystack.py`](haystack.py) — one massive in-memory `haystack` argument; the agent must chunk and delegate.
- [`filesystem.py`](filesystem.py) — 100 files under `haystack/`, searched with file tools, with the root and its workers on any harness (`repl`, `claude_code`, `pi`); `--all` checks every combination. See [On any harness](#on-any-harness).

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

The generated magic number is printed after the run, so the reported answer can be checked without trusting the model. The saved v7 graph is `examples/_runs/needle-haystack/`.

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

Both Needle variants render the live agent tree and checkpoint through a `ConsumerGroup`; pass `--no-viz` to disable terminal redraws. Pass `--docker-image rlmflow:local` to either variant to run agent code in the repository's Docker sandbox.

## On any harness

[`filesystem.py`](filesystem.py) is the needle over files, with the harness of the root and of its `searcher` workers as flags:

| Harness       | What runs the agent                        | Needs                                                                |
| ------------- | ------------------------------------------ | -------------------------------------------------------------------- |
| `repl`        | rlmflow's own model-and-REPL loop          | `OPENAI_API_KEY` (or a `claude*` `--model` with `ANTHROPIC_API_KEY`) |
| `claude_code` | one Claude Agent SDK session turn per step | `pip install "rlmflow[claude]"`, signed in to Claude Code            |
| `pi`          | one `pi -p --mode json` turn per step      | `npm i -g @mariozechner/pi-coding-agent`, plus that provider's key   |

```bash
python examples/needle/filesystem.py                                   # repl root, repl workers
python examples/needle/filesystem.py --root claude_code                # Claude Code root, repl workers
python examples/needle/filesystem.py --root repl --workers claude_code
python examples/needle/filesystem.py --root pi --workers pi --pi-provider openai --pi-model gpt-5-mini
python examples/needle/filesystem.py --all --num-files 40 --no-viz     # every combination this machine can run
```

`--all` skips a combination when a harness can't run here and says why. It prints a table and exits non-zero if any run got the wrong number. One run of each combination on 40 files (October 2026), with `gpt-5-mini` for the REPL root and for pi (`--pi-provider openai --pi-model gpt-5-mini`), `gpt-5-nano` for REPL searchers, and Claude Code on its default model:

```text
root         workers      correct  agents  seconds
repl         repl         True     5       75
repl         claude_code  True     9       41
repl         pi           True     6       32
claude_code  repl         True     5       77
claude_code  claude_code  True     5       20
pi           pi           True     9       59
pi           repl         False    5       36
```

The `pi`/`repl` miss was the model, not the harness. pi handed its searchers bare file names (`file_0032.txt` rather than `haystack/file_0032.txt`), the searcher holding the needle couldn't open its file and returned `None`, and pi submitted that. Expect a single run of any combination to miss now and then; run it again or use a stronger model.

With `--docker-image`, Claude Code and pi still run on your machine, so their prompt says where the REPL runs (`/workspace`, mounted from the run's directory) and that they reach its files through `rlmflow repl`.

Each run is saved in `examples/_runs/needle-filesystem/<root>-<workers>/`. Look at one with `rlmflow view show examples/_runs/needle-filesystem/claude_code-repl`.

A config's `model` means whatever its harness takes: for `repl`, the model behind rlmflow's own loop (`--model` for the root, `--worker-model` for searchers); for `claude_code`, Claude's `--model` (`--claude-model`); for `pi`, pi's `--model` (`--pi-model`, with `--pi-provider`). If `pi` isn't on your PATH, pass the binary: `--pi-command /path/to/node_modules/.bin/pi`.

Your own Claude Code session can be the root too, through the plugin; the main [README](../../README.md#your-claude-code-session-as-the-root-agent) has a recorded run on this haystack.
