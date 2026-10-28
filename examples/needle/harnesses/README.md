# Needle across harnesses

[`run.py`](run.py) is the filesystem needle ([`../filesystem.py`](../filesystem.py)) written against the Python API, with the harness of the root and of its workers as flags. Use it to check that every harness works end to end, alone or mixed:

| Harness       | What runs the agent                        | Needs                                                                |
| ------------- | ------------------------------------------ | -------------------------------------------------------------------- |
| `repl`        | rlmflow's own model-and-REPL loop          | `OPENAI_API_KEY` (or a `claude*` `--model` with `ANTHROPIC_API_KEY`) |
| `claude_code` | one Claude Agent SDK session turn per step | `pip install "rlmflow[claude]"`, signed in to Claude Code            |
| `pi`          | one `pi -p --mode json` turn per step      | `npm i -g @mariozechner/pi-coding-agent`, plus that provider's key   |

The root gets a task to split 100 files into batches and give each to a `searcher` agent. The searchers' harness is a separate flag, so for example a Claude Code root can launch REPL workers, or a REPL root can launch Claude Code workers.

## Run

```bash
python examples/needle/harnesses/run.py                                   # repl root, repl workers
python examples/needle/harnesses/run.py --root claude_code                # Claude Code root, repl workers
python examples/needle/harnesses/run.py --root repl --workers claude_code
python examples/needle/harnesses/run.py --root pi --workers pi --pi-provider openai --pi-model gpt-5-mini
python examples/needle/harnesses/run.py --all --num-files 40 --no-viz     # every combination this machine can run
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

Add `--docker-image rlmflow:local` (build it with `docker build -t rlmflow:local .`) to run every REPL in a container. Claude Code and pi still run on your machine, so their prompt says where the REPL runs (`/workspace`, mounted from the run's directory) and that they reach its files through `rlmflow repl`. On Docker, with the same models and 40 files, `claude_code`/`repl`, `repl`/`claude_code`, `claude_code`/`claude_code` and `repl`/`pi` were correct; `pi`/`repl` missed because pi's task said "matching exactly: The magic number is <number>." and the `gpt-5-nano` searcher put that final period in its regex.

Each run is saved in `examples/_runs/needle-harnesses/<root>-<workers>/`. Look at one with `rlmflow view show examples/_runs/needle-harnesses/claude_code-repl`.

## How it is built

The whole setup is a few lines of the Python API. The searcher config lives on the Flow, and the root's config is passed to the run:

```python
searcher = AgentConfig(
    name="searcher", harness="claude_code", max_depth=0, description="searches one batch of files"
)
flow = Flow(
    configs=[searcher],
    runtime=SubprocessRuntime(working_directory=workdir),
    tools=NEEDLE_FILE_TOOLS,
)

g = Graph()
root = g.create_agent(
    "There are 100 text files in haystack/ ... give each batch to a `searcher` agent."
)
flow.run(
    g,
    root_config=AgentConfig(
        harness="pi", model="gpt-5-mini", max_depth=1, harness_options={"provider": "openai"}
    ),
)
```

A config's `model` means whatever its harness takes:

- for `repl`, the model behind rlmflow's own loop (`--model` for the root, `--worker-model` for searchers);
- for `claude_code`, Claude's `--model` (`--claude-model`; `default` keeps your Claude setting);
- for `pi`, pi's `--model` (`--pi-model`, with `--pi-provider`).

Claude Code and pi agents run in the same working directory, use their own read tools on the files, and reach the agent tools (`launch_agent`, `submit`, ...) through `rlmflow repl`. Their cells land in the graph as `code` nodes like a REPL agent's. To wait for searchers they either `await` the handles in a cell or end their turn and get resumed with the results; a handle's `.done()` is a snapshot from the start of the cell, so polling it in a loop never sees a searcher finish.

If `pi` isn't on your PATH, install it anywhere and pass the binary: `--pi-command /path/to/node_modules/.bin/pi`.

## Other ways to run the same search

- [`../filesystem.py --harness claude_code`](../filesystem.py) and [`../haystack.py --harness pi`](../haystack.py) change only the root's harness.
- [`../fork_halfway.py`](../fork_halfway.py) forks a Claude Code root halfway through and checks that the fork reuses the finished workers.
- Your own Claude Code session as the root, through the plugin: `claude --plugin-dir "$(rlmflow plugin path)"`, then `/rlmflow:run There are 100 text files in haystack/ ...`. See [Usage](../../../docs/usage.md#your-claude-code-session-as-the-root).
