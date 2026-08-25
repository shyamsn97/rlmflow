# Command line

`rlmflow` runs an agent and reads the graph it leaves behind. Everything here
also works as `python -m rlmflow …`, and `rlmflow --help` (or `--help` on any
command) prints the same information from the source.

Every command is a Fire class: constructor arguments are flags, methods are
verbs. `rlmflow run --model gpt-5 tui "fix the tests"` instantiates `Run` with
`--model`, then calls `tui` with the query.

```bash
rlmflow tui                                   # coding agent, in the dashboard
rlmflow run print "fix the failing test"      # same agent, no dashboard
rlmflow view show runs/coding/graph           # the tree, then the timeline
rlmflow render gif runs/coding/graph run.gif  # the run, animated
rlmflow config show                           # what a run would use, and why
rlmflow version                               # version, Python, installed extras
```

## `rlmflow tui` and `rlmflow run`

Both drive a coding agent with `FILE_TOOLS` over `--workdir` (the current
directory by default), checkpointing the graph to `<workdir>/graph`. The
dashboard needs `pip install "rlmflow[tui]"`.

```bash
rlmflow tui                                     # dashboard, waiting for a query
rlmflow tui --query "add a test for parse_args" # dashboard, first turn underway
rlmflow run tui "add a test for parse_args"     # same, query as a positional
rlmflow run print "fix the failing test"        # headless: stream it, print the answer
rlmflow run --workdir ./proj --model gpt-5 tui
rlmflow tui --docker-image rlmflow:local
rlmflow tui --resume ./proj/graph               # continue a saved run
rlmflow tui --agent mypkg.agents:build          # your own Flow instead of this one
```

| Flag | What it changes |
|---|---|
| `--query` | Opening turn. On `run`, the verb can take this as a positional instead. |
| `--model`, `--fast-model` | The main client and the one registered as `fast`. `claude*` goes to Anthropic, anything else to OpenAI. |
| `--reasoning-effort` | `minimal`, `low`, `medium`, `high`. Reaches OpenAI reasoning models; ignored elsewhere. |
| `--workdir` | The directory the agent reads and edits, and where `graph/` lands. |
| `--docker-image` | Swaps `LocalRuntime` for `DockerRuntime` on that image. |
| `--max-depth`, `--max-iters` | The `AgentConfig` every root picks up. |
| `--workers` | How many agents may run at once. |
| `--tools` | `files` (the default) or `none` for a bare REPL. |
| `--resume DIR` | Continue a saved run instead of starting a new one. |
| `--agent module:factory` | Import a zero-argument callable returning a `Flow`. |

`run print` is the one to reach for in a pipe, a CI job, or anywhere without a
terminal. It streams the same nodes through `LiveTreeRenderer` and exits with
`0` on success, `1` on a handled error, `2` on bad usage.

Anything the flags cannot say stays in Python: write a factory that returns the
`Flow` you want — custom prompts, tools, runtimes, profiles — and point
`--agent` at it.

```python
# mypkg/agents.py
from rlmflow import FILE_TOOLS, AgentConfig, Flow, LocalRuntime
from rlmflow.llm import client_for

def build() -> Flow:
    return Flow(
        client_for("gpt-5", reasoning_effort="low"),
        runtime=LocalRuntime(working_directory="."),
        tools=[*FILE_TOOLS, my_tool],
    root_config=AgentConfig(max_depth=2, max_iters=40),
    )
```

## `rlmflow view` and `rlmflow render`

`view` reads a saved run and prints; `render` writes it somewhere, so each verb
takes a destination.

```bash
RUN=runs/coding/graph
rlmflow view show $RUN                # the agent tree, then the numbered timeline
rlmflow view show $RUN --tree         # just the tree
rlmflow view show $RUN --step 12      # one step, with its content
rlmflow view show $RUN --frames-only  # the tree as it stood at every step

rlmflow render svg $RUN run.svg       # the figure
rlmflow render html $RUN run.html     # a single-file stepper
rlmflow render browser $RUN           # the Gradio viewer      (rlmflow[viewer])
rlmflow render gif $RUN run.gif       # animated, --every N    (rlmflow[image])
rlmflow render frames $RUN out/       # a PNG per step         (rlmflow[image])
```

See [Observability](observability.md) for the Python equivalents and for what
each view is good at.

## Configuration

Settings resolve in this order, first one wins:

1. flags on the command
2. environment: `RLMFLOW_MODEL`, `RLMFLOW_WORKDIR`, `RLMFLOW_MAX_ITERS`, … (one
   per flag, upper-cased with the prefix)
3. `./rlmflow.toml` in the directory you are in
4. `~/.config/rlmflow/config.toml`, or wherever `RLMFLOW_CONFIG` points
5. the defaults

```toml
# rlmflow.toml
[run]
model = "gpt-5"
fast_model = "gpt-5-mini"
reasoning_effort = "low"
max_iters = 30
```

```bash
rlmflow config show   # every setting, its value, which source won, its env var
rlmflow config path   # the files that are read, and whether they exist
rlmflow config init   # write a starter rlmflow.toml holding what is in effect
```

Provider keys stay the clients' business: set `OPENAI_API_KEY` or
`ANTHROPIC_API_KEY` as usual. `run` / `tui` check the one your model needs is
present and say so plainly if it is not, rather than failing inside the SDK.

## Extending it

The CLI is a [Fire](https://github.com/google/python-fire) component in
[`rlmflow/cli/`](https://github.com/shyamsn97/rlmflow/tree/main/rlmflow/cli).
`RlmflowCLI` composes `RunCLI`, `TuiCLI`, `ViewCLI`, `RenderCLI`,
`ConfigCLI`, and `VersionCLI` onto one root. Fire turns each constructor into
flags, each method into a verb, and each docstring into help. Adding a command
is a class plus a line in `RlmflowCLI.__init__`.
