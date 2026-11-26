# Command line

`rlmflow` runs an agent and reads the graph it leaves behind. Everything here also works as `python -m rlmflow …`, and `rlmflow --help` (or `--help` on any command) prints the same information from the source.

Every command is a Fire class: constructor arguments are flags, methods are verbs. `rlmflow run --model gpt-6-sol print "fix the tests"` instantiates `Run` with `--model`, then calls `print` with the query.

```bash
rlmflow run print "fix the failing test"      # coding agent: stream it, print the answer
rlmflow view show runs/coding/graph           # the tree, then the timeline
rlmflow render gif runs/coding/graph run.gif  # the run, animated
rlmflow config show                           # what a run would use, and why
rlmflow version                               # version, Python, installed extras
```

## `rlmflow run`

Drives a coding agent with `FILE_TOOLS` over `--workdir` (the current directory by default), streaming the agent tree and checkpointing the graph to `<workdir>/graph`. Launched children submit against `CODING_HANDOFF_SCHEMA` (`done`, `diff_summary`, `concerns`, `followups`) so the parent sees a memo rather than `"done"`; the root itself stays untyped.

```bash
rlmflow run print "fix the failing test"           # stream it, print the answer
rlmflow run --query "add a test for parse_args"    # the same, query as a flag
rlmflow run --workdir ./proj --model gpt-6-sol print "add a test"
rlmflow run --docker-image rlmflow:local print "add a test"
rlmflow run --resume ./proj/graph print "now update the docs"   # continue a saved run
rlmflow run --flow mypkg.agents:build print "add a test"        # your own Flow
```

| Flag                         | What it changes                                                                                                               |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `--query`                    | Opening turn. `print` can take it as a positional instead.                                                                    |
| `--model`, `--fast-model`    | The root's model and the `worker` agent's. `claude*` goes to Anthropic, anything else to OpenAI.                              |
| `--reasoning-effort`         | `low` (default), `medium`, `high`, `xhigh`; Terra and Luna also take `none`, Sol does not. Ignored for Anthropic models.      |
| `--workdir`                  | The directory the agent reads and edits, and where `graph/` lands.                                                            |
| `--docker-image`             | Swaps `LocalRuntime` for `DockerRuntime` on that image.                                                                       |
| `--max-depth`, `--max-iters` | The `default` agent's: the root's, and its plain `launch_agent` calls'.                                                       |
| `--workers`                  | How many model calls may run at once.                                                                                         |
| `--tools`                    | `files` (the default) or `none` for a bare REPL.                                                                              |
| `--resume DIR`               | Continue a saved run instead of starting a new one.                                                                           |
| `--flow module:factory`      | A zero-argument callable returning a `Flow`, from the project or an installed package. Also `flow = "..."` in `rlmflow.toml`. |

It works in a pipe, a CI job, or anywhere without a terminal: it streams the nodes through a plain-text `LiveTree` and exits with `0` on success, `1` on a handled error, `2` on bad usage.

Anything the flags cannot say stays in Python: write a factory that returns the `Flow` you want — custom prompts, tools, runtimes, agents — and point `--flow` at it. Set `flow` in `rlmflow.toml` instead and `rlmflow serve` (so `/rlmflow:run` in your own Claude Code session) uses it too.

```python
# mypkg/agents.py
from rlmflow import Agent, Flow, LocalRuntime
from rlmflow.llm import client_for
from rlmflow.tools.files import CODING_HANDOFF_SCHEMA, FILE_TOOLS


def build() -> Flow:
    return Flow(
        agents={
            "default": Agent(model="gpt-6-sol", max_depth=1, max_iters=40),
            "worker": Agent(model="gpt-6.1-terra", output_schema=CODING_HANDOFF_SCHEMA),
        },
        llm_clients={
            "gpt-6-sol": client_for("gpt-6-sol", reasoning_effort="low")
        },
        runtime=LocalRuntime(working_directory="."),
        tools=[FILE_TOOLS, my_tool],
    )
```

## `rlmflow view` and `rlmflow render`

`view` reads a saved run and prints; `render` writes it somewhere, so each verb takes a destination.

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

See [Observability](observability.md) for the Python equivalents and for what each view is good at.

## Configuration

Settings resolve in this order, first one wins:

1. flags on the command
1. environment: `RLMFLOW_MODEL`, `RLMFLOW_WORKDIR`, `RLMFLOW_MAX_ITERS`, … (one per flag, upper-cased with the prefix)
1. `./rlmflow.toml` in the directory you are in
1. `~/.config/rlmflow/config.toml`, or wherever `RLMFLOW_CONFIG` points
1. the defaults

```toml
# rlmflow.toml
[run]
model = "gpt-6-sol"
fast_model = "gpt-6.1-terra"
reasoning_effort = "low"
max_iters = 30
```

```bash
rlmflow config show   # every setting, its value, which source won, its env var
rlmflow config path   # the files that are read, and whether they exist
rlmflow config init   # write a starter rlmflow.toml holding what is in effect
```

Provider keys stay the clients' business: set `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` as usual. `run` checks the one your model needs is present and say so plainly if it is not, rather than failing inside the SDK.

## Extending it

The CLI is a [Fire](https://github.com/google/python-fire) component in [`rlmflow/cli/`](https://github.com/shyamsn97/rlmflow/tree/main/rlmflow/cli). `RlmflowCLI` composes `RunCLI`, `ViewCLI`, `RenderCLI`, `ConfigCLI`, and `VersionCLI` onto one root. Fire turns each constructor into flags, each method into a verb, and each docstring into help. Adding a command is a class plus a line in `RlmflowCLI.__init__`.
