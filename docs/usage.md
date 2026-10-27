# Usage

How to run rlmflow, configure it, and work with what it produces. What an agent does once it's running (its REPL tools, how it waits, what a Claude Code session sees) is in [Agents](agents.md).

## Install

```bash
pip install "rlmflow[openai]"        # or [anthropic], [claude], [tui], [mcp], [all]
```

Set `OPENAI_API_KEY` (or `ANTHROPIC_API_KEY` for `claude*` models). The Claude Code harness and plugin need the `claude` [CLI](https://docs.claude.com/en/docs/claude-code), signed in.

## Run a task

```python
from rlmflow import Flow
from rlmflow.llm import client_for

flow = Flow(client_for("gpt-5-mini"))
flow.run("Add a and b", a=2, b=3)       # 5; keyword arguments become REPL variables
```

To keep the run, build the graph yourself:

```python
from rlmflow import AgentConfig, Graph

g = Graph()
root = g.create_agent("Count the x's in doc.", config=AgentConfig(max_depth=1), doc="xxaaxbxx", name="root")
flow.run(g)                             # returns the graph
g.result(root)                   # 5
g.save("runs/count")
```

Inside async code, use `await flow.arun(g)` and `await flow.aclose()` when done.

## Configs

An `AgentConfig` is a kind of agent. Named configs go on the graph, and agents launch each other by config name.

| Field             | Default     | Meaning                                                                |
| ----------------- | ----------- | ---------------------------------------------------------------------- |
| `name`            | `"default"` | What `launch_agent(config=...)` refers to.                             |
| `harness`         | `"repl"`    | `"repl"`, `"claude_code"`, `"pi"`, `"interactive"`, or an import path. |
| `model`           | `"default"` | A model registered on the Flow.                                        |
| `description`     | `""`        | Shown to agents choosing a config.                                     |
| `max_depth`       | `1`         | Levels of agents below this one; `0` can't launch.                     |
| `max_iters`       | `None`      | Model turns before it must finish.                                     |
| `output_schema`   | `None`      | JSON schema or Pydantic model the result must match.                   |
| `harness_options` | `{}`        | Harness settings, e.g. `{"tools": "loose"}`.                           |
| `max_budget`      | `None`      | Whole-run token threshold; crossing it gives every agent a final turn. |

```python
from pydantic import BaseModel

class Finding(BaseModel):
    file: str
    line: int

g.add_config(AgentConfig(name="searcher", model="fast", max_depth=0, description="searches one batch of files", output_schema=Finding))
root = g.create_agent("Find the magic number in haystack/.", config=AgentConfig(max_depth=1))
```

## Models

```python
from rlmflow.llm import AnthropicClient, client_for

flow = Flow(
    client_for("gpt-5"),                                  # the "default" model
    llm_clients={"fast": client_for("gpt-5-mini"), "claude": AnthropicClient("claude-sonnet-4-5")},
)
```

`client_for` picks Anthropic for `claude*` names and OpenAI otherwise. Any object with `chat(messages)` works as a client.

## Tools

```python
from rlmflow import FILE_TOOLS, tool

@tool("Look up a customer by id; returns a dict.")
def customer(customer_id: str) -> dict:
    return db.get(customer_id)

flow = Flow(client_for("gpt-5"), tools=[FILE_TOOLS, customer])
flow.inject("TODAY", "2026-10-25")      # a value bound in every REPL
```

Prompts and skills: [Prompt customization](prompt_customization.md), [Skills](skills.md).

## Runtimes and workspaces

The runtime is where agent code runs; the workspace is where each agent's files live.

```python
from rlmflow import DockerRuntime, GitWorktrees

flow = Flow(client_for("gpt-5"), runtime=DockerRuntime("rlmflow:local", working_directory="./proj"), workspaces=GitWorktrees("."))
```

| Runtime                  | For                                                  |
| ------------------------ | ---------------------------------------------------- |
| `LocalRuntime` (default) | A worker process per agent, with your privileges.    |
| `SubprocessRuntime`      | The same, with a chosen interpreter and environment. |
| `DockerRuntime`          | Untrusted code.                                      |
| `ModalRuntime`           | Remote sandboxes (`rlmflow[modal]`).                 |

`GitWorktrees` gives each agent its own worktree and checks a fork out at its fork point. More in [Runtimes](runtimes.md) and [Security](security.md).

## Stream and stop

```python
async for step in flow.run_streaming(g, until="idle"):
    print(g.path(step.agent), step.node.type)
```

`until` is `"done"` (default), `"next"`, `"idle"` (the next `code`, `waiting` or `done`), or a function of the step. Stop, look or change the graph, then call `run_streaming` again. Consumers handle the stream for you:

```python
from rlmflow import ConsumerGroup, GraphCheckpointer, LiveGraphTree

consumers = ConsumerGroup([LiveGraphTree(), GraphCheckpointer("runs/x")])
consumers.init(flow)
async for step in flow.run_streaming(g):
    consumers.handle(step, g)
consumers.close()
```

More in [Streaming](streaming.md).

## Read the graph

```python
g.result(root)       # None until done
g.status(root)       # "running" | "waiting" | "done"
g.history(root)      # the agent's nodes
g.children(root)     # agents it launched
g.usage()              # tokens; g.cost_usd()
g.nodes("code", agent=root)
```

Node types and edges: [Node model](node_model.md).

## Change a run

```python
from rlmflow import Input

g.send(root, "Also report the file name.")      # a message; reopens a finished agent
writer = g.create_agent("Summarize research.", inputs={"research": Input(research_agent)})  # runs after it
retry = g.fork(some_node, name="retry", model="fast")   # branch from any node
flow.run(g)
```

A fork shares the history up to its node and doesn't redo work that already finished. More in [Control](control.md) and [Injections](injections.md).

## Save and resume

```python
g.save("runs/x")
g = Graph.load("runs/x")
flow.run(g)                 # unfinished agents continue
```

## Claude Code and pi as agents

The harness is per config, so one run can mix them:

```python
g.add_config(AgentConfig(name="claude", harness="claude_code", harness_options={"tools": "loose"}))
root = g.create_agent(task, config=AgentConfig(harness="claude_code", max_depth=1))
```

| `claude_code` option       | Meaning                                                               |
| -------------------------- | --------------------------------------------------------------------- |
| `tools="strict"` (default) | Changes only through `rlmflow repl`, so every change is in the graph. |
| `tools="loose"`            | Also Claude's edit tools and any shell command.                       |
| `mcp=True`                 | Also offer the agent tools over MCP.                                  |
| `command`                  | A `claude` executable instead of the SDK's bundled one.               |

`pi` takes `tools`, `provider`, `session_dir` and `command`. Unknown options are an error. To write your own harness, subclass `Harness`; see [Internals](internals.md) and [`claude_code.py`](https://github.com/shyamsn97/rlmflow/blob/main/rlmflow/harnesses/claude_code.py).

## Your Claude Code session as the root

```text
$ cd my-project
$ claude --plugin-dir "$(rlmflow plugin path)"
> /rlmflow:run Find the magic number in haystack/ and tell me which file it's in.
```

`rlmflow plugin path` prints the plugin directory inside the installed package; `--plugin-dir` loads it for that session. Your session becomes the root agent and can launch workers; the run is saved under `.rlmflow/runs/<timestamp>/`. Workers use the models in `rlmflow.toml`. The plugin does nothing in sessions that didn't run `/rlmflow:run`. What the session does during a run is in [Agents](agents.md#your-own-claude-code-session).

## Fork from the command line

```bash
rlmflow view show runs/x --step 14                          # the header has the node id
rlmflow fork runs/x NODE --harness claude_code              # headless, saved back (or --out)
rlmflow fork runs/x NODE --interactive                      # prints the claude command to continue it
```

## Command line

```bash
rlmflow tui                                  # a coding agent over ., in the dashboard
rlmflow run print "fix the failing test"     # headless
rlmflow run --resume ./graph print "now update the docs"
rlmflow view show runs/x                     # tree and timeline
rlmflow render html runs/x run.html          # or gif, browser
rlmflow config show
```

Settings come from flags, then `RLMFLOW_*`, then `./rlmflow.toml`, then `~/.config/rlmflow/config.toml`; `rlmflow config init` writes a starter file. Every flag: [Command line](cli.md).

## See a run

```python
from rlmflow import render_tree, save_html, save_svg

print(render_tree(g))
save_svg(g, "graph.svg")
save_html(g, "run.html")
```

More in [Observability](observability.md).

## rlmflow as an LLM

```python
from rlmflow.flow import FlowLLM

agent = FlowLLM(Flow(client_for("gpt-5-mini")), AgentConfig(max_depth=1))
agent.chat([{"role": "user", "content": "Summarize this repo's README."}])
agent.last_graph
```

## Troubleshooting

| Symptom                                               | Fix                                                                                                               |
| ----------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `OPENAI_API_KEY is not set`                           | Set it, or use a `claude*` model with `ANTHROPIC_API_KEY`.                                                        |
| `claude_code needs pip install 'rlmflow[claude]'`     | Install the extra; it bundles the Claude Code CLI. Sign in with `claude` or set `ANTHROPIC_API_KEY`.              |
| Claude Code bills your API key, not your subscription | Unset `ANTHROPIC_API_KEY` for `claude`.                                                                           |
| `/rlmflow:run`: serve bound to another session        | One serve per project; exit the other session first.                                                              |
| A run fails                                           | `rlmflow view show RUN --step N` on the last step prints the error; `rlmflow serve` logs to `.rlmflow/serve.log`. |
