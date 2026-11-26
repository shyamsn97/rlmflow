# Usage

How to run rlmflow, configure it, and work with what it produces. What an agent does once it's running (its REPL tools, how it waits, what a Claude Code session sees) is in [Agents](agents.md).

## Install

```bash
pip install "rlmflow[openai]"        # or [anthropic], [claude], [mcp], [all]
```

Set `OPENAI_API_KEY` (or `ANTHROPIC_API_KEY` for `claude*` models). The Claude Code harness and plugin need the `claude` [CLI](https://docs.claude.com/en/docs/claude-code), signed in.

## Run a task

```python
from rlmflow import Agent, Flow

flow = Flow(agents={"default": Agent(model="gpt-6.1-terra")})
flow.run("default", "Add a and b", a=2, b=3)  # 5; keyword arguments become REPL variables
```

To keep the run, start the graph yourself:

```python
g = flow.start("default", "Count the x's in doc.", doc="xxaaxbxx")
flow.run(g)  # returns the graph
g.root.final_result()  # 5
g.save("runs/count")
```

`Graph().start(task, agent="default", **args)` does the same without a Flow.

Inside async code, use `await flow.arun(g)` and `await flow.aclose()` when done.

## Agents

An `Agent` is a kind of agent: its harness (the class), model, limits and prompt. The Flow holds them by name in `agents`, and runs launch each other by that name. The graph only records each run's agent name; running a saved graph uses the agents of the Flow you run it with. Without `agents`, a Flow has one, `"default": Agent()`.

| Argument        | Default       | Meaning                                                                                                    |
| --------------- | ------------- | ---------------------------------------------------------------------------------------------------------- |
| `model`         | `"gpt-6-sol"` | A model name, like `"gpt-6.1-terra"` or `"claude-sonnet-4-5"`. `ClaudeCode` and `Pi` default to their own. |
| `description`   | `""`          | Shown to agents choosing one to launch.                                                                    |
| `max_depth`     | `1`           | Levels of agents below this one; `0` can't launch.                                                         |
| `max_iters`     | `30`          | Steps before it must finish; `None` for no limit.                                                          |
| `output_schema` | `None`        | JSON schema or Pydantic model the result must match.                                                       |
| `prompt`        | `{}`          | Sections merged over the system prompt by name; see [Prompts](#prompts).                                   |
| `system`        | `None`        | The whole system prompt, replacing the sections.                                                           |
| `keep`          | `None`        | Keep only the last `keep` messages of history.                                                             |

A whole run's tokens, dollars and time are limited per call with `flow.run(g, hooks=[Budget(...)])`; see [Control](control.md#run-budgets).

```python
from pydantic import BaseModel


class Finding(BaseModel):
    file: str
    line: int


searcher = Agent(
    model="gpt-6.1-terra",
    max_depth=0,
    description="searches one batch of files",
    output_schema=Finding,
)
lead = Agent(model="gpt-6-sol", max_depth=1)
flow = Flow(agents={"lead": lead, "searcher": searcher})
flow.run("lead", "Find the magic number in haystack/.")
```

Every run names one of the Flow's agents; `g.start(task)` without `agent=` names `"default"`. `launch_agent(...)` without `agent=` launches the launcher's own agent. An unknown name is a `KeyError` when the Flow reaches the run.

## Models

```python
from rlmflow.llm.openai import OpenAIClient

flow = Flow(
    agents={
        "default": Agent(model="gpt-6-sol"),
        "cheap": Agent(model="gpt-6.1-terra"),
        "claude": Agent(model="claude-sonnet-4-5"),
        "local": Agent(model="local"),
    },
    llm_clients={"local": OpenAIClient(model="qwen3", base_url="http://localhost:8000/v1")},
)
```

An agent's `model` is a model name; the Flow builds the client the first time it's used, Anthropic for `claude*` names and OpenAI otherwise. `Agent(..., reasoning_effort="low")` sets that agent's OpenAI reasoning effort; `"auto"` (the default) resolves to `"low"` for GPT-6 and the provider default for other models, while `None` always requests the provider default. `llm_clients` is for clients a name can't describe, such as a local server; an agent names them by their key. `OpenAIClient` talks to OpenAI through the Responses API, where GPT-6 can reason and call tools in one request, and to any other `base_url` through Chat Completions; `api="chat"` or `api="responses"` overrides that. Any OpenAI-compatible server works through `OpenAIClient(model, base_url=...)`, since the agent's one tool, `repl(code)`, is an OpenAI function tool (vLLM needs `--enable-auto-tool-choice` and a `--tool-call-parser` for it). Any object with `chat(messages, tools=None)` also works as a client: it gets OpenAI chat messages, tool calls and `tool` results included, and returns text, or a `rlmflow.llm.Reply` whose `calls` hold the `repl` call. `rlmflow.agents.repl_reply(code)` builds that reply for a scripted model. An `Agent` with `model=None` raises when it takes a step; Claude Code and pi use their own default model.

## Tools

```python
from rlmflow import tool
from rlmflow.tools.files import FILE_TOOLS


@tool("Look up a customer by id; returns a dict.")
def customer(customer_id: str) -> dict:
    return db.get(customer_id)


flow = Flow(tools=[FILE_TOOLS, customer])
flow.add_tool("2026-10-25", name="TODAY")  # a value bound in every REPL
flow.repl(g.root).set(limit=3)  # a value bound in one run's REPL
```

## Prompts

Every harness builds its system prompt from the same named sections. An agent's `prompt` replaces a section by name, drops it with `None`, or adds a new name at the end; a value can be a function of `(node, repl)`, computed every step. `system` replaces the whole prompt:

```python
Agent(model="gpt-6.1-terra", prompt={"agents": None, "style": "Be terse."})
Agent(model="gpt-6.1-terra", system=SOKOBAN_RULES)
```

To change what the agent sees each step, subclass `Agent` and override `messages(node, repl, since)`. See [Prompt customization](prompt_customization.md) for the sections, and [Skills](skills.md).

## Runtimes and workspaces

The runtime is where agent code runs; the workspace is where each agent's files live.

```python
from rlmflow import DockerRuntime, GitWorktrees

flow = Flow(
    runtime=DockerRuntime("rlmflow:local", working_directory="./proj"), workspaces=GitWorktrees(".")
)
```

| Runtime                  | For                                                                                        |
| ------------------------ | ------------------------------------------------------------------------------------------ |
| `LocalRuntime` (default) | A worker process per agent, with your privileges; `python=` and `env=` choose another one. |
| `DockerRuntime`          | Untrusted code.                                                                            |
| `ModalRuntime`           | Remote sandboxes (`rlmflow[modal]`).                                                       |

`GitWorktrees` gives each agent its own worktree and checks a fork out at its fork point. More in [Runtimes](runtimes.md) and [Security](security.md).

## Stream and stop

```python
from rlmflow.view.text import kind

async for node in flow.stream(g, until="idle"):
    print(node.path(), kind(node))
```

`until` is `"done"` (default), `"next"`, `"idle"` (the next node that ran code or ended a turn), or a function of the node. Stop, look or change the graph, then call `stream` again. Hooks handle the stream for you:

```python
from rlmflow import Checkpoint
from rlmflow.view import LiveTree

async for node in flow.stream(g, hooks=[LiveTree(), Checkpoint("runs/x")]):
    ...
```

Hooks see every node before you do and are closed when the stream ends, however it ends. More in [Streaming](streaming.md#hooks).

## Read the graph

```python
root.final_result()  # None until done
root.status()  # "running" | "idle" | "done"
root.nodes()  # the agent's nodes, its root first
root.last_turn()  # the node that ended its latest turn
root.children()  # agents it launched; agent.parent() is the one that launched it
g.edges(kind="return", target=node.id)  # whose results reached a node
g.usage()  # tokens; g.usage().cost_usd
[n for n in root.nodes() if n.runs_code]  # the nodes that ran code
```

Nodes, agents and edges: [The graph](node_model.md).

## Change a run

```python
g.send(root, "Also report the file name.", ext=".py")  # a message; ext becomes a variable
helper = g.start("Collect the sources.", parent=root)  # a child, from host code
retry = g.fork(some_node)  # a copy of the run cut after some_node; g is untouched
g.edit(helper.nodes()[-1], result="...", dry_run=True)  # what changing a result would redo
g.replace(node, 0, "cached value")  # bind the node's first saved call; the node reruns
flow.run(g)
```

`flow.run(retry)` runs the copy. Its REPLs are rebuilt by running their agents' nodes again, so finished work isn't redone; children launched by `some_node` or after it aren't copied. Edits follow the edges to what used what they changed: see [The graph](node_model.md#editing). More in [Control](control.md) and [Injections](injections.md).

## Save and resume

```python
g.save("runs/x")
g = Graph.load("runs/x")
flow.run(g)  # unfinished agents continue, with this Flow's agents
```

## Claude Code and pi as agents

The harness is per agent, so one run can mix them:

```python
from rlmflow import ClaudeCode

flow = Flow(agents={"claude": ClaudeCode(options={"permission_mode": "acceptEdits"})})
flow.run("claude", task)
```

Claude runs with your own Claude Code settings (permissions, `CLAUDE.md`, MCP servers, plugins); rlmflow only adds permission to run `rlmflow` commands. It reaches the agent tools through `rlmflow repl`, and the code it runs there is recorded in the graph. Changes it makes with its own tools are not recorded, but the workspace snapshot at the end of each turn still has the files.

| `ClaudeCode` argument | Meaning                                                                                                                                          |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `mcp=True`            | Also offer the REPL as a `python` MCP tool.                                                                                                      |
| `options`             | Fields for `ClaudeAgentOptions` (`permission_mode`, `cli_path` for your own `claude`, `max_turns`, `plugins`, ...); `allowed_tools` is extended. |

`Pi` takes `command` (the executable) and `args` (extra command-line arguments, such as `["--provider", "anthropic"]`); its sessions are under `~/.rlmflow/pi-sessions`. To write your own harness, subclass `Agent` and override `step`; see [Internals](internals.md) and [`claude_code.py`](https://github.com/shyamsn97/rlmflow/blob/main/rlmflow/agents/claude_code.py).

## Your Claude Code session as the root

```text
$ cd my-project
$ claude --plugin-dir "$(rlmflow plugin path)"
> /rlmflow:run Find the magic number in haystack/ and tell me which file it's in.
```

`rlmflow plugin path` prints the plugin directory inside the installed package; `--plugin-dir` loads it for that session. Your session becomes the root agent and can launch workers; the run is saved under `.rlmflow/runs/<timestamp>/`. Workers use the coding Flow built from `rlmflow.toml`, or your own: set `flow = "mypkg.agents:build"` there (a function returning a `Flow`), and your session can launch that Flow's agents. The plugin does nothing in sessions that didn't run `/rlmflow:run`. What the session does during a run is in [Agents](agents.md#your-own-claude-code-session).

## Fork from the command line

```bash
rlmflow view show runs/x --step 14                          # the header has the node id
rlmflow fork runs/x NODE --agent worker                     # headless; the copy is saved beside runs/x (or --out)
rlmflow fork runs/x NODE --interactive                      # prints the claude command to continue it
```

## Command line

```bash
rlmflow run print "fix the failing test"     # a coding agent over .
rlmflow run --resume ./graph print "now update the docs"
rlmflow view show runs/x                     # tree and timeline
rlmflow render html runs/x run.html          # or gif, browser
rlmflow config show
```

Settings come from flags, then `RLMFLOW_*`, then `./rlmflow.toml`, then `~/.config/rlmflow/config.toml`; `rlmflow config init` writes a starter file. Every flag: [Command line](cli.md).

## See a run

```python
from rlmflow.view import render_tree, save_html, save_svg

print(render_tree(g))
save_svg(g, "graph.svg")
save_html(g, "run.html")
```

More in [Observability](observability.md).

## Troubleshooting

| Symptom                                               | Fix                                                                                                               |
| ----------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `OPENAI_API_KEY is not set`                           | Set it, or use a `claude*` model with `ANTHROPIC_API_KEY`.                                                        |
| `claude_code needs pip install 'rlmflow[claude]'`     | Install the extra; it bundles the Claude Code CLI. Sign in with `claude` or set `ANTHROPIC_API_KEY`.              |
| Claude Code bills your API key, not your subscription | Unset `ANTHROPIC_API_KEY` for `claude`.                                                                           |
| `/rlmflow:run`: serve bound to another session        | One serve per project; exit the other session first.                                                              |
| A run fails                                           | `rlmflow view show RUN --step N` on the last step prints the error; `rlmflow serve` logs to `.rlmflow/serve.log`. |
