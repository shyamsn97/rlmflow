# Agents

What an agent does once a run starts: the REPL it works in, how it launches and waits for other agents, and what a Claude Code or pi session sees. How to start and configure runs is in [Usage](usage.md).

## The REPL

Every agent has a persistent Python REPL. Its task is the variable `task`, and each keyword argument it was started with is a variable too. The prompt lists them by name, type and size, so a large input costs nothing until the agent reads it.

With the default `repl` harness, each model turn is one fenced ```` ```repl ```` block, which runs as one cell. An agent finishes by calling `submit(value)`, or by replying in plain text with no `repl` block. A reply with several blocks, an empty block or an unclosed one runs nothing; the agent sees why and tries again.

Limits come from its config: `max_iters` model turns, `max_depth` levels of agents below it, and `output_schema`, which the result is checked against. When the run crosses `max_budget` tokens, every agent gets one final turn.

## What's in the REPL

| Name                                                               | What it does                                                                      |
| ------------------------------------------------------------------ | --------------------------------------------------------------------------------- |
| `launch_agent(task, config=NAME, name=None, **args)`               | Start an agent; returns the `Agent` at once. `**args` become its variables.       |
| `await agent`, `await asyncio.gather(*agents)`                     | Wait for the results of agents you launched (or they launched).                   |
| `agent.send(message)`                                              | Message any agent, reopening it if it finished. For one you launched, `await` it for the reply. |
| `agent.status`, `agent.done()`, `agent.result()`                   | Check an agent without waiting.                                                   |
| `agent.parent`, `agent.children`, `agent.path`                     | Where it sits in the run; the same names as `g.parent(agent)` and so on.          |
| `llm_query(prompt, model=..., schema=...)`                         | One model call with no history; with `schema`, a validated JSON value. Awaitable. |
| `submit(value)`                                                    | Finish with a string or JSON value.                                               |
| `AGENTS`                                                           | Every agent in the run: `AGENTS["name"]` (name, path or id), `AGENTS.self`, `AGENTS.children`, `AGENTS.print_graph()`. |
| `Input(value, description=...)`                                    | Describe an argument's structure in the child's prompt.                           |

Tools passed to the Flow (`FILE_TOOLS`, your `@tool` functions) and values from `flow.inject` are also there, and listed in the prompt.

## Launching and waiting

A fan-out, as an agent writes it:

```python
chunks = [doc[i : i + 100_000] for i in range(0, len(doc), 100_000)]
workers = [
    launch_agent("Find the magic number in chunk.", config="worker", chunk=c) for c in chunks
]
results = await asyncio.gather(*workers)
submit(next(r for r in results if r))
```

Children can launch their own, up to `max_depth`, so one task becomes a tree. Launching and waiting are recorded on the cell that did them.

An `Agent`'s `status`, `done()`, `result()`, `parent` and `children` read the run as it was when the cell started, so an agent launched in this cell appears in `AGENTS` from the next one. In the cell that launched or awaited an agent, use the value `await agent` returns; in later cells, `.result()` is current. The status values are the host's: `"done"`, `"idle"` (a session waiting for messages) and `"running"`.

An agent can await only its descendants. Awaiting anyone else raises `NotYourAgent`: message them instead, and their reply arrives as a message at your next turn.

Describe a structured argument without copying it into the prompt, and ask for structured answers:

```python
child = launch_agent(
    "Label each record.", rows=Input(rows, description="list of {user_id, text} dicts")
)
facts = await llm_query(
    f"Extract the date and amount from: {email}",
    schema={
        "type": "object",
        "properties": {"date": {"type": "string"}, "amount": {"type": "number"}},
        "required": ["date", "amount"],
    },
)
```

## Messages

Agents message each other with `agent.send(message)`, and host code with `g.send(agent, text)`. A message waits until the agent's next turn and arrives as its next prompt; a `message` edge records which cell sent it. A finished agent that gets a message runs again with the same REPL, so for an agent you launched `await agent.send(...)` returns its next result: its reply. A review loop and a team of peers are both built on this:

```python
writer = launch_agent("Write a four-line poem about a lighthouse.", name="writer")
poem = await writer
poem = await writer.send("Tighten line two.")

AGENTS["backend"].send("I need GET /items to return ids.")  # a peer, by name; it replies by message
```

To pass one agent's result to another, await it and pass the value: `launch_agent("Review `code`.", code=await coder)`. Passing the agent itself is an error.

## Shared state

There is no built-in store. Agents share state through messages, through files in a shared working directory, or through host tools you write: a tool runs in your process, so a dict behind a lock is a store every agent sees. The [orchestrate skill](../rlmflow/plugin/skills/orchestrate/SKILL.md) has a task board built that way.

## Claude Code and pi sessions

With the `claude_code` or `pi` harness, each step of the agent is one headless session turn. The session reaches the same REPL with `rlmflow repl 'CODE'` (or `rlmflow repl -f file.py`, or a heredoc on stdin), and every call lands a `code` node. That is its only way in: `launch_agent`, `AGENTS`, `agent.send` and `submit` are called inside the REPL, as in our own harness. With `mcp=True` the same REPL is also its `python` MCP tool.

```text
$ rlmflow repl <<'EOF'
files = sorted(os.listdir("haystack"))
searchers = [launch_agent("Search these files.", config="searcher", files=files[i:i + 50]) for i in range(0, len(files), 50)]
notes = "launched searchers over 50-file batches"
EOF
```

The session's system prompt explains these commands, lists the configs it can launch and the Flow's own tools in the REPL, and asks it to keep its working state in REPL variables and a `notes` string. It also says that cells run through `rlmflow repl` are recorded in the run and changes made with its own tools are not, and, when the runtime is a Docker container or Modal sandbox, where the REPL runs and which paths it sees. Its own tools and permissions are whatever your Claude Code (or pi) settings allow; rlmflow only adds permission for `rlmflow` commands.

To wait for agents it launched, the session ends its turn without submitting; the agent lands `idle`. When they finish, the same session resumes (`claude --resume`) with their results. If it ends a turn with nothing to wait for and no `submit`, its last reply is the result.

## Your own Claude Code session

After `/rlmflow:run TASK`, your session is the root agent:

1. The plugin starts `rlmflow serve` for the project and creates the root agent with your task. The serve builds the project's Flow (the coding one, or `flow = "module:factory"` in `rlmflow.toml`) and registers an `interactive` config for your session on it.
1. Your session is given the same instructions as a harness session and works through `rlmflow repl`.
1. When it ends a turn while its workers run, the plugin's Stop hook waits for their results (up to about nine minutes) and hands them back to the session, which carries on.
1. Your later messages are messages to the root. Exiting the session stops the serve and saves the run.

## Forks

A fork is a new agent that shares another agent's history up to one node. Its REPL is rebuilt by rerunning the recorded cells; launches and `llm_query` calls in those cells return their recorded results, so finished workers aren't run again and the fork holds the same `Agent`s.

A Claude Code fork at the end of a session turn resumes that conversation as a new branch. A fork mid-turn starts a fresh session, briefed on the cells so far and the variables in its rebuilt REPL, and starts by reading `notes`.
