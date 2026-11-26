# Agents

What an agent does once a run starts: the REPL it works in, how it launches and waits for other agents, and what a Claude Code or pi session sees. How to start and configure runs is in [Usage](usage.md).

## The REPL

Every agent has a persistent Python REPL. Its task is the variable `task`, and each keyword argument it was started with is a variable too. The prompt lists them by name, type and size, so a large input costs nothing until the agent reads it.

With the default `Agent`, the model has one tool, `repl(code)`, and each model turn that calls it runs that code in the REPL; what the code prints comes back as the tool result. The functions the prompt lists under Tools are already defined in the REPL, and the model calls them from that code. An agent finishes by calling `submit(value)` in its code, or by replying in plain text without calling `repl`. A reply that calls `repl` more than once, with empty or missing code, or calls another tool name runs nothing; the agent sees why and tries again.

Limits come from its `Agent`: `max_iters` steps, `max_depth` levels of agents below it, and `output_schema`, which the result is checked against. `reasoning_effort` configures that agent's OpenAI calls; its default, `"auto"`, means `"low"` for GPT-6 and the provider default otherwise, while `None` always uses the provider default. When a run's `Budget` is nearly spent, every agent gets one final turn.

## What's in the REPL

| Name                                                 | What it does                                                                                                                     |
| ---------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `launch_agent(task, agent=NAME, name=None, **args)`  | Start an agent; returns a handle at once. `**args` become its variables. Without `agent=`, one of your own kind (a Claude Code session through the plugin starts `default` agents). |
| `await agent`, `await asyncio.gather(*agents)`       | Wait for the results of agents you launched (or they launched).                                                                  |
| `agent.send(message, **inputs)`                      | Message any agent, reopening it if it finished; `**inputs` become its variables. For one you launched, `await` it for the reply. |
| `agent.status`, `agent.done()`, `agent.result()`     | Check an agent without waiting.                                                                                                  |
| `agent.parent`, `agent.children`, `agent.path`       | Where it sits in the run.                                                                                                       |
| `llm_query(prompt, model=..., schema=...)`           | One model call with no conversation; with `schema`, a validated JSON value. Awaitable.                                           |
| `submit(value)`                                      | Finish with a string or JSON value.                                                                                              |
| `AGENTS`                                             | Every agent in the run: `AGENTS["name"]` (name, path or id), `AGENTS.self`, `AGENTS.children`, `AGENTS.print_graph()`.           |
| `Input(value, description=...)`                      | Describe an argument's structure in the child's prompt.                                                                          |

Tools passed to the Flow (`FILE_TOOLS`, your `@tool` functions) and values from `flow.inject` are also there, and listed in the prompt.

## Launching and waiting

A fan-out, as an agent writes it:

```python
chunks = [doc[i : i + 100_000] for i in range(0, len(doc), 100_000)]
workers = [
    launch_agent("Find the magic number in chunk.", agent="worker", chunk=c) for c in chunks
]
results = await asyncio.gather(*workers)
submit(next(r for r in results if r))
```

Children can launch their own, up to `max_depth`, so one task becomes a tree. Launching and waiting are recorded on the node whose code did them.

It works like `asyncio` tasks: `launch_agent` is `create_task`, and a handle is awaited like a task.

### What `launch_agent` does

- It starts the agent and returns its handle at once. The agent runs in parallel from that moment, even while the code that launched it is still running.
- `agent` defaults to the launcher's own agent. The name defaults to `<agent>-<n>`, e.g. `default-0`. `**args` become variables in its REPL.
- Its files start as a copy of the launcher's; `workspace="shared"` gives it the launcher's own directory instead.
- It raises in the launcher's code past the launcher's `max_depth`, for a task over `MAX_TASK` characters, or when a hook refuses it (a spent `Budget`).

### How a result reaches you

1. **`await agent`.** It waits for that agent in the same block and returns its result; `await asyncio.gather(*agents)` waits for several. Awaiting in a later turn an agent that already finished returns its result at once.
2. **Not awaited, while you keep working.** Before your next turn, the agents that finished since your last one reach you in one message, "Agents you started have finished", with each result. Each result is also bound as a variable named after its agent (`default-0` becomes `default_0`). The agents still running are listed at the end of your messages.
3. **You finish first (fire and forget).** A plain-text reply from the REPL agent, or `submit(value)` from any harness, ends your run at once, even with agents you launched still running. They run to the end: `flow.run` and `flow.stream` return only when every run is done. Their results stay on their runs. If a message later reopens you, they reach you at your next turn, as in 2.
4. **A Claude Code, pi or person session ends its turn without `submit`.** With agents it launched still running, it waits and is resumed with their results. That is how a session waits, since a long `await` in `rlmflow repl` can hit its tool timeout.

How a step that ran no code ends, by harness:

| Harness and step | Agents it launched still running | None running |
| --- | --- | --- |
| REPL agent, plain-text reply | finishes; they keep going | finishes |
| Claude Code, pi or person, turn ends | waits; resumed with their results | finishes with its last reply |
| Any harness, `submit(value)` | finishes; they keep going | finishes |

### Agents you didn't launch

An agent can await only its descendants: the agents it launched and the ones they launched. Awaiting anyone else, a sibling or a parent, raises `NotYourAgent`; two agents awaiting each other would wait forever. To get something from another agent:

- **Read its result**, once it's done: `AGENTS["B"].result()`. `AGENTS["B"].done()` says whether it is.
- **Ask it.** `AGENTS["B"].send("What total did you get?")`, then finish the block or end the turn. It reads the message at its next turn (reopening if it had finished) and answers with `AGENTS["A"].send(...)`, which you read at your next turn.
- **Take its files**: `AGENTS["B"].commit` is its files after its last code ran, and `checkout(...)` replaces yours with them.

A handle's `status`, `done()`, `result()`, `parent` and `children` read the run as it was when the code started, so an agent launched in this turn appears in `AGENTS` from the next one, and an agent that finishes during a block still reads as running in that block. In the code that launched or awaited an agent, use the value `await agent` returns; in later turns, `.result()` is current. The status values are the host's: `"done"`, `"idle"` (a session waiting for messages) and `"running"`.

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

Agents message each other with `agent.send(message, **inputs)`, and host code with `g.send(agent, text, **inputs)`. A message is a node added to the agent: the agent reads its text at its next turn, and its inputs become variables in its REPL. A `write` edge records which node sent it. A finished agent that gets a message runs again with the same REPL, so for an agent you launched `await agent.send(...)` returns its next result: its reply. A review loop and a team of peers are both built on this:

```python
writer = launch_agent("Write a four-line poem about a lighthouse.", name="writer")
poem = await writer
poem = await writer.send("Tighten line two.")

AGENTS["backend"].send(
    "I need GET /items to return ids.", fields=["id"]
)  # a peer; it replies by message
```

To pass one agent's result to another, await it and pass the value: `launch_agent("Review `code`.", code=await coder)`. Passing the agent itself is an error.

## Shared state

There is no built-in store. Agents share state through messages, through files in a shared working directory, or through host tools you write: a tool runs in your process, so a dict behind a lock is a store every agent sees. The [orchestrate skill](https://github.com/shyamsn97/rlmflow/blob/main/rlmflow/plugin/skills/orchestrate/SKILL.md) has a task board built that way.

## Claude Code and pi sessions

With the `claude_code` or `pi` harness, each newly launched agent starts its own fresh native session; a child never inherits its parent's conversation. Each step is one headless session turn, and later turns resume that same agent's session. The session reaches the same REPL with `rlmflow repl 'CODE'` (or `rlmflow repl -f file.py`, or a heredoc on stdin), and every call lands a node with that code. That is its only way in: `launch_agent`, `AGENTS`, `agent.send` and `submit` are called inside the REPL, as in our own harness. With `mcp=True` the same REPL is also its `python` MCP tool.

```text
$ rlmflow repl <<'EOF'
files = sorted(os.listdir("haystack"))
searchers = [launch_agent("Search these files.", agent="searcher", files=files[i:i + 50]) for i in range(0, len(files), 50)]
notes = "launched searchers over 50-file batches"
EOF
```

The session's system prompt explains these commands, lists the Flow agents it can launch and the Flow's own tools in the REPL, and asks it to keep its working state in REPL variables and a `notes` string. It also says that code run through `rlmflow repl` is recorded in the run and changes made with its own tools are not, and, when the runtime is a Docker container or Modal sandbox, where the REPL runs and which paths it sees. Its own tools and permissions are whatever your Claude Code (or pi) settings allow; rlmflow only adds permission for `rlmflow` commands.

To wait for agents it launched, the session ends its turn without submitting; its turn ends `idle`. When they finish, the same session resumes (`claude --resume`) with their results. If it ends a turn with nothing to wait for and no `submit`, its last reply is the result.

## Your own Claude Code session

After `/rlmflow:run TASK`, your session is the root agent:

1. The plugin starts `rlmflow serve` for the project and creates the root agent with your task. The server builds the project's Flow (the coding one, or `flow = "module:factory"` in `rlmflow.toml`) and registers an `interactive` agent for your session.
1. Your session is given the same instructions as a harness session and works through `rlmflow repl`.
1. When it ends a turn while its workers run, the plugin's Stop hook waits for their results (up to about nine minutes) and hands them back to the session, which carries on.
1. Your later messages are messages to the root. Exiting the session stops the serve and saves the run. If the root already submitted, agents still running in the background finish first, and the serve saves the run when they're done. If it hadn't, they stop with the session and stay unfinished in the saved run.

## Forks

A fork is a copy of the run cut after one node (`g.fork(node)`, or `rlmflow fork RUN NODE`); the original is left alone. An agent in the copy gets its REPL rebuilt by running its nodes again; launches and `llm_query` calls in it return their recorded results, so finished workers aren't run again and the copy holds copies of them. Workers launched at or after the fork node aren't copied.

A Claude Code agent forked at the end of a session turn resumes that conversation as a new branch (`--fork-session`). Forked mid-turn, it starts a fresh session, briefed on the code run so far and the variables in its rebuilt REPL, and starts by reading `notes`.
