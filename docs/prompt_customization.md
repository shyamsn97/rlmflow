# Prompt customization

An agent's prompt has two parts, and both belong to the agent:

| part          | how to change it                                                               |
| ------------- | ------------------------------------------------------------------------------ |
| system prompt | `Agent(prompt={...})` to change sections, `Agent(system=...)` to replace it all |
| messages      | subclass `Agent` and override `messages(node, repl, since)`                    |

The same named sections are used on every harness: `Agent` sends them as the system prompt, while Claude Code and pi keep their own system prompt and get rlmflow's appended.

## System prompt sections

`sections(node, repl)` returns the default system prompt as a dict, in order, and `system_prompt(node, repl)` merges the agent's `prompt` over it and joins the parts:

```python
parts = {**agent.sections(node, repl), **agent.prompt}
"\n\n".join(text.strip() for text in parts.values() if text and text.strip())
```

So a `prompt` entry does one of four things by name:

- a string under an existing name replaces that section, in place;
- `None` drops it;
- a new name is added at the end;
- a function of `(node, repl)` is called every step, for text that changes during the run.

```python
from rlmflow import Agent, Flow

reviewer = Agent(
    model="gpt-6.1-terra",
    prompt={
        "agents": None,  # drop the AGENTS section
        "style": "Report only actionable findings, most severe first.",  # added last
        "skills": lambda node, repl: library.render(),  # recomputed every step
    },
)
flow = Flow(agents={"reviewer": reviewer})
```

To own the whole system prompt, pass `system`; the sections and `prompt` are then ignored:

```python
Agent(model="gpt-6.1-terra", system=SOKOBAN_RULES)
```

Override `sections` or `system_prompt` only for logic a dict can't hold. [`examples/features/skills.py`](https://github.com/shyamsn97/rlmflow/blob/main/examples/features/skills.py) uses a function section.

## Messages

`messages(node, repl, since=None)` is what the agent sees each step: the system prompt, then the run's history up to `node` (after `since`, for a session that already has the rest). It ends the last user message with the runs the agent launched and how they are, and a notice on its last step. Override it for whatever the agent should see every step, such as a board, a budget, or files that change during the run:

```python
from rlmflow import Agent
from rlmflow.agents import steps


class Budgeted(Agent):
    def messages(self, node, repl, since=None):
        left = self.max_iters - steps(node)
        return [*super().messages(node, repl, since), {"role": "user", "content": f"{left} steps left."}]
```

`Agent` sends all of it on each model call. Claude Code and pi append the system prompt to the session's own on each step, and send only the messages since the session's last step as its next turn. With `keep` set, `Agent` keeps only the last `keep` history messages and puts a notice in place of the rest.

## Default sections

| name          | what it says                                                                                                |
| ------------- | ----------------------------------------------------------------------------------------------------------- |
| `protocol`    | `Agent`: one `repl(code)` tool call per reply; `submit(value)`, or a reply without a call, finishes. Claude Code and pi: running code with `rlmflow repl`, and how to wait for agents |
| `location`    | Claude Code and pi: where the REPL runs, for Docker and Modal runtimes                                     |
| `tools`       | the REPL tools this agent can call (for Claude Code and pi, only the Flow's own)                           |
| `agents`      | `Agent`: agent handles and `AGENTS`. Claude Code and pi: the agents `launch_agent` takes                   |
| `verify`      | check the answer against the inputs; surface failures, don't paper over them                               |
| `state`       | keep working state in REPL variables                                                                        |
| `orchestrate` | the orchestrate skill, for agents that can launch others                                                   |
| `arguments`   | the task arguments by name and size; their values are REPL variables                                       |
| `schema`      | the output schema, when the agent has one                                                                   |

## What the prompt doesn't change

An agent's behaviour is in its `step()`, not its prompt. `Agent` offers the model one tool, `repl(code)`, runs the code of exactly one call per reply, and takes a reply without a call as the final answer, whatever the prompt says, so a prompt that replaces `protocol` has to explain that to the model.
