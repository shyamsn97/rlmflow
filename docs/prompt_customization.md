# Prompt customization

Every harness builds its system prompt from named sections. `prompt_sections(flow, step)` returns them as a dict, in order, and `prompt(flow, step)` puts the config's `prompt` overrides on top and joins them:

```python
sections = {**harness.prompt_sections(flow, step), **step.agent.config.prompt}
"\n\n".join(text.strip() for text in sections.values() if text and text.strip())
```

So a config override does one of three things by key:

- a string under an existing key replaces that section, in place;
- `None` drops it;
- a new key is added at the end.

## From config

`AgentConfig.prompt` is plain JSON, so it's saved with the run, works from `rlmflow.toml`, and applies to every agent launched with that config:

```python
from rlmflow import AgentConfig, Flow

reviewer = AgentConfig(
    name="reviewer",
    model="gpt-5-mini",
    prompt={
        "agents": None,                                      # drop the AGENTS section
        "style": "Report only actionable findings, most severe first.",  # added last
    },
)
flow = Flow(configs=[reviewer])
```

## From code

Subclass the harness when the text depends on the step, such as a board, a budget, or files that change during the run:

```python
from rlmflow.harnesses.repl import ReplHarness, llm_turns


class BudgetHarness(ReplHarness):
    def prompt_sections(self, flow, step):
        left = step.agent.config.max_iters - llm_turns(step)
        return {**super().prompt_sections(flow, step), "budget": f"{left} turns left."}


AgentConfig(name="worker", model="gpt-5-mini", max_iters=20, harness=BudgetHarness)
```

The config's `prompt` still applies on top of a subclass's sections. To own the whole prompt, return only your own sections:

```python
class RulesOnly(ReplHarness):
    def prompt_sections(self, flow, step):
        return {}


AgentConfig(name="worker", model="gpt-5-mini", harness=RulesOnly, prompt={"rules": RULES})
```

Or override `prompt(flow, step)` to return the final string, which skips the config's overrides as well. Settings a subclass needs can go in `harness_options`, read with `self.options(step)`; see [`examples/skills.py`](https://github.com/shyamsn97/rlmflow/blob/main/examples/skills.py) and [`examples/dynamic_skills.py`](https://github.com/shyamsn97/rlmflow/blob/main/examples/dynamic_skills.py).

## Default sections

The REPL harness (`rlmflow.harnesses.repl`):

| key         | what it says                                                                                   |
| ----------- | ---------------------------------------------------------------------------------------------- |
| `protocol`  | one fenced `repl` block per turn; `submit(value)`, or a reply with no block, finishes          |
| `verify`    | check the answer against the inputs before submitting; surface failures, don't paper over them |
| `tools`     | the REPL tools this agent can call, including the configs `launch_agent` takes                 |
| `agents`    | `AGENTS`, `send_agent` and `ask_agent`                                                         |
| `arguments` | the task arguments by name and size; their values are REPL variables                           |
| `schema`    | the output schema, when the config has one                                                     |

The session harnesses (`claude_code`, `pi`, `interactive`) append their sections to the session's own system prompt:

| key        | what it says                                                                 |
| ---------- | ---------------------------------------------------------------------------- |
| `protocol` | running code with `rlmflow repl`, the REPL tools, and how to wait for agents |
| `configs`  | the configs `launch_agent` takes                                             |
| `tools`    | the Flow's own REPL tools                                                    |
| `location` | where the REPL runs, for Docker and Modal runtimes                           |
| `state`    | keeping working state in REPL variables                                      |
| `record`   | which changes the run records                                                |
| `schema`   | the output schema, when the config has one                                   |

## What the prompt doesn't change

The harness's behaviour is in its `step()`, not its prompt. The REPL harness runs exactly one fenced `repl` block per reply and takes a reply with no block as the final answer, whatever the prompt says, so a prompt that replaces `protocol` has to explain that to the model.

Turn-level notes aren't part of the system prompt either. The final-turn notice, the list of background agents, and the truncation notice for `keep_n_messages` are added to the messages.
