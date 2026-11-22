# Prompt customization

An agent's prompt has three parts:

| part          | who owns it | how to change it                                   |
| ------------- | ----------- | -------------------------------------------------- |
| system prompt | the harness | `AgentConfig(prompt={...})`, or a harness subclass |
| guidance      | the Flow    | override `Flow.guidance(step)`                     |
| turn prompt   | the Flow    | override `Flow.turn_prompt(step)`                  |

The harness owns the system prompt because it describes the harness's own mechanics: the REPL harness writes the whole system prompt, while Claude Code and pi keep their own and get rlmflow's sections appended. Guidance (how to work well) and the turn prompt (what the agent is told each turn) are the same on every harness, so they belong to the Flow.

## System prompt sections

Every harness builds its system prompt from named sections. `prompt_sections(flow, step)` returns them as a dict, in order, including the Flow's guidance, and `prompt(flow, step)` puts the config's `prompt` overrides on top and joins them:

```python
sections = {**harness.prompt_sections(flow, step), **step.config.prompt}
"\n\n".join(text.strip() for text in sections.values() if text and text.strip())
```

So a config override does one of three things by key:

- a string under an existing key replaces that section, in place;
- `None` drops it;
- a new key is added at the end.

`AgentConfig.prompt` is plain JSON, so it's saved with the run, works from `rlmflow.toml`, and applies to every agent launched with that config:

```python
from rlmflow import AgentConfig, Flow

reviewer = AgentConfig(
    name="reviewer",
    model="gpt-5-mini",
    prompt={
        "agents": None,  # drop the AGENTS section
        "style": "Report only actionable findings, most severe first.",  # added last
    },
)
flow = Flow(configs=[reviewer])
```

To own the whole system prompt, subclass the harness and return only your own sections:

```python
from rlmflow.harnesses.repl import ReplHarness


class RulesOnly(ReplHarness):
    def prompt_sections(self, flow, step):
        return {}


AgentConfig(name="worker", model="gpt-5-mini", harness=RulesOnly, prompt={"rules": RULES})
```

Or override `prompt(flow, step)` to return the final string, which skips the config's overrides as well. Settings a subclass needs can go in `harness_options`, read with `self.options(step)`; see [`examples/features/skills.py`](https://github.com/shyamsn97/rlmflow/blob/main/examples/features/skills.py).

## Guidance

`Flow.guidance(step)` returns the sections every harness includes: `verify`, `state`, and `orchestration` (the orchestrate skill, only for agents that can launch others). Override it to change them for every agent the Flow runs, on any harness:

```python
class TerseFlow(Flow):
    def guidance(self, step):
        return {**super().guidance(step), "state": None, "style": "Be terse."}
```

The config's `prompt` still applies on top.

## Turn prompt

`Flow.turn_prompt(step)` is what the agent is told at the start of each turn, after anything it was sent. By default it lists the agents it launched and whether their results are ready. Override it for whatever changes between turns, such as a board, a budget, or files that change during the run:

```python
from rlmflow.harnesses.repl import turns


class BudgetFlow(Flow):
    def turn_prompt(self, step):
        left = step.config.max_iters - turns(step)
        return "\n\n".join(p for p in (super().turn_prompt(step), f"{left} turns left.") if p)
```

The REPL harness appends it to the last user message of each request. Claude Code and pi append it to the message that opens each turn. The REPL harness also appends its own final-turn notice after it, and replaces dropped turns with a notice when `keep_n_messages` is set.

## Default sections

The REPL harness (`rlmflow.harnesses.repl`):

| key             | what it says                                                                           |
| --------------- | -------------------------------------------------------------------------------------- |
| `protocol`      | one fenced `repl` block per turn; `submit(value)`, or a reply with no block, finishes  |
| `tools`         | the REPL tools this agent can call, including the configs `launch_agent` takes         |
| `agents`        | agent handles and `AGENTS`                                                             |
| `verify`        | guidance: check the answer against the inputs; surface failures, don't paper over them |
| `state`         | guidance: keep working state in REPL variables                                         |
| `orchestration` | guidance: the orchestrate skill, for agents that can launch others                     |
| `arguments`     | the task arguments by name and size; their values are REPL variables                   |
| `schema`        | the output schema, when the config has one                                             |

The session harnesses (`claude_code`, `pi`, `interactive`) append their sections to the session's own system prompt:

| key                                | what it says                                                                 |
| ---------------------------------- | ---------------------------------------------------------------------------- |
| `protocol`                         | running code with `rlmflow repl`, the REPL tools, and how to wait for agents |
| `configs`                          | the configs `launch_agent` takes                                             |
| `tools`                            | the Flow's own REPL tools                                                    |
| `location`                         | where the REPL runs, for Docker and Modal runtimes                           |
| `record`                           | which changes the run records                                                |
| `verify`, `state`, `orchestration` | the Flow's guidance, as above                                                |
| `schema`                           | the output schema, when the config has one                                   |

## What the prompt doesn't change

The harness's behaviour is in its `step()`, not its prompt. The REPL harness runs exactly one fenced `repl` block per reply and takes a reply with no block as the final answer, whatever the prompt says, so a prompt that replaces `protocol` has to explain that to the model.
