# Skills

## The bundled skill

rlmflow ships one skill, [`orchestrate`](https://github.com/shyamsn97/rlmflow/blob/main/rlmflow/plugin/skills/orchestrate/SKILL.md), which the Claude Code plugin loads. It covers the ways to split work across agents (subagents, fan-out, recursive trees, pipelines, review loops, peers, a DeLM-style task board on the shared store), with code and when each pays off, and the same shapes built from host code. Every agent that can launch others gets it as the `orchestrate` prompt section, on every harness. Drop it with `prompt={"orchestrate": None}`.

## Your own skills

Skills are ordinary files that become part of an agent's prompt: project style guides, domain playbooks, a contract for workers, lessons from earlier runs. Each one goes in a prompt section of its own, so it can change without touching the rest of the prompt ([Prompt customization](prompt_customization.md)).

### Fixed skills: a text section

When the skill text is known before the run, put it in the agent's `prompt`. It applies to every run of that agent:

```python
from pathlib import Path

from rlmflow import Agent, Flow

style = Path("skills/project-style/SKILL.md").read_text()
contract = Path("skills/worker-contract/SKILL.md").read_text()

flow = Flow(
    agents={
        "default": Agent(model="gpt-6-sol", prompt={"skills": style}),
        "worker": Agent(model="gpt-6.1-terra", prompt={"skills": f"{style}\n\n{contract}"}),
    }
)
```

Different agents get different skills, so a worker-only contract is a section on the worker agent.

### Live skills: a function section

When the files can change during the run, make the section a function of `(node, repl)`; it is called every step:

```python
from pathlib import Path

from rlmflow import Agent


def skills(node, repl):
    directory = Path("skills/run-memory")
    return "\n\n".join(
        f"### {path.stem}\n\n{path.read_text()}" for path in sorted(directory.glob("*.md"))
    )


Agent(model="gpt-6.1-terra", prompt={"skills": skills})
```

[`examples/features/skills.py`](https://github.com/shyamsn97/rlmflow/blob/main/examples/features/skills.py) reads skill files from a directory this way, and adds an `add_skill` tool that writes into the same directory, so the agent sees the skill it installed in its own prompt on the next turn.
