# Skills

Skills are ordinary files that become part of an agent's prompt: project style guides, domain playbooks, a contract for workers, lessons from earlier runs. Each one goes in a prompt section of its own, so it can change without touching the rest of the prompt ([Prompt customization](prompt_customization.md)).

## Fixed skills: a config section

When the skill text is known before the run, put it in the config's `prompt`. It applies to every agent launched with that config and is saved with the run:

```python
from pathlib import Path

from rlmflow import AgentConfig, Flow

style = Path("skills/project-style/SKILL.md").read_text()
contract = Path("skills/worker-contract/SKILL.md").read_text()

flow = Flow(
    configs=[
        AgentConfig(model="gpt-5", prompt={"skills": style}),
        AgentConfig(name="worker", model="gpt-5-mini", prompt={"skills": f"{style}\n\n{contract}"}),
    ]
)
```

Different configs get different skills, so a worker-only contract is a section on the worker config.

## Live skills: a harness section

When the files can change during the run, read them each turn in a harness subclass. Put their location in `harness_options`, which the run records:

```python
from dataclasses import dataclass
from pathlib import Path

from rlmflow import AgentConfig
from rlmflow.harnesses.repl import ReplHarness


@dataclass(frozen=True)
class SkillsOptions:
    directory: str


class SkillsHarness(ReplHarness):
    Options = SkillsOptions

    def prompt_sections(self, flow, step):
        directory = Path(self.options(step).directory)
        skills = "\n\n".join(f"### {path.stem}\n\n{path.read_text()}" for path in sorted(directory.glob("*.md")))
        return {**super().prompt_sections(flow, step), "skills": skills or None}


AgentConfig(model="gpt-5-mini", harness=SkillsHarness, harness_options={"directory": "skills/run-memory"})
```

[`examples/skills.py`](https://github.com/shyamsn97/rlmflow/blob/main/examples/skills.py) loads `SKILL.md` files this way. [`examples/dynamic_skills.py`](https://github.com/shyamsn97/rlmflow/blob/main/examples/dynamic_skills.py) adds an `add_skill` tool that writes into the directory, so the agent sees the skill it installed in its own prompt on the next turn.
