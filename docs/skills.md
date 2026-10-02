# Customizable Skills

Skills are ordinary repo files that become part of an agent's prompt when they matter. Use them for stable guidance you want to reuse across runs: project style guides, domain playbooks, child-agent contracts, benchmark heuristics, or lessons distilled from previous traces.

rlmflow keeps skills as files. Add them to the system `Prompt` under stable keys so a live library can update one skill without rewriting the protocol. See [`examples/skills.py`](https://github.com/shyamsn97/rlmflow/blob/main/examples/skills.py) for a small runnable version.

## Suggested Layout

```text
skills/
+-- project-style/
|   `-- SKILL.md
+-- numpy-linear-algebra/
|   `-- SKILL.md
+-- child-agent-contract/
|   `-- SKILL.md
`-- run-memory/
    +-- debugging.md
    `-- eval-lessons.md
```

Each `SKILL.md` should be short, concrete, and action-oriented. Prefer rules the agent can follow during a run over long background explanations.

## Always-On Project Skills

Load project conventions into every agent:

```python
from pathlib import Path

import rlmflow
from rlmflow.llm import OpenAIClient


def add_project_skill(prompt, flow=None, node=None):
    skill = Path("skills/project-style/SKILL.md").read_text(encoding="utf-8")
    prompt.add_section("skills.project", skill, order=800)


flow = rlmflow.Flow(
    OpenAIClient(model="gpt-4o-mini"),
    system_prompt_modifiers=add_project_skill,
)
```

## Query-Selected Skills

Choose domain skills from the current task:

```python
from pathlib import Path

import rlmflow
from rlmflow.llm import OpenAIClient

SKILL_DIR = Path("skills")


def _read_skill(name: str) -> str:
    path = SKILL_DIR / name / "SKILL.md"
    if not path.exists():
        return ""
    body = path.read_text(encoding="utf-8").strip()
    return f"### {name}\n{body}"


def add_workspace_skills(prompt, flow=None, node=None):
    agent = None if node is None else node.parent_agent
    query = (agent.content if agent is not None else "").lower()
    skills = [_read_skill("project-style")]
    if "numpy" in query or "linear algebra" in query:
        skills.append(_read_skill("numpy-linear-algebra"))
    if agent is not None and agent.config.depth > 0:
        skills.append(_read_skill("child-agent-contract"))
    extra = "\n\n".join(skill for skill in skills if skill)
    if extra:
        prompt.add_section("skills.workspace", extra, order=800)


flow = rlmflow.Flow(
    OpenAIClient(model="gpt-4o-mini"),
    system_prompt_modifiers=add_workspace_skills,
)
```

## Child-Only Skills

Give spawned agents a tighter contract than the root planner:

```python
from pathlib import Path

import rlmflow
from rlmflow.llm import OpenAIClient


def add_child_contract(prompt, flow=None, node=None):
    agent = None if node is None else node.parent_agent
    if agent is None or agent.config.depth == 0:
        return
    contract = Path("skills/child-agent-contract/SKILL.md").read_text(
        encoding="utf-8"
    )
    prompt.add_section("skills.child_contract", contract, order=800)


flow = rlmflow.Flow(
    OpenAIClient(model="gpt-4o-mini"),
    system_prompt_modifiers=add_child_contract,
)
```

## Run-Memory Skills

Turn lessons from previous runs into reusable guidance:

```python
from pathlib import Path

import rlmflow
from rlmflow.llm import OpenAIClient

MEMORY_DIR = Path("skills/run-memory")


def add_run_memory(prompt, flow=None, node=None):
    blocks = []
    for path in sorted(MEMORY_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8").strip()
        if text:
            blocks.append(f"### {path.stem}\n{text}")
    extra = "\n\n".join(blocks)
    if extra:
        prompt.add_section("memory.run", extra, order=800)


flow = rlmflow.Flow(
    OpenAIClient(model="gpt-4o-mini"),
    system_prompt_modifiers=add_run_memory,
)
```

## Combining Skills With Other Prompt Changes

Skills compose by stable key. Different sources add different keys; a later source updates an existing non-pinned key:

```python
def add_combined_prompt(prompt, flow=None, node=None):
    prompt.add_section(
        "skills.workspace", workspace_skills(flow, node), order=800
    )
    prompt.add_section("memory.run", run_memory(), order=810)
```

For lower-level prompt mechanics, see [`prompt_customization.md`](prompt_customization.md).
