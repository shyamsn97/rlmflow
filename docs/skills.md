# Customizable Skills

Skills are ordinary repo files that become part of an agent's prompt when they
matter. Use them for stable guidance you want to reuse across runs: project
style guides, domain playbooks, child-agent contracts, benchmark heuristics, or
lessons distilled from previous traces.

rlmflow keeps skills as files. Add a callable prompt section that decides which
files belong in the current agent's context. See
[`examples/skills.py`](../examples/skills.py) for a small runnable version.

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

Each `SKILL.md` should be short, concrete, and action-oriented. Prefer rules the
agent can follow during a run over long background explanations.

## Always-On Project Skills

Load project conventions into every agent:

```python
from pathlib import Path

import rflow
from rflow import SystemPromptBuilder


def project_skill(flow: rflow.Flow, graph: rflow.Graph) -> str:
    return Path("skills/project-style/SKILL.md").read_text(encoding="utf-8")


flow = rflow.Flow(rflow.OpenAIClient(model="gpt-4o-mini"))
prompt = SystemPromptBuilder()
prompt.sections.add("project_skill", project_skill, title="Project Skill", before="tools")
flow.system_prompt = prompt
```

## Query-Selected Skills

Choose domain skills from the current task:

```python
from pathlib import Path

import rflow
from rflow import SystemPromptBuilder

SKILL_DIR = Path("skills")


def _read_skill(name: str) -> str:
    path = SKILL_DIR / name / "SKILL.md"
    if not path.exists():
        return ""
    body = path.read_text(encoding="utf-8").strip()
    return f"### {name}\n{body}"


def workspace_skills(flow: rflow.Flow, graph: rflow.Graph) -> str:
    query = graph.query.lower()
    skills = [_read_skill("project-style")]

    if "numpy" in query or "linear algebra" in query:
        skills.append(_read_skill("numpy-linear-algebra"))
    if graph.depth > 0:
        skills.append(_read_skill("child-agent-contract"))

    return "\n\n".join(skill for skill in skills if skill)


flow = rflow.Flow(rflow.OpenAIClient(model="gpt-4o-mini"))
prompt = SystemPromptBuilder()
prompt.sections.add(
    "workspace_skills", workspace_skills, title="Workspace Skills", before="tools"
)
flow.system_prompt = prompt
```

## Child-Only Skills

Give spawned agents a tighter contract than the root planner:

```python
from pathlib import Path

import rflow
from rflow import SystemPromptBuilder


def child_contract(flow: rflow.Flow, graph: rflow.Graph) -> str:
    if graph.depth == 0:
        return ""
    return Path("skills/child-agent-contract/SKILL.md").read_text(encoding="utf-8")


flow = rflow.Flow(rflow.OpenAIClient(model="gpt-4o-mini"))
prompt = SystemPromptBuilder()
prompt.sections.add(
    "child_contract", child_contract, title="Child Agent Contract", after="strategy"
)
flow.system_prompt = prompt
```

## Run-Memory Skills

Turn lessons from previous runs into reusable guidance:

```python
from pathlib import Path

import rflow
from rflow import SystemPromptBuilder

MEMORY_DIR = Path("skills/run-memory")


def run_memory(flow: rflow.Flow, graph: rflow.Graph) -> str:
    blocks = []
    for path in sorted(MEMORY_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8").strip()
        if text:
            blocks.append(f"### {path.stem}\n{text}")
    return "\n\n".join(blocks)


flow = rflow.Flow(rflow.OpenAIClient(model="gpt-4o-mini"))
prompt = SystemPromptBuilder()
prompt.sections.add("run_memory", run_memory, title="Run Memory", before="examples")
flow.system_prompt = prompt
```

## Combining Skills With Other Prompt Changes

Skills are prompt sections, so they compose with the rest of the prompt builder —
just keep editing `.sections`:

```python
prompt = SystemPromptBuilder()
prompt.sections.add(
    "workspace_skills", workspace_skills, title="Workspace Skills", before="tools"
)
prompt.sections.add("run_memory", run_memory, title="Run Memory", before="examples")
flow.system_prompt = prompt
```

For lower-level prompt mechanics, see
[`prompt_customization.md`](prompt_customization.md).
