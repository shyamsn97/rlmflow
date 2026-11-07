"""Prompt pieces the harnesses share."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rlmflow.graph.config import can_spawn

if TYPE_CHECKING:
    from rlmflow.graph.nodes import Step

SKILLS_DIR = Path(__file__).resolve().parent.parent / "plugin" / "skills"

#: What a session harness (Claude Code, pi) is told about working in a run.
TOOLS_PROMPT = """\
You are one agent in a multi-agent run, with a persistent Python REPL.

Run code with `rlmflow repl 'CODE'`, or for several lines:
    rlmflow repl <<'EOF'
    ...
    EOF
Quote the heredoc marker ('EOF') so the shell leaves the code alone. Variables persist
between calls. Your inputs are already variables: {arguments}.
Only printed output comes back, so print summaries, not whole values.

In the REPL:
- launch_agent(task, config=NAME, **args) starts an agent and returns it as an `Agent`.
  `await agent` waits for the result of an agent you launched. An Agent's `.status`,
  `.done()` and `.result()` are as of when the cell started, so they never change inside a loop.
- agent.send(message) messages any agent, reopening it if it finished. For an agent you
  launched, `await agent.send(...)` also waits for its reply; others reply with a message.
- AGENTS["name"] is any agent in the run, by name, path or id. AGENTS.self is you,
  AGENTS.children the agents you launched, and AGENTS.print_graph(show_results=True)
  prints them all with their status and results.
- await llm_query(prompt) asks a model and returns its answer.
- submit(value) records your final result. Call it exactly once, when you're finished.

To wait for agents you started, end your turn without submitting. You'll be resumed
with their results. Don't poll. A cell that awaits long work can hit your tool timeout,
so launch agents and end the turn instead."""

VERIFY_PROMPT = """\
Before submitting, check the answer against the inputs, in code where possible. Do not \
turn missing data, failed reads, tool errors, or an agent's empty or "None" result into \
a plausible answer: surface the failure unless you validate a recovery."""

STATE_PROMPT = """\
Keep your working state in REPL variables, not only in your replies: put intermediate
results in clearly named variables (for example `candidate_files`, `matches_by_chunk`),
and keep a `notes` string with your plan and what you've ruled out. Update `notes`
before waiting on agents. You may later continue from this REPL without your earlier
conversation."""

RECORD_PROMPT = """\
Cells you run with `rlmflow repl` are recorded in this run, so it can be replayed and
forked from them; changes you make with your own tools are not."""

LOCATION_PROMPT = """\
Your REPL runs in {location}. Your own tools run on this machine; reach the REPL's
files and programs through `rlmflow repl`, with paths as the REPL sees them."""

RESUME_PROMPT = """\
You are continuing an earlier session on this task. Its conversation is gone, but its
REPL is intact. Cells run so far (outputs truncated):

{cells}

REPL variables: {variables}
Start with `rlmflow repl 'print(notes)'`."""


@cache
def skill(name: str) -> str:
    """The body of a bundled skill, without its front matter."""
    text = (SKILLS_DIR / name / "SKILL.md").read_text()
    if text.startswith("---"):
        text = text.split("\n---", 1)[1]
    return text.strip()


def guidance(step: Step) -> dict[str, str | None]:
    """How to work well, the same on every harness; each harness adds its own mechanics.

    The orchestration skill goes only to agents that can launch agents.
    """
    return {
        "verify": VERIFY_PROMPT,
        "state": STATE_PROMPT,
        "orchestration": skill("orchestrate") if can_spawn(step.graph, step.agent) else None,
    }


def argument_manifest(step: Step) -> str:
    """Arguments by name and size; the values stay in the REPL."""
    agent = step.agent
    values = agent.resolved_args()
    if not values:
        return ""
    descriptions = agent.argument_descriptions()
    lines = []
    for name, value in values.items():
        if isinstance(value, str):
            size = f"str, {len(value)} chars"
        elif isinstance(value, list):
            size = f"list, {len(value)} items"
        elif isinstance(value, dict):
            size = f"object, {len(value)} fields"
        else:
            size = type(value).__name__
        line = f"- `{name}`: {size}"
        if description := descriptions.get(name):
            line += f" — {' '.join(description.split())}"
        lines.append(line)
    return "Your REPL has these task arguments as direct variables:\n" + "\n".join(lines)


def preview(value: Any, limit: int = 160) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


__all__ = [
    "LOCATION_PROMPT",
    "RECORD_PROMPT",
    "RESUME_PROMPT",
    "SKILLS_DIR",
    "STATE_PROMPT",
    "TOOLS_PROMPT",
    "VERIFY_PROMPT",
    "argument_manifest",
    "guidance",
    "preview",
    "skill",
]
