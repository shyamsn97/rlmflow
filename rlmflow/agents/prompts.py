"""What agents are told: the prompt text every harness shares, and the pieces built from a run."""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

from rlmflow.graph.nodes import Node

SKILLS_DIR = Path(__file__).resolve().parent.parent / "plugin" / "skills"

#: The one tool rlmflow's own harness acts through.
REPL_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "repl",
        "description": "Run Python in your persistent REPL. Returns what it printed.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python to run. Top-level `await` works."}
            },
            "required": ["code"],
            "additionalProperties": False,
        },
    },
}

#: How rlmflow's own harness acts: one ``repl`` call per reply.
PROTOCOL = """\
You work through a persistent Python REPL, then give the final result.

To act, call the `repl` tool with Python code. It runs in your REPL, and what it prints comes back as the result. The functions listed under Tools are already defined in that REPL: call them from the code you pass to `repl`, which is how you read files, launch agents and do the work. Never answer that you cannot access files, tools or agents: run code and find out.

Call `repl` once per response, and the run continues from its result. A response that doesn't call `repl` is your final answer: its text completes the run. Plan the immediate next action from results already obtained, then execute it. Compute answers from your inputs, not from assumptions. Avoid calling `submit(...)` in code that prints output. When printed output must be inspected as evidence, read the result first, then submit in the next call. Printing and submitting together is allowed only when the output is incidental and the answer is already determined. If you don't yet know the inputs' size, format, or contents, look first and decide after reading the result. Prefer bounded, decision-relevant observations over broad dumps. When no next step remains, finish: call `submit(value)` in your code to return a computed or structured value, or reply in plain text without calling `repl`. Do not reply in plain text while work remains.

The REPL persists across calls. The invocation `task` and any named JSON arguments are direct Python variables. Top-level `await` is available. Only printed stdout is returned, and outputs over ~20K characters are truncated, so inspect bounded samples or summaries."""

#: How a CLI agent (Claude Code, pi, a person's session) reaches the REPL.
USE_RLMFLOW_REPL = """\
You are one agent in a multi-agent run, with a persistent Python REPL.

Run code with `rlmflow repl 'CODE'`, or for several lines:
    rlmflow repl <<'EOF'
    ...
    EOF
Quote the heredoc marker ('EOF') so the shell leaves the code alone. Variables persist
between calls. Your inputs are already variables: `task`, and any arguments listed below.
Only printed output comes back, so print summaries, not whole values.

In the REPL:
- launch_agent(task, agent=NAME, **args) starts an agent and returns it as an `Agent`.
  `await agent` waits for the result of an agent you launched. An Agent's `.status`,
  `.done()` and `.result()` are as of when the code started, so they never change inside a loop.
- agent.send(message) messages any agent, reopening it if it finished. For an agent you
  launched, `await agent.send(...)` also waits for its reply; others reply with a message.
- AGENTS["name"] is any agent in the run, by name, path or id. AGENTS.self is you,
  AGENTS.children the agents you launched, and AGENTS.print_graph(show_results=True)
  prints them all with their status and results.
- await llm_query(prompt) asks a model and returns its answer.
- submit(value) records your final result. Call it exactly once, when you're finished.

To wait for agents you started, end your turn without submitting. You'll be resumed
with their results. Don't poll. Code that awaits long work can hit your tool timeout,
so launch agents and end the turn instead.

Code you run with `rlmflow repl` is recorded in this run, so it can be replayed and
forked from; changes you make with your own tools are not."""

LOCATION = """\
Your REPL runs in {location}. Your own tools run on this machine; reach the REPL's
files and programs through `rlmflow repl`, with paths as the REPL sees them."""

AGENTS_HELP = """\
An `Agent` is what `launch_agent` returns and what `AGENTS[name]` looks up (by name, path or id). \
`await agent` waits for the result of an agent you launched. `agent.send(message)` queues a \
message for any agent, reopening it if it finished; for an agent you launched, \
`await agent.send(message)` also waits for its reply. Agents you didn't launch answer with a \
message at your next turn, so send and finish the block. `agent.status`, `agent.result()`, \
`agent.parent` and `agent.children` are as of when the block started.
`AGENTS` is this run's agents: `AGENTS.self` is you, `AGENTS.children` the agents you \
launched, and `AGENTS.print_graph(show_results=True)` prints the tree with statuses and results."""

VERIFY = """\
Before submitting, check the answer against the inputs, in code where possible. Do not \
turn missing data, failed reads, tool errors, or an agent's empty or "None" result into \
a plausible answer: surface the failure unless you validate a recovery."""

STATE = """\
Keep your working state in REPL variables, not only in your replies: put intermediate
results in clearly named variables (for example `candidate_files`, `matches_by_chunk`),
and keep a `notes` string with your plan and what you've ruled out. Update `notes`
before waiting on agents. You may later continue from this REPL without your earlier
conversation."""

FINAL_TURN = """\
This is your last turn. Give the best supported final answer now: call `submit(value)` \
in one `repl` call, or reply in plain text without calling `repl`. Do not start work \
that requires another observation."""

TRUNCATED = """\
[earlier turns omitted to fit the context window; the most recent turns follow. The \
REPL kept running, so variables, imports, and helpers defined in those turns are still \
bound — reuse them instead of redefining them.]"""


@cache
def skill(name: str) -> str:
    """The body of a bundled skill, without its front matter."""
    text = (SKILLS_DIR / name / "SKILL.md").read_text()
    if text.startswith("---"):
        text = text.split("\n---", 1)[1]
    return text.strip()


def arguments(run: Node) -> dict[str, Any]:
    """The run's arguments: its task node's inputs, apart from ``task``."""
    return {name: value for name, value in run.root.input.items() if name != "task"}


def argument_manifest(run: Node) -> str:
    """Arguments by name and size; the values stay in the REPL."""
    values = arguments(run)
    if not values:
        return ""
    descriptions = run.root.arg_descriptions
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
    "AGENTS_HELP",
    "FINAL_TURN",
    "LOCATION",
    "PROTOCOL",
    "REPL_TOOL",
    "SKILLS_DIR",
    "STATE",
    "TRUNCATED",
    "USE_RLMFLOW_REPL",
    "VERIFY",
    "argument_manifest",
    "arguments",
    "preview",
    "skill",
]
