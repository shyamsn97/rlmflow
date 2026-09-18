"""Official-shaped system prompt: one template, one callable builder."""

from __future__ import annotations

import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from rlmflow.graph.config import can_spawn
from rlmflow.graph.prompts import EXECUTION_POLICY, WORKER_ADDENDUM
from rlmflow.prompts.messages import RenderFn, build_inputs_manifest
from rlmflow.structured import system_prompt_hint

#: Ceiling for the statically rendered ``SYSTEM_PROMPT`` (no flow/node bound).
MAX_STATIC_PROMPT_CHARS = 4_000

RLM_SYSTEM_PROMPT = textwrap.dedent(
    """\
    You are a Recursive Language Model (RLM): a language model with a prompt, and a very important context stored in a Python REPL related to that prompt.
    You can iteratively interact with the Python REPL, which has access to LLM calls as a function. You will be queried turn-by-turn until you have an answer to the query.

    {execution_policy}

    To use the REPL, write code in ```repl``` blocks; the REPL persists across turns. Available in the REPL:
    - `INPUTS: dict[str, str]`: the important, potentially very long information related to the prompt.
    - Top-level `await` is available; do not start another event loop.
    {tools}

    Operating rules:
    - Inspect bounded samples or summaries of `INPUTS` before substantial work.
    - Execute one ```repl``` block, observe its output, then revise the next step.
    - Only printed stdout is returned; wrap inspections in `print(...)`.
    - Outputs over ~20K characters are truncated. Keep bulky values in the REPL or pass them through `inputs=`.
    - Call `finish(...)` only when the requested answer is supported. Do not finish on turn 1 before inspecting `INPUTS`.
    """
)

STRUCTURED_OUTPUT_TEXT = """\
This run requires structured output. When complete, call `finish(value)` with a
JSON-compatible Python value matching this JSON Schema exactly — the value
itself, not a status string:

```json
{schema_hint}
```

```repl
finish({{"n": 41}})
```
"""


class PromptBuilder:
    """The default system prompt. Equivalent to a ``(flow, node) -> str`` function."""

    def __call__(self, flow: Any = None, node: Any = None) -> str:
        if flow is None:
            return RLM_SYSTEM_PROMPT.format(
                execution_policy=EXECUTION_POLICY,
                tools="",
            )
        agent = as_agent(node)
        execution_policy = EXECUTION_POLICY if can_spawn(agent) else WORKER_ADDENDUM
        parts = [
            RLM_SYSTEM_PROMPT.format(
                execution_policy=execution_policy,
                tools=flow.render_tools(node),
            ),
            flow.render_inputs(node),
        ]
        return "\n\n".join(part for part in parts if part)


SystemPromptFn = Callable[[Any, Any], str]
SystemPromptSource = str | SystemPromptFn | PromptBuilder


def as_system_prompt_fn(source: Any) -> SystemPromptFn:
    """Normalize a system-prompt source to a ``(flow, node) -> str`` function."""
    if isinstance(source, str):
        if "{tools}" in source:

            def formatted(flow: Any = None, node: Any = None) -> str:
                tools = flow.render_tools(node) if flow is not None else ""
                return source.format(tools=tools)

            return formatted
        return lambda _flow=None, _node=None, text=source: text
    if callable(source):
        return source
    raise TypeError(f"unsupported system prompt source: {type(source)!r}")


@dataclass
class PromptProfile:
    """A named system prompt and current-frontier renderer for a child agent.

    ``None`` on either side means *inherit the flow's default* for that side, so a
    profile can override just the system prompt. ``description`` is a short summary
    of when to use the profile; it is advertised next to ``launch_subagent`` when
    the flow has a registry.
    """

    system: SystemPromptSource | None = None
    render_fn: RenderFn | None = None
    description: str = ""


def as_agent(node: Any) -> Any:
    """The ``AgentStart`` for ``node``. ``parent_agent`` is a stored field, O(1)."""
    if node is None:
        return None
    parent = getattr(node, "parent_agent", None)
    if parent is not None:
        return parent
    return node if getattr(node, "config", None) is not None else None


def render_inputs_text(flow: Any, node: Any) -> str:
    """Per-agent INPUTS sizes, output schema, and depth — official metadata analog."""
    agent = as_agent(node)
    if agent is None:
        return ""
    parts: list[str] = []
    manifest = build_inputs_manifest(agent.resolved_inputs())
    if manifest:
        parts.append(manifest)
    schema = agent.config.output_schema
    if schema is not None:
        hint = system_prompt_hint(schema)
        parts.append(STRUCTURED_OUTPUT_TEXT.format(schema_hint=hint).strip())
    max_depth = agent.config.max_depth
    if max_depth == 0:
        parts.append("Baseline mode: no sub-agents available.")
    else:
        note = (
            f"You are using model key **`{agent.config.model}`** at recursion depth "
            f"**{agent.config.depth}** of max **{max_depth}**."
        )
        if agent.config.depth >= max_depth:
            note += " You cannot spawn sub-agents."
        parts.append(note)
    return "\n\n".join(parts)


SYSTEM_PROMPT = PromptBuilder()()


__all__ = [
    "MAX_STATIC_PROMPT_CHARS",
    "RLM_SYSTEM_PROMPT",
    "STRUCTURED_OUTPUT_TEXT",
    "SYSTEM_PROMPT",
    "PromptBuilder",
    "PromptProfile",
    "SystemPromptFn",
    "SystemPromptSource",
    "as_agent",
    "as_system_prompt_fn",
    "can_spawn",
    "render_inputs_text",
]
