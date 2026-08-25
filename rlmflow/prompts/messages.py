"""Current-node rendering and input manifests.

Keeps message shaping out of ``flow.py``. Inspect, plan, final, continue,
truncation, and a dead REPL are node types; this module re-exports their
default strings and holds dynamic manifests. Canonical history projection
lives on ``Node``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from rlmflow.graph.nodes import (
    COLD_REPL_NOTE,
    CONTINUE_NUDGE,
    FINAL_ANSWER_ACTION,
    INSPECTION_ACTION,
    ORCHESTRATION_ACTION,
    TRUNCATION_SUMMARY,
    AgentStart,
    Node,
)

if TYPE_CHECKING:
    from rlmflow.runtime import Runtime

RenderFn = Callable[["Runtime", Node], list[dict[str, str]]]


def build_background_agents_manifest(agent: AgentStart) -> str:
    """Render direct children and current result readiness from the graph."""
    children = agent.sub_agents
    if not children:
        return ""

    lines = ["Background subagents:"]
    for child in children:
        state = "result ready" if child.terminal else "running"
        lines.append(f"- `{child.config.name}` ({child.config.path}): {state}")
    return "\n".join(lines)


def default_render(_runtime: Runtime, node: Node) -> list[dict[str, str]]:
    """Render the frontier and add live background-agent status."""
    messages = node.render()
    agent = node.parent_agent
    if agent is None:
        return messages
    manifest = build_background_agents_manifest(agent)
    if not manifest:
        return messages
    return [
        {"role": "user", "content": manifest},
        *messages,
    ]


def build_inputs_manifest(inputs: dict[str, str]) -> str:
    """List inputs by name and size without copying their values into the prompt.

    ``INPUTS`` is the RLM context boundary: the model should inspect values
    programmatically in the persistent REPL, not receive a second copy in its
    context window. Returns ``""`` when there are no inputs.
    """
    if not inputs:
        return ""
    lines = [f"- {name}: str, {len(value)} chars" for name, value in inputs.items()]
    total = sum(len(value) for value in inputs.values())
    return (
        "Your REPL `INPUTS` contain important information related to the query:\n"
        + "\n".join(lines)
        + f"\nTotal input chars: {total}."
    )


class PromptBuilder:
    """Shared base for prompt callables that return chat messages.

    A ``PromptBuilder`` is a ``(flow, agent) -> list[dict]`` callable.
    ``SystemPromptBuilder`` returns a single ``system`` message.
    """

    def __call__(self, flow: Any, agent: Node) -> list[dict[str, str]]:
        raise NotImplementedError


__all__ = [
    "COLD_REPL_NOTE",
    "CONTINUE_NUDGE",
    "FINAL_ANSWER_ACTION",
    "INSPECTION_ACTION",
    "ORCHESTRATION_ACTION",
    "TRUNCATION_SUMMARY",
    "PromptBuilder",
    "RenderFn",
    "build_background_agents_manifest",
    "build_inputs_manifest",
    "default_render",
]
