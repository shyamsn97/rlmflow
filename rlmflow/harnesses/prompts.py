"""Prompt pieces every harness shares."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from rlmflow.graph.graph import DEPENDS_ON

if TYPE_CHECKING:
    from rlmflow.graph.nodes import Step


def argument_manifest(step: Step) -> str:
    """Arguments by name and size; the values stay in the REPL."""
    g, agent = step.graph, step.agent
    values = {**agent.resolved_args(), **g.inputs(agent)}
    if not values:
        return ""
    descriptions = {
        **agent.argument_descriptions(),
        **{
            edge.data["as"]: edge.data["description"]
            for edge in g.edges(kind=DEPENDS_ON, source=agent.id)
            if edge.data.get("as") and edge.data.get("description")
        },
    }
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


__all__ = ["argument_manifest", "preview"]
