"""Typed specifications for recursive sub-agent calls."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypeAlias


@dataclass(slots=True)
class Subagent:
    """One child requested through ``launch_subagents``."""

    goal: str
    name: str | None = None
    inputs: dict[str, str] = field(default_factory=dict)
    model: str | None = None
    output_schema: Any = None
    prompt_profile: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the mapping form consumed by the launch tool."""
        return {
            "goal": self.goal,
            "name": self.name,
            "inputs": self.inputs,
            "model": self.model,
            "output_schema": self.output_schema,
            "prompt_profile": self.prompt_profile,
        }


SubagentSpec: TypeAlias = Subagent | dict[str, Any]


def normalize_subagent(spec: SubagentSpec) -> dict[str, Any]:
    """Normalize typed and mapping specs while preserving ``query`` compatibility."""
    if isinstance(spec, Subagent):
        normalized = spec.to_dict()
    elif isinstance(spec, dict):
        normalized = dict(spec)
    else:
        raise TypeError("subagent spec must be a Subagent or dict")

    goal = normalized.pop("goal", None)
    query = normalized.get("query")
    if goal is not None:
        if query is not None and query != goal:
            raise ValueError("subagent spec cannot set different values for 'goal' and 'query'")
        normalized["query"] = goal
    return normalized


__all__ = ["Subagent", "SubagentSpec"]
