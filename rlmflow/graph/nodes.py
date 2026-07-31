"""Compatibility surface: the node model lives in :mod:`rlmflow.minimal.nodes`."""

from dataclasses import dataclass
from typing import Any, ClassVar

from rlmflow.minimal.nodes import (
    DEFAULT_MAX_QUERY_CHARS,
    DEFAULT_QUERY,
    AgentConfig,
    AgentStart,
    DoneOutput,
    ErrorOutput,
    ExecAction,
    ExecOutput,
    LLMOutput,
    LLMUsage,
    Node,
    UserQuery,
    new_agent_id,
    new_node_id,
    start,
    validate_agent_name,
)


@dataclass
class SupervisingOutput(Node):
    """A delegating turn's partial output, and the node its children hung off.

    Only this package writes it. Children now branch off the ``ExecAction`` that
    launched them, so a delegating step no longer moves its agent's frontier — see
    ``docs/internal/supervising_node.md``. Runs saved before that load through
    :mod:`rlmflow.minimal.persistence`, which reads the type as an ``ExecOutput``.
    """

    type: ClassVar[str] = "supervising_output"

    def to_dict(self, *, nested: bool = True) -> dict[str, Any]:
        data = super().to_dict(nested=nested)
        data["payload"]["output"] = self.content
        data["payload"]["waiting_on"] = [
            child.config.path for child in self.children if isinstance(child, AgentStart)
        ]
        return data

__all__ = [
    "DEFAULT_MAX_QUERY_CHARS",
    "DEFAULT_QUERY",
    "AgentConfig",
    "AgentStart",
    "DoneOutput",
    "ErrorOutput",
    "ExecAction",
    "ExecOutput",
    "LLMOutput",
    "LLMUsage",
    "Node",
    "SupervisingOutput",
    "UserQuery",
    "new_agent_id",
    "new_node_id",
    "start",
    "validate_agent_name",
]
