"""A small recursive agent engine: nodes, one flow, one task queue."""

from typing import Any

from rlmflow.minimal import boundaries, persistence
from rlmflow.minimal.nodes import (
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
    start,
)
from rlmflow.minimal.task import TaskQueue

_LAZY = {"Flow", "StepUntil", "code_block"}


def __getattr__(name: str) -> Any:
    # Flow pulls in the shared prompt stack, which imports these nodes back;
    # loading it on demand keeps that from becoming an import cycle.
    if name in _LAZY:
        from rlmflow.minimal import flow

        return getattr(flow, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AgentConfig",
    "AgentStart",
    "DoneOutput",
    "ErrorOutput",
    "ExecAction",
    "ExecOutput",
    "Flow",
    "LLMOutput",
    "LLMUsage",
    "Node",
    "StepUntil",
    "TaskQueue",
    "UserQuery",
    "boundaries",
    "code_block",
    "persistence",
    "start",
]
