"""Named streaming boundaries over published graph nodes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, TypeAlias

from rlmflow.minimal.nodes import DoneOutput, ErrorOutput, ExecOutput, Node

Boundary: TypeAlias = Callable[[Node, Node], bool]
StepUntil: TypeAlias = Literal["next", "idle", "done", "finished", "error"] | Boundary

_FULL_RUN = frozenset({"done", "finished"})
_BOUNDARIES: dict[str, Boundary] = {
    "next": lambda node, root: True,
    "idle": lambda node, root: isinstance(node, (ExecOutput, DoneOutput)),
    "error": lambda node, root: isinstance(node, ErrorOutput),
}


def register(name: str, boundary: Boundary) -> None:
    """Register a process-wide named streaming boundary."""
    if name in _FULL_RUN or name in _BOUNDARIES:
        raise ValueError(f"boundary {name!r} is already defined")
    _BOUNDARIES[name] = boundary


def resolve(until: StepUntil) -> Boundary | None:
    """Resolve a built-in, registered, or callable boundary."""
    if until in _FULL_RUN:
        return None
    if callable(until):
        return until
    try:
        return _BOUNDARIES[until]
    except (KeyError, TypeError):
        raise ValueError(f"unknown until boundary: {until!r}") from None


__all__ = ["Boundary", "StepUntil", "register", "resolve"]
