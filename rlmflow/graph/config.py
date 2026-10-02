"""Per-agent limits and identity for a run."""

from __future__ import annotations

import keyword
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any

DEFAULT_QUERY = "Complete the requested task."
AGENT_REF_KEY = "$rlmflow_agent"
AgentArgValue = Any
#: Official RLM truncates each REPL block at 20K characters (`format_iteration`).
DEFAULT_MAX_OUTPUT_CHARS = 20_000
DEFAULT_MAX_QUERY_CHARS = 20_000


def agent_ref_id(value: object) -> str | None:
    """Return the agent id from one exact persisted input reference."""
    if (
        isinstance(value, dict)
        and set(value) == {AGENT_REF_KEY}
        and isinstance(value.get(AGENT_REF_KEY), str)
        and value[AGENT_REF_KEY]
    ):
        return value[AGENT_REF_KEY]
    return None


def argument_references(args: Mapping[str, object]) -> tuple[tuple[str, str], ...]:
    """Return ``(argument_path, agent_id)`` pairs from nested JSON arguments."""
    references: list[tuple[str, str]] = []

    def visit(path: str, value: object) -> None:
        agent_id = agent_ref_id(value)
        if agent_id is not None:
            references.append((path, agent_id))
        elif isinstance(value, dict):
            for key, item in value.items():
                visit(f"{path}.{key}" if path else str(key), item)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(f"{path}[{index}]", item)

    for name, value in args.items():
        visit(name, value)
    return tuple(references)


def validate_agent_name(name: str) -> None:
    if (
        not isinstance(name, str)
        or not name
        or not name.isascii()
        or any(not (char.isalnum() or char in "_-") for char in name)
    ):
        raise ValueError(f"invalid child name {name!r}")


def validate_argument_name(name: str) -> None:
    """Require invocation arguments that can be bound as Python variables."""
    if not isinstance(name, str) or not name.isidentifier() or keyword.iskeyword(name):
        raise ValueError(f"invalid invocation argument name {name!r}")
    if name == "task":
        raise ValueError("'task' is the invocation task and cannot be passed twice")


@dataclass
class AgentConfig:
    name: str = "root"
    path: str = "root"
    depth: int = 0
    model: str = "default"
    prompt_profile: str = "default"
    output_schema: dict[str, Any] | None = None
    #: Default ``output_schema`` for launched children. Launch ``output_schema=``
    #: still wins. The root's own ``output_schema`` is independent.
    child_output_schema: dict[str, Any] | None = None
    #: Durable identity of a launch verifier whose callable remains in the parent worker.
    verifier_id: str | None = None
    verifier_hash: str | None = None
    #: Place a child in its caller's worker while keeping an isolated namespace.
    reuse_worker: bool = False
    max_depth: int = 1
    #: How many model turns this agent may take. ``None`` means unbounded.
    max_iters: int | None = None
    #: Whole-run soft token threshold. Crossing it finalizes every unfinished agent.
    max_budget: int | None = None
    #: How many of the agent's own transcript turns a prompt carries. ``None``
    #: keeps the full history. The system message and the truncation notice sit
    #: on top of this count when it is set.
    keep_n_messages: int | None = None
    max_output_length: int = DEFAULT_MAX_OUTPUT_CHARS
    max_query_chars: int = DEFAULT_MAX_QUERY_CHARS
    #: When a child ``finish()``es after this agent has already ``finish()``ed,
    #: append the child's result as a ``UserQuery`` and keep driving. Pass
    #: ``False`` on scored evals that must stay terminal once the parent submits.
    resume_on_child_done: bool = True

    def child(self, name: str, **overrides: Any) -> AgentConfig:
        validate_agent_name(name)
        schema = deepcopy(self.child_output_schema)
        values = {
            "name": name,
            "path": f"{self.path}.{name}",
            "depth": self.depth + 1,
            "output_schema": schema,
            "child_output_schema": deepcopy(schema),
            "verifier_id": None,
            "verifier_hash": None,
            "reuse_worker": False,
        }
        return replace(self, **{**values, **overrides})


def can_spawn(agent: Any) -> bool:
    """Whether this agent can create another recursion level."""
    if agent is None:
        return False
    config = getattr(agent, "config", None)
    if config is None:
        return False
    return config.max_depth > 0 and config.depth < config.max_depth


def at_leaf(flow: Any = None, node: Any = None) -> bool:
    """True when this agent cannot spawn — the usual place for implementation tools.

    Pass as ``@tool(..., when=at_leaf)``; ``when`` predicates take ``(flow, node)``.
    """
    agent = None if node is None else getattr(node, "parent_agent", None)
    return not can_spawn(agent)


__all__ = [
    "AGENT_REF_KEY",
    "DEFAULT_MAX_OUTPUT_CHARS",
    "DEFAULT_MAX_QUERY_CHARS",
    "DEFAULT_QUERY",
    "AgentConfig",
    "AgentArgValue",
    "agent_ref_id",
    "at_leaf",
    "can_spawn",
    "argument_references",
    "validate_agent_name",
]
