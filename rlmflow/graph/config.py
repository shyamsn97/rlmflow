"""Per-agent limits and identity for a run."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any

DEFAULT_QUERY = """Please read through the provided INPUTS if present and answer any
queries or respond to any instructions contained within it."""
AGENT_REF_KEY = "$rlmflow_agent"
AgentInputValue = str | dict[str, str]
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


def input_references(inputs: Mapping[str, object]) -> tuple[tuple[str, str], ...]:
    """Return ``(input_name, agent_id)`` pairs from top-level input values."""
    return tuple(
        (name, agent_id)
        for name, value in inputs.items()
        if (agent_id := agent_ref_id(value)) is not None
    )


def validate_agent_name(name: str) -> None:
    if (
        not isinstance(name, str)
        or not name
        or not name.isascii()
        or any(not (char.isalnum() or char in "_-") for char in name)
    ):
        raise ValueError(f"invalid child name {name!r}")


@dataclass
class AgentConfig:
    name: str = "root"
    path: str = "root"
    depth: int = 0
    model: str = "default"
    prompt_profile: str = "default"
    #: Persistent launch inputs. A child handle is stored as one exact
    #: ``{AGENT_REF_KEY: agent_id}`` marker and resolved only when seeding the REPL.
    inputs: dict[str, AgentInputValue] = field(default_factory=dict)
    output_schema: dict[str, Any] | None = None
    #: Default ``output_schema`` for launched children. Launch ``output_schema=``
    #: still wins. The root's own ``output_schema`` is independent.
    child_output_schema: dict[str, Any] | None = None
    #: Place a child in its caller's live Python worker.
    reuse_repl: bool = False
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
            "inputs": {},
            "output_schema": schema,
            "child_output_schema": deepcopy(schema),
            "reuse_repl": False,
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

    Pass as ``@tool(..., when=at_leaf)``. ``flow`` is accepted so the callable
    matches ``visible_tools`` / ``when`` ``(flow, node)`` signatures.
    """
    del flow
    agent = None if node is None else getattr(node, "parent_agent", None)
    return not can_spawn(agent)


__all__ = [
    "AGENT_REF_KEY",
    "DEFAULT_MAX_OUTPUT_CHARS",
    "DEFAULT_MAX_QUERY_CHARS",
    "DEFAULT_QUERY",
    "AgentConfig",
    "AgentInputValue",
    "agent_ref_id",
    "at_leaf",
    "can_spawn",
    "input_references",
    "validate_agent_name",
]
