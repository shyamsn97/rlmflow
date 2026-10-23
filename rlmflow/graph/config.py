"""Per-agent limits and identity for a run."""

from __future__ import annotations

import keyword
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field, fields, replace
from enum import StrEnum
from typing import Any

# Return this from a callback to visit a list or dictionary's children.
RECURSE = object()
ArgumentCallback = Callable[[str, Any], Any]


def transform_argument(
    value: Any,
    transform: ArgumentCallback,
    *,
    path: str,
) -> Any:
    """Transform an argument tree in pre-order, recursing on ``RECURSE``."""
    return _traverse(value, transform, path=path, rebuild=True)


def visit_argument(value: Any, visitor: ArgumentCallback, *, path: str) -> None:
    """Visit an argument tree in pre-order, recursing on ``RECURSE``."""
    _traverse(value, visitor, path=path, rebuild=False)


def _traverse(
    value: Any,
    callback: ArgumentCallback,
    *,
    path: str,
    rebuild: bool,
) -> Any:
    """Shared recursive implementation for argument mapping and inspection."""
    result = callback(path, value)
    if result is not RECURSE:
        return result
    if isinstance(value, list):
        if rebuild:
            return [
                _traverse(
                    item,
                    callback,
                    path=f"{path}[{index}]",
                    rebuild=True,
                )
                for index, item in enumerate(value)
            ]
        for index, item in enumerate(value):
            _traverse(item, callback, path=f"{path}[{index}]", rebuild=False)
        return value
    if isinstance(value, dict):
        result: dict[str, Any] | None = {} if rebuild else None
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"argument {path!r} contains a non-string object key")
            mapped_item = _traverse(
                item,
                callback,
                path=f"{path}.{key}",
                rebuild=rebuild,
            )
            if result is not None:
                result[key] = mapped_item
        return result if result is not None else value
    return value


DEFAULT_QUERY = "Complete the requested task."
INPUT_VALUE_KEY = "$rlmflow_input"
INPUT_DESCRIPTION_KEY = "$rlmflow_description"
_INPUT_MARKER_KEY = "$rlmflow_input_marker"
_INPUT_MARKER_VALUE = "v1"
AgentArgValue = Any
#: Official RLM truncates each REPL block at 20K characters (`format_iteration`).
DEFAULT_MAX_OUTPUT_CHARS = 20_000
DEFAULT_MAX_QUERY_CHARS = 20_000


class Input(dict[str, Any]):
    """A task argument paired with caller-supplied model context."""

    def __init__(self, value: Any, *, description: str) -> None:
        if not isinstance(description, str) or not description.strip():
            raise ValueError("Input description must be a non-empty string")
        super().__init__(
            {
                _INPUT_MARKER_KEY: _INPUT_MARKER_VALUE,
                INPUT_VALUE_KEY: value,
                INPUT_DESCRIPTION_KEY: description.strip(),
            }
        )

    @property
    def value(self) -> Any:
        return self[INPUT_VALUE_KEY]

    @property
    def description(self) -> str:
        return self[INPUT_DESCRIPTION_KEY]


def input_parts(value: object) -> tuple[Any, str] | None:
    """Return an Input marker's value and description."""
    if (
        isinstance(value, dict)
        and set(value) == {_INPUT_MARKER_KEY, INPUT_VALUE_KEY, INPUT_DESCRIPTION_KEY}
        and value.get(_INPUT_MARKER_KEY) == _INPUT_MARKER_VALUE
        and isinstance(value.get(INPUT_DESCRIPTION_KEY), str)
        and value[INPUT_DESCRIPTION_KEY].strip()
    ):
        return value[INPUT_VALUE_KEY], value[INPUT_DESCRIPTION_KEY]
    return None


class ContextPolicy(StrEnum):
    """What happens to an agent's live state once it is terminal."""

    #: Drop live state when terminal and refuse follow-up messages.
    EPHEMERAL = "ephemeral"
    #: Drop live state when terminal; replay history on a follow-up.
    REHYDRATABLE = "rehydratable"
    #: Keep live state; replay only after process loss.
    RESIDENT = "resident"


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


#: ``AgentConfig.backend`` for rlmflow's own model-and-REPL loop.
NATIVE_BACKEND = "rlmflow"


@dataclass
class AgentConfig:
    name: str = "root"
    #: Durable executor identity: ``NATIVE_BACKEND`` or a ``Flow(backends=...)`` key.
    backend: str = NATIVE_BACKEND
    model: str = "default"
    prompt_profile: str = "default"
    output_schema: dict[str, Any] | None = None
    #: Field overrides for agents this one launches, e.g. ``{"model": "fast",
    #: "backend": "claude-code", "output_schema": Schema}``. Every other field
    #: is inherited, and children inherit this table, so it applies all the way
    #: down. Launch ``schema=`` still wins over ``output_schema`` here.
    child_config: dict[str, Any] = field(default_factory=dict)
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
    #: When a child ``submit()``s after this agent has already ``submit()``ed,
    #: append the child's result as a ``UserQuery`` and keep driving. Pass
    #: ``False`` on scored evals that must stay terminal once the parent submits.
    resume_on_child_done: bool = True
    context_policy: ContextPolicy = ContextPolicy.RESIDENT

    def __post_init__(self) -> None:
        if not isinstance(self.backend, str) or not self.backend:
            raise ValueError("agent backend must be a non-empty string")
        if self.reuse_worker and self.backend != NATIVE_BACKEND:
            raise ValueError("an external backend cannot reuse its launcher's worker")
        unknown = set(self.child_config) - _CHILD_FIELDS
        if unknown:
            raise ValueError(f"child_config cannot set {sorted(unknown)}")
        self.context_policy = ContextPolicy(self.context_policy)

    def child(self, name: str, **overrides: Any) -> AgentConfig:
        """The config for a child named ``name``: this one, plus ``child_config``."""
        validate_agent_name(name)
        values = {
            "name": name,
            "output_schema": None,
            "verifier_id": None,
            "verifier_hash": None,
            "reuse_worker": False,
            **deepcopy(self.child_config),
        }
        return replace(self, **{**values, **overrides})


#: What ``child_config`` may override; identity and the table itself are not.
_CHILD_FIELDS = frozenset(f.name for f in fields(AgentConfig)) - {"name", "child_config"}


def can_spawn(graph: Any, agent: Any) -> bool:
    """Whether this agent can create another recursion level."""
    from rlmflow.graph import reads

    if agent is None:
        return False
    return agent.config.max_depth > 0 and reads.depth(graph, agent) < agent.config.max_depth


def at_leaf(flow: Any = None, node: Any = None) -> bool:
    """True when this agent cannot spawn — the usual place for implementation tools.

    Pass as ``@tool(..., when=at_leaf)``; ``when`` predicates take ``(flow, node)``.
    """
    from rlmflow.graph.nodes import current_step

    step = current_step()
    return step is None or not can_spawn(step.graph, step.agent)


__all__ = [
    "AgentArgValue",
    "AgentConfig",
    "ContextPolicy",
    "DEFAULT_MAX_OUTPUT_CHARS",
    "DEFAULT_MAX_QUERY_CHARS",
    "DEFAULT_QUERY",
    "Input",
    "NATIVE_BACKEND",
    "at_leaf",
    "can_spawn",
    "validate_agent_name",
]
