"""``AgentConfig``: plain data describing one kind of agent, plus argument helpers."""

from __future__ import annotations

import importlib
import keyword
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields
from typing import Any

# Return this from a callback to visit a list or dictionary's children.
RECURSE = object()
ArgumentCallback = Callable[[str, Any], Any]


def transform_argument(value: Any, transform: ArgumentCallback, *, path: str) -> Any:
    """Transform an argument tree in pre-order, recursing on ``RECURSE``."""
    return _traverse(value, transform, path=path, rebuild=True)


def visit_argument(value: Any, visitor: ArgumentCallback, *, path: str) -> None:
    """Visit an argument tree in pre-order, recursing on ``RECURSE``."""
    _traverse(value, visitor, path=path, rebuild=False)


def _traverse(value: Any, callback: ArgumentCallback, *, path: str, rebuild: bool) -> Any:
    result = callback(path, value)
    if result is not RECURSE:
        return result
    if isinstance(value, list):
        mapped = [
            _traverse(item, callback, path=f"{path}[{index}]", rebuild=rebuild)
            for index, item in enumerate(value)
        ]
        return mapped if rebuild else value
    if isinstance(value, dict):
        rebuilt: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"argument {path!r} contains a non-string object key")
            rebuilt[key] = _traverse(item, callback, path=f"{path}.{key}", rebuild=rebuild)
        return rebuilt if rebuild else value
    return value


DEFAULT_QUERY = "Complete the requested task."
INPUT_VALUE_KEY = "$rlmflow_input"
INPUT_DESCRIPTION_KEY = "$rlmflow_description"
_INPUT_MARKER_KEY = "$rlmflow_input_marker"
_INPUT_MARKER_VALUE = "v1"
#: Official RLM truncates each REPL block at 20K characters (`format_iteration`).
DEFAULT_MAX_OUTPUT_CHARS = 20_000
DEFAULT_MAX_QUERY_CHARS = 20_000
DEFAULT_MAX_ITERS = 30


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


def validate_agent_name(name: str) -> None:
    if (
        not isinstance(name, str)
        or not name
        or not name.isascii()
        or any(not (char.isalnum() or char in "_-") for char in name)
    ):
        raise ValueError(f"invalid agent name {name!r}")


def validate_argument_name(name: str) -> None:
    """Require invocation arguments that can be bound as Python variables."""
    if not isinstance(name, str) or not name.isidentifier() or keyword.iskeyword(name):
        raise ValueError(f"invalid invocation argument name {name!r}")
    if name == "task":
        raise ValueError("'task' is the invocation task and cannot be passed twice")


#: Short names for the harnesses that ship with rlmflow.
HARNESSES = {
    "repl": "rlmflow.harnesses.repl:ReplHarness",
    "claude_code": "rlmflow.harnesses.claude_code:ClaudeCodeHarness",
    "pi": "rlmflow.harnesses.pi:PiHarness",
    "interactive": "rlmflow.harnesses.interactive:InteractiveHarness",
}


def harness_path(harness: str | Callable[..., Any]) -> str:
    """The import path a harness is saved as."""
    if isinstance(harness, str):
        return HARNESSES.get(harness, harness)
    module, qualname = getattr(harness, "__module__", None), getattr(harness, "__qualname__", "")
    if not module or "<" in qualname:
        raise ValueError(f"harness {harness!r} must be a module-level class or function")
    return f"{module}:{qualname}"


def load_harness(path: str) -> Callable[..., Any]:
    """The harness at ``path``; a class is instantiated."""
    module, _, qualname = harness_path(path).partition(":")
    if not qualname:
        raise ValueError(f"harness path {path!r} must look like 'module:Harness'")
    target: Any = importlib.import_module(module)
    for part in qualname.split("."):
        target = getattr(target, part)
    return target() if isinstance(target, type) else target


@dataclass
class AgentConfig:
    """One kind of agent. Plain JSON data once ``harness`` is an import path."""

    #: The kind of agent, e.g. ``"worker"``; not the agent's own name.
    name: str = "default"
    #: A short name, a harness class or function, or its import path; saved as the path.
    harness: Any = "repl"
    #: A model name, or an ``llm_clients`` key on the Flow. ``None``: the harness's own
    #: default where it has one (Claude Code, pi); the REPL harness needs a model.
    model: str | None = None
    #: Overrides for the harness's ``prompt_sections()`` by key: a string replaces that
    #: section, ``None`` drops it, and a new key is added at the end.
    prompt: dict[str, str | None] = field(default_factory=dict)
    #: Shown to other agents choosing a config.
    description: str = ""
    max_iters: int | None = None
    max_depth: int = 1
    output_schema: dict[str, Any] | None = None
    #: Harness-specific settings, e.g. Claude Code's ``{"mcp": True}``.
    harness_options: dict[str, Any] = field(default_factory=dict)
    #: Whole-run soft token threshold. Crossing it gives every agent a final turn.
    max_budget: int | None = None
    #: How many of the agent's own transcript turns a prompt carries; ``None`` keeps all.
    keep_n_messages: int | None = None
    max_output_length: int = DEFAULT_MAX_OUTPUT_CHARS
    max_query_chars: int = DEFAULT_MAX_QUERY_CHARS

    def __post_init__(self) -> None:
        validate_agent_name(self.name)
        self.harness = harness_path(self.harness)
        if self.output_schema is not None:
            from rlmflow.schema import json_schema_for

            self.output_schema = json_schema_for(self.output_schema)
        self.harness_options = dict(self.harness_options)
        self.prompt = dict(self.prompt)

    def load_harness(self) -> Callable[..., Any]:
        return load_harness(self.harness)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentConfig:
        unknown = set(data) - _FIELDS
        if unknown:
            raise ValueError(f"unknown AgentConfig fields {sorted(unknown)}")
        return cls(**data)


_FIELDS = frozenset(item.name for item in fields(AgentConfig))


def can_spawn(graph: Any, agent: Any) -> bool:
    """Whether this agent can create another recursion level."""

    if agent is None:
        return False
    return agent.config.max_depth > 0 and graph.depth(agent) < agent.config.max_depth


def at_leaf(flow: Any = None, node: Any = None) -> bool:
    """True when this agent cannot spawn; pass as ``@tool(..., when=at_leaf)``."""
    from rlmflow.graph.nodes import current_step

    step = current_step()
    return step is None or not can_spawn(step.graph, step.agent)


__all__ = [
    "AgentConfig",
    "DEFAULT_MAX_ITERS",
    "DEFAULT_MAX_OUTPUT_CHARS",
    "DEFAULT_MAX_QUERY_CHARS",
    "DEFAULT_QUERY",
    "HARNESSES",
    "Input",
    "at_leaf",
    "can_spawn",
    "harness_path",
    "load_harness",
    "validate_agent_name",
    "validate_argument_name",
]
