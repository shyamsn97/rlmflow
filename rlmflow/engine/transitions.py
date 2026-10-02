"""Automatic transitions: one decorated function per edge, resolved first-match."""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import Any

from rlmflow.graph.nodes import Node, TransitionDecision

Condition = Callable[[Node], "bool | Awaitable[bool]"] | str
SemanticChoice = tuple[tuple[str, str, str], ...]
Judge = Callable[[Node, SemanticChoice], Awaitable[TransitionDecision]]


class TransitionPolicyError(ValueError):
    """Invalid transition policy declared by the host application."""


class TransitionProtocolError(RuntimeError):
    """The model broke the turn protocol a transition depends on."""


@dataclass(frozen=True, slots=True)
class TransitionFunction:
    """A producer carrying its complete automatic-transition declaration."""

    fn: Callable[..., Any]
    sources: tuple[type[Node], ...] | None
    when: Condition | None = None
    node_only: bool = False

    @classmethod
    def create(
        cls,
        fn: Callable[..., Any],
        *,
        source: type[Node] | tuple[type[Node], ...] | str,
        when: Condition | None = None,
        node_only: bool = False,
    ) -> TransitionFunction:
        _validate_condition(when)
        return cls(fn, _normalize_sources(source), when, node_only)

    @property
    def __name__(self) -> str:
        return self.fn.__name__

    @property
    def id(self) -> str:
        source = (
            "all"
            if self.sources is None
            else ",".join(f"{item.__module__}.{item.__qualname__}" for item in self.sources)
        )
        condition = (
            self.when
            if isinstance(self.when, str) or self.when is None
            else f"{self.when.__module__}.{self.when.__qualname__}"
        )
        fingerprint = json.dumps(
            {
                "condition": condition,
                "source": source,
                "handler": f"{self.fn.__module__}.{self.fn.__qualname__}",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(fingerprint.encode()).hexdigest()

    def accepts(self, node: Node) -> bool:
        return self.sources is None or isinstance(node, self.sources)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)

    async def produce(self, flow: Any, node: Node) -> Node:
        created = self.fn(node) if self.node_only else self.fn(flow, node)
        if inspect.isawaitable(created):
            created = await created
        if not isinstance(created, Node):
            raise TypeError(
                f"transition handler {self.__name__} returned "
                f"{type(created).__name__}, expected Node"
            )
        return created


def transition(
    *,
    source: type[Node] | tuple[type[Node], ...] | str,
    when: Condition | None = None,
):
    """Wrap one built-in producer with its automatic-transition declaration."""

    def decorate(fn: Callable[..., Any]) -> TransitionFunction:
        return TransitionFunction.create(fn, source=source, when=when)

    return decorate


async def resolve(
    transitions: Iterable[TransitionFunction],
    node: Node,
    judge: Judge,
) -> tuple[TransitionFunction, Node, Node]:
    """Return ``(function, anchor, source)`` for the selected transition.

    Callable conditions before the first semantic condition retain first-match
    priority. At the first string condition, all source-compatible string
    conditions before the next unconditional fallback become one structured
    choice. A selected semantic edge wins there; ``continue`` resumes deterministic
    first-match resolution after that choice. The one durable decision is reused
    on replay.
    """
    items = tuple(transitions)
    recorded: TransitionDecision | None = None
    source = node
    while isinstance(source, TransitionDecision):
        if recorded is not None:
            raise ValueError("expected one transition decision per source node")
        recorded = source
        source = source.prev
    if recorded is not None and recorded.source_node_id != source.id:
        raise ValueError("transition decision chain has inconsistent source nodes")

    anchor = node
    semantic_resolved = False
    for index, transition_fn in enumerate(items):
        if not transition_fn.accepts(source):
            continue
        when = transition_fn.when
        if isinstance(when, str):
            if semantic_resolved:
                continue
            semantic = _semantic_choices(items[index:], source)
            if recorded is not None:
                decision = recorded
                policy = tuple(
                    (rule_id, condition) for rule_id, _label, condition in semantic
                )
                if tuple(decision.candidates.items()) != policy:
                    raise ValueError("recorded transition decision no longer matches policy")
            else:
                decision = await judge(source, semantic)
                anchor.append(decision)
                anchor = decision
            semantic_resolved = True
            if decision.selected_rule_id is not None:
                selected = next(
                    (
                        candidate
                        for candidate in items
                        if candidate.id == decision.selected_rule_id
                        and candidate.accepts(source)
                        and isinstance(candidate.when, str)
                    ),
                    None,
                )
                if selected is None:
                    raise ValueError("semantic transition selected an unknown rule")
                return selected, anchor, source
            continue
        elif when is None:
            return transition_fn, anchor, source
        else:
            outcome = when(source)
            if bool(await outcome if inspect.isawaitable(outcome) else outcome):
                return transition_fn, anchor, source
    raise TypeError(f"cannot step {type(source).__name__}")


def _semantic_choices(
    transitions: tuple[TransitionFunction, ...],
    source: Node,
) -> SemanticChoice:
    """Semantic choices reachable before the next unconditional fallback."""
    choices: list[tuple[str, str, str]] = []
    for transition_fn in transitions:
        if not transition_fn.accepts(source):
            continue
        when = transition_fn.when
        if when is None:
            break
        if isinstance(when, str):
            choices.append((transition_fn.id, transition_fn.__name__, when))
    if not choices:
        raise ValueError("semantic transition choice requires at least one candidate")
    if len({rule_id for rule_id, _label, _condition in choices}) != len(choices):
        raise TransitionPolicyError("semantic transitions must have unique rule IDs")
    if len({label for _rule_id, label, _condition in choices}) != len(choices):
        raise TransitionPolicyError(
            "semantic transitions in one choice must have unique function names"
        )
    return tuple(choices)


def _validate_condition(when: Any) -> None:
    if isinstance(when, str) and not when.strip():
        raise TransitionPolicyError("semantic transition condition cannot be empty")
    if when is not None and not isinstance(when, str) and not callable(when):
        raise TransitionPolicyError("when must be a callable, string, or None")


def _normalize_sources(
    source: type[Node] | tuple[type[Node], ...] | str,
) -> tuple[type[Node], ...] | None:
    if source == "all":
        return None
    sources = source if isinstance(source, tuple) else (source,)
    if not sources or any(
        not isinstance(item, type) or not issubclass(item, Node) for item in sources
    ):
        raise TransitionPolicyError("source must be a Node class, tuple of Node classes, or 'all'")
    return sources


__all__ = [
    "TransitionFunction",
    "TransitionPolicyError",
    "TransitionProtocolError",
    "resolve",
    "transition",
]
