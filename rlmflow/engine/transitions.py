"""Immutable control-flow policy for model-selectable behavioral nodes."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, replace

from rlmflow.graph.nodes import (
    AgentStart,
    DoneOutput,
    ErrorOutput,
    ExecOutput,
    FinalQuery,
    LLMOutput,
    Node,
    PlanQuery,
    TruncationSummary,
    UserQuery,
)

Guard = Callable[[Node], bool]
SOURCES = (AgentStart, UserQuery, ErrorOutput, ExecOutput)
ACT_NAME = "act"
RESERVED_NAMES = frozenset({ACT_NAME, "finish"})
LEGACY_ACTION_NAMES = frozenset({"inspect", "plan"})


class TransitionPolicyError(ValueError):
    """Invalid transition policy declared by the host application."""


class TransitionProtocolError(RuntimeError):
    """Invalid transition selection produced by the model."""


class MissingTransitionError(TransitionProtocolError):
    """A successful action returned without selecting its next behavior."""


class InvalidTransitionError(TransitionProtocolError):
    """The model selected a transition unavailable from the current behavior."""

    def __init__(self, name: str, available: Iterable[str]) -> None:
        choices = tuple(available)
        super().__init__(f"transition {name!r} is unavailable; available: {list(choices)!r}")
        self.name = name
        self.available = choices


@dataclass(frozen=True)
class TransitionRule:
    """One graph edge, either host-selected or model-selectable."""

    source: type[Node] | tuple[type[Node], ...]
    target: type[UserQuery]
    when: Guard | None = None
    selectable: bool = False

    def matches(self, node: Node, *, behavior: UserQuery | None = None) -> bool:
        source = behavior if self.selectable else node
        return (
            source is not None
            and isinstance(source, self.source)
            and (self.when is None or self.when(node))
        )


@dataclass(frozen=True)
class TransitionOption:
    name: str
    description: str
    target: type[UserQuery] | None


@dataclass(frozen=True)
class Transitions:
    """Owner-layered transition rules, most-specific layer first."""

    layers: tuple[tuple[TransitionRule, ...], ...] = ((),)

    def derive(self) -> Transitions:
        """Add a most-specific owner layer without mutating inherited policy."""
        return replace(self, layers=((), *self.layers))

    def always(
        self,
        source: type[Node] | tuple[type[Node], ...],
        target: type[UserQuery],
        *,
        when: Guard | None = None,
    ) -> Transitions:
        _validate_sources(source)
        if not isinstance(target, type) or not issubclass(target, UserQuery):
            raise TransitionPolicyError("transition target must be a UserQuery class")
        head, *rest = self.layers
        rule = TransitionRule(source=source, target=target, when=when)
        return replace(self, layers=((*head, rule), *rest))

    def on(
        self,
        current: type[UserQuery],
        targets: Iterable[type[UserQuery]],
        *,
        when: Guard | None = None,
    ) -> Transitions:
        _validate_behavior(current)
        normalized = tuple(targets)
        if not normalized:
            raise TransitionPolicyError("a transition choice list cannot be empty")
        for target in normalized:
            _validate_behavior(target)
        names = [target.name for target in normalized]
        if len(names) != len(set(names)):
            raise TransitionPolicyError(
                f"duplicate transition names from {current.__name__}: {names!r}"
            )
        head, *rest = self.layers
        existing = {rule.target.name for rule in head if rule.selectable and rule.source is current}
        repeated = existing.intersection(names)
        if repeated:
            raise TransitionPolicyError(
                f"duplicate transition names from {current.__name__}: {sorted(repeated)!r}"
            )
        rules = tuple(
            TransitionRule(
                source=current,
                target=target,
                when=when,
                selectable=True,
            )
            for target in normalized
        )
        return replace(self, layers=((*head, *rules), *rest))

    def rules(self, *, selectable: bool | None = None) -> Iterator[TransitionRule]:
        for layer in self.layers:
            for rule in layer:
                if selectable is None or rule.selectable is selectable:
                    yield rule

    def resolve_automatic(self, node: Node) -> TransitionRule | None:
        return next(
            (rule for rule in self.rules(selectable=False) if rule.matches(node)),
            None,
        )

    def available(self, node: Node) -> tuple[TransitionOption, ...]:
        behavior = self.current_behavior(node)
        if behavior is None:
            return ()
        found: dict[str, TransitionOption] = {}
        action_description = getattr(behavior, "action_description", None)
        if action_description is not None:
            found[ACT_NAME] = TransitionOption(
                name=ACT_NAME,
                description=action_description,
                target=None,
            )
        for rule in self.rules(selectable=True):
            if not rule.matches(node, behavior=behavior):
                continue
            target = rule.target
            found.setdefault(
                target.name,
                TransitionOption(
                    name=target.name,
                    description=target.transition_description,
                    target=target,
                ),
            )
        return tuple(found.values())

    def current_behavior(self, node: Node) -> UserQuery | None:
        """Return the latest selectable or action-capable query."""
        behavior_types: dict[type[UserQuery], None] = {}
        for rule in self.rules(selectable=True):
            sources = rule.source if isinstance(rule.source, tuple) else (rule.source,)
            for source in sources:
                if issubclass(source, UserQuery):
                    behavior_types.setdefault(source, None)
            behavior_types.setdefault(rule.target, None)
        classes = tuple(behavior_types)
        return next(
            (
                item
                for item in node.iter_backwards()
                if (
                    classes
                    and isinstance(item, classes)
                    or isinstance(item, UserQuery)
                    and getattr(item, "action_description", None) is not None
                )
            ),
            None,
        )

    def resolve(self, node: Node, name: str) -> TransitionOption | None:
        options = self.available(node)
        selected = next((option for option in options if option.name == name), None)
        if selected is not None:
            return selected
        if name in LEGACY_ACTION_NAMES:
            return next((option for option in options if option.name == ACT_NAME), None)
        return None

    def __str__(self) -> str:
        lines = []
        for rule in self.rules():
            source = _source_name(rule.source)
            guard = f" when {rule.when.__name__}" if rule.when is not None else ""
            mode = "on" if rule.selectable else "always"
            lines.append(f"{source} -> {rule.target.__name__}{guard} ({mode})")
        return "\n".join(lines)


def _validate_sources(source: type[Node] | tuple[type[Node], ...]) -> None:
    sources = source if isinstance(source, tuple) else (source,)
    if not sources or any(
        not isinstance(item, type) or not issubclass(item, Node) for item in sources
    ):
        raise TransitionPolicyError("source must contain Node classes")


def _validate_behavior(target: type[UserQuery]) -> None:
    if not isinstance(target, type) or not issubclass(target, UserQuery):
        raise TransitionPolicyError("selectable states must be UserQuery classes")
    name = getattr(target, "name", None)
    if not isinstance(name, str) or not name.strip():
        raise TransitionPolicyError(f"{target.__name__} has no transition name")
    if name in RESERVED_NAMES:
        raise TransitionPolicyError(f"transition name {name!r} is reserved")
    description = getattr(target, "transition_description", None)
    if not isinstance(description, str) or not description.strip():
        raise TransitionPolicyError(f"{target.__name__} has no transition description")
    action_description = getattr(target, "action_description", None)
    if action_description is not None and (
        not isinstance(action_description, str) or not action_description.strip()
    ):
        raise TransitionPolicyError(
            f"{target.__name__}.action_description must be a non-empty string"
        )


def _source_name(source: type[Node] | tuple[type[Node], ...]) -> str:
    if isinstance(source, tuple):
        return "|".join(item.__name__ for item in source)
    return source.__name__


def at_final(agent: AgentStart) -> bool:
    limit = agent.config.max_iters
    return limit is not None and agent.llm_turns() == limit - 1


def budget_nearly_spent(node: Node) -> bool:
    """Whether one more average turn would exhaust the token budget."""
    limit = node.parent_agent.config.max_budget
    root = node.root
    if limit is None or root is None:
        return False
    turns = root.stats.node_counts.get(LLMOutput.type, 0)
    if turns == 0:
        return False
    spent = root.usage.total
    return spent + spent / turns >= limit


def out_of_room(node: Node) -> bool:
    return not isinstance(node, FinalQuery) and (
        at_final(node.parent_agent) or budget_nearly_spent(node)
    )


def child_returned(node: Node) -> bool:
    return type(node) is UserQuery and isinstance(node.prev, DoneOutput)


def needs_truncation(node: Node) -> bool:
    if isinstance(node, (FinalQuery, TruncationSummary)):
        return False
    keep = node.parent_agent.config.keep_n_messages
    if keep is None:
        return False
    keep = max(keep, 1)
    return len(node.project(keep=keep + 1)) > keep


DEFAULT_TRANSITIONS = (
    Transitions()
    .always(SOURCES, FinalQuery, when=out_of_room)
    .always(AgentStart, PlanQuery)
    .always(UserQuery, PlanQuery, when=child_returned)
    .always(SOURCES, TruncationSummary, when=needs_truncation)
)


__all__ = [
    "ACT_NAME",
    "DEFAULT_TRANSITIONS",
    "SOURCES",
    "InvalidTransitionError",
    "MissingTransitionError",
    "RESERVED_NAMES",
    "TransitionOption",
    "TransitionPolicyError",
    "TransitionProtocolError",
    "TransitionRule",
    "Transitions",
    "at_final",
    "budget_nearly_spent",
    "child_returned",
    "needs_truncation",
    "out_of_room",
]
