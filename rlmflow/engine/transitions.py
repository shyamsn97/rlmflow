"""Empty table of ``@transitions.on`` producers and named choice menus."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass

from rlmflow.graph.nodes import Node, UserQuery

Guard = Callable[[Node], bool]
Producer = Callable[..., Node]
RESERVED_NAMES = frozenset({"finish"})


class TransitionPolicyError(ValueError):
    """Invalid transition policy declared by the host application."""


class TransitionProtocolError(RuntimeError):
    """Invalid transition selection produced by the model."""


class InvalidTransitionError(TransitionProtocolError):
    """The model selected a transition unavailable from the current behavior."""

    def __init__(self, name: str, available: Iterable[str]) -> None:
        choices = tuple(available)
        super().__init__(f"transition {name!r} is unavailable; available: {list(choices)!r}")
        self.name = name
        self.available = choices


@dataclass(frozen=True, slots=True)
class Row:
    sources: tuple[type[Node], ...]
    fn: Producer
    when: Guard | None = None

    def matches(self, node: Node) -> bool:
        return isinstance(node, self.sources) and (self.when is None or self.when(node))


@dataclass(frozen=True)
class TransitionOption:
    name: str
    description: str
    target: type[UserQuery]


@dataclass(frozen=True, slots=True)
class ChoiceSet:
    current: type[UserQuery]
    targets: tuple[type[UserQuery], ...]
    when: Guard | None = None


class Transitions:
    """Producer rows plus named ``transition("...")`` choices. Starts empty."""

    def __init__(self) -> None:
        self._rows: list[Row] = []
        self._choices: list[ChoiceSet] = []
        self._base: Transitions | None = None

    def derive(self) -> Transitions:
        """Child table: own rows win, then this table."""
        child = Transitions()
        child._base = self
        return child

    def on(self, *sources: type[Node], when: Guard | None = None):
        """Register a producer. ``@table.on(UserQuery)`` or stacked."""
        _validate_sources(sources)

        def register(fn: Producer) -> Producer:
            self._rows.append(Row(sources=sources, fn=fn, when=when))
            return fn

        return register

    def choices(
        self,
        current: type[UserQuery],
        *targets: type[UserQuery],
        when: Guard | None = None,
    ) -> Transitions:
        """Allow ``transition("...")`` to select a target from ``current``."""
        _validate_behavior(current)
        if not targets:
            raise TransitionPolicyError("a transition choice list cannot be empty")
        for target in targets:
            _validate_behavior(target)
        names = [target.name for target in targets]
        if len(names) != len(set(names)):
            raise TransitionPolicyError(
                f"duplicate transition names from {current.__name__}: {names!r}"
            )
        existing = {
            target.name for choices in self._own_choices(current) for target in choices.targets
        }
        repeated = existing.intersection(names)
        if repeated:
            raise TransitionPolicyError(
                f"duplicate transition names from {current.__name__}: {sorted(repeated)!r}"
            )
        self._choices.append(ChoiceSet(current=current, targets=targets, when=when))
        return self

    def _own_choices(self, current: type[UserQuery]) -> Iterator[ChoiceSet]:
        for choices in self._choices:
            if choices.current is current:
                yield choices

    def resolve(self, node: Node) -> Producer:
        for row in self._rows:
            if row.matches(node):
                return row.fn
        if self._base is not None:
            return self._base.resolve(node)
        raise TypeError(f"cannot step {type(node).__name__}")

    def available(self, node: Node) -> tuple[TransitionOption, ...]:
        behavior = self.current_behavior(node)
        if behavior is None:
            return ()
        found: dict[str, TransitionOption] = {}
        for choices in self._walk_choices():
            if not isinstance(behavior, choices.current):
                continue
            if choices.when is not None and not choices.when(node):
                continue
            for target in choices.targets:
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
        """Return the latest query participating in a choice set."""
        classes: dict[type[UserQuery], None] = {}
        for choices in self._walk_choices():
            classes.setdefault(choices.current, None)
            for target in choices.targets:
                classes.setdefault(target, None)
        if not classes:
            return None
        kinds = tuple(classes)
        return next(
            (item for item in node.iter_backwards() if isinstance(item, kinds)),
            None,
        )

    def resolve_choice(self, node: Node, name: str) -> TransitionOption | None:
        return next((option for option in self.available(node) if option.name == name), None)

    def _walk_choices(self) -> Iterator[ChoiceSet]:
        current: Transitions | None = self
        while current is not None:
            yield from current._choices
            current = current._base

    def __str__(self) -> str:
        lines = []
        for row in self._rows:
            source = "|".join(item.__name__ for item in row.sources)
            guard = f" when {row.when.__name__}" if row.when is not None else ""
            lines.append(f"{source} -> {row.fn.__name__}{guard}")
        return "\n".join(lines)


def _validate_sources(sources: tuple[type[Node], ...]) -> None:
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


__all__ = [
    "InvalidTransitionError",
    "RESERVED_NAMES",
    "TransitionOption",
    "TransitionPolicyError",
    "TransitionProtocolError",
    "Transitions",
]
