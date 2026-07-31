"""Tiny decorator-backed component registry."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class Registered(Generic[T]):
    name: str
    factory: Callable[..., T]
    aliases: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    defaults: dict = field(default_factory=dict)

    def make(self, **kwargs) -> T:
        return self.factory(**{**self.defaults, **kwargs})


class Registry(Generic[T]):
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, Registered[T]] = {}
        self._aliases: dict[str, list[str] | str] = {}

    def register(
        self,
        name: str,
        factory: Callable[..., T],
        *,
        aliases: Iterable[str] = (),
        tags: Iterable[str] = (),
        defaults: dict | None = None,
    ) -> Callable[..., T]:
        item = Registered(
            name=name,
            factory=factory,
            aliases=tuple(aliases),
            tags=tuple(tags),
            defaults=defaults or {},
        )
        self._items[name] = item
        for alias in item.aliases:
            self._aliases[alias] = name
        return factory

    def decorator(
        self,
        name: str,
        *,
        aliases: Iterable[str] = (),
        tags: Iterable[str] = (),
        defaults: dict | None = None,
    ):
        def wrap(factory: Callable[..., T]) -> Callable[..., T]:
            self.register(name, factory, aliases=aliases, tags=tags, defaults=defaults)
            if isinstance(factory, type):
                factory.name = name
            return factory

        return wrap

    def alias(self, name: str, values: Iterable[str]) -> None:
        self._aliases[name] = list(values)

    def make(self, component: str, **kwargs) -> T:
        resolved = self.resolve_one(component)
        try:
            return self._items[resolved].make(**kwargs)
        except KeyError as exc:
            raise ValueError(
                f"unknown {self.kind} {component!r}; available: {', '.join(self.names())}"
            ) from exc

    def names(self, *, tags: Iterable[str] | None = None) -> list[str]:
        if tags is None:
            return sorted(self._items)
        wanted = set(tags)
        return sorted(name for name, item in self._items.items() if wanted.intersection(item.tags))

    def expand(self, values: Iterable[str]) -> list[str]:
        expanded: list[str] = []
        for value in values:
            for part in str(value).split(","):
                name = part.strip()
                if not name:
                    continue
                alias = self._aliases.get(name)
                if isinstance(alias, list):
                    expanded.extend(alias)
                elif isinstance(alias, str):
                    expanded.append(alias)
                else:
                    expanded.append(name)
        unknown = [name for name in expanded if name not in self._items]
        if unknown:
            raise ValueError(
                f"unknown {self.kind}(s): {', '.join(unknown)}; "
                f"available: {', '.join(self.names())}"
            )
        return list(dict.fromkeys(expanded))

    def resolve_one(self, name: str) -> str:
        alias = self._aliases.get(name)
        if isinstance(alias, str):
            return alias
        return name


__all__ = ["Registered", "Registry"]
