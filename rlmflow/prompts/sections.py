"""A mutable prompt assembled from ordered, named sections."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass


class PromptSectionError(ValueError):
    """Invalid section composition."""


class PinnedPromptSectionError(PromptSectionError):
    """A caller attempted to replace or remove a pinned section."""


@dataclass(frozen=True, slots=True)
class PromptSection:
    """One named contribution to a prompt.

    ``key`` is the update identity; it is intentionally independent of display
    order. ``source`` is provenance for inspection and harness feedback and is
    not rendered into the prompt.
    """

    key: str
    content: str
    order: int = 0
    source: str = ""
    pinned: bool = False

    def __post_init__(self) -> None:
        if not self.key or not self.key.strip():
            raise PromptSectionError("prompt section key cannot be empty")


class Prompt:
    """One mutable system or user prompt, assembled from named sections.

    System and user prompts use the same class. The chat-message role belongs
    to the caller that renders the prompt, not to a section.
    """

    def __init__(self, sections: Iterable[PromptSection] = ()) -> None:
        self._sections: dict[str, PromptSection] = {}
        self._positions: dict[str, int] = {}
        self._next_position = 0
        self.extend(sections)

    def __iter__(self) -> Iterator[PromptSection]:
        return iter(self.sections())

    def __contains__(self, key: object) -> bool:
        return key in self._sections

    def __str__(self) -> str:
        return self.render()

    def get(self, key: str) -> PromptSection | None:
        return self._sections.get(key)

    def sections(self) -> tuple[PromptSection, ...]:
        return tuple(
            sorted(
                self._sections.values(),
                key=lambda section: (section.order, self._positions[section.key]),
            )
        )

    def add_section(
        self,
        key: str,
        content: str,
        *,
        order: int = 0,
        source: str = "",
        pinned: bool = False,
    ) -> Prompt:
        """Add or replace a named section and return ``self``."""
        return self.add(
            PromptSection(
                key,
                content,
                order=order,
                source=source,
                pinned=pinned,
            )
        )

    def add(self, section: PromptSection) -> Prompt:
        """Add an already constructed section and return ``self``."""
        existing = self._sections.get(section.key)
        if existing is not None and existing.pinned and section != existing:
            raise PinnedPromptSectionError(
                f"cannot replace pinned prompt section {section.key!r} "
                f"from {existing.source or 'unknown source'}"
            )
        if existing is None:
            self._positions[section.key] = self._next_position
            self._next_position += 1
        self._sections[section.key] = section
        return self

    def extend(self, sections: Iterable[PromptSection]) -> Prompt:
        for section in sections:
            self.add(section)
        return self

    def update_section(
        self,
        key: str,
        content: str,
        *,
        order: int | None = None,
        source: str | None = None,
        pinned: bool | None = None,
    ) -> Prompt:
        """Update an existing section while retaining unspecified metadata."""
        existing = self._sections.get(key)
        if existing is None:
            raise KeyError(key)
        return self.add_section(
            key,
            content,
            order=existing.order if order is None else order,
            source=existing.source if source is None else source,
            pinned=existing.pinned if pinned is None else pinned,
        )

    def remove_section(self, key: str) -> PromptSection | None:
        """Remove and return one section; pinned sections cannot be removed."""
        existing = self._sections.get(key)
        if existing is None:
            return None
        if existing.pinned:
            raise PinnedPromptSectionError(
                f"cannot remove pinned prompt section {key!r} "
                f"from {existing.source or 'unknown source'}"
            )
        del self._sections[key]
        del self._positions[key]
        return existing

    def render(self) -> str:
        """Join non-empty sections with stable blank lines."""
        return "\n\n".join(
            section.content.strip()
            for section in self.sections()
            if section.content.strip()
        )

    def copy(self) -> Prompt:
        return Prompt(self.sections())


__all__ = [
    "PinnedPromptSectionError",
    "Prompt",
    "PromptSection",
    "PromptSectionError",
]
