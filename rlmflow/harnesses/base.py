"""The harness interface: ``await harness(flow, step)`` takes one turn for ``step.agent``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from rlmflow.graph.nodes import Node, Step

if TYPE_CHECKING:
    from rlmflow.flow import Flow


@dataclass(frozen=True)
class NoOptions:
    """A harness that takes no ``harness_options``."""


class Harness:
    """Takes one turn for one agent.

    Subclasses set ``name`` and implement ``step``, which appends the turn's
    nodes with ``flow.append`` (running code with ``flow.run_node``); the node
    that ends the turn carries ``seen``, how much of the history it read. A
    config names a harness by short name or by import path; a class is
    instantiated when it's loaded, and a plain ``async (flow, step)`` function
    works as a harness too.
    """

    name: ClassVar[str]
    #: The dataclass an agent's ``config.harness_options`` are read into.
    Options: ClassVar[type] = NoOptions

    async def __call__(self, flow: Flow, step: Step) -> Node | None:
        return await self.step(flow, step)

    async def step(self, flow: Flow, step: Step) -> Node | None:
        raise NotImplementedError

    def prompt_sections(self, flow: Flow, step: Step) -> dict[str, str | None]:
        """The default system prompt as named sections, in order."""
        return {}

    def prompt(self, flow: Flow, step: Step) -> str:
        """``prompt_sections`` with ``config.prompt`` on top, joined into the system prompt."""
        sections = {**self.prompt_sections(flow, step), **step.config.prompt}
        return "\n\n".join(text.strip() for text in sections.values() if text and text.strip())

    def options(self, step: Step) -> Any:
        try:
            return self.Options(**step.config.harness_options)
        except TypeError as exc:
            raise ValueError(f"invalid {self.name} harness_options: {exc}") from exc


__all__ = ["Harness", "NoOptions"]
