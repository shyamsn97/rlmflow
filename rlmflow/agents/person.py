"""A person's own session as a harness, through ``rlmflow serve`` and the Claude Code plugin's hooks.

A step waits for the session's Stop hook (``/stop``); meanwhile its
``rlmflow repl`` runs land as code nodes like any harness's. The plugin's
prompt hook (``/prompt``) gives the session the same system prompt and
messages any harness gets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rlmflow.agents.agent import LLM, Agent
from rlmflow.agents.prompts import USE_RLMFLOW_REPL
from rlmflow.graph.nodes import Node

if TYPE_CHECKING:
    from rlmflow.repl import Repl


class Person(Agent):
    """Your own Claude Code session; a person takes as many steps as they like."""

    protocol = USE_RLMFLOW_REPL
    # A launched person would wait for a session nobody opens.
    child_agent = "default"

    def __init__(self, model: str | None = None, **shared: Any) -> None:
        shared.setdefault("max_iters", None)
        super().__init__(model, **shared)

    async def step(self, node: Node, llm: LLM, repl: Repl) -> Node:
        out = await repl.turn_ended()
        return self.done(out, out.text, repl, since=node)


__all__ = ["Person"]
