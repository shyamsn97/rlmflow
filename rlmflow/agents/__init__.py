"""Agents: the harnesses that drive a model for one run."""

from __future__ import annotations

from typing import Any

from rlmflow.agents.agent import (
    LLM,
    MAX_ITERS_EXCEEDED,
    Agent,
    as_messages,
    as_prompt,
    final_turn,
    last_session,
    parse_code,
    repl_reply,
    steps,
)


def __getattr__(name: str) -> Any:
    if name in ("ClaudeCode", "Pi", "Person"):
        from importlib import import_module

        module = {"ClaudeCode": "claude_code", "Pi": "pi", "Person": "person"}[name]
        return getattr(import_module(f"rlmflow.agents.{module}"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "LLM",
    "MAX_ITERS_EXCEEDED",
    "Agent",
    "ClaudeCode",
    "Person",
    "Pi",
    "as_messages",
    "as_prompt",
    "final_turn",
    "last_session",
    "parse_code",
    "repl_reply",
    "steps",
]
