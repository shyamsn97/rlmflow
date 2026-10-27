"""A harness for an agent that a person's own session drives, through ``rlmflow serve``.

The agent's turn opens when Flow steps it and stays open until the session's
Stop hook closes it with ``/stop``; meanwhile its ``rlmflow repl`` cells land as
``code`` nodes like any harness's. ``outcome`` then lands ``done`` or
``waiting``, so handoff, forking and replay work as they do for ``claude_code``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rlmflow.graph.nodes import Step
from rlmflow.harnesses.sessions import SessionHarness, TurnResult

if TYPE_CHECKING:
    from rlmflow.agent_tools.server import Turn
    from rlmflow.flow import Flow


def read_transcript(path: str | Path, start: int = 0) -> tuple[dict[str, int], str, int]:
    """Usage and the last assistant text in a Claude Code transcript from line ``start``, and its length."""
    lines = Path(path).read_text(encoding="utf-8").splitlines() if Path(path).is_file() else []
    usage_by_request: dict[str, dict[str, Any]] = {}
    text = ""
    for line in lines[start:]:
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if not isinstance(entry, dict) or entry.get("type") != "assistant":
            continue
        message = entry.get("message") or {}
        usage_by_request[entry.get("requestId") or entry.get("uuid") or str(len(usage_by_request))] = (
            message.get("usage") or {}
        )
        texts = [block.get("text", "") for block in message.get("content") or [] if block.get("type") == "text"]
        if any(texts):
            text = "\n".join(t for t in texts if t)
    usage = {"input_tokens": 0, "output_tokens": 0}
    for counts in usage_by_request.values():
        usage["input_tokens"] += sum(
            int(counts.get(key) or 0)
            for key in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
        )
        usage["output_tokens"] += int(counts.get("output_tokens") or 0)
    return usage, text, len(lines)


class InteractiveHarness(SessionHarness):
    name = "interactive"

    async def run_turn(self, flow: Flow, step: Step, turn: Turn) -> TurnResult:
        return await turn.ended


__all__ = ["InteractiveHarness", "read_transcript"]
