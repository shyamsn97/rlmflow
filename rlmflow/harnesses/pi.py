"""pi as a harness: one ``pi -p --mode json`` session turn per step.

pi has no MCP and no permission system, so it reaches the agent tools through
the ``rlmflow`` CLI and strict tools are prompt guidance only. Its sessions are
JSONL files, so resuming passes the same file and forking copies it.

``harness_options``: ``provider`` (pi's ``--provider``), ``tools`` (``"strict"``
or ``"loose"``), ``session_dir`` (default ``~/.rlmflow/pi-sessions``), and
``command`` (``"pi"`` by default).
"""

from __future__ import annotations

import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rlmflow.graph.nodes import Step
from rlmflow.harnesses.sessions import (
    SESSION_NAMESPACE,
    Session,
    SessionHarness,
    SessionOptions,
    TurnResult,
    json_payload,
    next_prompt,
    system_prompt,
)

if TYPE_CHECKING:
    from rlmflow.agent_tools.server import Turn
    from rlmflow.flow import Flow


@dataclass(frozen=True)
class PiOptions(SessionOptions):
    command: str = "pi"
    provider: str | None = None
    session_dir: str = "~/.rlmflow/pi-sessions"


class PiHarness(SessionHarness):
    name = "pi"
    Options = PiOptions

    async def run_turn(self, flow: Flow, step: Step, turn: Turn) -> TurnResult:
        options, config = self.options(step), step.agent.config
        session = Session.for_step(step)
        path = self.session_file(step, session)
        argv = [options.command, "-p", await next_prompt(flow, step, session), "--mode", "json"]
        argv += ["--session", str(path)]
        argv += ["--append-system-prompt", system_prompt(flow, step, strict=options.strict)]
        if options.provider:
            argv += ["--provider", options.provider]
        if config.model != "default":
            argv += ["--model", config.model]

        stdout = await self.run_command(flow, step, argv, turn.env)
        events = [json_payload(line, "pi") for line in stdout.splitlines() if line.strip()]
        text, usage = self.final_reply(events)
        return TurnResult(session_id=str(path), text=text, usage=usage)

    def session_file(self, step: Step, session: Session) -> Path:
        """The JSONL file this turn runs in; a fork gets a copy of the one it resumes."""
        if session.resume and not session.fork:
            return Path(session.id)
        directory = Path(self.options(step).session_dir).expanduser()
        directory.mkdir(parents=True, exist_ok=True)
        if not session.resume:
            return directory / f"{session.id}.jsonl"
        copy = directory / f"{uuid.uuid5(SESSION_NAMESPACE, f'{step.agent.id}:fork')}.jsonl"
        shutil.copyfile(session.id, copy)
        return copy

    @staticmethod
    def final_reply(events: list[dict[str, Any]]) -> tuple[str, dict[str, int]]:
        """The last assistant message's text, and the tokens every assistant message used."""
        messages = next((e["messages"] for e in reversed(events) if e.get("type") == "agent_end"), None)
        if messages is None:
            messages = [e["message"] for e in events if e.get("type") == "message_end" and "message" in e]
        replies = [m for m in messages if m.get("role") == "assistant"]
        if not replies:
            raise RuntimeError("pi returned no assistant message")
        last = replies[-1]
        if last.get("stopReason") == "error":
            raise RuntimeError(f"pi failed: {last.get('errorMessage') or 'unknown error'}")
        text = "".join(part.get("text", "") for part in last.get("content") or [] if part.get("type") == "text")
        usage = {
            "input_tokens": sum(int((m.get("usage") or {}).get("input") or 0) for m in replies),
            "output_tokens": sum(int((m.get("usage") or {}).get("output") or 0) for m in replies),
        }
        return text.strip(), usage


__all__ = ["PiHarness", "PiOptions"]
