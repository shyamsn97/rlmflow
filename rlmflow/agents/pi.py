"""pi as a harness: one ``pi -p --mode json`` per step, resuming its session file.

pi has no MCP, so it reaches the agent tools through the ``rlmflow`` CLI. Its
sessions are JSONL files under ``~/.rlmflow/pi-sessions``, so resuming passes
the same file and a fork copies it.
"""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rlmflow.agents.agent import LLM, Agent, as_prompt, forked, last_session
from rlmflow.agents.claude_code import SESSION_NAMESPACE
from rlmflow.agents.prompts import USE_RLMFLOW_REPL
from rlmflow.graph.nodes import Node

if TYPE_CHECKING:
    from rlmflow.repl import Repl

SESSION_DIR = Path("~/.rlmflow/pi-sessions")


class Pi(Agent):
    """pi. ``command`` is the executable; ``args`` are extra arguments, e.g. ``["--provider", "openai"]``."""

    protocol = USE_RLMFLOW_REPL

    def __init__(
        self,
        model: str | None = None,
        *,
        command: str = "pi",
        args: list[str] | tuple[str, ...] = (),
        **shared: Any,
    ) -> None:
        super().__init__(model, **shared)
        self.command = command
        self.args = list(args)

    async def step(self, node: Node, llm: LLM, repl: Repl) -> Node:
        last = last_session(node)
        system, *new = self.messages(node, repl, since=last)
        session = session_file(last, repl)
        argv = [self.command, "-p", as_prompt(new), "--mode", "json", "--session", str(session)]
        argv += ["--append-system-prompt", system["content"]]
        if self.model is not None:
            argv += ["--model", self.model]
        argv += self.args
        stdout = await repl.run_command(argv)
        events = []
        for line in stdout.splitlines():
            if line.strip():
                try:
                    events.append(json.loads(line))
                except ValueError as exc:
                    raise RuntimeError(f"pi returned invalid JSON: {exc}") from exc
        text, usage = final_reply(events)
        out = Node(
            "assistant", text, data={"session": str(session), "repl": repl.id, "usage": usage}
        )
        return self.done(out, text, repl, since=node)


def session_file(last: Node | None, repl: Repl) -> Path:
    """The JSONL file a step runs in: the last one to resume, a copy of it after a fork, or a new one."""
    if last is not None and not forked(last, repl):
        return Path(last.data["session"])
    directory = SESSION_DIR.expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    if last is None:
        name = uuid.uuid5(SESSION_NAMESPACE, f"{repl.id}:{len(repl.root.nodes())}")
        return directory / f"{name}.jsonl"
    copy = directory / f"{uuid.uuid5(SESSION_NAMESPACE, f'{repl.id}:fork')}.jsonl"
    shutil.copyfile(last.data["session"], copy)
    return copy


def final_reply(events: list[dict[str, Any]]) -> tuple[str, dict[str, int]]:
    """The last assistant message's text, and the tokens every assistant message used."""
    messages = next((e["messages"] for e in reversed(events) if e.get("type") == "agent_end"), None)
    if messages is None:
        messages = [
            e["message"] for e in events if e.get("type") == "message_end" and "message" in e
        ]
    replies = [m for m in messages if m.get("role") == "assistant"]
    if not replies:
        raise RuntimeError("pi returned no assistant message")
    last = replies[-1]
    if last.get("stopReason") == "error":
        raise RuntimeError(f"pi failed: {last.get('errorMessage') or 'unknown error'}")
    text = "".join(
        part.get("text", "") for part in last.get("content") or [] if part.get("type") == "text"
    )
    usage = {
        "input_tokens": sum(int((m.get("usage") or {}).get("input") or 0) for m in replies),
        "output_tokens": sum(int((m.get("usage") or {}).get("output") or 0) for m in replies),
    }
    return text.strip(), usage


__all__ = ["Pi", "final_reply", "session_file"]
