"""``rlmflow hook prompt|stop|end``: the Claude Code hooks of the rlmflow plugin.

Each reads the hook's JSON from stdin and talks to the project's ``rlmflow
serve``. Without a serve, or from a session the serve isn't bound to, a hook
prints nothing, except that ``/rlmflow:run TASK`` starts a serve whose root is
this session. A hook never fails the session: errors go to stderr, exit 0.

- ``prompt`` (UserPromptSubmit): opens the root's turn; prints the tools prompt,
  which Claude Code adds as context, on the session's first turn.
- ``stop`` (Stop): ends the turn. If the root is waiting on agents it started,
  holds until their results arrive and prints a ``block`` decision carrying
  them, so Claude continues with them in the same session.
- ``end`` (SessionEnd): shuts the serve down; the run stays on disk.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from rlmflow.agent_tools.cli import ToolError, find_serve, request

EVENTS = ("prompt", "stop", "end")
RUN_COMMAND = "/rlmflow:run"
#: The plugin directory, for ``claude --plugin-dir``.
PLUGIN_DIR = Path(__file__).resolve().parents[1] / "plugin"

#: How long a ``/rlmflow:run`` waits for the serve it started.
START_TIMEOUT = 60.0
#: How long ``stop`` holds for agents' results; under the Stop hook's 600 s timeout.
STOP_WAIT = 540.0


def start_serve(project: Path, args: list[str] | None = None) -> dict[str, Any]:
    """Start ``rlmflow serve`` for ``project`` in the background and wait for its serve.json."""
    log_path = project / ".rlmflow" / "serve.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "ab") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "rlmflow", "serve", "--project", str(project), *(args or [])],
            cwd=project,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
    deadline = time.monotonic() + START_TIMEOUT
    while time.monotonic() < deadline:
        found = find_serve(project)
        if found is not None and found.get("pid") == process.pid:
            return found
        if process.poll() is not None:
            tail = log_path.read_text(errors="replace").strip().splitlines()[-5:]
            raise ToolError("rlmflow serve exited: " + (" | ".join(tail) or f"status {process.returncode}"))
        time.sleep(0.1)
    raise ToolError(f"rlmflow serve did not start within {START_TIMEOUT:.0f}s; see {log_path}")


def call(serve: dict[str, Any], route: str, body: dict[str, Any], timeout: float) -> dict[str, Any]:
    return request(route, body, url=serve["url"], token=serve["serve_token"], timeout=timeout)


def prompt(data: dict[str, Any]) -> str:
    text = data.get("prompt") or ""
    project = Path(data.get("cwd") or Path.cwd())
    starting = text.startswith(RUN_COMMAND)
    serve = find_serve(project)
    body = {key: data.get(key) for key in ("session_id", "prompt", "transcript_path")}
    if serve is not None:
        try:
            reply = call(serve, "prompt", body, timeout=90)
        except ToolError as exc:
            if not starting or "cannot reach" not in str(exc):
                raise
            Path(serve["path"]).unlink(missing_ok=True)
            serve = None
    if serve is None:
        if not starting:
            return ""
        reply = call(start_serve(project), "prompt", body, timeout=90)
    if not reply.get("bound"):
        if starting:
            return "rlmflow: this project's serve is bound to another Claude session; end that one first."
        return ""
    return reply.get("context") or ""


def stop(data: dict[str, Any]) -> str:
    serve = find_serve(data.get("cwd") or Path.cwd())
    if serve is None:
        return ""
    body = {key: data.get(key) for key in ("session_id", "transcript_path", "last_assistant_message")}
    reply = call(serve, "stop", {**body, "timeout": STOP_WAIT}, timeout=STOP_WAIT + 60)
    if reply.get("block"):
        return json.dumps({"decision": "block", "reason": reply["block"]})
    return ""


def end(data: dict[str, Any]) -> str:
    serve = find_serve(data.get("cwd") or Path.cwd())
    if serve is None:
        return ""
    call(serve, "shutdown", {"session_id": data.get("session_id")}, timeout=30)
    return ""


def run(event: str, data: dict[str, Any]) -> str:
    """What the hook prints for ``event``."""
    if event not in EVENTS:
        raise ToolError(f"unknown hook {event!r}; expected one of {', '.join(EVENTS)}")
    return {"prompt": prompt, "stop": stop, "end": end}[event](data)


def main(argv: list[str]) -> int:
    """``rlmflow hook EVENT`` with the hook's JSON on stdin."""
    try:
        if len(argv) != 1:
            raise ToolError(f"usage: rlmflow hook {{{','.join(EVENTS)}}}")
        raw = sys.stdin.read()
        data = json.loads(raw) if raw.strip() else {}
        output = run(argv[0], data if isinstance(data, dict) else {})
    except (ToolError, ValueError, OSError) as exc:
        print(f"rlmflow hook: {exc}", file=sys.stderr)
        return 0
    if output:
        print(output)
    return 0


__all__ = ["EVENTS", "PLUGIN_DIR", "RUN_COMMAND", "STOP_WAIT", "end", "main", "prompt", "run", "start_serve", "stop"]
