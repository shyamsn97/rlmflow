"""``rlmflow repl/spawn/wait/send/done/agents``: the agent tools from a shell.

A harness starts its session with ``RLMFLOW_URL``, ``RLMFLOW_AGENT`` and
``RLMFLOW_TOKEN``; each command is one request to that Flow's agent-tools server.
Outside such a session, the commands find ``rlmflow serve`` through the nearest
``.rlmflow/serve.json`` above the working directory.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

COMMANDS = ("repl", "spawn", "wait", "send", "done", "agents")

#: How long ``rlmflow repl`` waits for a cell by default, in seconds.
REPL_TIMEOUT = 600.0

SERVE_FILE = Path(".rlmflow") / "serve.json"


class ToolError(Exception):
    """A request the server refused or couldn't serve."""


def find_serve(start: str | Path | None = None) -> dict[str, Any] | None:
    """The nearest ``.rlmflow/serve.json`` at or above ``start``, if any."""
    here = Path(start or Path.cwd()).resolve()
    for directory in (here, *here.parents):
        path = directory / SERVE_FILE
        if not path.is_file():
            continue
        if path.stat().st_mode & 0o077:
            raise ToolError(f"{path} must be readable only by you (chmod 600)")
        return {**json.loads(path.read_text()), "path": str(path)}
    return None


def endpoint() -> tuple[str, str]:
    """The URL and token to call: the turn in the environment, else the nearest serve."""
    url, token = os.environ.get("RLMFLOW_URL"), os.environ.get("RLMFLOW_TOKEN")
    if url and token:
        return url, token
    serve = find_serve()
    if serve is None:
        raise ToolError(
            "not inside an rlmflow agent turn: RLMFLOW_URL and RLMFLOW_TOKEN are unset "
            "and no .rlmflow/serve.json was found"
        )
    return serve["url"], serve["agent_token"]


def request(
    tool: str,
    body: dict[str, Any],
    *,
    timeout: float = 120.0,
    url: str | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    """POST ``body`` to ``tool``, on ``url`` with ``token`` or wherever ``endpoint`` points."""
    if url is None or token is None:
        url, token = endpoint()
    req = urllib.request.Request(
        f"{url}/{tool}",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        try:
            message = json.loads(exc.read()).get("error") or exc.reason
        except ValueError:
            message = exc.reason
        raise ToolError(str(message)) from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        reason = getattr(exc, "reason", exc)
        raise ToolError(f"cannot reach the rlmflow server at {url}: {reason}") from exc


def json_or_text(value: str) -> Any:
    try:
        return json.loads(value)
    except ValueError:
        return value


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="rlmflow", description="The agent tools of an rlmflow run.")
    commands = root.add_subparsers(dest="command", required=True)

    repl = commands.add_parser("repl", help="run Python in this agent's persistent REPL")
    repl.add_argument("code", nargs="?", help="the code; read from stdin when omitted")
    repl.add_argument("-f", "--file", help="read the code from this file")
    repl.add_argument(
        "--timeout", type=float, default=REPL_TIMEOUT, help="seconds to wait for the cell"
    )

    spawn = commands.add_parser("spawn", help="start an agent; prints its id")
    spawn.add_argument("task")
    spawn.add_argument("--config", help="a config name on the run; defaults to your own")
    spawn.add_argument("--name")
    spawn.add_argument(
        "--arg",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="an argument for the agent; VALUE is JSON or text",
    )

    wait = commands.add_parser("wait", help="wait up to a minute for agents; prints JSON")
    wait.add_argument(
        "agents", nargs="*", help="agent ids, paths or names; default: your running children"
    )
    wait.add_argument("--timeout", type=float, default=60.0)

    send = commands.add_parser("send", help="put a message in another agent's inbox")
    send.add_argument("agent")
    send.add_argument("text")

    done = commands.add_parser("done", help="record your final result")
    done.add_argument("result", nargs="?", help="JSON or text; read from stdin when omitted")

    commands.add_parser("agents", help="the run's agent tree with status and results")
    return root


def run(argv: list[str]) -> int:
    args = parser().parse_args(argv)
    match args.command:
        case "repl":
            if args.file:
                with open(args.file, encoding="utf-8") as handle:
                    code = handle.read()
            else:
                code = args.code if args.code is not None else sys.stdin.read()
            ran = request("repl", {"code": code}, timeout=args.timeout)
            if ran["output"]:
                print(ran["output"], end="" if ran["output"].endswith("\n") else "\n")
            if ran["submitted"]:
                print("[submitted]")
            return 1 if ran["error"] else 0
        case "spawn":
            values = {}
            for item in args.arg:
                name, sep, value = item.partition("=")
                if not sep:
                    raise ToolError(f"--arg takes NAME=VALUE, not {item!r}")
                values[name] = json_or_text(value)
            body = {"task": args.task, "config": args.config, "name": args.name, "args": values}
            print(request("spawn", body)["id"])
        case "wait":
            waited = request(
                "wait", {"agents": args.agents, "timeout": args.timeout}, timeout=args.timeout + 30
            )
            print(json.dumps(waited["results"], indent=2, default=str))
            if waited.get("note"):
                print(waited["note"])
        case "send":
            request("send", {"agent": args.agent, "text": args.text})
        case "done":
            value = args.result if args.result is not None else sys.stdin.read()
            request("done", {"result": json_or_text(value)})
            print("[submitted]")
        case "agents":
            print(request("agents", {})["text"])
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(list(sys.argv[1:] if argv is None else argv))
    except ToolError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


__all__ = ["COMMANDS", "SERVE_FILE", "ToolError", "endpoint", "find_serve", "main", "request"]
