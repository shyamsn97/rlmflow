"""Container-side REPL server — the JSON-over-stdio bridge.

Runs *inside* the sandbox (container image must ``pip install rlmflow``):

    python -m rflow.runtime.repl_server [--workdir DIR]

It reads one JSON command per stdin line and writes one JSON response per stdout
line. Code execution is delegated to the in-process :class:`rflow.runtime.repl.REPL`
(top-level-await detection, suspension, stdout capture, ``DoneSignal``) — this
module only adds the wire protocol and the host-call proxies.

Host-bound tools (``done`` writes the host's ``env``; private child spawn mutates
the host graph; ``HISTORY`` reads the host's graph) are installed as **proxies**.
Calling one ships ``{"proxy": name, ...}`` to the host and blocks on the reply.
Ordinary tools (``FILE_TOOLS`` and friends) are shipped *in* and run here against
the sandbox's own working directory. ``launch_subagents`` is built locally from
the private spawn proxy, and suspension is decided by the local coroutine driver.

Commands: ``inject`` (literal), ``inject_proxy`` (host-call function),
``inject_object`` (host-call object), ``inject_import`` / ``inject_source`` (ship a
local tool in), ``build_launcher`` (wire up ``launch_subagents``), ``run``,
``resume``, ``reset``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from types import SimpleNamespace
from typing import TextIO

from rflow.runtime.repl import REPL, DoneSignal
from rflow.runtime.runtime import deserialize, serialize
from rflow.tools.builtins import (
    DEFAULT_MAX_QUERY_CHARS,
    make_launch_subagents,
)


class ReplServer:
    """Drive an in-process :class:`REPL` over a JSON-line stdin/stdout protocol."""

    def __init__(
        self, protocol_in: TextIO | None = None, protocol_out: TextIO | None = None
    ) -> None:
        # Capture the protocol channel BEFORE constructing REPL(), which wraps
        # sys.stdout with its own capture proxy. Proxy writes must reach the real
        # stdout, not the per-turn capture buffer.
        self._in = protocol_in or sys.stdin
        self._out = protocol_out or sys.stdout
        self.repl = REPL()

    # ── wire protocol ─────────────────────────────────────────────────

    def _write(self, msg: dict) -> None:
        self._out.write(json.dumps(msg) + "\n")
        self._out.flush()

    def make_proxy(self, name: str):
        """A local function that forwards its call to the host and blocks."""

        def proxy(*args, **kwargs):
            self._write(
                {
                    "proxy": name,
                    "args": [serialize(a) for a in args],
                    "kwargs": {k: serialize(v) for k, v in kwargs.items()},
                }
            )
            resp = json.loads(self._in.readline())
            if resp.get("done"):
                raise DoneSignal()
            if "error" in resp:
                raise RuntimeError(resp["error"])
            return deserialize(resp["value"])

        return proxy

    def _format(self, suspended: bool, payload: object) -> dict:
        if suspended:
            request, pre_output = payload  # type: ignore[misc]
            resp: dict = {
                "suspended": True,
                "agent_ids": request.agent_ids,
                "launch_id": request.launch_id,
                "launch_specs": request.launch_specs,
                "launch_names": request.launch_names,
            }
            if pre_output:
                resp["pre_output"] = pre_output
        else:
            resp = {"suspended": False, "output": payload}
        if self.repl.errored:
            resp["errored"] = True
        return resp

    def handle(self, msg: dict) -> dict:
        """Process one command, returning the response dict."""
        cmd = msg.get("cmd")
        if cmd == "run":
            return self._format(*self.repl.start(msg["code"]))
        if cmd == "resume":
            return self._format(*self.repl.resume(msg.get("value")))
        if cmd == "inject":
            self.repl.namespace[msg["name"]] = eval(msg["value"], self.repl.namespace)
            return {"ok": True}
        if cmd == "set_env":
            os.environ.update({str(k): str(v) for k, v in msg["values"].items()})
            return {"ok": True}
        if cmd == "inject_proxy":
            self.repl.namespace[msg["name"]] = self.make_proxy(msg["name"])
            return {"ok": True}
        if cmd == "inject_object":
            name = msg["name"]
            obj = SimpleNamespace()
            for method in msg["methods"]:
                setattr(obj, method, self.make_proxy(f"{name}.{method}"))
            self.repl.namespace[name] = obj
            return {"ok": True}
        if cmd == "inject_import":
            module = importlib.import_module(msg["module"])
            target = module
            for part in msg["qualname"].split("."):
                target = getattr(target, part)
            self.repl.namespace[msg["name"]] = target
            return {"ok": True}
        if cmd == "inject_source":
            scope = dict(self.repl.namespace)
            # Provide the @tool decorator so a shipped, decorated source defines
            # cleanly even if the host module's imports didn't come along.
            from rflow.tools import tool

            scope.setdefault("tool", tool)
            exec(msg["source"], scope)  # noqa: S102 - trusted host-shipped source
            self.repl.namespace[msg["name"]] = scope[msg["func_name"]]
            return {"ok": True}
        if cmd == "build_launcher":
            ns = self.repl.namespace
            ns["launch_subagents"] = make_launch_subagents(
                ns["_rflow_spawn_child"],
                graph_wait=self.repl.graph_wait,
                max_query_chars=msg.get("max_query_chars") or DEFAULT_MAX_QUERY_CHARS,
            )
            return {"ok": True}
        if cmd == "reset":
            self.repl = REPL()
            return {"ok": True}
        return {"error": f"unknown command: {cmd!r}"}

    def serve(self) -> None:
        """Read commands from stdin until EOF, writing one response each."""
        while True:
            line = self._in.readline()
            if not line:
                break
            self._write(self.handle(json.loads(line)))


def main() -> None:
    parser = argparse.ArgumentParser(description="rlmflow remote REPL server")
    parser.add_argument("--workdir", help="chdir before starting")
    args = parser.parse_args()
    if args.workdir:
        os.chdir(args.workdir)
    ReplServer().serve()


if __name__ == "__main__":
    main()
