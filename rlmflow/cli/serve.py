"""``rlmflow serve``: a long-lived Flow whose root run is your own Claude Code session.

The root runs ``Person()``, registered as the ``person`` agent. The plugin's
hooks call ``/prompt`` when you send a message and ``/stop`` when Claude
finishes replying. A prompt starts the root or is attached to it as a user
node; Flow steps it, which waits for ``/stop``, and your session's ``rlmflow
repl`` runs land in that step through the root's token in
``.rlmflow/serve.json``. If the root is left waiting on runs it launched,
``/stop`` holds until their results start its next step and returns them, so
the Stop hook can hand them to Claude.

Between prompts the run loop idles; the graph is checkpointed under
``.rlmflow/runs/<ts>/`` and a restarted serve continues from it.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import time
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from rlmflow.agent_tools.cli import SERVE_FILE
from rlmflow.agent_tools.hooks import RUN_COMMAND, STOP_WAIT
from rlmflow.agents.agent import as_prompt, last_session
from rlmflow.agents.person import Person
from rlmflow.graph.graph import Graph
from rlmflow.graph.nodes import SELF, AgentRoot, Node

if TYPE_CHECKING:
    from rlmflow.flow import Flow

T = TypeVar("T")

#: The agent ``serve`` registers on its Flow for its root, and what ``fork --interactive`` picks.
PERSON = "person"

#: How long ``/prompt`` waits for Flow to start the root's step.
OPEN_TIMEOUT = 30.0
#: How long ``/stop`` waits for the step's end to land.
LAND_TIMEOUT = 60.0


class Serve:
    """One project's serve: a Flow, its graph, and the root run bound to one session."""

    def __init__(
        self,
        flow: Flow,
        graph: Graph,
        *,
        graph_dir: str | Path,
        project: str | Path,
        root: AgentRoot | None = None,
        session_id: str | None = None,
    ) -> None:
        from rlmflow.budget import Checkpoint

        self.flow, self.graph = flow, graph
        self.graph_dir = Path(graph_dir)
        self.project = Path(project).resolve()
        flow.agents[PERSON] = person_agent(flow)
        self.root = root if root is not None else person_root(graph)
        if self.root is not None:
            self.root.agent = PERSON
        self.session_id = session_id or (current_session(self.root) if self.root else None)
        self.checkpointer = Checkpoint(self.graph_dir)
        self.wakeup = asyncio.Event()
        self.stopping = asyncio.Event()
        self.draining = False
        self.error: BaseException | None = None
        self.agent_token: str | None = None
        self.host_token: str | None = None
        self.url: str | None = None
        self._briefed: set[str] = set()
        self._shown: set[str] = set()
        self._transcript_line = 0

    @property
    def serve_file(self) -> Path:
        return self.project / SERVE_FILE

    # -- lifecycle ----------------------------------------------------------

    async def start(self) -> None:
        server = self.flow.agent_tools
        self.url = await server.start()
        self.host_token = server.serve(self)
        if self.root is not None:
            self.agent_token = server.token(self.root)
        self.write_serve_file()

    def write_serve_file(self) -> None:
        payload = {
            "url": self.url,
            "serve_token": self.host_token,
            "agent_token": self.agent_token,
            "graph_dir": str(self.graph_dir),
            "root": self.root.id if self.root else None,
            "session_id": self.session_id,
            "pid": os.getpid(),
        }
        path = self.serve_file
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".tmp")
        fd = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2)
        os.replace(partial, path)

    async def run(self) -> None:
        """Serve until ``/shutdown``: step whatever is ready, then idle until a prompt arrives.

        A shutdown after the root finished lets the runs still going finish first.
        """
        await self.start()
        stopping = asyncio.ensure_future(self.stopping.wait())
        try:
            while not self.stopping.is_set():
                self.wakeup.clear()
                streaming = asyncio.ensure_future(self._stream())
                await asyncio.wait({streaming, stopping}, return_when=asyncio.FIRST_COMPLETED)
                if self.stopping.is_set():
                    streaming.cancel()
                    with suppress(asyncio.CancelledError):
                        await streaming
                    break
                if self.draining:
                    break
                woken = asyncio.ensure_future(self.wakeup.wait())
                await asyncio.wait({woken, stopping}, return_when=asyncio.FIRST_COMPLETED)
                woken.cancel()
        finally:
            stopping.cancel()
            self.graph.save(self.graph_dir)
            self.release_serve_file()
            await self.flow.aclose()

    def release_serve_file(self) -> None:
        """Remove serve.json if it is still this serve's; a newer serve may have replaced it."""
        try:
            if json.loads(self.serve_file.read_text()).get("pid") == os.getpid():
                self.serve_file.unlink()
        except (FileNotFoundError, ValueError):
            pass

    async def _stream(self) -> None:
        try:
            await self.flow.arun(self.graph, hooks=[self.checkpointer])
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - serve outlives a failed run
            self.error = exc
            print(f"rlmflow serve: the run failed: {type(exc).__name__}: {exc}", file=sys.stderr)

    def kick(self) -> None:
        """New work: wake the running loop, or start the next one."""
        self.wakeup.set()
        self.flow.wake(self.graph)

    # -- the hooks' routes --------------------------------------------------

    def bound(self, session_id: Any) -> bool:
        return self.session_id is None or session_id == self.session_id

    def open(self) -> bool:
        """Whether the root's step is open, waiting for the session's turn to end."""
        return self.root is not None and self.flow.agent_tools.waiting(self.root)

    async def prompt(self, body: dict[str, Any]) -> dict[str, Any]:
        session_id, text = body.get("session_id"), body.get("prompt") or ""
        if not isinstance(text, str):
            raise ValueError("prompt must be a string")
        if not self.bound(session_id):
            return {"bound": False}
        if self.root is None:
            task = text.removeprefix(RUN_COMMAND).strip() if text.startswith(RUN_COMMAND) else text
            self.root = self.graph.start(task, name="root", agent=PERSON)
            self.agent_token = self.flow.agent_tools.token(self.root)
        elif not self.open():
            self.graph.send(self.root, text, sender="user")
        self.session_id = session_id
        self.write_serve_file()
        if body.get("transcript_path"):
            self._transcript_line = read_transcript(body["transcript_path"])[2]
        self.kick()
        if await until(lambda: True if self.open() else None, OPEN_TIMEOUT) is None:
            raise RuntimeError(
                f"the root's step did not start: {self.error or 'nothing stepped it'}"
            )
        return {"bound": True, "root": self.root.id, "context": self._context(session_id)}

    def _context(self, session_id: Any) -> str:
        """What the session hasn't seen: the system prompt, the whole run if no session has it, and news.

        A session resumed from the run's last session step (or forked from it)
        already has the run up to there.
        """
        root = self.root
        node = root.nodes()[-1]
        agent, repl = self.flow.agent(root), self.flow.repl(root)
        parts = []
        if session_id not in self._briefed:
            self._briefed.add(session_id)
            parts.append(agent.system_prompt(node, repl))
            if last_session(node) is None and len(root.nodes()) > 1:
                parts.append(as_prompt(agent.messages(node, repl)[1:]))
                self._shown.update(n.id for n in root.nodes())
        parts += self._news()
        return "\n\n".join(part for part in parts if part)

    def _news(self) -> list[str]:
        """What others wrote since the root's last step began, not yet shown to the session."""
        turn = self.root.last_turn()
        start = turn.data["seen"] if turn is not None else 1
        news = [
            node
            for node in self.root.nodes()[start:]
            if node.producer not in (SELF, "user") and node.text and node.id not in self._shown
        ]
        self._shown.update(node.id for node in news)
        return [node.text for node in news]

    async def stop(self, body: dict[str, Any]) -> dict[str, Any]:
        session_id = body.get("session_id")
        if self.root is None or not self.bound(session_id):
            return {"bound": False}
        if not self.open():
            return {"bound": True, "landed": None}
        usage, text, self._transcript_line = read_transcript(
            body.get("transcript_path") or "", self._transcript_line
        )
        out = Node(
            "assistant",
            body.get("last_assistant_message") or text,
            data={"session": session_id, "repl": self.root.repl_id, "usage": usage},
        )
        self.flow.agent_tools.end_step(self.root, out)
        landed = await until(lambda: out if out.graph is not None else None, LAND_TIMEOUT)
        if landed is None:
            raise RuntimeError(f"the root's step did not land: {self.error or 'timed out'}")
        ended = "done" if out.terminal else "idle" if out.idle else "running"
        reply: dict[str, Any] = {"bound": True, "landed": ended}
        if ended == "idle":
            timeout = float(body.get("timeout", STOP_WAIT))
            if await until(lambda: True if self.open() else None, timeout) is not None:
                reply["block"] = "\n\n".join(self._news()) or "Continue."
        return reply

    async def shutdown(self, body: dict[str, Any]) -> dict[str, Any]:
        if body.get("session_id") != self.session_id:
            return {"bound": False}
        stream = self.flow._streams.get(self.graph.id)
        if self.root is not None and self.root.status() == "done" and stream and stream.tasks:
            self.draining = True
            self.release_serve_file()
            return {"bound": True, "stopping": True, "draining": len(stream.tasks)}
        asyncio.get_running_loop().call_later(0.05, self.stopping.set)
        return {"bound": True, "stopping": True}


async def until(probe: Callable[[], T | None], timeout: float, interval: float = 0.02) -> T | None:
    """Poll ``probe`` until it returns something, or ``None`` after ``timeout`` seconds."""
    deadline = time.monotonic() + timeout
    while True:
        found = probe()
        if found is not None or time.monotonic() >= deadline:
            return found
        await asyncio.sleep(interval)


def person_agent(flow: Flow) -> Person:
    """``Person()`` with the ``default`` agent's model (for ``llm_query``), prompt and limits."""
    base = flow.agents.get("default")
    if base is None:
        return Person()
    return Person(
        base.model,
        prompt=base.prompt,
        description=base.description,
        max_depth=base.max_depth,
        output_schema=base.output_schema,
    )


def person_root(graph: Graph) -> AgentRoot | None:
    """The latest run in ``graph`` a person drives."""
    found = [run for run in graph.agent_roots() if run.agent == PERSON]
    return found[-1] if found else None


def current_session(run: AgentRoot) -> str | None:
    """The session that ended ``run``'s latest step in its own REPL; a fork has none."""
    last = last_session(run.nodes()[-1])
    if last is None or last.data.get("repl", last.root_id) != (run.repl_id or run.id):
        return None
    return last.data["session"]


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
        usage_by_request[
            entry.get("requestId") or entry.get("uuid") or str(len(usage_by_request))
        ] = message.get("usage") or {}
        texts = [
            block.get("text", "")
            for block in message.get("content") or []
            if block.get("type") == "text"
        ]
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


def serve_main(
    project: str | Path,
    *,
    graph_dir: str | Path | None = None,
    root: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> None:
    """Build the project's Flow (the coding one, or ``flow = "module:factory"``'s) and serve it."""
    from rlmflow.cli.app import resolve_options
    from rlmflow.cli.run import build_flow

    project = Path(project).expanduser().resolve()
    options, _ = resolve_options({**(overrides or {}), "workdir": str(project)}, cwd=project)
    if graph_dir is not None and Path(graph_dir).exists():
        graph = Graph.load(graph_dir)
    else:
        graph_dir = graph_dir or project / ".rlmflow" / "runs" / time.strftime("%Y%m%d-%H%M%S")
        graph = Graph()
    root_run = None
    if root is not None:
        from rlmflow.tools.agents import resolve_agent

        root_run = resolve_agent(graph, root)
    flow = build_flow(options, workdir=project)

    async def main() -> None:
        serve = Serve(flow, graph, graph_dir=graph_dir, project=project, root=root_run)
        loop = asyncio.get_running_loop()
        for signum in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(signum, serve.stopping.set)
        await serve.run()
        print(f"run saved to {serve.graph_dir}")

    asyncio.run(main())


__all__ = [
    "PERSON",
    "Serve",
    "current_session",
    "person_agent",
    "person_root",
    "read_transcript",
    "serve_main",
    "until",
]
