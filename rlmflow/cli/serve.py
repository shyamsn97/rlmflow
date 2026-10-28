"""``rlmflow serve``: a long-lived Flow whose root agent is your own Claude Code session.

The plugin's hooks call ``/prompt`` when you send a message and ``/stop`` when
Claude finishes replying. A prompt creates the root (``harness="interactive"``)
or becomes a user message to it; Flow steps it, which opens the root's turn,
and your session's ``rlmflow repl`` cells land in that turn through the agent
token in ``.rlmflow/serve.json``. ``/stop`` ends the turn. If the root is left
waiting on agents it started, ``/stop`` holds until their results open its next
turn and returns them, so the Stop hook can hand them to Claude.

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
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from rlmflow.agent_tools.cli import SERVE_FILE
from rlmflow.agent_tools.hooks import RUN_COMMAND, STOP_WAIT
from rlmflow.agent_tools.server import Turn
from rlmflow.graph.config import harness_path
from rlmflow.graph.graph import Graph
from rlmflow.graph.nodes import CODE, DONE, MESSAGE, WAITING, Node, Step
from rlmflow.harnesses.interactive import read_transcript
from rlmflow.harnesses.sessions import (
    Session,
    TurnResult,
    pending_messages,
    resume_prompt,
)

if TYPE_CHECKING:
    from rlmflow.flow import Flow
    from rlmflow.graph.config import AgentConfig

T = TypeVar("T")

INTERACTIVE = harness_path("interactive")
#: The config ``serve`` registers on its Flow: its root's, and what ``fork --interactive`` picks.
INTERACTIVE_CONFIG = "interactive"

#: How long ``/prompt`` waits for Flow to open the root's turn.
OPEN_TIMEOUT = 30.0
#: How long ``/stop`` waits for the turn's end to land.
LAND_TIMEOUT = 60.0


class Serve:
    """One project's serve: a Flow, its graph, and the interactive root bound to one session."""

    def __init__(
        self,
        flow: Flow,
        graph: Graph,
        *,
        graph_dir: str | Path,
        project: str | Path,
        root: Node | None = None,
        root_config: AgentConfig | None = None,
        session_id: str | None = None,
    ) -> None:
        from rlmflow.consumers import GraphCheckpointer

        self.flow, self.graph = flow, graph
        self.graph_dir = Path(graph_dir)
        self.project = Path(project).resolve()
        self.root = root if root is not None else interactive_root(graph)
        self.root_config = flow.add_config(
            interactive_config(root_config or flow.configs["default"])
        )
        self.session_id = session_id or (last_session(graph, self.root) if self.root else None)
        self.checkpointer = GraphCheckpointer(self.graph_dir)
        self.wakeup = asyncio.Event()
        self.stopping = asyncio.Event()
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
            self.agent_token = server.attach(self.root)
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
        """Serve until ``/shutdown``: step whatever is ready, then idle until a prompt arrives."""
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
                woken = asyncio.ensure_future(self.wakeup.wait())
                await asyncio.wait({woken, stopping}, return_when=asyncio.FIRST_COMPLETED)
                woken.cancel()
        finally:
            stopping.cancel()
            self.graph.save(self.graph_dir)
            with suppress(FileNotFoundError):
                self.serve_file.unlink()
            await self.flow.aclose()

    async def _stream(self) -> None:
        try:
            async for step in self.flow.run_streaming(self.graph, root_config=self.root_config):
                self.checkpointer.handle(step, self.graph)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - serve outlives a failed run
            self.error = exc
            print(f"rlmflow serve: the run failed: {type(exc).__name__}: {exc}", file=sys.stderr)

    def kick(self) -> None:
        """New work: wake the running loop, or start the next one."""
        self.wakeup.set()
        self.flow.wake()

    # -- the hooks' routes --------------------------------------------------

    def bound(self, session_id: Any) -> bool:
        return self.session_id is None or session_id == self.session_id

    async def prompt(self, body: dict[str, Any]) -> dict[str, Any]:
        session_id, text = body.get("session_id"), body.get("prompt") or ""
        if not isinstance(text, str):
            raise ValueError("prompt must be a string")
        if not self.bound(session_id):
            return {"bound": False}
        if self.root is None:
            task = text.removeprefix(RUN_COMMAND).strip() if text.startswith(RUN_COMMAND) else text
            self.root = self.graph.create_agent(task, name="root")
            self.agent_token = self.flow.agent_tools.attach(self.root)
        elif self.flow.agent_tools.open_turn(self.root) is None:
            self.graph.send(self.root, text, sender="user")
        self.session_id = session_id
        self.write_serve_file()
        if body.get("transcript_path"):
            self._transcript_line = read_transcript(body["transcript_path"])[2]
        self.kick()
        turn = await until(lambda: self.flow.agent_tools.open_turn(self.root), OPEN_TIMEOUT)
        if turn is None:
            raise RuntimeError(
                f"the root's turn did not open: {self.error or 'nothing stepped it'}"
            )
        return {
            "bound": True,
            "root": self.root.id,
            "context": await self._context(session_id, turn),
        }

    async def _context(self, session_id: Any, turn: Turn) -> str:
        """What the session hasn't seen: the tools prompt, old cells if it can't resume, and agents' results."""
        parts = []
        if session_id not in self._briefed:
            self._briefed.add(session_id)
            parts.append(turn.step.agent.config.load_harness().prompt(self.flow, turn.step))
            has_cells = any(node.type == CODE for node in self.graph.history(turn.step.node))
            if has_cells and not Session.for_step(turn.step).resume:
                parts.append(await resume_prompt(self.flow, turn.step))
        if turn.token not in self._shown:
            self._shown.add(turn.token)
            parts += [
                node.data["text"]
                for node in pending(turn.step)
                if node.data.get("sender") != "user"
            ]
        return "\n\n".join(parts)

    async def stop(self, body: dict[str, Any]) -> dict[str, Any]:
        session_id = body.get("session_id")
        if self.root is None or not self.bound(session_id):
            return {"bound": False}
        turn = self.flow.agent_tools.open_turn(self.root)
        if turn is None:
            return {"bound": True, "landed": None}
        usage, text, self._transcript_line = read_transcript(
            body.get("transcript_path") or "", self._transcript_line
        )
        mark = len(self.graph)
        turn.ended.set_result(
            TurnResult(
                session_id=session_id, text=body.get("last_assistant_message") or text, usage=usage
            )
        )
        landed = await until(lambda: self._turn_end_since(mark), LAND_TIMEOUT)
        if landed is None:
            raise RuntimeError(f"the root's turn did not land: {self.error or 'timed out'}")
        reply: dict[str, Any] = {"bound": True, "landed": landed.type}
        if landed.type == WAITING:
            timeout = float(body.get("timeout", STOP_WAIT))
            following = await until(lambda: self._next_turn(turn), timeout)
            if following is not None:
                self._shown.add(following.token)
                reply["block"] = "\n\n".join(pending_messages(following.step)) or "Continue."
        return reply

    async def shutdown(self, body: dict[str, Any]) -> dict[str, Any]:
        if body.get("session_id") != self.session_id:
            return {"bound": False}
        asyncio.get_running_loop().call_later(0.05, self.stopping.set)
        return {"bound": True, "stopping": True}

    def _turn_end_since(self, mark: int) -> Node | None:
        for node in reversed(self.graph.nodes(agent=self.root)):
            if node.seq < mark:
                return None
            if node.type in (WAITING, DONE):
                return node
        return None

    def _next_turn(self, ended: Turn) -> Turn | None:
        turn = self.flow.agent_tools.open_turn(self.root)
        return turn if turn is not None and turn is not ended else None


async def until(probe: Callable[[], T | None], timeout: float, interval: float = 0.02) -> T | None:
    """Poll ``probe`` until it returns something, or ``None`` after ``timeout`` seconds."""
    deadline = time.monotonic() + timeout
    while True:
        found = probe()
        if found is not None or time.monotonic() >= deadline:
            return found
        await asyncio.sleep(interval)


def pending(step: Step) -> list[Node]:
    """The messages at the end of the step's history, which the turn opened on."""
    found: list[Node] = []
    for node in reversed(step.graph.history(step.node)):
        if node.type != MESSAGE:
            break
        found.append(node)
    return found[::-1]


def interactive_config(base: AgentConfig | None = None) -> AgentConfig:
    from rlmflow.graph.config import AgentConfig

    return replace(
        base or AgentConfig(), name=INTERACTIVE_CONFIG, harness=INTERACTIVE, max_iters=None
    )


def ran_interactively(agent: Node) -> bool:
    """An ``interactive`` fork, or a root a serve ran with the interactive harness."""
    if agent.config_name == INTERACTIVE_CONFIG:
        return True
    recorded = agent.metadata.get("config") or {}
    return agent.config_name is None and recorded.get("harness") == INTERACTIVE


def interactive_root(graph: Graph) -> Node | None:
    """The latest agent in ``graph`` your session drives."""
    found = [agent for agent in graph.agents() if ran_interactively(agent)]
    return found[-1] if found else None


def last_session(graph: Graph, agent: Node) -> str | None:
    """The session that ended ``agent``'s latest turn; a fork has none of its own."""
    for node in reversed(graph.own_history(agent)):
        if node.type in (WAITING, DONE) and node.data.get("session_id"):
            return node.data["session_id"]
    return None


def serve_main(
    project: str | Path,
    *,
    graph_dir: str | Path | None = None,
    root: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> None:
    """Build the project's Flow (the coding one, or ``flow = "module:factory"``'s) and serve it."""
    from rlmflow.cli import resolve_options
    from rlmflow.cli.run import build_flow

    project = Path(project).expanduser().resolve()
    options, _ = resolve_options({**(overrides or {}), "workdir": str(project)}, cwd=project)
    if graph_dir is not None and Path(graph_dir).exists():
        graph = Graph.load(graph_dir)
    else:
        graph_dir = graph_dir or project / ".rlmflow" / "runs" / time.strftime("%Y%m%d-%H%M%S")
        graph = Graph()
    root_agent = None
    if root is not None:
        from rlmflow.agent_tools.tools import resolve

        root_agent = resolve(graph, root)
    flow = build_flow(options, workdir=project)

    async def main() -> None:
        serve = Serve(
            flow,
            graph,
            graph_dir=graph_dir,
            project=project,
            root=root_agent,
        )
        loop = asyncio.get_running_loop()
        for signum in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(signum, serve.stopping.set)
        await serve.run()
        print(f"run saved to {serve.graph_dir}")

    asyncio.run(main())


__all__ = [
    "INTERACTIVE_CONFIG",
    "Serve",
    "interactive_config",
    "interactive_root",
    "last_session",
    "serve_main",
    "until",
]
