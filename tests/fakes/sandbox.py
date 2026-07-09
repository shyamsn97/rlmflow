"""Fake sandbox providers + an in-process file-bridge for runtime tests.

Adapted from the original ``tests/fakes/sandbox.py`` to the minimal stack: the
remote REPL now speaks :mod:`rflow.runtime.repl_server` over the file bridge in
:class:`rflow.runtime.sandbox.remote.RemoteFileRuntime`.

A fake provider's ``commands.run(command)`` is routed through :func:`run_local`,
which:

* intercepts the persistent REPL start command and, instead of launching
  ``tail -f in.jsonl | python -m rflow.runtime.repl_server``, starts a real
  :class:`ReplServer` in a daemon thread whose ``stdin`` *tails* ``in.jsonl`` and
  whose ``stdout`` appends to ``out.jsonl`` — faithfully reproducing the bridge,
  including mid-execution host proxies;
* runs everything else (the transport's pure-stdlib ``python -c`` append/poll
  snippets, ``mkdir``, ``rm``) as a real subprocess. They only touch files, so
  they share state with the in-thread server via the filesystem and — crucially
  — never swap the process-global ``sys.stdout`` the server's capture relies on.
"""

from __future__ import annotations

import re
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from rflow.minimal.clients.llm import LLMClient
from rflow.runtime.repl_server import ReplServer

_REPL_DETECT = "from rflow.runtime.repl_server import main"
_REMOTE_DIR_RE = re.compile(r"mkdir -p (/[^\s]*rlmflow-[a-f0-9]+)\s+(\S+)")
_KILL_PID_RE = re.compile(r"kill \$\(cat ([^)]+)/pid\)")

_sessions: dict[str, "_BridgeSession"] = {}


class _TailReader:
    """A line stream over a growing file (like ``tail -f``), with a stop signal."""

    def __init__(self, path: Path, stop: threading.Event) -> None:
        self._path = path
        self._stop = stop
        self._offset = 0

    def readline(self) -> str:
        while not self._stop.is_set():
            text = self._path.read_text() if self._path.exists() else ""
            idx = text.find("\n", self._offset)
            if idx >= 0:
                line = text[self._offset : idx + 1]
                self._offset = idx + 1
                return line
            time.sleep(0.005)
        return ""  # EOF → ReplServer.serve() loop ends


class _AppendWriter:
    """A minimal write/flush sink that appends to ``out.jsonl``."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def write(self, data: str) -> int:
        with self._path.open("a") as f:
            f.write(data)
        return len(data)

    def flush(self) -> None:  # pragma: no cover - trivial
        pass


class _BridgeSession:
    """A persistent ReplServer wired to in/out files in a daemon thread."""

    def __init__(self, remote_dir: str, workdir: str | None = None) -> None:
        self.remote_dir = Path(remote_dir)
        self.input_path = self.remote_dir / "in.jsonl"
        self.output_path = self.remote_dir / "out.jsonl"
        self.remote_dir.mkdir(parents=True, exist_ok=True)
        if workdir:
            Path(workdir).mkdir(parents=True, exist_ok=True)
        for path in (self.input_path, self.output_path, self.remote_dir / "stderr.log"):
            path.write_text("")
        (self.remote_dir / "pid").write_text("0")
        self._stop = threading.Event()
        server = ReplServer(
            protocol_in=_TailReader(self.input_path, self._stop),
            protocol_out=_AppendWriter(self.output_path),
        )
        self._thread = threading.Thread(target=server.serve, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()


def _maybe_handle_repl_lifecycle(command: str) -> tuple[int, str, str] | None:
    if _REPL_DETECT in command:
        match = _REMOTE_DIR_RE.search(command)
        if match is None:
            return None
        remote_dir, workdir = match.group(1), match.group(2)
        if remote_dir not in _sessions:
            _sessions[remote_dir] = _BridgeSession(remote_dir, workdir=workdir)
        return 0, "", ""
    match = _KILL_PID_RE.search(command)
    if match is not None:
        session = _sessions.pop(match.group(1), None)
        if session is not None:
            session.stop()
        return 0, "", ""
    return None


def run_local(command: str, *, timeout: float | None = None) -> tuple[int, str, str]:
    lifecycle = _maybe_handle_repl_lifecycle(command)
    if lifecycle is not None:
        return lifecycle
    # The transport's append/poll snippets are pure stdlib (pathlib/sys/time):
    # run them for real so they read/write the same files the in-thread server
    # uses, without touching the host process's sys.stdout.
    proc = subprocess.run(
        command, shell=True, text=True, capture_output=True, timeout=timeout, check=False
    )
    return proc.returncode, proc.stdout, proc.stderr


# ── fake E2B provider ─────────────────────────────────────────────────


class FakeE2BCommands:
    def run(self, command: str, timeout: float | None = None):
        code, stdout, stderr = run_local(command, timeout=timeout)
        return SimpleNamespace(exit_code=code, stdout=stdout, stderr=stderr)


class FakeE2BSandbox:
    def __init__(self):
        self.commands = FakeE2BCommands()
        self.killed = False

    def kill(self):
        self.killed = True


class FakeE2BSandboxFactory:
    created: list[FakeE2BSandbox] = []

    @classmethod
    def create(cls, **kwargs):
        sandbox = FakeE2BSandbox()
        cls.created.append(sandbox)
        return sandbox


class NoopLLM(LLMClient):
    def chat(self, messages, *args, **kwargs) -> str:
        return '```repl\ndone("ok")\n```'


__all__ = [
    "FakeE2BSandbox",
    "FakeE2BSandboxFactory",
    "NoopLLM",
    "run_local",
]
