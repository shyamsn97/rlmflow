"""Subprocess/container transport for the remote REPL protocol.

``PopenConnection`` is shared by the subprocess and Docker runtimes.
"""

from __future__ import annotations

import selectors
import subprocess as sp
import threading
from collections import deque
from contextlib import suppress
from pathlib import Path

from rlmflow.runtime.protocol import (
    ProxyCall,
    ReplResponse,
    WireModel,
    dump_message,
    parse_client_message,
)
from rlmflow.runtime.repl_client import ReplConnection

#: Nothing drains the sandbox's stderr while a block runs, so an unbounded pipe
#: would fill and wedge the process. One owner reads it continuously into this
#: many trailing lines.
STDERR_LINES = 200

#: A sandbox that stops answering must not hang the host forever. ``None`` is
#: still accepted as an explicit "wait indefinitely".
DEFAULT_REPL_TIMEOUT = 300.0


class PopenConnection(ReplConnection):
    """Connect RemoteRepl to a repl_server subprocess/container."""

    def __init__(
        self,
        argv: list[str],
        *,
        cwd: str | Path | None = None,
        env: dict[str, str] | None = None,
        label: str = "REPL subprocess",
        repl_timeout: float | None = DEFAULT_REPL_TIMEOUT,
    ) -> None:
        self.argv = argv
        self.cwd = cwd
        self.env = env
        self.label = label
        self.repl_timeout = repl_timeout
        self.proc: sp.Popen | None = None
        self._stderr_tail: deque[str] = deque(maxlen=STDERR_LINES)
        self._stderr_reader: threading.Thread | None = None

    def _process(self) -> sp.Popen:
        if self.proc is None:
            self.proc = sp.Popen(
                self.argv,
                stdin=sp.PIPE,
                stdout=sp.PIPE,
                stderr=sp.PIPE,
                cwd=str(self.cwd) if self.cwd is not None else None,
                env=self.env,
                bufsize=0,
            )
            self._drain_stderr(self.proc)
        return self.proc

    def _drain_stderr(self, proc: sp.Popen) -> None:
        """Give stderr a single dedicated reader so the pipe can never fill."""
        if proc.stderr is None:
            return

        def pump(stream) -> None:
            with suppress(Exception):
                for line in iter(stream.readline, b""):
                    self._stderr_tail.append(line.decode(errors="replace").rstrip("\n"))

        self._stderr_reader = threading.Thread(
            target=pump, args=(proc.stderr,), daemon=True, name=f"{self.label} stderr"
        )
        self._stderr_reader.start()

    def stderr_tail(self) -> str:
        """The sandbox's recent stderr, for diagnosing a failed or dead run."""
        return "\n".join(self._stderr_tail)

    def send(self, msg: WireModel) -> None:
        stdin = self._stdin()
        stdin.write((dump_message(msg) + "\n").encode())
        stdin.flush()

    def recv(self) -> ReplResponse | ProxyCall:
        stdout = self._stdout()
        self._wait_until_readable(stdout)
        line = stdout.readline()
        if not line:
            raise self._exited_error()
        return parse_client_message(line.decode())

    def _stdin(self):
        proc = self._process()
        if proc.stdin is None:
            raise RuntimeError(f"{self.label} stdin is not available")
        return proc.stdin

    def _stdout(self):
        proc = self._process()
        if proc.stdout is None:
            raise RuntimeError(f"{self.label} stdout is not available")
        return proc.stdout

    def _wait_until_readable(self, stream) -> None:
        if self.repl_timeout is None:
            return
        selector = selectors.DefaultSelector()
        try:
            selector.register(stream, selectors.EVENT_READ)
            events = selector.select(timeout=self.repl_timeout)
        finally:
            selector.close()
        if not events:
            tail = self.stderr_tail()
            self.close(force=True)
            detail = f" stderr: {tail}" if tail else ""
            raise TimeoutError(f"{self.label} did not respond within {self.repl_timeout}s.{detail}")

    def _exited_error(self) -> RuntimeError:
        # stderr was drained all along by its reader; join briefly so a message
        # written just before the exit has landed.
        if self._stderr_reader is not None:
            self._stderr_reader.join(timeout=0.5)
        return RuntimeError(
            f"{self.label} {self.argv!r} exited unexpectedly. stderr: {self.stderr_tail()}"
        )

    def close(self, *, force: bool = False) -> None:
        proc, self.proc = self.proc, None
        if proc is None:
            return
        if force and proc.poll() is None:
            self._terminate(proc)
        self._close_stdin(proc)
        try:
            proc.wait(timeout=0.5 if force else 2)
        except sp.TimeoutExpired:
            self._kill(proc)
        for stream in (proc.stdout, proc.stderr):
            with suppress(Exception):
                if stream is not None and not stream.closed:
                    stream.close()

    def _close_stdin(self, proc: sp.Popen) -> None:
        with suppress(Exception):
            if proc.stdin is not None and not proc.stdin.closed:
                proc.stdin.close()

    def _terminate(self, proc: sp.Popen) -> None:
        with suppress(Exception):
            proc.terminate()

    def _kill(self, proc: sp.Popen) -> None:
        for action in (proc.terminate, proc.kill):
            with suppress(Exception):
                action()
                proc.wait(timeout=0.5)
                return


__all__ = ["DEFAULT_REPL_TIMEOUT", "STDERR_LINES", "PopenConnection"]
