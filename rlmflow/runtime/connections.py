"""Subprocess/container transport for the minimal remote REPL protocol.

``PopenConnection`` is shared by the subprocess and Docker runtimes.
"""

from __future__ import annotations

import selectors
import subprocess as sp
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


class PopenConnection(ReplConnection):
    """Connect RemoteRepl to a repl_server subprocess/container."""

    def __init__(
        self,
        argv: list[str],
        *,
        cwd: str | Path | None = None,
        env: dict[str, str] | None = None,
        label: str = "minimal REPL subprocess",
        repl_timeout: float | None = None,
    ) -> None:
        self.argv = argv
        self.cwd = cwd
        self.env = env
        self.label = label
        self.repl_timeout = repl_timeout
        self.proc: sp.Popen | None = None

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
        return self.proc

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
            self.close(force=True)
            raise TimeoutError(
                f"{self.label} did not respond within {self.repl_timeout}s"
            )

    def _exited_error(self) -> RuntimeError:
        proc = self._process()
        err = b""
        if proc.stderr is not None:
            with suppress(Exception):
                err = proc.stderr.read() or b""
        return RuntimeError(
            f"{self.label} {self.argv!r} exited unexpectedly. "
            f"stderr: {err.decode(errors='replace')}"
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


__all__ = ["PopenConnection"]
