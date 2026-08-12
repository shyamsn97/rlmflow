"""Process-stream transports for lightweight Python workers."""

from __future__ import annotations

import subprocess
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

STDERR_LINES = 200
DEFAULT_REPL_TIMEOUT = 300.0


class PopenConnection:
    """A lazy subprocess connected by protected JSON-line stdio."""

    def __init__(
        self,
        argv: list[str],
        *,
        cwd: str | Path | None = None,
        env: dict[str, str] | None = None,
        label: str = "REPL worker",
    ) -> None:
        self.argv = argv
        self.cwd = cwd
        self.env = env
        self.label = label
        self.proc: subprocess.Popen[bytes] | None = None
        self._start_lock = threading.Lock()
        self._stderr_tail: deque[str] = deque(maxlen=STDERR_LINES)
        self._stderr_reader: threading.Thread | None = None

    def start(self) -> None:
        self._process()

    def _process(self) -> subprocess.Popen[bytes]:
        with self._start_lock:
            if self.proc is None:
                self.proc = subprocess.Popen(
                    self.argv,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=str(self.cwd) if self.cwd is not None else None,
                    env=self.env,
                    bufsize=0,
                )
                self._drain_stderr(self.proc)
            return self.proc

    def _drain_stderr(self, proc: subprocess.Popen[bytes]) -> None:
        if proc.stderr is None:
            return

        def pump() -> None:
            with suppress(Exception):
                for line in iter(proc.stderr.readline, b""):
                    self._stderr_tail.append(line.decode(errors="replace").rstrip("\n"))

        self._stderr_reader = threading.Thread(
            target=pump,
            daemon=True,
            name=f"{self.label} stderr",
        )
        self._stderr_reader.start()

    def stderr_tail(self) -> str:
        return "\n".join(self._stderr_tail)

    def send(self, message: WireModel) -> None:
        proc = self._process()
        if proc.stdin is None:
            raise RuntimeError(f"{self.label} stdin is unavailable")
        proc.stdin.write((dump_message(message) + "\n").encode())
        proc.stdin.flush()

    def recv(self) -> ReplResponse | ProxyCall:
        proc = self._process()
        if proc.stdout is None:
            raise RuntimeError(f"{self.label} stdout is unavailable")
        line = proc.stdout.readline()
        if not line:
            raise self._exited_error()
        return parse_client_message(line.decode())

    def _exited_error(self) -> RuntimeError:
        if self._stderr_reader is not None:
            self._stderr_reader.join(timeout=0.2)
        tail = self.stderr_tail()
        detail = f" stderr: {tail}" if tail else ""
        return RuntimeError(f"{self.label} exited unexpectedly.{detail}")

    def close(self, *, force: bool = False) -> None:
        with self._start_lock:
            proc, self.proc = self.proc, None
        if proc is None:
            return
        if force and proc.poll() is None:
            with suppress(Exception):
                proc.terminate()
        with suppress(Exception):
            if proc.stdin is not None and not proc.stdin.closed:
                proc.stdin.close()
        try:
            proc.wait(timeout=0.5 if force else 2)
        except subprocess.TimeoutExpired:
            with suppress(Exception):
                proc.kill()
                proc.wait(timeout=1)
        for stream in (proc.stdout, proc.stderr):
            with suppress(Exception):
                if stream is not None and not stream.closed:
                    stream.close()


__all__ = ["DEFAULT_REPL_TIMEOUT", "PopenConnection", "STDERR_LINES"]
