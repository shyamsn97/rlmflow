"""Tiny async Python REPL for minimal rflow."""

from __future__ import annotations

import ast
import asyncio
import contextvars
import inspect
import io
import os
import sys
import threading
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


class MissingReplError(Exception):
    """Raised when the assistant response has no executable REPL block."""


class DoneSignal(BaseException):
    """Raised by ``done(...)`` to end a block immediately."""


@runtime_checkable
class ReplLike(Protocol):
    namespace: dict[str, Any]
    env: dict[str, Any]
    done_result: str | None
    errored: bool

    def seed(
        self, tools: dict[str, Callable[..., object]], inputs: dict[str, str]
    ) -> None: ...

    def inject(self, name: str, fn: Any) -> None: ...

    def remove_tool(self, name: str) -> None: ...

    def update_env(self, values: dict[str, Any]) -> None: ...

    async def run(self, code: str) -> str: ...

    def drain(self) -> str: ...

    def close(self) -> None: ...


_stdout_buf: contextvars.ContextVar[io.StringIO | None] = contextvars.ContextVar(
    "minimal_rflow_stdout", default=None
)
_CWD_LOCK = threading.RLock()


class _Stdout:
    def __init__(self, real):
        self.real = real

    def write(self, text):
        buf = _stdout_buf.get()
        return buf.write(text) if buf is not None else self.real.write(text)

    def flush(self):
        self.real.flush()

    def __getattr__(self, name):
        return getattr(self.real, name)


def has_top_level_await(tree: ast.AST) -> bool:
    boundaries = (
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.Lambda,
        ast.ClassDef,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp,
    )
    stack = [tree]
    while stack:
        node = stack.pop()
        if isinstance(node, ast.Await):
            return True
        stack.extend(
            child
            for child in ast.iter_child_nodes(node)
            if not isinstance(child, boundaries)
        )
    return False


class Repl:
    """Stateful namespace with top-level await support."""

    def __init__(self, working_directory: str | Path | None = None) -> None:
        self.namespace: dict[str, Any] = {"__builtins__": __builtins__}
        # Per-REPL environment: a flat key/value channel distinct from the
        # Python namespace. The host seeds it (agent RFLOW_* metadata) and REPL
        # code reads/writes it via the injected ``ENV`` handle
        # (e.g. ``ENV["RFLOW_AGENT_ID"]`` or ``ENV["solved"] = True``). Because
        # it is per-REPL it stays isolated across concurrent local agents.
        self.env: dict[str, Any] = {}
        self.namespace["ENV"] = self.env
        self.done_result: str | None = None
        self.errored = False
        self._buf = io.StringIO()
        self.working_directory = (
            Path(working_directory).resolve() if working_directory is not None else None
        )
        if self.working_directory is not None:
            self.working_directory.mkdir(parents=True, exist_ok=True)
        if not isinstance(sys.stdout, _Stdout):
            sys.stdout = _Stdout(sys.stdout)

    @property
    def output(self) -> str:
        return self._buf.getvalue().strip()

    def drain(self) -> str:
        output = self.output
        self._buf = io.StringIO()
        _stdout_buf.set(self._buf)
        return output

    def seed(
        self, tools: dict[str, Callable[..., object]], inputs: dict[str, str]
    ) -> None:
        self.namespace.update(tools)
        self.namespace["INPUTS"] = dict(inputs)

    def inject(self, name: str, fn: Any) -> None:
        """Bind any Python object (function, class, or value) into the live
        namespace under ``name``. The general injection primitive; dynamic tools
        are just one use of it."""
        self.namespace[name] = fn

    def remove_tool(self, name: str) -> None:
        self.namespace.pop(name, None)

    def update_env(self, values: dict[str, Any]) -> None:
        """Merge ``values`` into this REPL's ``env`` (host -> REPL metadata)."""
        self.env.update(values)

    def close(self) -> None:
        """Release runtime resources; local minimal REPLs have none."""

    @contextmanager
    def capture(self):
        self._buf = io.StringIO()
        self.errored = False
        token = _stdout_buf.set(self._buf)
        previous_cwd: str | None = None
        if self.working_directory is not None:
            _CWD_LOCK.acquire()
            previous_cwd = os.getcwd()
            os.chdir(self.working_directory)
        try:
            yield
        except DoneSignal:
            pass
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            # Never swallow control-flow exceptions: a cancelled agent must
            # actually unwind (otherwise run/interpreter shutdown hangs on it).
            raise
        except BaseException as exc:  # noqa: BLE001
            self.errored = True
            self._buf.write(f"{type(exc).__name__}: {exc}")
        finally:
            _stdout_buf.reset(token)
            if previous_cwd is not None:
                os.chdir(previous_cwd)
                _CWD_LOCK.release()

    async def run(self, code: str) -> str:
        with self.capture():
            if not code.strip():
                raise MissingReplError("missing ```repl``` block")
            try:
                tree = ast.parse(code)
            except SyntaxError as exc:
                self.errored = True
                self._buf.write(f"SyntaxError: {exc}")
                return self.output
            if not has_top_level_await(tree):
                exec(compile(tree, "<minimal-rflow>", "exec"), self.namespace)
            else:
                compiled = compile(
                    tree,
                    "<minimal-rflow>",
                    "exec",
                    flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
                )
                result = eval(compiled, self.namespace)
                if inspect.iscoroutine(result):
                    await result
        return self.output


__all__ = [
    "DoneSignal",
    "MissingReplError",
    "Repl",
    "ReplLike",
    "has_top_level_await",
]
