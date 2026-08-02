"""Tiny async Python REPL for minimal rlmflow."""

from __future__ import annotations

import ast
import asyncio
import contextvars
import inspect
import io
import os
import sys
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


class MissingReplError(Exception):
    """Raised when the assistant response has no executable REPL block."""


class DoneSignal(BaseException):
    """Raised by ``done(...)`` to end a block immediately."""


class ReplStatus(StrEnum):
    """How a block of agent code ended."""

    #: It ran to the end.
    OK = "ok"
    #: It raised; the traceback is the output.
    ERROR = "error"
    #: It called ``done(...)``, so its agent has an answer.
    DONE = "done"
    #: The REPL itself died, taking the namespace with it.
    DEAD = "dead"


@dataclass
class ReplRun:
    """The outcome of one block of agent code.

    Every ending is a value, including the REPL dying: ``Runtime.execute`` catches
    that and reports it as ``DEAD`` rather than raising at whoever asked for a step.
    """

    output: str = ""
    status: ReplStatus = ReplStatus.OK
    #: What ``done(...)`` was called with; set only when the status is ``DONE``.
    answer: Any = None
    #: What killed the REPL; set only when the status is ``DEAD``.
    error: BaseException | None = None


class Repl(ABC):
    """Abstract REPL contract the ``Flow`` drives, regardless of where code runs.

    Two concrete subclasses implement it: :class:`LocalRepl` (in this process)
    and :class:`~rlmflow.runtime.repl_client.RemoteRepl` (a stub that speaks the
    wire protocol to a sandbox). Being an ABC (not a ``Protocol``) makes the
    relationship explicit — you can see both subclasses inherit from it, and
    Python enforces that each implements every abstract method.

    Implementations expose four state attributes: ``namespace`` (the Python
    globals code runs against), ``env`` (the host<->REPL metadata channel),
    ``done_result`` (where the injected ``done(...)`` leaves its answer), and
    ``errored`` (whether the last run raised). The last two are how a run reports
    back from inside; callers read :class:`ReplRun` instead.
    """

    namespace: dict[str, Any]
    env: dict[str, Any]
    done_result: str | None
    errored: bool

    @abstractmethod
    def seed(
        self, tools: dict[str, Callable[..., object]], inputs: dict[str, str]
    ) -> None: ...

    @abstractmethod
    def inject(self, name: str, fn: Any) -> None: ...

    @abstractmethod
    def get_var(self, name: str) -> Any: ...

    @abstractmethod
    def get_env_var(self, name: str) -> Any: ...

    @abstractmethod
    def remove_tool(self, name: str) -> None: ...

    @abstractmethod
    def update_env(self, values: dict[str, Any]) -> None: ...

    @abstractmethod
    async def run(self, code: str) -> ReplRun: ...

    @abstractmethod
    def read(self, *, clear: bool = False) -> str: ...

    @abstractmethod
    def close(self) -> None: ...

    def outcome(self, output: str) -> ReplRun:
        """Classify a finished run from the state it left behind."""
        if self.done_result is not None:
            return ReplRun(
                output=output, status=ReplStatus.DONE, answer=self.done_result
            )
        status = ReplStatus.ERROR if self.errored else ReplStatus.OK
        return ReplRun(output=output, status=status)


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


class LocalRepl(Repl):
    """Stateful in-process REPL: a namespace with top-level await support.

    Runs agent code in *this* Python process, so injected objects are the real
    live objects (by reference, zero-copy) and ``get_var`` returns them as-is.
    Returned by :class:`~rlmflow.runtime.runtime.LocalRuntime`.
    """

    def __init__(self, working_directory: str | Path | None = None) -> None:
        self.namespace: dict[str, Any] = {"__builtins__": __builtins__}
        # Per-REPL environment: a flat key/value channel distinct from the
        # Python namespace. The host seeds it (agent RLMFLOW_* metadata) and REPL
        # code reads/writes it via the injected ``ENV`` handle
        # (e.g. ``ENV["RLMFLOW_AGENT_ID"]`` or ``ENV["solved"] = True``). Because
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

    def read(self, *, clear: bool = False) -> str:
        """Return captured stdout so far; with ``clear=True``, also reset the buffer."""
        output = self.output
        if clear:
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

    def get_var(self, name: str) -> Any:
        """Return the live object bound to ``name`` in the Python namespace
        (in-process: the real object, not a copy). Raises ``KeyError`` if unbound."""
        if name not in self.namespace:
            raise KeyError(name)
        return self.namespace[name]

    def get_env_var(self, name: str) -> Any:
        """Return a value from this REPL's ``env`` metadata channel (``ENV``).

        Distinct from :meth:`get_var`, which reads the Python namespace. Raises
        ``KeyError`` if ``name`` is unset.
        """
        if name not in self.env:
            raise KeyError(name)
        return self.env[name]

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

    async def run(self, code: str) -> ReplRun:
        # Cleared here rather than by the caller: an answer belongs to one run.
        self.done_result = None
        with self.capture():
            if not code.strip():
                raise MissingReplError("missing ```repl``` block")
            try:
                tree = ast.parse(code)
            except SyntaxError as exc:
                self.errored = True
                self._buf.write(f"SyntaxError: {exc}")
                return self.outcome(self.output)
            if not has_top_level_await(tree):
                exec(  # noqa: S102 - executing agent code is the REPL's purpose
                    compile(tree, "<minimal-rlmflow>", "exec"),
                    self.namespace,
                )
            else:
                compiled = compile(
                    tree,
                    "<minimal-rlmflow>",
                    "exec",
                    flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
                )
                result = eval(compiled, self.namespace)
                if inspect.iscoroutine(result):
                    await result
        return self.outcome(self.output)


__all__ = [
    "DoneSignal",
    "LocalRepl",
    "MissingReplError",
    "Repl",
    "ReplRun",
    "ReplStatus",
    "has_top_level_await",
]
