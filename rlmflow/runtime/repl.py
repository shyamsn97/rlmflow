"""Tiny async Python REPL for rlmflow."""

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
    """Raised by ``finish(...)`` to end a block immediately, carrying its answer.

    The answer rides on the exception because that is the one path certain to
    reach whoever started the block. A REPL therefore never has to hold "the last
    answer" as state, and a finished run is classified from what happened rather
    than from what was left behind.
    """

    def __init__(self, answer: Any = None) -> None:
        super().__init__(answer)
        self.answer = answer


class ReplStatus(StrEnum):
    """How a block of agent code ended."""

    #: It ran to the end.
    OK = "ok"
    #: It raised; the traceback is the output.
    ERROR = "error"
    #: It called ``finish(...)``, so its agent has an answer.
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
    #: What ``finish(...)`` was called with; set only when the status is ``DONE``.
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

    Implementations expose two pieces of state: ``namespace`` (the Python globals
    code runs against) and ``env`` (the host<->REPL metadata channel). How a run
    ended is not state — it is the :class:`ReplRun` that :meth:`run` returns.
    """

    namespace: dict[str, Any]
    env: dict[str, Any]

    @abstractmethod
    def seed(self, tools: dict[str, Callable[..., object]], inputs: dict[str, str]) -> None: ...

    @abstractmethod
    def inject(self, name: str, fn: Any) -> None: ...

    @abstractmethod
    def get_var(self, name: str) -> Any: ...

    @abstractmethod
    def remove_tool(self, name: str) -> None: ...

    @abstractmethod
    def update_env(self, values: dict[str, Any]) -> None: ...

    @abstractmethod
    async def run(self, code: str) -> ReplRun: ...

    @abstractmethod
    def close(self) -> None: ...

    def get_env_var(self, name: str) -> Any:
        """Return a value from the ``env`` metadata channel (``ENV``).

        Distinct from :meth:`get_var`, which reads the Python namespace. Concrete
        here because ``env`` is part of this contract, so both implementations
        read it the same way. Raises ``KeyError`` if ``name`` is unset.
        """
        if name not in self.env:
            raise KeyError(name)
        return self.env[name]


# The pseudo-filename agent code is compiled under. Appears in every traceback
# and syntax error the agent reads back, so it names the tool, not the file.
REPL_FILENAME = "<rlmflow>"

_stdout_buf: contextvars.ContextVar[io.StringIO | None] = contextvars.ContextVar(
    "rlmflow_stdout", default=None
)


class WorkingDirectoryConflict(RuntimeError):
    """Two live runs asked for different working directories.

    The current directory belongs to the process, so it cannot be two things at
    once. Raised rather than silently running in the wrong place.
    """


class _CwdScope:
    """Refcounted ``chdir``: first run in changes, last run out restores.

    A mutex held across a run would serialize concurrent agents and deadlock
    delegation — a parent waiting on a child would hold the directory the child
    needs to start. Since every run sharing a REPL wants the *same* directory,
    there is nothing to arbitrate: count the runs instead, and hold the lock only
    while touching the counter.
    """

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._depth = 0
        self._previous: str | None = None
        self._active: Path | None = None

    def _enter(self, target: Path) -> None:
        with self._guard:
            if self._depth == 0:
                self._previous, self._active = os.getcwd(), target
                os.chdir(target)
            elif self._active is not None and target.resolve() != self._active.resolve():
                raise WorkingDirectoryConflict(
                    f"cannot run in {str(target)!r}: another run is active in "
                    f"{str(self._active)!r}, and the working directory is per-process"
                )
            self._depth += 1

    def _exit(self) -> None:
        with self._guard:
            self._depth -= 1
            if self._depth == 0:
                if self._previous is not None:
                    os.chdir(self._previous)
                self._previous = self._active = None

    @contextmanager
    def held(self, target: Path | None):
        if target is None:
            yield
            return
        self._enter(target)
        try:
            yield
        finally:
            self._exit()


_CWD = _CwdScope()


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
    )
    stack = [tree]
    while stack:
        node = stack.pop()
        if isinstance(node, ast.Await):
            return True
        stack.extend(
            child for child in ast.iter_child_nodes(node) if not isinstance(child, boundaries)
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
        self.working_directory = (
            Path(working_directory).resolve() if working_directory is not None else None
        )
        if self.working_directory is not None:
            self.working_directory.mkdir(parents=True, exist_ok=True)
        if not isinstance(sys.stdout, _Stdout):
            sys.stdout = _Stdout(sys.stdout)

    def seed(self, tools: dict[str, Callable[..., object]], inputs: dict[str, str]) -> None:
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

    def remove_tool(self, name: str) -> None:
        self.namespace.pop(name, None)

    def update_env(self, values: dict[str, Any]) -> None:
        """Merge ``values`` into this REPL's ``env`` (host -> REPL metadata)."""
        self.env.update(values)

    def close(self) -> None:
        """Release runtime resources; local in-process REPLs have none."""

    async def run(self, code: str) -> ReplRun:
        try:
            # Held outside ``_capture``: a directory conflict is a refusal to start,
            # not an error raised by agent code, so it is not this block's output.
            with _CWD.held(self.working_directory):
                return await self._capture(code)
        except WorkingDirectoryConflict as exc:
            return ReplRun(output=f"{type(exc).__name__}: {exc}", status=ReplStatus.ERROR)

    async def _capture(self, code: str) -> ReplRun:
        """Run one block, collecting what it printed and how it ended.

        Private because it must run inside the working-directory scope that
        :meth:`run` holds around it.
        """
        buf = io.StringIO()
        token = _stdout_buf.set(buf)
        try:
            await self._execute(code)
        except DoneSignal as signal:
            return ReplRun(
                output=buf.getvalue().strip(),
                status=ReplStatus.DONE,
                answer=signal.answer,
            )
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            # Never swallow control-flow exceptions: a cancelled agent must
            # actually unwind (otherwise run/interpreter shutdown hangs on it).
            raise
        except BaseException as exc:  # noqa: BLE001
            buf.write(f"{type(exc).__name__}: {exc}")
            return ReplRun(output=buf.getvalue().strip(), status=ReplStatus.ERROR)
        finally:
            _stdout_buf.reset(token)
        return ReplRun(output=buf.getvalue().strip(), status=ReplStatus.OK)

    async def _execute(self, code: str) -> None:
        """Compile and run ``code`` against the namespace, awaiting top-level await.

        Raises whatever the block raises, ``SyntaxError`` included, for
        :meth:`_capture` to classify.
        """
        if not code.strip():
            raise MissingReplError("missing ```repl``` block")
        tree = ast.parse(code, filename=REPL_FILENAME)
        if not has_top_level_await(tree):
            exec(  # noqa: S102 - executing agent code is the REPL's purpose
                compile(tree, REPL_FILENAME, "exec"),
                self.namespace,
            )
            return
        compiled = compile(
            tree,
            REPL_FILENAME,
            "exec",
            flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
        )
        result = eval(compiled, self.namespace)
        if inspect.iscoroutine(result):
            await result


__all__ = [
    "DoneSignal",
    "LocalRepl",
    "MissingReplError",
    "Repl",
    "ReplRun",
    "ReplStatus",
    "WorkingDirectoryConflict",
    "has_top_level_await",
]
