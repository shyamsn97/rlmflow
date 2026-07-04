"""Minimal REPL — stateful code execution with normal asyncio.

One :class:`REPL` per agent. It runs an LLM code block in a persistent
namespace and captures stdout. If the block has a *top-level* ``await``, the
block is compiled as a coroutine and run on this REPL's asyncio event loop.
Ordinary async Python is handled by asyncio; graph delegation is just another
async tool awaited by agent code.

stdout is captured through a thread-local buffer so REPLs running in
parallel threads don't clobber each other's output.
"""

from __future__ import annotations

import ast
import asyncio
import contextvars
import inspect
import io
import os
import re
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from rflow.runtime.context import EngineContext

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_capture_buf: contextvars.ContextVar[io.StringIO | None] = contextvars.ContextVar(
    "rflow_capture_buf", default=None
)

# ``os.chdir`` mutates process-global state, so local REPLs serialize code blocks
# that need a working-directory overlay. Environment overlays are best-effort
# metadata for subprocesses; holding the lock across async child delegation would
# deadlock the scheduler.
_PROCESS_STATE_LOCK = threading.RLock()


class DoneSignal(BaseException):
    """Raised by ``done()`` to stop the block. BaseException so a broad
    ``except Exception`` in agent code can't swallow a successful finish."""


def _has_top_level_await(tree: ast.AST) -> bool:
    """True iff ``tree`` has an ``await`` outside any nested scope."""

    boundary = (
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.Lambda,
        ast.ClassDef,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp,
    )
    stack: list[ast.AST] = [tree]
    while stack:
        node = stack.pop()
        if isinstance(node, ast.Await):
            return True
        for child in ast.iter_child_nodes(node):
            if not isinstance(child, boundary):
                stack.append(child)
    return False


class _StdoutProxy:
    """Thread-aware stdout: routes to a per-thread buffer when one is active."""

    def __init__(self, real):
        self.real = real

    def write(self, s):
        buf = _capture_buf.get()
        return buf.write(s) if buf is not None else self.real.write(s)

    def flush(self):
        self.real.flush()

    def __getattr__(self, name):
        return getattr(self.real, name)


class REPL:
    """A stateful Python namespace that runs top-level await on asyncio.

    :meth:`start` returns captured stdout after the block completes. Errors are
    captured into stdout and flagged on :attr:`errored`.
    """

    def __init__(self, working_directory: str | Path | None = None) -> None:
        self.namespace: dict[str, Any] = {"__builtins__": __builtins__}
        self.engine_context = EngineContext()
        self.process_env: dict[str, str] = {}
        self.loop = asyncio.new_event_loop()
        self.task: asyncio.Task | None = None
        self.errored = False
        self._buf = io.StringIO()
        # ``None`` → run in the process cwd as-is (the default; no chdir, no
        # lock). A path → each code block runs with the cwd switched into it
        # (created if missing), serialized via ``_CWD_LOCK`` across agents.
        self.working_directory: Path | None = None
        if working_directory is not None:
            self.working_directory = Path(working_directory).resolve()
            self.working_directory.mkdir(parents=True, exist_ok=True)
        if not isinstance(sys.stdout, _StdoutProxy):
            sys.stdout = _StdoutProxy(sys.stdout)

    # ── stdout capture ────────────────────────────────────────────────

    @contextmanager
    def _capture(self, *, process_overlay: bool = True):
        """Run a fresh step with this thread's stdout routed into a buffer.

        Resets the buffer/error state on entry, and turns any agent error
        into captured output (``done()`` finishes cleanly).
        """
        self._buf = io.StringIO()
        self.errored = False
        token = _capture_buf.set(self._buf)
        prev_cwd: str | None = None
        old_env: dict[str, str] = {}
        missing_env: set[str] = set()
        needs_process_lock = process_overlay and self.working_directory is not None
        if needs_process_lock:
            _PROCESS_STATE_LOCK.acquire()
        if process_overlay and self.process_env:
            for key, value in self.process_env.items():
                if key in os.environ:
                    old_env[key] = os.environ[key]
                else:
                    missing_env.add(key)
                os.environ[key] = value
        if process_overlay and self.working_directory is not None:
            prev_cwd = os.getcwd()
            os.chdir(self.working_directory)
        try:
            yield
        except DoneSignal:
            pass
        except (GeneratorExit, KeyboardInterrupt):
            raise
        except BaseException as exc:  # noqa: BLE001 - agent errors become output
            self._buf.write(f"\n{type(exc).__name__}: {exc}")
            self.errored = True
        finally:
            _capture_buf.reset(token)
            if prev_cwd is not None:
                os.chdir(prev_cwd)
            if process_overlay:
                for key in self.process_env:
                    if key in missing_env:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = old_env[key]
            if needs_process_lock:
                _PROCESS_STATE_LOCK.release()

    @property
    def _output(self) -> str:
        return _ANSI_RE.sub("", self._buf.getvalue()).strip()

    def drain_output(self) -> str:
        """Return captured output so far and reset the active capture buffer."""
        out = self._output
        self._buf = io.StringIO()
        if _capture_buf.get() is not None:
            _capture_buf.set(self._buf)
        return out

    # ── execution ─────────────────────────────────────────────────────

    def start(self, code: str) -> str:
        """Run a fresh code block in the persistent namespace."""
        if self.task is not None and not self.task.done():
            self.errored = True
            return "RuntimeError: REPL already has a running async task"
        self.task = None
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            self.errored = True
            return f"SyntaxError: {exc}"
        with self._capture(process_overlay=False):
            task = self._exec_block(tree)
            if task is not None:
                self._run_to_completion(task)
        return self._output

    async def start_async(self, code: str) -> str:
        """Run a code block on the current asyncio loop.

        This is the scheduler path. Top-level ``await`` yields the scheduler
        loop normally, so ``await launch_subagents(...)`` can wait on child graph
        work without suspending the Python frame back to Flow.
        """
        if self.task is not None and not self.task.done():
            self.errored = True
            return "RuntimeError: REPL already has a running async task"
        self.task = None
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            self.errored = True
            return f"SyntaxError: {exc}"
        task: asyncio.Task | None = None
        with self._capture():
            try:
                task = self._exec_block_on_current_loop(tree)
                if task is not None:
                    await task
            finally:
                self.task = None
        return self._output

    def close(self) -> None:
        """Cancel pending async work and close this REPL's event loop.

        Present so :class:`REPL` satisfies the ``ReplBackend`` protocol that
        remote backends (Docker/Modal) implement with real teardown.
        """
        if self.loop.is_closed():
            return
        asyncio.set_event_loop(self.loop)
        self._cancel_background_tasks()
        self.loop.close()
        self.task = None

    def _exec_block(self, tree: ast.AST):
        """Execute the block. Return its coroutine if it has a top-level
        ``await``, else ``None`` (a plain block that already ran)."""
        if not _has_top_level_await(tree):
            exec(compile(tree, "<rlm>", "exec"), self.namespace)
            return None
        code = compile(tree, "<rlm>", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        result = eval(code, self.namespace)  # creates the coroutine; runs nothing yet
        if not inspect.iscoroutine(result):
            return None
        asyncio.set_event_loop(self.loop)
        self.task = self.loop.create_task(result)
        return self.task

    def _exec_block_on_current_loop(self, tree: ast.AST):
        """Execute a block on the running scheduler loop."""
        if not _has_top_level_await(tree):
            exec(compile(tree, "<rlm>", "exec"), self.namespace)
            return None
        code = compile(tree, "<rlm>", "exec", flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        result = eval(code, self.namespace)
        if not inspect.iscoroutine(result):
            return None
        self.task = asyncio.create_task(result)
        return self.task

    def _cancel_background_tasks(self, *, exclude: asyncio.Task | None = None) -> None:
        """Cancel detached tasks; they are not durable REPL/graph state."""
        tasks = [
            task
            for task in asyncio.all_tasks(self.loop)
            if task is not exclude and not task.done()
        ]
        for task in tasks:
            task.cancel()
        if tasks:
            self.loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))

    def _run_to_completion(self, task: asyncio.Task) -> None:
        """Run one asyncio task to completion."""
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(task)
        except DoneSignal:
            pass
        except (GeneratorExit, KeyboardInterrupt):
            raise
        except BaseException as exc:  # noqa: BLE001 - agent errors are output
            self._buf.write(f"\n{type(exc).__name__}: {exc}")
            self.errored = True
        self._cancel_background_tasks(exclude=task)
        self.task = None


__all__ = ["DoneSignal", "REPL"]
