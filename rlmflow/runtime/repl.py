"""Execution outcomes and the stateful Python worker contract."""

from __future__ import annotations

import ast
import asyncio
import contextvars
import importlib
import inspect
import io
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, MutableMapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


class DoneSignal(BaseException):
    """Internal control flow raised by the host completion tool."""

    def __init__(self, answer: Any = None, *, acceptance: dict[str, Any] | None = None) -> None:
        super().__init__(answer)
        self.answer = answer
        self.acceptance = acceptance or {}


class VerificationSignal(BaseException):
    """Internal control flow for a rejected finish candidate."""

    def __init__(
        self,
        candidate: Any,
        *,
        kind: str,
        feedback: str,
        schema_hash: str | None = None,
        verifier_hash: str | None = None,
    ) -> None:
        super().__init__(feedback)
        self.candidate = candidate
        self.kind = kind
        self.feedback = feedback
        self.schema_hash = schema_hash
        self.verifier_hash = verifier_hash


class _CellStopIteration(Exception):
    """StopIteration from cell code, lifted out of the async runner (PEP 479)."""


#: Observation returned for an invalid action reply, so the retry knows what to fix.
MISSING_REPL_NOTE = """MissingReplError: expected exactly one non-empty
fenced ```repl``` block. Nothing ran and this turn produced no observation.
Reply with one block that carries out the next step or calls finish(value)."""


class ReplStatus(StrEnum):
    OK = "ok"
    ERROR = "error"
    DONE = "done"
    REJECTED = "rejected"
    DEAD = "dead"


@dataclass
class ReplRun:
    """Outcome of one agent execution."""

    output: str = ""
    status: ReplStatus = ReplStatus.OK
    answer: Any = None
    verification: VerificationSignal | None = None
    acceptance: dict[str, Any] | None = None
    error: BaseException | None = None


class Repl(ABC):
    """One agent tenant in a Python worker."""

    namespace: dict[str, Any]
    env: dict[str, Any]

    @abstractmethod
    def seed(self, tools: dict[str, Callable[..., object]], args: dict[str, Any]) -> None: ...

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
        if name not in self.env:
            raise KeyError(name)
        return self.env[name]


REPL_FILENAME = "<rlmflow>"

#: Modules bound in every REPL namespace before any agent code runs, under their own
#: name. Models reach for these reflexively — ``json.loads``, ``re.findall`` — and a
#: ``NameError`` costs a whole turn to recover from. Keep the default set to cheap,
#: side-effect-free stdlib: importing is not free and the namespace is shared by every
#: tenant in a worker. An explicit ``preimports`` may name anything importable,
#: third-party packages included.
DEFAULT_PREIMPORTS: tuple[str, ...] = (
    "asyncio",
    "collections",
    "csv",
    "datetime",
    "itertools",
    "json",
    "math",
    "os",
    "pathlib",
    "re",
    "statistics",
    "sys",
    "textwrap",
    "time",
)


def base_namespace(preimports: Iterable[str] | None = None) -> dict[str, Any]:
    """Build a fresh REPL global namespace with ``preimports`` bound by module name.

    ``None`` means ``DEFAULT_PREIMPORTS``; an empty iterable means bind nothing. An
    unimportable name raises rather than being skipped, so a bad configuration fails
    at worker startup instead of surfacing later as a puzzling ``NameError``.
    """
    names = DEFAULT_PREIMPORTS if preimports is None else tuple(preimports)
    namespace: dict[str, Any] = {"__builtins__": __builtins__}
    for name in names:
        namespace[name] = importlib.import_module(name)
    return namespace


_BINDING: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "rlmflow_binding", default=None
)
_OUTPUT: contextvars.ContextVar[io.StringIO | None] = contextvars.ContextVar(
    "rlmflow_output", default=None
)


def current_binding() -> dict[str, Any]:
    binding = _BINDING.get()
    if binding is None:
        raise RuntimeError("agent bindings are only available during execution")
    return binding


class CurrentMapping(MutableMapping[str, Any]):
    """Mapping view resolved from the execution thread's agent binding."""

    def __init__(self, name: str) -> None:
        self.name = name

    def _value(self) -> dict[str, Any]:
        return current_binding()[self.name]

    def __getitem__(self, key: str) -> Any:
        return self._value()[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._value()[key] = value

    def __delitem__(self, key: str) -> None:
        del self._value()[key]

    def __iter__(self):
        return iter(self._value())

    def __len__(self) -> int:
        return len(self._value())

    def copy(self) -> dict[str, Any]:
        return self._value().copy()

    def __repr__(self) -> str:
        return repr(self._value())


class CurrentObject:
    """Object proxy resolved from the execution thread's agent binding."""

    def __init__(self, name: str) -> None:
        object.__setattr__(self, "name", name)

    def _value(self) -> Any:
        return current_binding()[self.name]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._value(), name)

    def __repr__(self) -> str:
        return repr(self._value())


class _ContextStream:
    """Route Python writes to the active run without replacing stdout per cell."""

    def __init__(self, real: Any) -> None:
        self.real = real

    def write(self, text: str) -> int:
        buffer = _OUTPUT.get()
        return buffer.write(text) if buffer is not None else self.real.write(text)

    def flush(self) -> None:
        self.real.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.real, name)


class LocalRepl:
    """Worker-side cell executor with context-bound agent metadata."""

    def __init__(
        self,
        working_directory: str | Path | None = None,
        *,
        namespace: dict[str, Any] | None = None,
    ) -> None:
        self.namespace = namespace if namespace is not None else base_namespace()
        self.env: dict[str, Any] = {}
        self.args: dict[str, Any] = {}
        self._argument_names: set[str] = set()
        self.working_directory = (
            Path(working_directory).resolve() if working_directory is not None else None
        )
        if self.working_directory is not None:
            self.working_directory.mkdir(parents=True, exist_ok=True)
        self.namespace.setdefault("ENV", CurrentMapping("env"))
        if not isinstance(sys.stdout, _ContextStream):
            sys.stdout = _ContextStream(sys.stdout)
        if not isinstance(sys.stderr, _ContextStream):
            sys.stderr = _ContextStream(sys.stderr)

    def seed(self, tools: dict[str, Callable[..., object]], args: dict[str, Any]) -> None:
        reserved = (set(self.namespace) - self._argument_names) | tools.keys()
        collisions = sorted(args.keys() & reserved)
        if collisions:
            joined = ", ".join(collisions)
            raise ValueError(f"agent arguments collide with REPL bindings: {joined}")
        for name in self._argument_names - args.keys():
            self.namespace.pop(name, None)
        self.namespace.update(tools)
        self.namespace.update(args)
        self.args = dict(args)
        self._argument_names = set(args)

    def inject(self, name: str, value: Any) -> None:
        self.namespace[name] = value

    def get_var(self, name: str) -> Any:
        if name not in self.namespace:
            raise KeyError(name)
        return self.namespace[name]

    def remove_tool(self, name: str) -> None:
        self.namespace.pop(name, None)

    def update_env(self, values: dict[str, Any]) -> None:
        self.env.update(values)

    def close(self) -> None:
        pass

    async def run(self, code: str, *, binding: dict[str, Any] | None = None) -> ReplRun:
        if not code.strip():
            return ReplRun(output=MISSING_REPL_NOTE, status=ReplStatus.ERROR)
        current = binding or {"args": self.args, "env": self.env}
        current.setdefault("args", self.args)
        current.setdefault("env", self.env)
        buffer = io.StringIO()
        binding_token = _BINDING.set(current)
        output_token = _OUTPUT.set(buffer)
        try:
            await self._execute(code)
        except DoneSignal as signal:
            return ReplRun(
                output=buffer.getvalue().strip(),
                status=ReplStatus.DONE,
                answer=signal.answer,
                acceptance=signal.acceptance,
            )
        except VerificationSignal as signal:
            return ReplRun(
                output=buffer.getvalue().strip(),
                status=ReplStatus.REJECTED,
                verification=signal,
            )
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except _CellStopIteration:
            buffer.write("StopIteration")
            return ReplRun(output=buffer.getvalue().strip(), status=ReplStatus.ERROR)
        except BaseException as exc:  # noqa: BLE001 - agent exceptions are results
            buffer.write(f"{type(exc).__name__}: {exc}")
            return ReplRun(output=buffer.getvalue().strip(), status=ReplStatus.ERROR)
        finally:
            _OUTPUT.reset(output_token)
            _BINDING.reset(binding_token)
        return ReplRun(output=buffer.getvalue().strip(), status=ReplStatus.OK)

    async def _execute(self, code: str) -> None:
        compiled = compile(
            code,
            REPL_FILENAME,
            "exec",
            flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
        )
        try:
            result = eval(compiled, self.namespace)
            if inspect.isawaitable(result):
                await result
        except StopIteration:
            raise _CellStopIteration from None
        except RuntimeError as exc:
            if str(exc) != "coroutine raised StopIteration":
                raise
            raise _CellStopIteration from None


__all__ = [
    "DEFAULT_PREIMPORTS",
    "CurrentMapping",
    "CurrentObject",
    "MISSING_REPL_NOTE",
    "DoneSignal",
    "VerificationSignal",
    "LocalRepl",
    "Repl",
    "ReplRun",
    "ReplStatus",
    "base_namespace",
    "current_binding",
]
