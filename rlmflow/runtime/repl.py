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

    def __init__(self, answer: Any = None) -> None:
        super().__init__(answer)
        self.answer = answer


class TransitionSignal(BaseException):
    """Internal control flow raised by the host transition tool."""

    def __init__(self, transition: str) -> None:
        super().__init__(transition)
        self.transition = transition


class MissingReplError(ValueError):
    """Agent response did not contain executable REPL code."""


#: Observation returned for a reply with no code, so the retry knows what to fix.
MISSING_REPL_NOTE = f"""{MissingReplError.__name__}: missing ```repl``` block, so
nothing ran and this turn produced no observation. Reply with exactly
one fenced ```repl``` block that carries out the next step instead of describing it."""


class ReplStatus(StrEnum):
    OK = "ok"
    ERROR = "error"
    DONE = "done"
    TRANSITION = "transition"
    DEAD = "dead"


@dataclass
class ReplRun:
    """Outcome of one agent execution."""

    output: str = ""
    status: ReplStatus = ReplStatus.OK
    answer: Any = None
    transition: str | None = None
    error: BaseException | None = None


class Repl(ABC):
    """One agent tenant in a Python worker."""

    namespace: dict[str, Any]
    env: dict[str, Any]
    structured_output: bool

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
        self.inputs: dict[str, str] = {}
        self.working_directory = (
            Path(working_directory).resolve() if working_directory is not None else None
        )
        self.structured_output = False
        if self.working_directory is not None:
            self.working_directory.mkdir(parents=True, exist_ok=True)
        self.namespace.setdefault("INPUTS", CurrentMapping("inputs"))
        self.namespace.setdefault("ENV", CurrentMapping("env"))
        if not isinstance(sys.stdout, _ContextStream):
            sys.stdout = _ContextStream(sys.stdout)
        if not isinstance(sys.stderr, _ContextStream):
            sys.stderr = _ContextStream(sys.stderr)

    def seed(self, tools: dict[str, Callable[..., object]], inputs: dict[str, str]) -> None:
        self.namespace.update(tools)
        self.inputs = dict(inputs)

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
        current = binding or {"inputs": self.inputs, "env": self.env}
        current.setdefault("inputs", self.inputs)
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
            )
        except TransitionSignal as signal:
            return ReplRun(
                output=buffer.getvalue().strip(),
                status=ReplStatus.TRANSITION,
                transition=signal.transition,
            )
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
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
        result = eval(compiled, self.namespace)
        if inspect.isawaitable(result):
            await result


__all__ = [
    "DEFAULT_PREIMPORTS",
    "CurrentMapping",
    "CurrentObject",
    "MISSING_REPL_NOTE",
    "DoneSignal",
    "TransitionSignal",
    "LocalRepl",
    "MissingReplError",
    "Repl",
    "ReplRun",
    "ReplStatus",
    "base_namespace",
    "current_binding",
]
