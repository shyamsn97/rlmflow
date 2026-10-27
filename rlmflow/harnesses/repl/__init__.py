"""Our harness: a model plus a persistent Python REPL, one node per call."""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    # Lazy, so the REPL worker can import ``runtime`` without the harness.
    if name == "ReplHarness":
        from rlmflow.harnesses.repl.loop import ReplHarness

        return ReplHarness
    raise AttributeError(name)


__all__ = ["ReplHarness"]
