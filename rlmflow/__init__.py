"""Let coding agents build multi-agent systems in code, and record every run as a graph you can edit."""

from importlib import import_module
from typing import Any

#: Names resolved on first access, so importing ``rlmflow`` doesn't start the REPL
#: runtime or load optional dependencies.
_LAZY = {
    "Flow": "rlmflow.flow",
    "Graph": "rlmflow.graph",
    "Node": "rlmflow.graph",
    "Edge": "rlmflow.graph",
    "Input": "rlmflow.graph",
    "Agent": "rlmflow.agents",
    "ClaudeCode": "rlmflow.agents",
    "Pi": "rlmflow.agents",
    "Person": "rlmflow.agents",
    "tool": "rlmflow.tools",
    "Toolkit": "rlmflow.tools",
    "current_repl": "rlmflow.repl",
    "attach_image": "rlmflow.repl",
    "Budget": "rlmflow.budget",
    "Checkpoint": "rlmflow.budget",
    "Stop": "rlmflow.hooks",
    "LocalRuntime": "rlmflow.runtime",
    "DockerRuntime": "rlmflow.runtime",
    "ModalRuntime": "rlmflow.runtime",
    "Copies": "rlmflow.workspaces",
    "GitWorktrees": "rlmflow.workspaces",
}


def __getattr__(name: str) -> Any:
    if name == "view":
        return import_module("rlmflow.view")
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module), name)


__all__ = sorted(_LAZY)
