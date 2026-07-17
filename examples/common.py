"""Shared helpers for the example scripts.

Examples run as standalone scripts at varying directory depths, so each one
starts with a tiny bootstrap that puts the ``examples/`` directory on
``sys.path`` before importing this module:

    import sys
    from pathlib import Path

    examples_dir = next(
        p for p in Path(__file__).resolve().parents if p.name == "examples"
    )
    if str(examples_dir) not in sys.path:
        sys.path.insert(0, str(examples_dir))

    from common import build_client, example_run_dir, save_example_graph

This module centralizes the bits every example was duplicating: building an LLM
client from a model name, resolving a per-example ``_runs`` directory, saving a
graph, and adding the common CLI flags. Run directories are anchored to this
file's location, so callers pass a name only (never ``__file__``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rflow import Graph
from rflow.clients import AnthropicClient, OpenAIClient

# The directory this module lives in, i.e. the repo's ``examples/`` folder.
EXAMPLES_DIR = Path(__file__).resolve().parent


def build_client(model: str):
    """Return an LLM client for ``model`` (Anthropic for ``claude*``, else OpenAI)."""
    if model.startswith("claude"):
        return AnthropicClient(model)
    return OpenAIClient(model)


def example_run_dir(name: str) -> Path:
    """Resolve ``examples/_runs/<name>``."""
    return EXAMPLES_DIR / "_runs" / name


def save_example_graph(
    graph: Graph,
    name: str,
    *,
    out_dir: str | Path | None = None,
    label: str = "Graph saved to",
) -> Path:
    """Save ``graph`` to ``out_dir`` (or the default ``_runs`` dir) and print the path."""
    path = graph.save(Path(out_dir) if out_dir is not None else example_run_dir(name))
    print(f"{label} {path}")
    return path


def add_model_args(
    parser: argparse.ArgumentParser,
    *,
    default: str = "gpt-5-mini",
    fast_default: str | None = None,
) -> None:
    """Add ``--model`` (and optionally ``--fast-model``) to a parser."""
    parser.add_argument("--model", default=default, help="Primary LLM model.")
    if fast_default is not None:
        parser.add_argument(
            "--fast-model", default=fast_default, help="Cheaper model for subagents."
        )


def add_flow_args(
    parser: argparse.ArgumentParser,
    *,
    max_depth: int = 1,
    max_iters: int = 8,
) -> None:
    """Add ``--max-depth`` / ``--max-iters`` with per-example defaults."""
    parser.add_argument("--max-depth", type=int, default=max_depth)
    parser.add_argument("--max-iters", type=int, default=max_iters)


def add_viz_arg(parser: argparse.ArgumentParser) -> None:
    """Add the common ``--no-viz`` flag."""
    parser.add_argument(
        "--no-viz", action="store_true", help="Disable the live terminal graph view."
    )


def add_out_dir_arg(
    parser: argparse.ArgumentParser,
    name: str,
    *,
    help: str | None = None,
) -> None:
    """Add ``--out-dir`` defaulting to this example's ``_runs`` directory."""
    default = example_run_dir(name)
    parser.add_argument(
        "--out-dir",
        default=str(default),
        help=help or f"Save/checkpoint the run here (default: {default}).",
    )


__all__ = [
    "EXAMPLES_DIR",
    "add_flow_args",
    "add_model_args",
    "add_out_dir_arg",
    "add_viz_arg",
    "build_client",
    "example_run_dir",
    "save_example_graph",
]
