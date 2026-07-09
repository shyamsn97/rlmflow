"""Run Flow with Tinker inference.

Requires Tinker credentials and optional dependencies:

    export TINKER_API_KEY=...
    pip install -e ".[tinker]"
    python examples/providers/tinker_agent.py
"""

from __future__ import annotations

from pathlib import Path

import argparse
import asyncio

from rflow.minimal.clients import TinkerClient
from rflow.minimal import Flow, Graph, LiveTreeRenderer


def _example_run_dir(source_file: str | Path, name: str) -> Path:
    source = Path(source_file).resolve()
    for parent in (source.parent, *source.parents):
        if parent.name == "examples":
            return parent / "_runs" / name
    return source.parent / "_runs" / name


def _save_example_graph(
    graph,
    source_file: str | Path,
    name: str,
    *,
    out_dir: str | Path | None = None,
    label: str = "Graph saved to",
) -> Path:
    path = graph.save(
        Path(out_dir) if out_dir is not None else _example_run_dir(source_file, name)
    )
    print(f"{label} {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a tiny Flow task with Tinker."
    )
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen3-8B",
        help="Tinker base model for inference.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional saved Tinker weights path, e.g. tinker://run/weights/checkpoint.",
    )
    parser.add_argument(
        "--renderer",
        default="qwen3",
        help="Tinker cookbook renderer name matching the model family.",
    )
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-iters", type=int, default=4)
    parser.add_argument(
        "--query",
        default="Use Python to compute 17 * 23, then call done() with the answer.",
    )
    args = parser.parse_args()

    llm = TinkerClient(
        base_model=None if args.model_path else args.base_model,
        model_path=args.model_path,
        renderer=args.renderer,
        max_tokens=args.max_tokens,
    )
    flow = Flow(llm, max_iters=args.max_iters)
    print(f"Query: {args.query}\n")
    graph = Graph(query=args.query)

    async def drive() -> None:
        renderer = LiveTreeRenderer()
        async for event in flow.run_streaming(graph):
            renderer.handle(event, graph)

    asyncio.run(drive())
    print(graph.result())
    _save_example_graph(graph, __file__, "tinker-agent")
    flow.close_repls(graph.graph_id)


if __name__ == "__main__":
    main()
