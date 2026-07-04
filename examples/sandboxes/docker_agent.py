"""Run a platformer-building Flow task inside Docker.

Each agent's code runs in a Docker container via ``rflow.minimal.DockerRuntime``.

Setup:
    docker build -t rlmflow:local .
    export OPENAI_API_KEY=...

Run:
    python examples/sandboxes/docker_agent.py --docker-image rlmflow:local
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rflow.clients import AnthropicClient, OpenAIClient  # noqa: E402
from rflow.minimal import DockerRuntime, Flow, Graph, LiveTreeRenderer  # noqa: E402


PLATFORMER_QUERY = """\
Build a simple 2D side-scrolling platformer in plain HTML/CSS/JS under output/.
No build tools, no libraries, no ES modules. Write files with plain Python
(e.g. `open(path, "w").write(...)`) in the sandbox.

Files:
- output/index.html
- output/styles.css
- output/scripts/engine.js
- output/scripts/main.js
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run minimal Flow inside Docker.")
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--fast-model", default="gpt-5-mini")
    parser.add_argument("--docker-image", default="rlmflow:local")
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--max-iters", type=int, default=5)
    parser.add_argument("--no-live", action="store_true")
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "examples" / "_runs" / "sandbox-docker"),
        help="Save the final run here.",
    )
    return parser.parse_args()


def build_llm(model: str):
    return (
        AnthropicClient(model)
        if model.startswith("claude")
        else OpenAIClient(model)
    )


async def run_turn(flow: Flow, query: str, *, use_live: bool) -> Graph:
    graph = flow.start(Graph(query=query))
    renderer = LiveTreeRenderer(clear=use_live)
    async for event in flow.run_streaming(graph):
        renderer.handle(event, graph)
    return graph


def main() -> None:
    args = parse_args()
    runtime = DockerRuntime(args.docker_image)
    flow = Flow(
        build_llm(args.model),
        llm_clients={"fast": build_llm(args.fast_model)},
        runtime=runtime,
        max_depth=args.max_depth,
        max_iters=args.max_iters,
    )
    try:
        graph = asyncio.run(run_turn(flow, PLATFORMER_QUERY, use_live=not args.no_live))
        print(graph.result())
        path = graph.save(Path(args.out_dir))
        print(f"Graph saved to {path}")
    finally:
        flow.close_repls()


if __name__ == "__main__":
    main()
