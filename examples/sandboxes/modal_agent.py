"""Run a platformer-building Flow task inside a Modal Sandbox.

Each agent's code runs remotely in a Modal Sandbox (via a :class:`ModalRuntime`);
files are written with plain Python inside the sandbox working directory.

Setup:
    pip install -e ".[openai,modal]"
    export OPENAI_API_KEY=...
    modal setup

Run:
    python examples/sandboxes/modal_agent.py --model gpt-5
"""

# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rlmflow.clients import OpenAIClient  # noqa: E402
from rlmflow import (  # noqa: E402
    Flow,
    ModalRuntime,
    Node,
    start,
)
from rlmflow.consumers import ConsumerGroup, GraphCheckpointer  # noqa: E402
from rlmflow.view import LiveTreeRenderer  # noqa: E402

from examples.common import save_example_graph  # noqa: E402

REMOTE_REPO = "/opt/rlmflow"

PLATFORMER_QUERY = """\
Build a simple 2D side-scrolling platformer in plain HTML/CSS/JS under output/.
No build tools, no libraries, no ES modules. Write files with plain Python
(e.g. `open(path, "w").write(...)`) in the sandbox.

Files:
- output/index.html
- output/styles.css
- output/scripts/engine.js   — state, input, physics
- output/scripts/main.js     — level, render, requestAnimationFrame loop

index.html loads engine.js then main.js. Canvas with left/right movement, jump,
gravity, platform collision, scrolling camera, and restart.
"""


def log(message: str) -> None:
    print(f"[modal-agent] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Flow inside Modal.")
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument(
        "--fast-model",
        default="gpt-5-mini",
        help="Cheaper model exposed to delegates as `model='fast'`.",
    )
    parser.add_argument("--max-iters", type=int, default=5)
    parser.add_argument(
        "--max-depth",
        type=int,
        default=1,
        help="Recursive sub-agent depth. Defaults to 1 so delegation is enabled.",
    )
    parser.add_argument("--app-name", default="rlmflow")
    parser.add_argument(
        "--sandbox-timeout",
        type=int,
        default=3600,
        help="Modal sandbox lifetime in seconds. Multi-agent LLM turns can exceed 5 minutes.",
    )
    parser.add_argument("--repl-timeout", type=float, default=30)
    parser.add_argument("--remote-workdir", default="/workspace")
    parser.add_argument(
        "--no-live",
        action="store_true",
        help="Disable the live terminal graph view.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "examples" / "_runs" / "sandbox-modal"),
        help="Save the final run here (default: examples/_runs/sandbox-modal/).",
    )
    return parser.parse_args()


async def run_turn(flow: Flow, query: str, *, use_live: bool, out_dir: Path) -> Node:
    graph = start(query=query)
    consumers = ConsumerGroup(
        [
            LiveTreeRenderer(clear=use_live),
            GraphCheckpointer(out_dir / "graph"),
        ]
    )
    try:
        async for event in flow.run_streaming(graph):
            consumers.handle(event)
    finally:
        consumers.close()
    return graph


def local_rlmflow_image():
    import modal

    log(f"preparing Modal image from local checkout: {REPO_ROOT} -> {REMOTE_REPO}")
    return (
        modal.Image.debian_slim()
        .add_local_dir(
            REPO_ROOT,
            remote_path=REMOTE_REPO,
            copy=True,
            ignore=[
                ".git",
                ".venv",
                "__pycache__",
                ".pytest_cache",
                ".ruff_cache",
                "media",
                "docs",
                "examples/_runs",
            ],
        )
        .run_commands(f"python -m pip install -e {REMOTE_REPO}")
    )


def main() -> None:
    args = parse_args()
    image = local_rlmflow_image()

    # One sandbox per agent; created lazily when the agent first runs code.
    runtime = ModalRuntime(
        app_name=args.app_name,
        remote_workdir=args.remote_workdir,
        image=image,
        timeout=args.sandbox_timeout,
        repl_timeout=args.repl_timeout,
    )

    log(
        f"creating Flow with model={args.model}, fast_model={args.fast_model}, "
        f"max_iters={args.max_iters}, max_depth={args.max_depth}"
    )
    flow = Flow(
        OpenAIClient(model=args.model),
        llm_clients={"fast": OpenAIClient(model=args.fast_model)},
        runtime=runtime,
        max_depth=args.max_depth,
        max_iters=args.max_iters,
    )
    log("running platformer task; first run may build/start Modal sandbox")
    try:
        graph = asyncio.run(
            run_turn(
                flow,
                PLATFORMER_QUERY,
                use_live=not args.no_live,
                out_dir=Path(args.out_dir),
            )
        )
        print(graph.agent_result())
        save_example_graph(graph, "sandbox-modal", out_dir=args.out_dir)
    finally:
        log("closing Flow")
        flow.runtime.close_repls()


if __name__ == "__main__":
    main()
