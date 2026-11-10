"""Build a small platformer with every agent's code running in a sandbox.

``--runtime docker`` runs each agent's REPL in a Docker container
(``DockerRuntime``); ``--runtime modal`` runs it in a Modal Sandbox
(``ModalRuntime``), built from this checkout. Files are written with plain
Python inside the sandbox's working directory.

Setup:
    export OPENAI_API_KEY=...
    docker build -t rlmflow:local .               # for --runtime docker
    pip install -e ".[openai,modal]"; modal setup  # for --runtime modal

Run:
    python examples/sandboxes/platformer.py --runtime docker --docker-image rlmflow:local
    python examples/sandboxes/platformer.py --runtime modal --model gpt-5
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

from examples.common import print_run  # noqa: E402
from rlmflow import AgentConfig, DockerRuntime, Flow, Graph, ModalRuntime  # noqa: E402

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runtime", choices=("docker", "modal"), default="docker")
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument(
        "--fast-model",
        default="gpt-5-mini",
        help="Cheaper model for the `fast` config delegates can launch.",
    )
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--max-iters", type=int, default=5)
    parser.add_argument("--docker-image", default="rlmflow:local", help="--runtime docker")
    parser.add_argument("--app-name", default="rlmflow", help="--runtime modal")
    parser.add_argument(
        "--sandbox-timeout",
        type=int,
        default=3600,
        help="--runtime modal: sandbox lifetime in seconds; multi-agent turns can exceed 5 min.",
    )
    parser.add_argument("--repl-timeout", type=float, default=30, help="--runtime modal")
    parser.add_argument("--remote-workdir", default="/workspace", help="--runtime modal")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Save the final run here (default: examples/_runs/sandbox-<runtime>/).",
    )
    return parser.parse_args()


def modal_runtime(args: argparse.Namespace) -> ModalRuntime:
    import modal

    print(f"preparing Modal image from local checkout: {REPO_ROOT} -> {REMOTE_REPO}")
    image = (
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
    return ModalRuntime(
        app_name=args.app_name,
        remote_workdir=args.remote_workdir,
        image=image,
        timeout=args.sandbox_timeout,
        repl_timeout=args.repl_timeout,
    )


def main() -> None:
    args = parse_args()
    runtime = DockerRuntime(args.docker_image) if args.runtime == "docker" else modal_runtime(args)
    flow = Flow(
        configs=[
            AgentConfig(model=args.model, max_depth=args.max_depth, max_iters=args.max_iters),
            AgentConfig(
                name="fast",
                model=args.fast_model,
                description="cheaper model for a bounded piece of the game",
                max_depth=args.max_depth,
                max_iters=args.max_iters,
            ),
        ],
        runtime=runtime,
    )
    g = Graph()
    root = g.start(PLATFORMER_QUERY)
    out_dir = Path(args.out_dir or REPO_ROOT / "examples" / "_runs" / f"sandbox-{args.runtime}")
    try:
        asyncio.run(print_run(flow, g, out_dir))
        print(root.final_result())
        print(f"Run saved to {out_dir}")
    finally:
        flow.runtime.close_repls()


if __name__ == "__main__":
    main()
