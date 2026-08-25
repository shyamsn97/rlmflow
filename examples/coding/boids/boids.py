"""Run the boids coding task with rlmflow, official RLM, or both.

Examples:
    python examples/coding/boids/boids.py --runner rlmflow
    python examples/coding/boids/boids.py --runner rlm
    python examples/coding/boids/boids.py --runner both

The two runners use the same query and write into suffixed directories so their
artifacts and trajectories can be compared directly:
`{out_dir}-rlmflow` and `{out_dir}-rlm-official`.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import os
import shutil
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from pydantic import BaseModel

examples_dir = next(p for p in Path(__file__).resolve().parents if p.name == "examples")
if str(examples_dir) not in sys.path:
    sys.path.insert(0, str(examples_dir))

TASK = """Create a runnable browser-based boids simulation in plain HTML, CSS, and JavaScript."""

CONTEXT = """
Requirements:
- The main runnable interface is `index.html`. No build tools and no external
    libraries. Do not use ES modules; wire scripts with `<script src="..."></script>`
    tags. Do not add configuration controls, just the canvas.
- Render 2000 boids on a full-viewport 2D canvas over a dark background, at 60fps.
- Modules and their public contracts. The names are fixed, and each file defines
    exactly the global it is listed with:
    - `vec.js` defines `Vec` with add, sub, scale, limit, mag, normalize, and dist.
    - `spatial.js` defines `SpatialGrid(cellSize)` with insert(boid), clear(), and
        neighbors(boid, radius). It must serve every boid each frame without
        all-pairs scanning.
    - `rules.js` defines `Rules` with align, cohere, separate, and
        avoid(boid, obstacles). Each returns a steering `Vec`.
    - `species.js` defines `SPECIES`: four entries, each with its own color,
        maxSpeed, perception radius, and rule weights.
    - `render.js` defines `Renderer(ctx)` with draw(boids, obstacles, fps): HiDPI
        scaling, fading trails, orientation-aware boid shapes, and an on-canvas
        FPS readout.
    - `main.js` boots the boids across the four species, rebuilds the grid every
        frame, applies the rules, wraps movement at the edges, and drives the
        requestAnimationFrame loop.
    - `index.html` loads the scripts in dependency order.
    - `style.css` styles the full-viewport dark canvas.
- Verify before returning: every file exists, every listed global is defined in
    its own file, script tags are in dependency order, and no file uses a global
    it was not given.
"""


@contextmanager
def pushd(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def build_rlmflow_llm(model: str):
    from common import build_client

    return build_client(model)


def reset_run_dir(path: Path, *, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(
                f"{path} already exists. Pass --force to replace it, or use --out-dir."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def summarize_files(path: Path) -> list[str]:
    files = []
    for candidate in sorted(path.rglob("*")):
        if candidate.is_file():
            files.append(f"{candidate.relative_to(path)} ({candidate.stat().st_size} bytes)")
    return files


def suffixed_dir(base: Path, suffix: str) -> Path:
    return base.with_name(f"{base.name}-{suffix}")


def supported_kwargs(callable_obj, kwargs: dict) -> dict:
    parameters = inspect.signature(callable_obj).parameters
    accepted = {name for name, param in parameters.items() if param.kind != param.VAR_KEYWORD}
    if any(param.kind == param.VAR_KEYWORD for param in parameters.values()):
        return kwargs
    dropped = sorted(set(kwargs) - accepted)
    if dropped:
        print(f"Skipping unsupported official RLM kwargs: {', '.join(dropped)}")
    return {name: value for name, value in kwargs.items() if name in accepted}


class BoidsSimulation(BaseModel):
    """The final answer: what was written and what was checked.

    The files themselves live in the workspace, so echoing their contents back
    through the schema would only duplicate them into the transcript.
    """

    files: list[str]
    verification: str


def run_rlmflow(
    run_dir: Path,
    *,
    model: str,
    fast_model: str,
    max_depth: int,
    max_iters: int,
    max_concurrency: int,
) -> None:
    from rlmflow import (
        FILE_TOOLS,
        AgentStart,
        Flow,
        LocalRuntime,
        json_schema_for,
    )

    reset_run_dir(run_dir, force=True)
    (run_dir / "task.txt").write_text(TASK)

    runtime = LocalRuntime(working_directory=run_dir)
    flow = Flow(
        build_rlmflow_llm(model),
        llm_clients={"fast": build_rlmflow_llm(fast_model)},
        runtime=runtime,
        tools=FILE_TOOLS,
        workers=max_concurrency,
    )

    print(f"\n=== rlmflow run ===\nworkdir: {run_dir}\nmodel: {model}\n")
    try:
        root = flow.start(
            TASK,
            inputs={"context": CONTEXT},
            output_schema=json_schema_for(BoidsSimulation),
            max_depth=max_depth,
            max_iters=max_iters,
        )
        graph_dir = run_dir / "graph"

        async def drive(root: AgentStart) -> None:
            async for node in flow.run_streaming(root):
                print(f"{node.parent_agent.config.path:<20} {node.type}")
                root.save(graph_dir)

        asyncio.run(drive(root))

        result = root.result() or ""
        (run_dir / "response.txt").write_text(str(result))
        print("\nrlmflow response:")
        print(result or "(no result)")
        print(f"\nrlmflow graph: {graph_dir}")
    finally:
        flow.runtime.close_repls()

    print("\nrlmflow files:")
    for item in summarize_files(run_dir):
        print(f"- {item}")


def run_official_rlm(
    run_dir: Path,
    *,
    model: str,
    max_depth: int,
    max_iters: int,
    max_concurrency: int,
    verbose: bool,
) -> None:
    try:
        from rlm import RLM  # type: ignore[reportMissingImports]
        from rlm.logger import RLMLogger  # type: ignore[reportMissingImports]
    except ImportError as exc:
        raise SystemExit(
            "Official RLM is not installed. Install it with `python -m pip install rlms`."
        ) from exc

    reset_run_dir(run_dir, force=True)
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "task.txt").write_text(TASK)

    print(f"\n=== official RLM run ===\nworkdir: {run_dir}\nmodel: {model}\n")
    logger = RLMLogger(log_dir=str(log_dir), file_name="boids")
    rlm_kwargs = supported_kwargs(
        RLM,
        {
            "backend": "openai",
            "backend_kwargs": {"model_name": model},
            "max_depth": max_depth,
            "max_iterations": max_iters,
            "max_concurrent_subcalls": max_concurrency,
            "logger": logger,
            "verbose": verbose,
        },
    )
    rlm = RLM(**rlm_kwargs)

    # Official RLM's local REPL runs in-process, so cwd controls where normal
    # Python file writes land.
    with pushd(run_dir):
        completion = rlm.completion(CONTEXT, root_prompt=TASK)

    response = str(completion.response)
    (run_dir / "response.txt").write_text(response)
    if completion.metadata is not None:
        (run_dir / "metadata_type.txt").write_text(type(completion.metadata).__name__)

    print("\nofficial RLM response:")
    print(response or "(no result)")
    print(f"\nofficial RLM logs: {log_dir}")
    print("\nofficial RLM files:")
    for item in summarize_files(run_dir):
        print(f"- {item}")


def parse_args() -> argparse.Namespace:
    examples_root = Path(__file__).resolve().parents[2]
    default_out = examples_root / "_runs" / "coding" / "boids"

    parser = argparse.ArgumentParser(description="Compare rlmflow and official RLM on boids.")
    parser.add_argument(
        "--runner",
        choices=("rlmflow", "rlm", "both"),
        default="both",
        help="Which runner to execute.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=default_out,
        help="Base output path. Runner outputs use -rlmflow and -rlm-official suffixes.",
    )
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--fast-model", default="gpt-5-mini")
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--max-iters", type=int, default=30)
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument(
        "--rlm-quiet",
        action="store_true",
        help="Disable official RLM verbose console output.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace selected runner output directories before running.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    rlmflow_dir = suffixed_dir(out_dir, "rlmflow")
    official_dir = suffixed_dir(out_dir, "rlm-official")

    if args.runner in ("rlmflow", "both"):
        if rlmflow_dir.exists() and args.force:
            shutil.rmtree(rlmflow_dir)
        run_rlmflow(
            rlmflow_dir,
            model=args.model,
            fast_model=args.fast_model,
            max_depth=args.max_depth,
            max_iters=args.max_iters,
            max_concurrency=args.max_concurrency,
        )

    if args.runner in ("rlm", "both"):
        if official_dir.exists() and args.force:
            shutil.rmtree(official_dir)
        run_official_rlm(
            official_dir,
            model=args.model,
            max_depth=args.max_depth,
            max_iters=args.max_iters,
            max_concurrency=args.max_concurrency,
            verbose=not args.rlm_quiet,
        )

    print("\nDone. Compare outputs:")
    if args.runner in ("rlmflow", "both"):
        print(f"- rlmflow: {rlmflow_dir}")
    if args.runner in ("rlm", "both"):
        print(f"- official RLM: {official_dir}")


if __name__ == "__main__":
    main()
