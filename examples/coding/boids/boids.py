"""Run the boids coding task with rflow, official RLM, or both.

Examples:
    python examples/coding/boids/boids.py --runner rflow
    python examples/coding/boids/boids.py --runner rlm
    python examples/coding/boids/boids.py --runner both

The two runners use the same query and write into suffixed directories so their
artifacts and trajectories can be compared directly:
`{out_dir}-rflow` and `{out_dir}-rlm-official`.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import os
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from pydantic import BaseModel

examples_dir = next(p for p in Path(__file__).resolve().parents if p.name == "examples")
if str(examples_dir) not in sys.path:
    sys.path.insert(0, str(examples_dir))

TASK = """Create a runnable browser-based boids simulation in plain HTML, CSS, and JavaScript.
Requirements:
- The main runnable interface is `index.html`.
- Write separate files:
    - `index.html`
    - `style.css`
    - `boids.js`
- Do not use build tools or external libraries.
- Use a dark color background.
- Do not use ES modules; wire scripts with `<script src="..."></script>` tags.
- Render 100s of colorful boids on a 2D canvas. Do not add configurations, just the canvas.
- Verify that all files exist, script tags are ordered correctly, and the JavaScript has no obvious syntax/runtime wiring errors before returning.
"""


@contextmanager
def pushd(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def build_rflow_llm(model: str):
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
    index_html: str
    style_css: str
    boids_js: str

def run_rflow(
    run_dir: Path,
    *,
    model: str,
    fast_model: str,
    max_depth: int,
    max_iters: int,
    max_concurrency: int,
    no_viz: bool,
) -> None:
    from rflow import (
        FILE_TOOLS,
        ConsumerGroup,
        Flow,
        Graph,
        GraphCheckpointer,
        LiveTreeRenderer,
        LocalRuntime,
    )

    reset_run_dir(run_dir, force=True)
    (run_dir / "task.txt").write_text(TASK)

    runtime = LocalRuntime(working_directory=run_dir)
    runtime.register_tools(FILE_TOOLS)
    flow = Flow(
        build_rflow_llm(model),
        llm_clients={"fast": build_rflow_llm(fast_model)},
        runtime=runtime,
        max_depth=max_depth,
        max_iters=max_iters,
        workers=max_concurrency,
    )

    print(f"\n=== rflow run ===\nworkdir: {run_dir}\nmodel: {model}\n")
    try:
        graph = Graph(query=TASK)
        graph_dir = run_dir / "graph"
        consumers = ConsumerGroup(
            [
                LiveTreeRenderer(clear=not no_viz),
                GraphCheckpointer(graph_dir),
            ]
        )

        async def drive() -> None:
            try:
                async for event in flow.run_streaming(
                    graph=graph,
                    output_schema=BoidsSimulation,
                ):
                    consumers.handle(event, graph)
            finally:
                consumers.close()

        asyncio.run(drive())

        result = graph.result() or ""
        (run_dir / "response.txt").write_text(str(result))
        print("\nrflow response:")
        print(result or "(no result)")
        print(f"\nrflow graph: {graph_dir}")
    finally:
        flow.close_repls()

    print("\nrflow files:")
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
        completion = rlm.completion(TASK)

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

    parser = argparse.ArgumentParser(description="Compare rflow and official RLM on boids.")
    parser.add_argument(
        "--runner",
        choices=("rflow", "rlm", "both"),
        default="both",
        help="Which runner to execute.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=default_out,
        help="Base output path. Runner outputs use -rflow and -rlm-official suffixes.",
    )
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--fast-model", default="gpt-5-mini")
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--max-iters", type=int, default=30)
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--no-viz", action="store_true", help="Disable rflow live tree.")
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
    rflow_dir = suffixed_dir(out_dir, "rflow")
    official_dir = suffixed_dir(out_dir, "rlm-official")

    if args.runner in ("rflow", "both"):
        if rflow_dir.exists() and args.force:
            shutil.rmtree(rflow_dir)
        run_rflow(
            rflow_dir,
            model=args.model,
            fast_model=args.fast_model,
            max_depth=args.max_depth,
            max_iters=args.max_iters,
            max_concurrency=args.max_concurrency,
            no_viz=args.no_viz,
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
    if args.runner in ("rflow", "both"):
        print(f"- rflow: {rflow_dir}")
    if args.runner in ("rlm", "both"):
        print(f"- official RLM: {official_dir}")


if __name__ == "__main__":
    main()
