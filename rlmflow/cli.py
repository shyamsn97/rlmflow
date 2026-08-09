"""``rlmflow`` command-line entry point.

Read or render a saved graph without writing a script::

    rlmflow view runs/coding/graph                # the tree, then the timeline
    rlmflow view runs/coding/graph --step 3       # just that step, with its content
    rlmflow view runs/coding/graph --frames-only  # every step as an ASCII tree
    rlmflow view runs/coding/graph --svg g.svg    # the figure
    rlmflow view runs/coding/graph --html r.html  # steppable, single file
    rlmflow view runs/coding/graph --browser      # the Gradio viewer
    rlmflow view runs/coding/graph --frames out/  # a PNG per step
    rlmflow view runs/coding/graph --gif run.gif  # the run, animated

``python -m rlmflow view …`` works the same.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rlmflow",
        description="Inspect and render rlmflow graphs from the terminal.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    view = sub.add_parser("view", help="Read or render a saved graph directory.")
    view.add_argument("source", type=Path, help="Graph checkpoint directory.")
    view.add_argument(
        "--step",
        type=int,
        default=None,
        metavar="N",
        help="Print step N (1-based) with its content, instead of the whole timeline.",
    )
    view.add_argument(
        "--frames-only",
        action="store_true",
        help="Print the ASCII agent tree as it stood at every step.",
    )
    view.add_argument(
        "--tree", action="store_true", help="Print only the agent tree, not the timeline."
    )
    view.add_argument(
        "--svg", type=Path, default=None, metavar="PATH", help="Write the figure as SVG."
    )
    view.add_argument(
        "--html",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write a single-file stepper: the graph per step, with node content.",
    )
    view.add_argument(
        "--browser",
        action="store_true",
        help="Open the run in the Gradio viewer (needs rlmflow[viewer]).",
    )
    view.add_argument(
        "--frames",
        type=Path,
        default=None,
        metavar="DIR",
        help="Write a PNG per step into DIR (needs rlmflow[image]).",
    )
    view.add_argument(
        "--gif",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write the run as an animated GIF (needs rlmflow[image]).",
    )
    view.add_argument(
        "--every",
        type=int,
        default=1,
        metavar="N",
        help="With --frames or --gif, keep every Nth step (default: every step).",
    )
    view.set_defaults(func=_cmd_view)

    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


def _cmd_view(args: argparse.Namespace) -> int:
    from rlmflow import persistence, render_tree
    from rlmflow.view import render_steps, save_html, save_svg, steps

    source: Path = args.source
    if not source.exists():
        print(f"error: path not found: {source}", file=sys.stderr)
        return 1
    try:
        graph = persistence.load(source)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"error: cannot read a graph from {source}: {exc}", file=sys.stderr)
        return 1

    walked = steps(graph)
    if args.step is not None:
        if args.step < 1 or args.step > len(walked):
            print(f"error: --step {args.step} out of range (1..{len(walked)})", file=sys.stderr)
            return 1
        step = walked[args.step - 1]
        print(f"=== step {args.step}/{len(walked)} · {step.title} · +{step.elapsed:.2f}s ===")
        print(step.detail or step.summary or "(no content)")
        return 0

    if args.frames_only:
        for index, frame in enumerate(render_steps(graph), 1):
            print(f"=== step {index}/{len(walked)} ===\n{frame}\n")
    else:
        print(render_tree(graph))
        if not args.tree:
            print(f"\n{len(walked)} steps")
            for step in walked:
                print(
                    f"  {step.index + 1:>4}  +{step.elapsed:7.2f}s  "
                    f"{step.title:<34}  {step.summary}"
                )

    if args.svg is not None:
        print(f"\nwrote {save_svg(graph, args.svg)}")
    if args.html is not None:
        print(f"wrote {save_html(graph, args.html)}")

    if args.frames is not None or args.gif is not None:
        try:
            from rlmflow.view.images import save_frames, save_gif
        except ImportError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        if args.frames is not None:
            written = save_frames(graph, args.frames, every=args.every)
            print(f"wrote {len(written)} frames to {args.frames}")
        if args.gif is not None:
            print(f"wrote {save_gif(graph, args.gif, every=args.every)}")

    if args.browser:
        try:
            from rlmflow.view.app import open_viewer
        except ImportError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        open_viewer(graph)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
