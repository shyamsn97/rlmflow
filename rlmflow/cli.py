"""``rlmflow`` command-line entry point.

Render a saved graph directory (or any :class:`~rlmflow.utils.viewer.ViewSource`)
as ASCII step frames, a Gradio stepper, PNGs, or a GIF::

    rlmflow view runs/coding/graph
    rlmflow view runs/coding/graph --step 3
    rlmflow view runs/coding/graph --browser
    rlmflow view runs/coding/graph --frames blog/frames/
    rlmflow view runs/coding/graph --image hero.png
    rlmflow view runs/coding/graph --gif run.gif
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

    view = sub.add_parser(
        "view",
        help="Render a saved graph directory (ASCII steps by default).",
    )
    view.add_argument(
        "source",
        type=Path,
        help="Graph checkpoint directory, or a path Graph.load can read.",
    )
    view.add_argument(
        "--step",
        type=int,
        default=None,
        metavar="N",
        help="Print only step N (1-based). Default: print every step.",
    )
    view.add_argument(
        "--browser",
        action="store_true",
        help="Open the Gradio stepper (needs: pip install rlmflow[viewer]).",
    )
    view.add_argument(
        "--image",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write the final graph as PNG/SVG/PDF (needs: rlmflow[image]).",
    )
    view.add_argument(
        "--frames",
        type=Path,
        default=None,
        metavar="DIR",
        help="Write one PNG per step into DIR (needs: rlmflow[image]).",
    )
    view.add_argument(
        "--gif",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write an animated GIF of the run (needs: rlmflow[image]).",
    )
    view.set_defaults(func=_cmd_view)

    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


def _cmd_view(args: argparse.Namespace) -> int:
    source = args.source
    if not source.exists():
        print(f"error: path not found: {source}", file=sys.stderr)
        return 1

    wants_export = (
        args.image is not None or args.frames is not None or args.gif is not None
    )
    if args.browser:
        from rlmflow.utils.viewer import open_viewer

        open_viewer(source)
        return 0

    if wants_export:
        return _export(source, args)

    from rlmflow.utils.viewer import render_steps

    frames = render_steps(source)
    if not frames:
        print("(empty graph)", file=sys.stderr)
        return 1
    if args.step is not None:
        if args.step < 1 or args.step > len(frames):
            print(
                f"error: --step {args.step} out of range (1..{len(frames)})",
                file=sys.stderr,
            )
            return 1
        print(frames[args.step - 1])
        return 0
    for i, frame in enumerate(frames, 1):
        print(f"=== step {i}/{len(frames)} ===")
        print(frame)
        if i != len(frames):
            print()
    return 0


def _export(source: Path, args: argparse.Namespace) -> int:
    if args.image is not None:
        from rlmflow.utils.viewer import save_image

        path = save_image(source, args.image)
        print(f"wrote {path}")
    if args.frames is not None:
        from rlmflow.utils.viewer import save_steps

        paths = save_steps(source, args.frames)
        print(f"wrote {len(paths)} frames under {args.frames}")
    if args.gif is not None:
        from rlmflow.utils.viewer import save_gif

        path = save_gif(source, args.gif)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
