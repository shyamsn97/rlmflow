"""Animate what a rewind actually does: play the jam, undo it push by push, recover.

The per-branch GIFs each show one timeline going forwards, which is the one thing
a rewind is not. This stitches three phases into a single strip so the undo is
visible: the worker shoving a box to the wall, those pushes coming back off the
board one at a time down to the depth the shepherd chose, and the winning branch
replaying from there.

The reverse phase is the worker's own frames read backwards. That is honest — a
rewind returns the board to a state it really held, and the frame for that state
is the one the worker recorded on the way out.

Needs Pillow and the sprite tiles (see ``sprites.py``). Run a shepherd first, then:

    python examples/shepherd/render_rewind.py
    python examples/shepherd/render_rewind.py --out docs/shepherd_rewind.gif
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

import sprites  # noqa: E402
from render_graph import RUN_DIR, board_rows, picked_summary, rewind_depth  # noqa: E402

from rlmflow import persistence  # noqa: E402

JAM_COLOR = (217, 59, 48)
REWIND_COLOR = (245, 165, 36)
RECOVERY_COLOR = (52, 168, 83)
BAR_BG = (24, 26, 32)
BAR_TEXT = (236, 239, 244)


@dataclass
class Frame:
    board: str
    caption: str
    color: tuple[int, int, int]
    hold_ms: int
    phase: str


def load_frames(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return [f for f in data if isinstance(f, dict) and board_rows(f.get("board", ""))]


def initial_board() -> str:
    """The board before anything moved, from the example's own definition of it."""
    import shepherd
    from sokoban import Sokoban

    return Sokoban(shepherd.BOARD).render(ids=True)


def plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}es"


def push_boundaries(frames: list[dict]) -> list[int]:
    """Frame index of each push, so a rewind measured in pushes can find its frame."""
    bounds = []
    for index, frame in enumerate(frames):
        label = str(frame.get("label", ""))
        if label.startswith("push"):
            bounds.append(index)
    return bounds


def build_strip(
    *,
    jam: list[dict],
    recovery: list[dict],
    branch_name: str,
    rewind: int,
    outcome: str,
    step_ms: int,
    pause_ms: int,
) -> list[Frame]:
    start = initial_board()
    pushes = len(push_boundaries(jam)) or len(jam)
    keep = max(0, pushes - rewind)
    solved = outcome == "solved"
    recovery_color = RECOVERY_COLOR if solved else JAM_COLOR

    strip = [Frame(start, f"jam · 0 of {pushes} pushes", JAM_COLOR, pause_ms, "jam")]
    for index, frame in enumerate(jam, start=1):
        strip.append(
            Frame(
                frame["board"],
                f"jam · push {index} of {pushes} · {frame.get('label', '')}",
                JAM_COLOR,
                step_ms,
                "jam",
            )
        )
    strip[-1] = Frame(
        strip[-1].board,
        f"jam · push {pushes} of {pushes} · box against the wall, board dead",
        JAM_COLOR,
        pause_ms * 2,
        "jam",
    )

    # Walk the jam back out: the board after n pushes is the frame the worker left
    # for n, and the pre-push board for n == 0.
    boards = [start] + [frame["board"] for frame in jam]
    undo_from = len(strip)
    for remaining in range(len(boards) - 2, keep - 1, -1):
        undone = pushes - remaining
        strip.append(
            Frame(
                boards[remaining],
                f"rewind · undid {undone} of {rewind} · {plural(remaining, 'push')} left",
                REWIND_COLOR,
                step_ms,
                "rewind",
            )
        )
    if len(strip) > undo_from:
        strip[-1] = Frame(
            strip[-1].board,
            f"rewind · {plural(keep, 'push')} kept, {branch_name} starts here",
            REWIND_COLOR,
            pause_ms * 2,
            "rewind",
        )

    # A branch replays the history it inherited before making its own first move, and
    # the rewind just walked back through exactly those pushes, so drop them rather
    # than animate the jam twice.
    bounds = push_boundaries(recovery)
    own = recovery[bounds[keep - 1] + 1 :] if keep and len(bounds) >= keep else recovery
    total = keep + sum(1 for f in own if str(f.get("label", "")).startswith("push"))
    done = keep
    for index, frame in enumerate(own, start=1):
        label = str(frame.get("label", ""))
        if label.startswith("push"):
            done += 1
            caption = f"{branch_name} · push {done} of {total} · {label}"
        else:
            caption = f"{branch_name} · {label}"
        strip.append(
            Frame(
                frame["board"],
                caption,
                recovery_color,
                pause_ms if index == len(own) else step_ms,
                "recovery",
            )
        )
    if own:
        ending = f"solved in {total} pushes" if solved else f"{outcome} after {total} pushes"
        strip[-1] = Frame(
            strip[-1].board, f"{branch_name} · {ending}", recovery_color, pause_ms * 5, "recovery"
        )
    return strip


def compose(strip: list[Frame], tile: int, bar_h: int):
    """Board plus a caption bar per frame, as PIL images."""
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.load_default(size=max(11, int(bar_h * 0.5)))
    images = []
    for frame in strip:
        board = sprites.render_image(frame.board, tile)
        if board is None:
            continue
        canvas = Image.new("RGB", (board.width, board.height + bar_h), BAR_BG)
        canvas.paste(board, (0, 0))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle([0, board.height, board.width, board.height + bar_h], fill=BAR_BG)
        draw.rectangle([0, board.height, 4, board.height + bar_h], fill=frame.color)
        draw.text(
            (11, board.height + bar_h / 2),
            frame.caption,
            font=font,
            fill=BAR_TEXT,
            anchor="lm",
        )
        images.append(canvas)
    return images


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR, help="Directory shepherd saved.")
    parser.add_argument("--out", type=Path, default=None, help="Default <run-dir>/rewind.gif.")
    parser.add_argument("--branch", default=None, help="Branch to recover with (default: picked).")
    parser.add_argument("--tile", type=int, default=32, help="Sprite tile size in pixels.")
    parser.add_argument("--step-ms", type=int, default=260)
    parser.add_argument("--pause-ms", type=int, default=650)
    args = parser.parse_args()

    if not sprites.available():
        raise SystemExit("needs Pillow and the sprite tiles: pip install pillow")
    run_dir: Path = args.run_dir
    if not (run_dir / "shepherd").exists():
        raise SystemExit(
            f"no saved shepherd graph under {run_dir}. Run: python examples/shepherd/shepherd.py"
        )

    shepherd_graph = persistence.load(run_dir / "shepherd")
    summary = picked_summary(shepherd_graph)
    found = re.search(r"Picked (\S+?):", summary)
    name = args.branch or (found.group(1) if found else "")
    agent = next((a for a in shepherd_graph.sub_agents if a.config.name == name), None)
    if agent is None:
        names = ", ".join(a.config.name for a in shepherd_graph.sub_agents)
        raise SystemExit(f"no branch {name!r} in that run. Available: {names}")

    jam = load_frames(run_dir / "traces" / "worker.json")
    recovery = load_frames(run_dir / "traces" / f"{name}.json")
    if not jam or not recovery:
        raise SystemExit(
            "need traces/worker.json and traces/<branch>.json from a run that exported them"
        )

    strip = build_strip(
        jam=jam,
        recovery=recovery,
        branch_name=name,
        rewind=rewind_depth(agent) or len(jam),
        outcome=str(agent.result() or "unfinished"),
        step_ms=args.step_ms,
        pause_ms=args.pause_ms,
    )
    images = compose(strip, args.tile, bar_h=max(20, args.tile - 6))
    if not images:
        raise SystemExit("no frame could be rendered")

    out: Path = args.out or (run_dir / "rewind.gif")
    out.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        out,
        format="GIF",
        save_all=True,
        append_images=images[1:],
        duration=[frame.hold_ms for frame in strip][: len(images)],
        loop=0,
        optimize=True,
    )
    counts = {phase: sum(1 for f in strip if f.phase == phase) for phase in ("jam", "rewind")}
    counts[name] = sum(1 for f in strip if f.phase == "recovery")
    print(summary or "no branch picked")
    print(f"frames: {len(images)} ({', '.join(f'{k} {v}' for k, v in counts.items())})")
    print(f"gif: {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
