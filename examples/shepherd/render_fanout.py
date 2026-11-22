"""Animate the whole recovery as a tree: the jam, the rewind, then every branch at once.

``rewind.gif`` follows the picked branch alone. This one keeps the camera on the
whole run: the worker jams the board, the dead board shrinks into the root of a
tree, one child per branch spawns below it, each child takes pushes back off the
board down to the depth its plan chose, and then every child plays out side by
side until it solves or sticks. The pick and its edge are outlined at the end.

Needs Pillow and the sprite tiles (see ``sprites.py``). Run a shepherd first, then:

    python examples/shepherd/render_fanout.py
    python examples/shepherd/render_fanout.py --out docs/static/shepherd/fanout.gif
"""

from __future__ import annotations

import argparse
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
from render_graph import (  # noqa: E402
    RUN_DIR,
    load_branches,
    load_run,
    picked_summary,
    rewind_depth,
)
from render_rewind import (  # noqa: E402
    BAR_BG,
    BAR_TEXT,
    JAM_COLOR,
    RECOVERY_COLOR,
    REWIND_COLOR,
    initial_board,
    load_frames,
    push_boundaries,
)

BG = (13, 15, 20)
LABEL = (160, 166, 178)
EDGE = (70, 74, 84)
COLS = 4
COL_GAP = 28
ROW_GAP = 30
MARGIN = 14
ROOT_SCALE = 1.4
EDGE_DROP = 44
TWEEN_MS = 50


@dataclass
class Branch:
    name: str
    rewind: int
    start: str
    frames: list[dict]
    outcome: str
    pushes: int

    @property
    def solved(self) -> bool:
        return self.outcome == "solved"


def load(run_dir: Path) -> tuple[list[str], list[Branch], str]:
    """The jam's boards (initial board first), every branch, and the picked branch's name."""
    _, worker, meta, shepherd = load_run(run_dir)
    found = re.search(r"Picked (\S+?):", picked_summary(meta, shepherd))
    jam = load_frames(run_dir / "traces" / "worker.json")
    boards = [initial_board()] + [frame["board"] for frame in jam]
    pushes = len(boards) - 1
    branches = []
    for name, graph, agent in load_branches(run_dir, worker):
        frames = load_frames(run_dir / "traces" / f"{name}.json")
        rewind = rewind_depth(graph, agent) or pushes
        keep = max(0, pushes - rewind)
        bounds = push_boundaries(frames)
        own = frames[bounds[keep - 1] + 1 :] if keep and len(bounds) >= keep else frames
        total = keep + sum(1 for f in own if str(f.get("label", "")).startswith("push"))
        outcome = str(agent.final_result() or "unfinished")
        branches.append(Branch(name, rewind, boards[keep], own, outcome, total))
    return boards, branches, found.group(1) if found else ""


Box = tuple[int, int, int, int]


class Tree:
    """Where the root and each child sit, and how to draw a board, an edge or a caption."""

    def __init__(self, count: int, tile: int) -> None:
        from PIL import ImageFont

        self.tile = tile
        cw, ch = sprites.render_image(initial_board(), tile).size
        self.bar_h = max(28, tile + 12)
        self.label_h = max(18, tile + 4)
        cols = min(COLS, count)
        rows = -(-count // cols)
        rw, rh = int(cw * ROOT_SCALE), int(ch * ROOT_SCALE)
        width = cols * cw + (cols - 1) * COL_GAP + 2 * MARGIN
        root_y = self.bar_h + MARGIN
        self.top = root_y + rh + self.label_h + EDGE_DROP
        row_h = ch + self.label_h + ROW_GAP
        self.size = (width, self.top + rows * row_h - ROW_GAP + MARGIN)
        self.root: Box = ((width - rw) // 2, root_y, rw, rh)
        self.children: list[Box] = [
            (MARGIN + (i % cols) * (cw + COL_GAP), self.top + (i // cols) * row_h, cw, ch)
            for i in range(count)
        ]
        area_h = self.size[1] - self.bar_h - 2 * MARGIN
        hw = min(int(cw * area_h / ch), width - 2 * MARGIN)
        hh = int(ch * hw / cw)
        self.hero: Box = ((width - hw) // 2, self.bar_h + MARGIN, hw, hh)
        self.font = ImageFont.load_default(size=max(13, int(self.bar_h * 0.46)))
        self.small = ImageFont.load_default(size=max(11, int(self.label_h * 0.64)))
        self._boards: dict[tuple[str, int, int], object] = {}

    def frame(self, caption: str, color):
        from PIL import Image, ImageDraw

        image = Image.new("RGB", self.size, BG)
        draw = ImageDraw.Draw(image)
        draw.rectangle([0, 0, self.size[0], self.bar_h], fill=BAR_BG)
        draw.rectangle([0, 0, 5, self.bar_h], fill=color)
        draw.text((14, self.bar_h / 2), caption, font=self.font, fill=BAR_TEXT, anchor="lm")
        return image, draw

    def board(self, image, draw, text: str, box: Box, border, width=2, label="", color=LABEL):
        from PIL import Image

        x, y, w, h = box
        if w < 4 or h < 4:
            return
        key = (text, w, h)
        if key not in self._boards:
            big = sprites.render_image(text, self.tile * 2)
            self._boards[key] = big.resize((w, h), Image.LANCZOS)
        image.paste(self._boards[key], (x, y))
        draw.rectangle(
            [x - width, y - width, x + w + width - 1, y + h + width - 1],
            outline=border,
            width=width,
        )
        if label:
            draw.text(
                (x + w / 2, y + h + width + self.label_h / 2),
                label,
                font=self.small,
                fill=color,
                anchor="mm",
            )

    def edge(self, draw, index: int, color=EDGE, width=2, box: Box | None = None) -> None:
        """From the root down to child ``index`` (or to ``box``, where that child is while it
        spawns); lower rows run down the column gaps."""
        rx, ry, rw, rh = self.root
        x, y, w, _ = box or self.children[index]
        x0, y0 = rx + rw / 2, ry + rh + self.label_h
        bus = max(y0 + 4, min(self.top - EDGE_DROP / 2, y - 4))
        cx = x + w / 2
        if self.children[index][1] == self.top:
            points = [(x0, y0), (x0, bus), (cx, bus), (cx, y - 4)]
        else:
            lane = x - COL_GAP / 2 if x > MARGIN else MARGIN / 2
            above = max(bus, y - ROW_GAP / 2)
            points = [(x0, y0), (x0, bus), (lane, bus), (lane, above), (cx, above), (cx, y - 4)]
        draw.line(points, fill=color, width=width, joint="curve")


def lerp(a: tuple, b: tuple, t: float) -> tuple[int, ...]:
    return tuple(int(round(x + (y - x) * t)) for x, y in zip(a, b))


def ease(t: float) -> float:
    return t * t * (3 - 2 * t)


def build(
    boards: list[str],
    branches: list[Branch],
    picked: str,
    *,
    tile: int,
    step_ms: int,
    pause_ms: int,
    tween_frames: int,
):
    tree = Tree(len(branches), tile)
    pushes = len(boards) - 1
    dead = boards[-1]
    root_label = f"worker · jammed after {pushes} pushes"
    frames: list[tuple[object, int]] = []

    def root(image, draw):
        tree.board(image, draw, dead, tree.root, JAM_COLOR, label=root_label, color=JAM_COLOR)

    def finished(branch: Branch) -> tuple[str, str, tuple]:
        text = branch.frames[-1]["board"] if branch.frames else branch.start
        if branch.solved:
            return text, f"{branch.name} · solved, {branch.pushes}", RECOVERY_COLOR
        return text, f"{branch.name} · stuck", JAM_COLOR

    # The jam, full size.
    for n, text in enumerate(boards):
        image, draw = tree.frame(f"worker · push {n} of {pushes}", JAM_COLOR)
        tree.board(image, draw, text, tree.hero, JAM_COLOR, width=3)
        frames.append((image, step_ms * 2))
    image, draw = tree.frame(f"worker jammed after {pushes} pushes · board dead", JAM_COLOR)
    tree.board(image, draw, dead, tree.hero, JAM_COLOR, width=3)
    frames.append((image, pause_ms * 2))

    # It shrinks into the root.
    for i in range(1, tween_frames + 1):
        image, draw = tree.frame("the shepherd reads the stuck run", REWIND_COLOR)
        box = lerp(tree.hero, tree.root, ease(i / tween_frames))
        tree.board(image, draw, dead, box, JAM_COLOR, width=3)
        frames.append((image, TWEEN_MS))

    # One child per branch spawns below it, still holding the dead board.
    rx, ry, rw, rh = tree.root
    seed = (rx + rw // 2, ry + rh, 0, 0)
    for i in range(1, tween_frames + 1):
        t = ease(i / tween_frames)
        image, draw = tree.frame(f"the shepherd forks {len(branches)} branches", REWIND_COLOR)
        root(image, draw)
        boxes = [lerp(seed, child, t) for child in tree.children]
        for index, box in enumerate(boxes):
            tree.edge(draw, index, box=box)
        for box in boxes:
            tree.board(image, draw, dead, box, REWIND_COLOR)
        frames.append((image, TWEEN_MS))

    # Each child takes pushes back off its board down to its own depth.
    deepest = max(b.rewind for b in branches)
    for undone in range(deepest + 1):
        image, draw = tree.frame(
            f"each branch rewinds to its own depth · {undone} undone", REWIND_COLOR
        )
        root(image, draw)
        for index, (branch, child) in enumerate(zip(branches, tree.children)):
            back = min(undone, branch.rewind)
            tree.edge(draw, index)
            label = f"{branch.name} · rewind {back}"
            color = BAR_TEXT if back == branch.rewind else LABEL
            tree.board(
                image, draw, boards[pushes - back], child, REWIND_COLOR, label=label, color=color
            )
        frames.append((image, pause_ms if undone == deepest else step_ms * 2))

    # Then all of them play at once.
    longest = max(len(b.frames) for b in branches)
    for step in range(1, longest + 1):
        live = sum(1 for b in branches if len(b.frames) > step)
        image, draw = tree.frame(f"every branch at once · {live} still playing", RECOVERY_COLOR)
        root(image, draw)
        for index in range(len(branches)):
            tree.edge(draw, index)
        for branch, child in zip(branches, tree.children):
            if step < len(branch.frames):
                label = f"{branch.name} · rewind {branch.rewind}"
                text = branch.frames[step - 1]["board"]
                tree.board(image, draw, text, child, REWIND_COLOR, label=label)
            else:
                text, label, color = finished(branch)
                tree.board(image, draw, text, child, color, label=label, color=color)
        frames.append((image, step_ms))

    solved = sum(1 for b in branches if b.solved)
    winner = next((b for b in branches if b.name == picked), None)
    caption = f"{solved} of {len(branches)} solved · picked {picked}"
    if winner is not None:
        caption += f", {winner.pushes} pushes"
    image, draw = tree.frame(caption, RECOVERY_COLOR)
    root(image, draw)
    for index, (branch, child) in enumerate(zip(branches, tree.children)):
        if branch.name != picked:
            text, _, _ = finished(branch)
            tree.edge(draw, index)
            label = f"{branch.name} · {'solved' if branch.solved else 'stuck'}"
            tree.board(image, draw, text, child, EDGE, label=label)
    if winner is not None:
        index = branches.index(winner)
        text, _, _ = finished(winner)
        tree.edge(draw, index, RECOVERY_COLOR, width=4)
        label = f"{winner.name} · picked, {winner.pushes}"
        tree.board(
            image, draw, text, tree.children[index], RECOVERY_COLOR, 4, label, RECOVERY_COLOR
        )
    frames.append((image, pause_ms * 5))
    return frames


def shared_palette(images: list) -> list:
    """Every frame on one palette, so the GIF stores each frame as a diff from the last."""
    from PIL import Image

    picks = range(0, len(images), max(1, len(images) // 8))
    sample = Image.new("RGB", (images[0].width, images[0].height * len(picks)))
    for i, index in enumerate(picks):
        sample.paste(images[index], (0, i * images[0].height))
    palette = sample.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    return [image.quantize(palette=palette, dither=Image.Dither.NONE) for image in images]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR, help="Directory shepherd saved.")
    parser.add_argument("--out", type=Path, default=None, help="Default <run-dir>/fanout.gif.")
    parser.add_argument("--tile", type=int, default=16, help="Sprite tile size of a branch board.")
    parser.add_argument("--step-ms", type=int, default=150)
    parser.add_argument("--pause-ms", type=int, default=800)
    parser.add_argument("--tween-frames", type=int, default=20)
    args = parser.parse_args()

    if not sprites.available():
        raise SystemExit("needs Pillow and the sprite tiles: pip install pillow")
    run_dir: Path = args.run_dir
    if not (run_dir / "graph").exists():
        raise SystemExit(
            f"no saved shepherd graph under {run_dir}. Run: python examples/shepherd/shepherd.py"
        )
    boards, branches, picked = load(run_dir)
    if not branches:
        raise SystemExit("no branches in that run")

    frames = build(
        boards,
        branches,
        picked,
        tile=args.tile,
        step_ms=args.step_ms,
        pause_ms=args.pause_ms,
        tween_frames=args.tween_frames,
    )
    out: Path = args.out or (run_dir / "fanout.gif")
    out.parent.mkdir(parents=True, exist_ok=True)
    images = shared_palette([image for image, _ in frames])
    images[0].save(
        out,
        format="GIF",
        save_all=True,
        append_images=images[1:],
        duration=[ms for _, ms in frames],
        loop=0,
        optimize=True,
    )
    print(f"frames: {len(images)} · branches: {len(branches)} · picked: {picked}")
    print(f"gif: {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
