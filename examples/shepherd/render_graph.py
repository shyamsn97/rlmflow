"""Render a finished shepherd run as an agent-level SVG.

The saved tree has hundreds of nodes, which is the wrong picture for explaining
the run: what matters is the shape of the recovery — one jammed worker, one
shepherd that rewound it, and the branches that raced from those rewind points.
So this draws one box per agent and reads the labels back off the graph.

Run a shepherd first (``python examples/shepherd/shepherd.py``), then:

    python examples/shepherd/render_graph.py
    python examples/shepherd/render_graph.py --out docs/shepherd_graph.svg
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

from rlmflow import AgentStart, DoneOutput, UserQuery, persistence

EXAMPLES_DIR = Path(__file__).resolve().parents[1]
RUN_DIR = EXAMPLES_DIR / "_runs" / "shepherd"

INK = "#1b2330"
MUTED = "#6b7789"
EDGE = "#b3bdcc"
PAPER = "#ffffff"
SHEPHERD_FILL = "#dbeafe"
SHEPHERD_LINE = "#3b82f6"
JAM_FILL = "#fde7e7"
JAM_LINE = "#e06d6d"
SOLVED_FILL = "#dcf5e3"
SOLVED_LINE = "#4caf7d"
STUCK_FILL = "#eceff3"
STUCK_LINE = "#aab4c2"
WIN_LINE = "#1f9254"


@dataclass
class Branch:
    """One recovery attempt, as read back off the saved graph."""

    name: str
    rewind: int | None
    outcome: str
    turns: int
    nodes: int
    plan: str
    won: bool

    @property
    def solved(self) -> bool:
        return self.outcome == "solved"

    def inherits(self, jam_pushes: int | None) -> str:
        """How much of the jam this branch started from — the whole point of a rewind."""
        if jam_pushes is None or self.rewind is None:
            return ""
        kept = max(0, jam_pushes - self.rewind)
        return "fresh board" if kept == 0 else f"keeps {kept} push{'es' if kept > 1 else ''}"


def rewind_depth(agent: AgentStart) -> int | None:
    """How far this branch rewound, per the note ``prepare_branch`` left it."""
    for node in agent.walk():
        if isinstance(node, UserQuery):
            found = re.search(r"Rewound (\d+) pushes", node.content or "")
            if found:
                return int(found.group(1))
    return None


def picked_summary(shepherd: AgentStart) -> str:
    for node in shepherd.walk():
        if isinstance(node, DoneOutput) and node.parent_agent is shepherd:
            return str(node.result or node.content or "")
    return ""


def read_branches(shepherd: AgentStart, winner: str) -> list[Branch]:
    return [
        Branch(
            name=agent.config.name,
            rewind=rewind_depth(agent),
            outcome=str(agent.result() or "unfinished"),
            turns=agent.llm_turns(),
            nodes=sum(1 for _ in agent.walk()),
            plan=str(agent.config.inputs.get("_shepherd_order", "")),
            won=agent.config.name == winner,
        )
        for agent in shepherd.sub_agents
    ]


class Canvas:
    """The few SVG primitives this picture needs."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.parts: list[str] = []

    def box(self, x: float, y: float, w: float, h: float, fill: str, line: str, lw: float = 1.4):
        self.parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="8" '
            f'fill="{fill}" stroke="{line}" stroke-width="{lw}"/>'
        )

    def text(
        self,
        x: float,
        y: float,
        body: str,
        *,
        size: float = 13,
        fill: str = INK,
        weight: str = "normal",
        anchor: str = "middle",
        mono: bool = False,
    ):
        family = (
            "ui-monospace, SFMono-Regular, Menlo, monospace"
            if mono
            else "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
        )
        self.parts.append(
            f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-family="{family}" '
            f'font-size="{size}" font-weight="{weight}" fill="{fill}">{escape(body)}</text>'
        )

    def elbow(self, x0: float, y0: float, x1: float, y1: float, *, dashed: bool = False):
        """A parent-to-child connector that drops, tracks sideways, then drops again."""
        mid = (y0 + y1) / 2
        dash = ' stroke-dasharray="4 3"' if dashed else ""
        self.parts.append(
            f'<path d="M {x0:.1f} {y0:.1f} V {mid:.1f} H {x1:.1f} V {y1:.1f}" fill="none" '
            f'stroke="{EDGE}" stroke-width="1.4"{dash}/>'
        )

    def arrow(self, x: float, y: float, *, label: str = "", dashed: bool = False, to: float = 0):
        dash = ' stroke-dasharray="4 3"' if dashed else ""
        self.parts.append(
            f'<path d="M {x:.1f} {y:.1f} V {to:.1f}" fill="none" stroke="{EDGE}" '
            f'stroke-width="1.6" marker-end="url(#tip)"{dash}/>'
        )
        if label:
            self.text(x + 8, (y + to) / 2 + 4, label, size=11, fill=MUTED, anchor="start")

    def render(self) -> str:
        return "\n".join(
            [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" '
                f'height="{self.height}" viewBox="0 0 {self.width} {self.height}">',
                "<defs>",
                '<marker id="tip" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
                'markerHeight="7" orient="auto-start-reverse">',
                f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{EDGE}"/>',
                "</marker>",
                "</defs>",
                f'<rect width="{self.width}" height="{self.height}" fill="{PAPER}"/>',
                *self.parts,
                "</svg>",
            ]
        )


def build_svg(
    *,
    branches: list[Branch],
    summary: str,
    worker_turns: int,
    worker_pushes: int | None,
    shepherd_turns: int,
    max_rewind: int | None,
    total_nodes: int,
) -> str:
    box_w, gap = 132.0, 14.0
    lane = len(branches) * box_w + (len(branches) - 1) * gap
    margin = 28.0
    width = int(lane + margin * 2)

    y_jam, jam_h = 64.0, 62.0
    y_shepherd = y_jam + jam_h + 52
    shepherd_h = 62.0
    y_branch = y_shepherd + shepherd_h + 60
    branch_h = 118.0
    y_picked = y_branch + branch_h + 56
    picked_h = 46.0
    height = int(y_picked + picked_h + margin)

    c = Canvas(width, height)
    centre = width / 2

    c.text(
        margin,
        30,
        "shepherd — one jam, eight rewinds, one winner",
        size=16,
        weight="600",
        anchor="start",
    )
    c.text(
        width - margin,
        30,
        f"{len(branches)} branches · {total_nodes} nodes in one tree",
        size=12,
        fill=MUTED,
        anchor="end",
    )

    jam_w = 330.0
    jam_note = f"jammed after {worker_pushes} pushes" if worker_pushes is not None else "jammed"
    c.box(centre - jam_w / 2, y_jam, jam_w, jam_h, JAM_FILL, JAM_LINE)
    c.text(centre, y_jam + 24, "worker", size=13, weight="600")
    c.text(centre, y_jam + 44, f"{jam_note} · {worker_turns} turns", size=11.5, fill=MUTED)

    c.arrow(centre, y_jam + jam_h + 4, to=y_shepherd - 6, label="stuck transcript", dashed=True)

    shepherd_w = 430.0
    previews = f"previewed {max_rewind} rewind depths, " if max_rewind else "previewed the board, "
    c.box(centre - shepherd_w / 2, y_shepherd, shepherd_w, shepherd_h, SHEPHERD_FILL, SHEPHERD_LINE)
    c.text(centre, y_shepherd + 24, "shepherd", size=13, weight="600")
    c.text(
        centre,
        y_shepherd + 44,
        f"{previews}wrote {len(branches)} plans · {shepherd_turns} turns",
        size=11.5,
        fill=MUTED,
    )

    left = margin
    for index, branch in enumerate(branches):
        x = left + index * (box_w + gap)
        fill = SOLVED_FILL if branch.solved else STUCK_FILL
        line = WIN_LINE if branch.won else (SOLVED_LINE if branch.solved else STUCK_LINE)
        c.elbow(centre, y_shepherd + shepherd_h, x + box_w / 2, y_branch)
        c.box(x, y_branch, box_w, branch_h, fill, line, lw=2.6 if branch.won else 1.4)
        c.text(x + box_w / 2, y_branch + 24, branch.name, size=12.5, weight="600")
        rewind = f"rewind {branch.rewind}" if branch.rewind is not None else "rewind ?"
        c.text(x + box_w / 2, y_branch + 45, rewind, size=11.5, fill=MUTED, mono=True)
        c.text(x + box_w / 2, y_branch + 62, branch.inherits(worker_pushes), size=10.5, fill=MUTED)
        c.text(
            x + box_w / 2,
            y_branch + 88,
            branch.outcome,
            size=12,
            weight="600",
            fill=WIN_LINE if branch.solved else MUTED,
        )
        c.text(x + box_w / 2, y_branch + 106, f"{branch.turns} turns", size=11, fill=MUTED)
        if branch.won:
            c.text(x + box_w / 2, y_branch - 10, "picked", size=11, weight="600", fill=WIN_LINE)

    winner = next((b for b in branches if b.won), None)
    if winner is not None:
        x = left + branches.index(winner) * (box_w + gap) + box_w / 2
        c.elbow(x, y_branch + branch_h, centre, y_picked - 7)
        c.arrow(centre, y_picked - 22, to=y_picked - 5)
    picked_w = min(width - margin * 2, max(360.0, len(summary) * 7.4))
    c.box(centre - picked_w / 2, y_picked, picked_w, picked_h, PAPER, WIN_LINE, lw=2.0)
    c.text(centre, y_picked + 29, summary or "no branch picked", size=13, weight="600", mono=True)

    return c.render()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR, help="Directory shepherd saved.")
    parser.add_argument(
        "--out", type=Path, default=None, help="SVG path (default <run-dir>/graph.svg)."
    )
    args = parser.parse_args()

    run_dir: Path = args.run_dir
    if not (run_dir / "shepherd").exists():
        raise SystemExit(
            f"no saved shepherd graph under {run_dir}. Run: python examples/shepherd/shepherd.py"
        )

    shepherd = persistence.load(run_dir / "shepherd")
    summary = picked_summary(shepherd)
    winner = ""
    found = re.search(r"Picked (\S+?):", summary)
    if found:
        winner = found.group(1)
    branches = read_branches(shepherd, winner)
    if not branches:
        raise SystemExit("that run has no recovery branches to draw")

    worker_turns, worker_pushes = shepherd.llm_turns(), None
    worker_dir = run_dir / "worker"
    if worker_dir.exists():
        worker = persistence.load(worker_dir)
        worker_turns = worker.llm_turns()
        # The jam is one push per turn, minus the turn that built the board.
        worker_pushes = max(0, worker_turns - 1)

    max_rewind = None
    raw_rewind = shepherd.config.inputs.get("max_rewind")
    if raw_rewind is not None and str(raw_rewind).strip().isdigit():
        max_rewind = int(str(raw_rewind).strip())

    svg = build_svg(
        branches=branches,
        summary=summary,
        worker_turns=worker_turns,
        worker_pushes=worker_pushes,
        shepherd_turns=shepherd.llm_turns(),
        max_rewind=max_rewind,
        total_nodes=sum(1 for _ in shepherd.walk()),
    )
    out: Path = args.out or (run_dir / "graph.svg")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(svg, encoding="utf-8")

    print(summary or "no branch picked")
    for branch in branches:
        mark = " <- picked" if branch.won else ""
        print(
            f"  {branch.name}: rewind={branch.rewind} {branch.outcome} "
            f"turns={branch.turns} nodes={branch.nodes}{mark}"
        )
        if branch.plan:
            print(f"      plan: {branch.plan}")
    print(f"\nsvg: {out}")


if __name__ == "__main__":
    main()
