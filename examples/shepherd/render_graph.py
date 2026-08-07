"""Render a finished shepherd run as SVG, at two zoom levels.

``graph.svg`` is the summary: one box per agent, labelled from the graph and
showing the board that agent finished on, because "solved" and "stuck" only mean
something next to the position that earned them.

``nodes.svg`` is the run itself — every saved ``Node``, typed and in order. That
picture is worth drawing because the shape is the claim: a short shepherd trunk,
one node fanning into eight branches, and each branch a chain whose length *is*
how long that recovery took. Nothing is summarised away.

Boards are drawn as vector cells rather than the ``sprites`` tiles, so the figures
stay small, stay crisp at any zoom, and need no Pillow. The colours mirror
``sprites`` so a board here is recognisably the same board as in the GIFs.

Run a shepherd first (``python examples/shepherd/shepherd.py``), then:

    python examples/shepherd/render_graph.py
    python examples/shepherd/render_graph.py --out docs/shepherd_graph.svg
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

from rlmflow import AgentStart, DoneOutput, UserQuery, persistence

EXAMPLES_DIR = Path(__file__).resolve().parents[1]
RUN_DIR = EXAMPLES_DIR / "_runs" / "shepherd"

INK = "#1b2330"
MUTED = "#6b7789"
EDGE = "#b9c2d0"
PAPER = "#ffffff"
SHEPHERD_FILL = "#e8f0fe"
SHEPHERD_LINE = "#4285f4"
SOLVED_FILL = "#f1faf4"
SOLVED_LINE = "#34a853"
FAIL_FILL = "#fdf1f0"
FAIL_LINE = "#d93b30"
WIN_LINE = "#137a3c"

# Board glyphs, coloured to match examples/shepherd/sprites.py.
WALL = "#7c4a3a"
WALL_LINE = "#95604d"
FLOOR = "#151515"
GOAL = "#e2483c"
PLAYER = "#5ec95e"
BOX_COLORS = (
    "#ffbe3c",  # B1 amber
    "#5abeff",  # B2 sky
    "#b982ff",  # B3 violet
    "#78e182",  # B4 green
    "#ff78aa",  # B5 pink
    "#5adccd",  # B6 teal
    "#ff8250",  # B7 ember
    "#cdcdeb",  # B8 pale
    "#d2e65a",  # B9 lime
)
BOX_IDS = "123456789"
BOX_ON_GOAL_IDS = "ABCDEFGHI"
BOARD_CHARS = frozenset("# .$*@+") | frozenset(BOX_IDS) | frozenset(BOX_ON_GOAL_IDS)

# One colour per Node type, for the node-level figure.
NODE_COLORS = {
    "agent_start": "#4285f4",
    "llm_output": "#a855f7",
    "exec_action": "#f5a524",
    "exec_output": "#94a3b8",
    "user_query": "#0ea5a4",
    "append_child": "#ec4899",
    "done_output": "#34a853",
}
NODE_FALLBACK = "#cbd5e1"

# A monospace glyph advance is a fixed share of the font size; used to keep text
# inside the boxes that surround it.
MONO_ADVANCE = 0.62
SANS_ADVANCE = 0.55


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
    board: str = ""

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


def final_boards(run_dir: Path) -> dict[str, str]:
    """The last board each agent stood on, from the traces shepherd exported."""
    boards: dict[str, str] = {}
    traces = run_dir / "traces"
    if not traces.is_dir():
        return boards
    for path in sorted(traces.glob("*.json")):
        try:
            frames = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for frame in reversed(frames if isinstance(frames, list) else []):
            board = frame.get("board", "") if isinstance(frame, dict) else ""
            if board_rows(board):
                boards[path.stem] = board
                break
    return boards


def board_rows(text: str) -> list[str]:
    """Only the grid rows, the way ``sprites`` picks them out of a status block."""
    return [line for line in text.splitlines() if line and set(line) <= BOARD_CHARS]


def read_branches(shepherd: AgentStart, winner: str, boards: dict[str, str]) -> list[Branch]:
    return [
        Branch(
            name=agent.config.name,
            rewind=rewind_depth(agent),
            outcome=str(agent.result() or "unfinished"),
            turns=agent.llm_turns(),
            nodes=sum(1 for _ in agent.walk()),
            plan=str(agent.config.inputs.get("_shepherd_order", "")),
            won=agent.config.name == winner,
            board=boards.get(agent.config.name, ""),
        )
        for agent in shepherd.sub_agents
    ]


class Canvas:
    """The few SVG primitives this picture needs."""

    def __init__(self, width: float, height: float) -> None:
        self.width = width
        self.height = height
        self.parts: list[str] = []

    def box(self, x: float, y: float, w: float, h: float, fill: str, line: str, lw: float = 1.4):
        self.parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="9" '
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
        if not body:
            return
        family = (
            "ui-monospace, SFMono-Regular, Menlo, monospace"
            if mono
            else "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
        )
        self.parts.append(
            f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-family="{family}" '
            f'font-size="{size:.1f}" font-weight="{weight}" fill="{fill}">{escape(body)}</text>'
        )

    def fitted_text(self, x: float, y: float, body: str, *, max_w: float, size: float, **kw):
        """Draw ``body`` centred at ``x``, shrunk if it would spill past ``max_w``."""
        advance = MONO_ADVANCE if kw.get("mono") else SANS_ADVANCE
        if body:
            size = min(size, max_w / (len(body) * advance))
        self.text(x, y, body, size=size, **kw)

    def rect(self, x: float, y: float, w: float, h: float, fill: str, **kw):
        extra = "".join(f' {k.replace("_", "-")}="{v}"' for k, v in kw.items())
        self.parts.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" fill="{fill}"{extra}/>'
        )

    def circle(self, cx: float, cy: float, r: float, fill: str):
        self.parts.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" fill="{fill}"/>')

    def diamond(self, cx: float, cy: float, r: float, fill: str):
        self.parts.append(
            f'<path d="M {cx:.2f} {cy - r:.2f} L {cx + r:.2f} {cy:.2f} '
            f'L {cx:.2f} {cy + r:.2f} L {cx - r:.2f} {cy:.2f} Z" fill="{fill}"/>'
        )

    def line(self, x0: float, y0: float, x1: float, y1: float):
        self.parts.append(
            f'<path d="M {x0:.1f} {y0:.1f} L {x1:.1f} {y1:.1f}" fill="none" '
            f'stroke="{EDGE}" stroke-width="1.4"/>'
        )

    def elbow(self, x0: float, y0: float, x1: float, y1: float):
        """A parent-to-child connector that drops, tracks sideways, then drops again."""
        mid = (y0 + y1) / 2
        self.parts.append(
            f'<path d="M {x0:.1f} {y0:.1f} V {mid:.1f} H {x1:.1f} V {y1:.1f}" fill="none" '
            f'stroke="{EDGE}" stroke-width="1.4"/>'
        )

    def arrow(self, x: float, y0: float, y1: float, *, label: str = "", dashed: bool = False):
        dash = ' stroke-dasharray="4 3"' if dashed else ""
        self.parts.append(
            f'<path d="M {x:.1f} {y0:.1f} V {y1:.1f}" fill="none" stroke="{EDGE}" '
            f'stroke-width="1.6" marker-end="url(#tip)"{dash}/>'
        )
        if label:
            self.text(x + 9, (y0 + y1) / 2 + 4, label, size=11, fill=MUTED, anchor="start")

    def board(self, x: float, y: float, rows: list[str], cell: float) -> tuple[float, float]:
        """Draw a Sokoban board as one rect per cell. Returns its (width, height)."""
        cols = max((len(row) for row in rows), default=0)
        self.rect(x, y, cols * cell, len(rows) * cell, FLOOR, rx=2)
        for r, row in enumerate(rows):
            for c in range(cols):
                glyph = row[c] if c < len(row) else " "
                cx, cy = x + c * cell, y + r * cell
                self._cell(cx, cy, cell, glyph)
        return cols * cell, len(rows) * cell

    def _cell(self, x: float, y: float, cell: float, glyph: str) -> None:
        mid_x, mid_y = x + cell / 2, y + cell / 2
        if glyph == "#":
            self.rect(x, y, cell, cell, WALL)
            return
        if glyph in ".+":
            self.diamond(mid_x, mid_y, cell * 0.3, GOAL)
        if glyph in "@+":
            self.circle(mid_x, mid_y, cell * 0.34, PLAYER)
            return
        index = BOX_IDS.find(glyph)
        locked = False
        if index < 0:
            index = BOX_ON_GOAL_IDS.find(glyph)
            locked = index >= 0
        if index < 0 and glyph in "$*":
            index, locked = 0, glyph == "*"
        if index < 0:
            return
        # A box on a goal is locked for good and keeps the white outline sprites.py
        # redraws it with; a box still loose gets a red one. Solved boards then read
        # as all-white and a stranded box stays visible at thumbnail size, which is
        # the whole difference between "solved" and "stuck".
        fill = BOX_COLORS[index % len(BOX_COLORS)]
        inset = cell * (0.12 if locked else 0.2)
        self.rect(
            x + inset,
            y + inset,
            cell - inset * 2,
            cell - inset * 2,
            fill,
            rx=1,
            stroke="#ffffff" if locked else FAIL_LINE,
            stroke_width=max(1.0, cell * 0.14),
        )

    def legend(self, x: float, y: float, items: list[tuple[str, str]], cell: float = 21) -> None:
        """Swatch-and-label pairs, so the boards need no caption to be readable."""
        for glyph, label in items:
            self.rect(x, y, cell, cell, FLOOR, rx=2)
            self._cell(x, y, cell, glyph)
            self.text(x + cell + 7, y + cell - 3, label, size=10.5, fill=MUTED, anchor="start")
            x += cell + 14 + len(label) * 10.5 * SANS_ADVANCE

    def render(self) -> str:
        return "\n".join(
            [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width:.0f}" '
                f'height="{self.height:.0f}" viewBox="0 0 {self.width:.0f} {self.height:.0f}">',
                "<defs>",
                '<marker id="tip" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
                'markerHeight="7" orient="auto-start-reverse">',
                f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{EDGE}"/>',
                "</marker>",
                "</defs>",
                f'<rect width="{self.width:.0f}" height="{self.height:.0f}" fill="{PAPER}"/>',
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
    worker_board: str,
    shepherd_turns: int,
    max_rewind: int | None,
    total_nodes: int,
) -> str:
    pad, margin, gap = 12.0, 28.0, 14.0
    branch_cell = 10.0
    branch_grid = max(
        (len(row) for branch in branches for row in board_rows(branch.board)),
        default=12,
    )
    box_w = max(132.0, branch_grid * branch_cell + pad * 2)
    lane = len(branches) * box_w + (len(branches) - 1) * gap
    width = lane + margin * 2
    centre = width / 2

    # Fixed offsets inside a branch box, so the board can never crowd the labels.
    board_top = 64.0
    branch_rows = max((len(board_rows(b.board)) for b in branches), default=0)
    board_h = branch_rows * branch_cell
    y_outcome = board_top + board_h + (16 if board_h else 0)
    y_turns = y_outcome + 16
    branch_h = y_turns + 10

    jam_cell = 14.0
    jam_rows = board_rows(worker_board)
    jam_board_w = max((len(r) for r in jam_rows), default=0) * jam_cell
    jam_board_h = len(jam_rows) * jam_cell
    jam_w = max(300.0, jam_board_w + pad * 2)
    jam_board_top = 52.0
    jam_h = jam_board_top + jam_board_h + (14 if jam_board_h else 0)

    y_jam = 52.0
    y_shepherd = y_jam + jam_h + 46
    shepherd_h = 62.0
    y_branch = y_shepherd + shepherd_h + 52
    y_picked = y_branch + branch_h + 52
    picked_h = 46.0
    height = y_picked + picked_h + margin

    c = Canvas(width, height)
    c.text(
        margin,
        32,
        "shepherd — one jam, eight rewinds, one winner",
        size=16,
        weight="600",
        anchor="start",
    )
    c.text(
        width - margin,
        32,
        f"{len(branches)} branches · {total_nodes} nodes in one tree",
        size=12,
        fill=MUTED,
        anchor="end",
    )

    # The jam.
    jam_note = f"jammed after {worker_pushes} pushes" if worker_pushes is not None else "jammed"
    c.box(centre - jam_w / 2, y_jam, jam_w, jam_h, FAIL_FILL, FAIL_LINE, lw=2.0)
    c.text(centre, y_jam + 23, "worker", size=13, weight="600")
    c.fitted_text(
        centre,
        y_jam + 41,
        f"{jam_note} · {worker_turns} turns",
        max_w=jam_w - pad * 2,
        size=11.5,
        fill=MUTED,
    )
    if jam_rows:
        c.board(centre - jam_board_w / 2, y_jam + jam_board_top, jam_rows, jam_cell)
    c.arrow(centre, y_jam + jam_h + 4, y_shepherd - 6, label="stuck transcript", dashed=True)

    # The shepherd.
    shepherd_w = 430.0
    previews = f"previewed {max_rewind} rewind depths, " if max_rewind else "previewed the board, "
    c.box(centre - shepherd_w / 2, y_shepherd, shepherd_w, shepherd_h, SHEPHERD_FILL, SHEPHERD_LINE)
    c.text(centre, y_shepherd + 25, "shepherd", size=13, weight="600")
    c.fitted_text(
        centre,
        y_shepherd + 44,
        f"{previews}wrote {len(branches)} plans · {shepherd_turns} turns",
        max_w=shepherd_w - pad * 2,
        size=11.5,
        fill=MUTED,
    )

    # The branches.
    for index, branch in enumerate(branches):
        x = margin + index * (box_w + gap)
        mid = x + box_w / 2
        fill = SOLVED_FILL if branch.solved else FAIL_FILL
        line = WIN_LINE if branch.won else (SOLVED_LINE if branch.solved else FAIL_LINE)
        c.elbow(centre, y_shepherd + shepherd_h, mid, y_branch)
        c.box(x, y_branch, box_w, branch_h, fill, line, lw=2.8 if branch.won else 1.6)
        if branch.won:
            c.text(mid, y_branch - 9, "picked", size=11, weight="600", fill=WIN_LINE)
        c.text(mid, y_branch + 22, branch.name, size=12.5, weight="600")
        rewind = f"rewind {branch.rewind}" if branch.rewind is not None else "rewind ?"
        c.fitted_text(
            mid, y_branch + 40, rewind, max_w=box_w - pad * 2, size=11.5, fill=MUTED, mono=True
        )
        c.fitted_text(
            mid,
            y_branch + 55,
            branch.inherits(worker_pushes),
            max_w=box_w - pad * 2,
            size=10.5,
            fill=MUTED,
        )
        rows = board_rows(branch.board)
        if rows:
            board_w = max(len(r) for r in rows) * branch_cell
            c.board(mid - board_w / 2, y_branch + board_top, rows, branch_cell)
        c.text(
            mid,
            y_branch + y_outcome,
            branch.outcome,
            size=12,
            weight="600",
            fill=WIN_LINE if branch.solved else FAIL_LINE,
        )
        c.fitted_text(
            mid,
            y_branch + y_turns,
            f"{branch.turns} turns",
            max_w=box_w - pad * 2,
            size=10.5,
            fill=MUTED,
        )

    # The pick.
    winner = next((b for b in branches if b.won), None)
    if winner is not None:
        x = margin + branches.index(winner) * (box_w + gap) + box_w / 2
        c.elbow(x, y_branch + branch_h, centre, y_picked - 20)
        c.arrow(centre, y_picked - 20, y_picked - 5)
    label = summary or "no branch picked"
    picked_w = min(width - margin * 2, len(label) * 13 * MONO_ADVANCE + pad * 4)
    c.box(centre - picked_w / 2, y_picked, picked_w, picked_h, PAPER, WIN_LINE, lw=2.0)
    c.fitted_text(
        centre, y_picked + 29, label, max_w=picked_w - pad * 2, size=13, weight="600", mono=True
    )
    c.legend(
        margin,
        y_picked + 15,
        [("A", "box on a goal, locked"), ("1", "box still loose"), (".", "empty goal")],
    )
    return c.render()


def agent_chain(agent: AgentStart) -> list:
    """The nodes this agent appended, in order, without descending into children."""
    return [node for node in agent.walk() if node is agent or node.parent_agent is agent]


def rewind_marker(chain: list) -> int:
    """Index of the note ``prepare_branch`` left, which is where inherited history ends."""
    for index, node in enumerate(chain):
        if isinstance(node, UserQuery) and re.search(r"Rewound \d+ pushes", node.content or ""):
            return index
    return 0


def build_nodes_svg(shepherd: AgentStart, branches: list[Branch]) -> str:
    """Every saved Node, one square each: the trunk, the fan, and eight chains."""
    margin, pitch, dot = 28.0, 9.0, 7.0
    label_w, row_h, note_size = 74.0, 26.0, 11.0

    trunk = agent_chain(shepherd)
    tracks = [(branch, agent_chain(agent)) for branch, agent in zip(branches, shepherd.sub_agents)]
    longest = max([len(trunk)] + [len(chain) for _, chain in tracks])

    trunk_note = f"{len(trunk)} nodes · plans the rewinds"
    notes = {
        branch.name: f"{branch.outcome} · {branch.turns} turns"
        + ("  ← picked" if branch.won else "")
        for branch, _ in tracks
    }
    note_w = max(len(n) for n in [trunk_note, *notes.values()]) * note_size * SANS_ADVANCE + 8

    lane_x = margin + label_w
    width = lane_x + longest * pitch + 14 + note_w + margin
    y_trunk = 62.0
    y_first = y_trunk + row_h + 22
    y_legend = y_first + len(tracks) * row_h + 26
    height = y_legend + 30

    c = Canvas(width, height)
    c.text(margin, 30, "the same run, every node", size=16, weight="600", anchor="start")
    c.text(
        width - margin,
        30,
        f"{sum(len(chain) for _, chain in tracks) + len(trunk)} nodes · "
        f"{len(tracks) + 1} agents · one tree",
        size=12,
        fill=MUTED,
        anchor="end",
    )

    def draw_chain(
        y: float, chain: list, label: str, note: str, note_fill: str, inherited: int = 0
    ) -> None:
        c.text(margin, y + dot, label, size=11.5, weight="600", anchor="start")
        c.rect(lane_x, y + dot / 2 - 0.5, max(len(chain) - 1, 0) * pitch + dot, 1.0, EDGE)
        for index, node in enumerate(chain):
            fill = NODE_COLORS.get(node.type, NODE_FALLBACK)
            # Nodes before the rewind marker were copied from the jam by the fork,
            # so fading them shows how much history each branch chose to keep.
            opacity = {"fill_opacity": 0.3} if index < inherited else {}
            c.rect(lane_x + index * pitch, y, dot, dot, fill, rx=1.5, **opacity)
        c.text(
            lane_x + longest * pitch + 14,
            y + dot,
            note,
            size=note_size,
            fill=note_fill,
            anchor="start",
        )

    draw_chain(y_trunk, trunk, "shepherd", trunk_note, MUTED)

    # The one node with more than one child is where the branches hang.
    fan = next((i for i, node in enumerate(trunk) if len(node.children) > 1), len(trunk) - 1)
    fan_x = lane_x + fan * pitch + dot / 2
    bus_x, bus_top = lane_x - 12, y_first - 9
    y_last = y_first + (len(tracks) - 1) * row_h + dot / 2
    c.line(fan_x, y_trunk + dot, fan_x, bus_top)
    c.line(fan_x, bus_top, bus_x, bus_top)
    c.line(bus_x, bus_top, bus_x, y_last)
    for row, (branch, chain) in enumerate(tracks):
        y = y_first + row * row_h
        c.line(bus_x, y + dot / 2, lane_x - 1, y + dot / 2)
        draw_chain(
            y,
            chain,
            branch.name,
            notes[branch.name],
            WIN_LINE if branch.solved else FAIL_LINE,
            inherited=rewind_marker(chain),
        )

    x = margin
    for node_type, color in NODE_COLORS.items():
        c.rect(x, y_legend, dot, dot, color, rx=1.5)
        c.text(x + dot + 6, y_legend + dot, node_type, size=10.5, fill=MUTED, anchor="start")
        x += dot + 12 + len(node_type) * 10.5 * SANS_ADVANCE
    c.rect(x, y_legend, dot, dot, NODE_FALLBACK, rx=1.5, fill_opacity=0.5)
    c.text(
        x + dot + 6,
        y_legend + dot,
        "faded: inherited from the jam",
        size=10.5,
        fill=MUTED,
        anchor="start",
    )
    return c.render()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR, help="Directory shepherd saved.")
    parser.add_argument(
        "--out", type=Path, default=None, help="Agent summary SVG (default <run-dir>/graph.svg)."
    )
    parser.add_argument(
        "--nodes-out", type=Path, default=None, help="Node-level SVG (default <run-dir>/nodes.svg)."
    )
    args = parser.parse_args()

    run_dir: Path = args.run_dir
    if not (run_dir / "shepherd").exists():
        raise SystemExit(
            f"no saved shepherd graph under {run_dir}. Run: python examples/shepherd/shepherd.py"
        )

    shepherd = persistence.load(run_dir / "shepherd")
    summary = picked_summary(shepherd)
    found = re.search(r"Picked (\S+?):", summary)
    boards = final_boards(run_dir)
    branches = read_branches(shepherd, found.group(1) if found else "", boards)
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
        worker_board=boards.get("worker", ""),
        shepherd_turns=shepherd.llm_turns(),
        max_rewind=max_rewind,
        total_nodes=sum(1 for _ in shepherd.walk()),
    )
    out: Path = args.out or (run_dir / "graph.svg")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(svg, encoding="utf-8")

    nodes_out: Path = args.nodes_out or (run_dir / "nodes.svg")
    nodes_out.parent.mkdir(parents=True, exist_ok=True)
    nodes_out.write_text(build_nodes_svg(shepherd, branches), encoding="utf-8")

    print(summary or "no branch picked")
    for branch in branches:
        mark = " <- picked" if branch.won else ""
        print(
            f"  {branch.name}: rewind={branch.rewind} {branch.outcome} "
            f"turns={branch.turns} nodes={branch.nodes}{mark}"
        )
        if branch.plan:
            print(f"      plan: {branch.plan}")
    missing = [b.name for b in branches if not board_rows(b.board)]
    if not boards.get("worker"):
        missing.append("worker")
    if missing:
        print(f"\nno final board in traces for: {', '.join(missing)}")
    print(f"\nsvg: {out}\nsvg: {nodes_out}")


if __name__ == "__main__":
    main()
