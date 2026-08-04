"""Board-panel UI helpers for the shepherd example.

:class:`PanelViewer` aggregates per-lane boards (terminal and/or Gradio via
``sink``). The shepherd host hands it every node its streams publish, alongside
the one-line-per-node trace it prints itself.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

from rlmflow import (
    AgentStart,
    DoneOutput,
    ExecAction,
    ExecOutput,
    Flow,
    LLMOutput,
    Node,
)
from rlmflow.consumers import StreamConsumer


def _looks_like_grid(text: str) -> bool:
    # A board observation contains at least one wall row (a line of glyphs
    # starting with '#'). The worker's status now prefixes the grid with a
    # "Current Grid:" header, so scan all lines rather than just the first.
    return any(line.strip().startswith("#") for line in text.splitlines())


def side_by_side(labeled: list[tuple[str, str]], *, gap: str = "    ") -> str:
    """Lay several headed text blocks next to each other in one row. ``labeled``
    is a list of ``(heading, block)`` pairs; blocks may be multi-line."""
    if not labeled:
        return ""
    cols = [[head, *block.splitlines()] for head, block in labeled]
    height = max(len(col) for col in cols)
    widths = [max((len(line) for line in col), default=0) for col in cols]
    rows = []
    for r in range(height):
        cells = [(col[r] if r < len(col) else "").ljust(w) for col, w in zip(cols, widths)]
        rows.append(gap.join(cells).rstrip())
    return "\n".join(rows)


def grid_of_blocks(labeled: list[tuple[str, str]], *, cols: int = 4) -> str:
    """``side_by_side`` wrapped into rows of at most ``cols`` blocks, so a large
    fan-out (e.g. 8 branches) renders as a grid instead of one very wide line."""
    cols = max(1, cols)
    rows = [side_by_side(labeled[i : i + cols]) for i in range(0, len(labeled), cols)]
    return "\n\n".join(row for row in rows if row)


def panel_status(flow: Flow, agent: AgentStart) -> str:
    """Live one-line status for a board panel, read from the agent's REPL env."""
    env = flow.runtime.repl_for(agent).env
    pushes = env.get("pushes")
    step = f"push {pushes}" if pushes is not None else "push ?"
    if env.get("solved"):
        return f"{step} · SOLVED"
    if env.get("blocked"):
        # A rejected push, not a terminal state — don't say DONE/BLOCKED.
        return f"{step} · illegal push (try another)"
    dist = env.get("dist")
    return f"{step} · dist {dist}" if pushes is not None else ""


def trace_line(node: Node, labels: dict[str, str]) -> str:
    """One compact line for the streamed node."""
    if isinstance(node, ExecAction):
        return ""
    agent = node.parent_agent
    lane = labels.get(agent.id, agent.config.name)
    if isinstance(node, LLMOutput):
        detail = node.code
    elif isinstance(node, DoneOutput):
        detail = f"done({node.result!r})"
    else:
        detail = node.content
    head = next((line for line in (detail or "").splitlines() if line.strip()), "")
    return f"[{lane:>9}] {node.type:<13} {head.strip()[:88]}"


def export_run_traces(root: Path, named_games: list[tuple[str, object]]) -> None:
    """Write frame JSON and optional GIFs for each game."""
    import sprites

    out = root / "traces"
    out.mkdir(parents=True, exist_ok=True)
    have_tiles = sprites.available()
    for name, game in named_games:
        steps = list(getattr(game, "step_frames", []))
        trace = [{"label": label, "board": board} for label, board in steps]
        (out / f"{name}.json").write_text(json.dumps(trace, indent=2))
        renders = [board for _label, board in steps]
        if have_tiles:
            sprites.save_gif(renders, out / f"{name}.gif")


class PanelViewer(StreamConsumer):
    """Live side-by-side boards, one panel per **agent**, updated push-by-push.

    One panel per agent, keyed by the agent's id in first-seen order. On each node
    the viewer walks that node's whole tree, so a root and any children it
    launched each get their own panel. Each panel shows that agent's board (via
    ``board_of``) or its latest ``ExecOutput`` grid, so the worker and every
    recovery branch animate as their streams advance. Safe to share across
    concurrently-gathered streams: node handling runs on one asyncio loop, so
    updates never interleave.
    """

    def __init__(
        self,
        *,
        title: str = "",
        cols: int = 4,
        every_s: float = 0.1,
        status_of: Callable[[AgentStart], str] | None = None,
        board_of: Callable[[AgentStart], str] | None = None,
        frames_of: Callable[[AgentStart], list[str]] | None = None,
        frame_ms: int = 70,
        sink: Callable[[list[tuple[str, str]]], None] | None = None,
        paint: bool = True,
    ) -> None:
        self.title = title
        self.cols = cols
        self.every_s = every_s
        self.status_of = status_of
        # When set, refresh each panel's board from the live game (or other
        # source) instead of scraping ExecOutput text for a grid.
        self.board_of = board_of
        # When set, this reads a lane's current-turn sub-step frames (e.g.
        # ``env["frames"]`` published by the game) and plays them out one-by-one, so
        # a box-level ``push`` visibly walks the player instead of snapping to the
        # end state (which looks teleporty). Frames are per-turn (replaced each
        # turn), so a plain "changed since last drawn?" check is all it takes.
        self.frames_of = frames_of
        self.frame_ms = frame_ms
        # Last frame-list drawn per lane, so a turn animates once (and a fork's
        # inherited frames are skipped on first sight — see ``_observe``).
        self._last_frames: dict[str, list[str]] = {}
        # When set, panels are routed here (e.g. a Gradio dashboard) in addition
        # to (or instead of, when paint=False) the terminal.
        self.sink = sink
        # When False, still aggregate + sink, but do not clear/print the TTY —
        # use this when something else (a node trace, say) owns the terminal.
        self.paint = paint
        # Panels are keyed per agent by the agent's own id, since every graph the
        # host drives is a separate root whose config name is just "root".
        # ``_root_labels`` holds host-supplied names for a graph's root agent
        # (children default to their config name).
        self._order: list[str] = []
        self._labels: dict[str, str] = {}
        self._blocks: dict[str, str] = {}
        self._agents: dict[str, AgentStart] = {}
        self._root_labels: dict[str, str] = {}
        self._last_paint = 0.0
        self._closed = False

    def label(self, agent: AgentStart, label: str) -> None:
        """Name a graph's root-agent panel (e.g. "worker"/"shepherd")."""
        self._root_labels[agent.id] = label
        if agent.id in self._labels:
            self._labels[agent.id] = label

    def _default_label(self, agent: AgentStart) -> str:
        return self._root_labels.get(agent.id) or agent.config.name

    def handle(self, node: Node) -> None:
        if self._closed:
            return
        # Fan out to every agent in this root's tree, so children get their own
        # lane rather than folding into the root's.
        agents = [n for n in node.root.walk() if isinstance(n, AgentStart)]
        for agent in agents:
            self._observe(agent, node if node.parent_agent is agent else None)
        # Play out this turn's per-sub-step frames so a box-level push visibly walks
        # the player around cell-by-cell instead of snapping (looking teleporty).
        if self.frames_of is not None:
            for agent in agents:
                self._animate(agent)
        now = time.monotonic()
        if self.every_s and now - self._last_paint < self.every_s:
            return
        self._paint()

    def _safe_frames(self, agent: AgentStart) -> list[str]:
        try:
            return list(self.frames_of(agent) or [])  # type: ignore[misc]
        except Exception:  # noqa: BLE001 - a viewer must never crash a run
            return []

    def _animate(self, agent: AgentStart) -> None:
        """Paint this turn's sub-step frames for ``agent`` in sequence (short delay
        each) so the walk-around before each shove is visible. ``frames`` is the
        current turn only (replaced each turn), so we play it once — when it differs
        from what we last drew for this lane."""
        key = agent.id
        frames = self._safe_frames(agent)
        if not frames or frames == self._last_frames.get(key):
            return
        self._last_frames[key] = frames
        delay = self.frame_ms / 1000.0
        for i, render in enumerate(frames):
            self._blocks[key] = render
            self._paint()
            if delay and i < len(frames) - 1:
                time.sleep(delay)

    def _observe(self, agent: AgentStart, changed: Node | None) -> None:
        key = agent.id
        if key not in self._order:
            self._order.append(key)
            self._labels.setdefault(key, self._default_label(agent))
            # Seed with the lane's current frames so a fork's inherited (replayed)
            # turn isn't re-animated on first sight — only turns from here play.
            if self.frames_of is not None:
                self._last_frames[key] = self._safe_frames(agent)
        self._agents[key] = agent
        if self.board_of is not None:
            try:
                board = self.board_of(agent)
            except Exception:  # noqa: BLE001 - a viewer must never crash a run
                board = None
            if board:
                self._blocks[key] = board
        elif isinstance(changed, ExecOutput):
            text = (changed.content or "").rstrip()
            if _looks_like_grid(text) or key not in self._blocks:
                self._blocks[key] = text

    def panels(self) -> list[tuple[str, str]]:
        """Current ``(heading, board)`` panels in first-seen order."""
        out = []
        for key in self._order:
            board = self._blocks.get(key)
            if not board:
                # No board for this lane yet (e.g. the shepherd, which drives from
                # its REPL and has no game) — don't render an empty panel.
                continue
            heading = self._labels.get(key) or key
            if self.status_of is not None and key in self._agents:
                try:
                    status = self.status_of(self._agents[key])
                except Exception:  # noqa: BLE001 - a viewer must never crash a run
                    status = ""
                if status:
                    heading = f"{heading}  [{status}]"
            out.append((heading, board))
        return out

    def render(self) -> str:
        """Plain-text board grid (for composing under a sibling live tree)."""
        panels = self.panels()
        if not panels:
            return ""
        header = f"{self.title}\n\n" if self.title else ""
        return header + grid_of_blocks(panels, cols=self.cols)

    def _paint(self) -> None:
        self._last_paint = time.monotonic()
        panels = self.panels()
        if not panels:
            return
        if self.sink is not None:
            self.sink(panels)
        if not self.paint:
            return
        header = f"{self.title}\n\n" if self.title else ""
        if sys.stdout.isatty():
            print("\033[2J\033[H" + header + grid_of_blocks(panels, cols=self.cols), flush=True)
        else:  # Non-TTY (CI, piped): compact status lines, no screen repaint.
            print("\n".join([self.title, *(h for h, _ in panels)]).strip(), flush=True)

    def close(self) -> None:
        """Paint the final frame once; later Nodes/closes are no-ops so the end
        state (and the report printed after it) is not wiped by a re-clear."""
        if self._closed:
            return
        self._last_paint = 0.0
        self._paint()
        self._closed = True


__all__ = [
    "PanelViewer",
    "export_run_traces",
    "grid_of_blocks",
    "panel_status",
    "side_by_side",
    "trace_line",
]
