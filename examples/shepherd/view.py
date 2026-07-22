"""Board-panel UI helpers for the shepherd example.

:class:`PanelViewer` aggregates per-lane boards (terminal and/or Gradio via
``sink``). The shepherd host calls it alongside :class:`rflow.LiveGraphTree`
on each stream event.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable

from rflow import AppendNode, ExecOutput, Flow, Graph, StreamConsumer


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
        cells = [
            (col[r] if r < len(col) else "").ljust(w) for col, w in zip(cols, widths)
        ]
        rows.append(gap.join(cells).rstrip())
    return "\n".join(rows)


def grid_of_blocks(labeled: list[tuple[str, str]], *, cols: int = 4) -> str:
    """``side_by_side`` wrapped into rows of at most ``cols`` blocks, so a large
    fan-out (e.g. 8 branches) renders as a grid instead of one very wide line."""
    cols = max(1, cols)
    rows = [
        side_by_side(labeled[i : i + cols]) for i in range(0, len(labeled), cols)
    ]
    return "\n\n".join(row for row in rows if row)


def panel_status(flow: Flow, graph: Graph) -> str:
    """Live one-line status for a board panel, read from the graph's REPL env."""
    env = flow.repl_for(graph).env
    pushes = env.get("pushes")
    step = f"push {pushes}" if pushes is not None else "push ?"
    if env.get("solved"):
        return f"{step} · SOLVED"
    if env.get("blocked"):
        # A rejected push, not a terminal state — don't say DONE/BLOCKED.
        return f"{step} · illegal push (try another)"
    dist = env.get("dist")
    return f"{step} · dist {dist}" if pushes is not None else ""


class PanelViewer(StreamConsumer):
    """Live side-by-side boards, one panel per **agent**, updated push-by-push.

    One panel per agent, keyed by ``(graph_id, agent_id)`` in first-seen order.
    On each event the viewer walks the passed graph's whole tree, so a root plus
    its ``launch_subagents`` / ``launch_subgraphs`` children — which all share the
    root's ``graph_id`` — each get their own panel (keying by ``graph_id`` alone
    would collapse them into one). Each panel shows that agent's board (via
    ``board_of``) or its latest ``ExecOutput`` grid, so the worker and every
    recovery branch animate as their streams advance. Safe to share across
    concurrently-gathered streams: event handling runs on one asyncio loop, so
    updates never interleave.
    """

    def __init__(
        self,
        *,
        title: str = "",
        cols: int = 4,
        every_s: float = 0.1,
        status_of: Callable[[Graph], str] | None = None,
        board_of: Callable[[Graph], str] | None = None,
        frames_of: Callable[[Graph], list[str]] | None = None,
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
        self._last_frames: dict[tuple[str, str], list[str]] = {}
        # When set, panels are routed here (e.g. a Gradio dashboard) in addition
        # to (or instead of, when paint=False) the terminal.
        self.sink = sink
        # When False, still aggregate + sink, but do not clear/print the TTY —
        # use this when a sibling LiveGraphTree owns the terminal frame.
        self.paint = paint
        # Panels are keyed per agent by ``(graph_id, agent_id)`` — agent_ids like
        # "root" collide across the worker and shepherd graphs, so the graph_id is
        # part of the key. ``_root_labels`` holds host-supplied names for a graph's
        # root agent (children default to their short agent_id).
        self._order: list[tuple[str, str]] = []
        self._labels: dict[tuple[str, str], str] = {}
        self._blocks: dict[tuple[str, str], str] = {}
        self._graphs: dict[tuple[str, str], Graph] = {}
        self._root_labels: dict[str, str] = {}
        self._last_paint = 0.0
        self._closed = False

    def label(self, graph_id: str, label: str) -> None:
        """Name a graph's root-agent panel (e.g. "worker"/"shepherd")."""
        self._root_labels[graph_id] = label
        key = (graph_id, graph_id and self._root_agent_id(graph_id))
        if key in self._labels:
            self._labels[key] = label

    def _root_agent_id(self, graph_id: str) -> str | None:
        for gid, agent_id in self._order:
            if gid == graph_id and self._graphs[(gid, agent_id)].parent_agent_id is None:
                return agent_id
        return None

    def _default_label(self, agent: Graph) -> str:
        if agent.parent_agent_id is None:
            return self._root_labels.get(agent.graph_id, agent.agent_id)
        # A child lane: show its short id ("root.b0" -> "b0").
        return agent.agent_id.rsplit(".", 1)[-1]

    def handle(self, event, graph: Graph | None) -> None:
        if graph is None or self._closed:
            return
        # Fan out to every agent in this root's tree so children that share the
        # root's graph_id (launch_subgraphs branches) each animate in their own lane.
        for agent in graph.walk():
            self._observe(agent, event if agent is graph else None)
        # Play out this turn's per-sub-step frames so a box-level push visibly walks
        # the player around cell-by-cell instead of snapping (looking teleporty).
        if self.frames_of is not None:
            for agent in graph.walk():
                self._animate(agent)
        now = time.monotonic()
        if self.every_s and now - self._last_paint < self.every_s:
            return
        self._paint()

    def _safe_frames(self, agent: Graph) -> list[str]:
        try:
            return list(self.frames_of(agent) or [])  # type: ignore[misc]
        except Exception:  # noqa: BLE001 - a viewer must never crash a run
            return []

    def _animate(self, agent: Graph) -> None:
        """Paint this turn's sub-step frames for ``agent`` in sequence (short delay
        each) so the walk-around before each shove is visible. ``frames`` is the
        current turn only (replaced each turn), so we play it once — when it differs
        from what we last drew for this lane."""
        key = (agent.graph_id, agent.agent_id)
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

    def _observe(self, agent: Graph, event) -> None:
        key = (agent.graph_id, agent.agent_id)
        if key not in self._order:
            self._order.append(key)
            self._labels.setdefault(key, self._default_label(agent))
            # Seed with the lane's current frames so a fork's inherited (replayed)
            # turn isn't re-animated on first sight — only turns from here play.
            if self.frames_of is not None:
                self._last_frames[key] = self._safe_frames(agent)
        self._graphs[key] = agent
        if self.board_of is not None:
            try:
                board = self.board_of(agent)
            except Exception:  # noqa: BLE001 - a viewer must never crash a run
                board = None
            if board:
                self._blocks[key] = board
        elif event is not None and isinstance(event, AppendNode) and isinstance(
            event.node, ExecOutput
        ):
            text = (event.node.content or "").rstrip()
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
            heading = self._labels.get(key) or key[1]
            if self.status_of is not None and key in self._graphs:
                try:
                    status = self.status_of(self._graphs[key])
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
        """Paint the final frame once; later events/closes are no-ops so the end
        state (and the report printed after it) is not wiped by a re-clear."""
        if self._closed:
            return
        self._last_paint = 0.0
        self._paint()
        self._closed = True


__all__ = ["PanelViewer", "grid_of_blocks", "panel_status", "side_by_side"]
