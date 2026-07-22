"""Event-stream UI consumers and compact graph rendering."""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from contextlib import suppress
from typing import Any

from rflow.consumers.base import StreamConsumer
from rflow.graph import (
    DoneOutput,
    ErrorOutput,
    ExecAction,
    ExecOutput,
    Graph,
    LLMOutput,
    ResumeAction,
    SupervisingOutput,
    UserQuery,
)
from rflow.graph.events import Event


class LiveTreeRenderer(StreamConsumer):
    """Redraw a compact graph tree in response to graph events."""

    def __init__(self, *, clear: bool = True) -> None:
        self.clear = clear

    def handle(self, event: Event, graph: Graph | None) -> None:
        if graph is None:
            return
        if self.clear:
            print("\033[2J\033[H", end="")
        print(render_tree(graph))


class LiveGraphTree(StreamConsumer):
    """Live agent-tree view with active/waiting child counts (and optional spinners).

    Tracks one or more roots by ``graph_id`` (first-seen order), so a parent with
    ``launch_subagents`` children *and* parallel fork forests both render. On each
    event it walks the tree and, for supervisors, reports ``children running
    active/total`` from ``SupervisingOutput.waiting_on``.

    Uses Rich ``Live`` + ``Spinner`` when Rich is installed and stdout is a TTY;
    otherwise falls back to a clear+``render_tree`` redraw (same active/total text).
    """

    def __init__(
        self,
        *,
        title: str = "",
        every_s: float = 0.05,
        spinners: bool = True,
        rich: bool | None = None,
        clear: bool = True,
        footer: Callable[[], str] | None = None,
    ) -> None:
        self.title = title
        self.every_s = every_s
        self.spinners = spinners
        self.clear = clear
        # Optional extra text under the tree (e.g. board panels from a sibling
        # PanelViewer) so multiple consumers can share one terminal frame.
        self.footer = footer
        self._rich_pref = rich
        self._order: list[str] = []
        self._labels: dict[str, str] = {}
        self._graphs: dict[str, Graph] = {}
        self._last_paint = 0.0
        self._closed = False
        self._live: Any = None
        self._use_rich = self._resolve_rich()

    def _resolve_rich(self) -> bool:
        if self._rich_pref is False:
            return False
        if self._rich_pref is True:
            return _rich_importable()
        return _rich_importable() and sys.stdout.isatty()

    def track(self, graph: Graph, *, label: str | None = None) -> None:
        """Register a root before the first event (optional)."""
        key = graph.graph_id
        if key not in self._order:
            self._order.append(key)
        self._graphs[key] = graph
        if label is not None:
            self._labels[key] = label
        else:
            self._labels.setdefault(key, graph.agent_id)

    def label(self, graph_id: str, label: str) -> None:
        if graph_id not in self._order:
            self._order.append(graph_id)
        self._labels[graph_id] = label

    def handle(self, event: Event, graph: Graph | None) -> None:
        if graph is None or self._closed:
            return
        key = graph.graph_id
        if key not in self._order:
            self._order.append(key)
            self._labels.setdefault(key, graph.agent_id)
        self._graphs[key] = graph
        now = time.monotonic()
        if self.every_s and now - self._last_paint < self.every_s:
            return
        self._paint()

    def close(self) -> None:
        if self._closed:
            return
        self._last_paint = 0.0
        self._paint(final=True)
        self._stop_live()
        self._closed = True

    def _paint(self, *, final: bool = False) -> None:
        self._last_paint = time.monotonic()
        if not self._order:
            return
        if self._use_rich:
            self._paint_rich(final=final)
            return
        text = self.render()
        if self.clear and sys.stdout.isatty() and not final:
            print("\033[2J\033[H" + text, flush=True)
        else:
            print(text, flush=True)

    def render(self) -> str:
        """Plain-text forest (also used as the non-Rich fallback)."""
        blocks: list[str] = []
        if self.title:
            blocks.append(self.title)
        for key in self._order:
            graph = self._graphs.get(key)
            if graph is None:
                continue
            heading = self._labels.get(key, graph.agent_id)
            body = render_tree(graph)
            blocks.append(f"[{heading}]\n{body}" if len(self._order) > 1 else body)
        foot = self._footer_text()
        if foot:
            blocks.append(foot)
        return "\n\n".join(blocks)

    def _footer_text(self) -> str:
        if self.footer is None:
            return ""
        try:
            return (self.footer() or "").rstrip()
        except Exception:  # noqa: BLE001 - a viewer must never crash a run
            return ""

    def _paint_rich(self, *, final: bool = False) -> None:
        from rich.console import Group
        from rich.live import Live
        from rich.panel import Panel
        from rich.text import Text

        trees = [
            self._rich_tree(self._graphs[key], key)
            for key in self._order
            if key in self._graphs
        ]
        if not trees:
            return
        body: Any = trees[0] if len(trees) == 1 else Group(*trees)
        foot = self._footer_text()
        if foot:
            body = Group(body, Text(""), Text(foot))
        if self.title:
            body = Panel(body, title=Text(self.title, style="bold"), border_style="dim")
        if self._live is None:
            self._live = Live(
                body,
                console=_rich_console(),
                refresh_per_second=12 if self.spinners else 4,
                vertical_overflow="visible",
                redirect_stdout=False,
                redirect_stderr=False,
            )
            self._live.start()
        else:
            self._live.update(body, refresh=True)
        if final:
            self._live.refresh()

    def _stop_live(self) -> None:
        if self._live is None:
            return
        with suppress(Exception):
            self._live.stop()
        self._live = None

    def _rich_tree(self, graph: Graph, key: str):
        from rich.spinner import Spinner
        from rich.text import Text
        from rich.tree import Tree

        def label_for(agent: Graph) -> Any:
            text = _status_text(agent)
            cur = agent.current()
            if isinstance(cur, DoneOutput):
                return Text.assemble(("✓ ", "green"), (text, "green"))
            if isinstance(cur, ErrorOutput):
                return Text.assemble(("✗ ", "red"), (text, "red"))
            if isinstance(cur, SupervisingOutput):
                return Text.assemble(("◌ ", "magenta"), (text, "magenta"))
            if self.spinners and not (cur and cur.terminal):
                return Spinner("dots", text=Text(text, style="cyan"))
            return Text.assemble(("• ", "cyan"), (text, "cyan"))

        root_label = self._labels.get(key)
        if root_label and root_label != graph.agent_id and len(self._order) > 1:
            tree = Tree(Text(root_label, style="bold"), guide_style="dim")
            node = tree.add(label_for(graph), guide_style="dim")
        else:
            tree = Tree(label_for(graph), guide_style="dim")
            node = tree

        def populate(parent, agent: Graph) -> None:
            children = list(agent.children.values())
            for child in children:
                branch = parent.add(label_for(child), guide_style="dim")
                populate(branch, child)

        populate(node, graph)
        return tree


def render_tree(graph: Graph) -> str:
    lines = [_clip(graph.query) or "root query"]
    _append_agent(lines, graph, prefix="", last=True)
    return "\n".join(lines)


def _append_agent(lines: list[str], graph: Graph, *, prefix: str, last: bool) -> None:
    lines.append(f"{prefix}{'└── ' if last else '├── '}{_status(graph)}")
    child_prefix = prefix + ("    " if last else "│   ")
    children = list(graph.children.values())
    for i, child in enumerate(children):
        _append_agent(lines, child, prefix=child_prefix, last=i == len(children) - 1)


def _status(graph: Graph) -> str:
    return _status_text(graph)


def _status_text(graph: Graph) -> str:
    current = graph.current()
    turns = sum(isinstance(node, LLMOutput) for node in graph.nodes)
    suffix = f" ({turns} turns)" if turns else ""
    if isinstance(current, DoneOutput):
        return f"{graph.agent_id}: done {_clip(current.result, limit=64)}{suffix}"
    if isinstance(current, ErrorOutput):
        return f"{graph.agent_id}: error {_clip(current.error or current.output, limit=64)}{suffix}"
    if isinstance(current, SupervisingOutput):
        active, total = _child_progress(graph, current.waiting_on)
        return f"{graph.agent_id}: children running {active}/{total}{suffix}"
    if isinstance(current, ResumeAction):
        return f"{graph.agent_id}: resumed{suffix}"
    if isinstance(current, ExecAction):
        return f"{graph.agent_id}: running code{suffix}"
    if isinstance(current, LLMOutput):
        return f"{graph.agent_id}: planning{suffix}"
    if isinstance(current, ExecOutput):
        return f"{graph.agent_id}: observed output{suffix}"
    if isinstance(current, UserQuery):
        # A committed user turn is the transient between-turns state: the agent has
        # queued its next prompt and is awaiting the model. If it has already taken
        # turns it's actively working; only a fresh (0-turn) agent is truly pending.
        return (
            f"{graph.agent_id}: working{suffix}"
            if turns
            else f"{graph.agent_id}: pending"
        )
    return f"{graph.agent_id}: pending"


def _child_progress(graph: Graph, waiting_on: list[str]) -> tuple[int, int]:
    """Count still-running children among ``waiting_on`` (active, total)."""
    active = 0
    for child_id in waiting_on:
        child = graph.agents.get(child_id)
        if child is None:
            active += 1
            continue
        cur = child.current()
        if not (cur and cur.terminal):
            active += 1
    return active, len(waiting_on)


def _clip(text: str, *, limit: int = 80) -> str:
    text = " ".join(text.strip().split())
    return text if len(text) <= limit else text[: limit - 1] + "..."


def _rich_importable() -> bool:
    try:
        import rich  # noqa: F401

        return True
    except ImportError:
        return False


def _rich_console():
    from rich.console import Console

    return Console(stderr=False)


__all__ = [
    "LiveGraphTree",
    "LiveTreeRenderer",
    "render_tree",
]
