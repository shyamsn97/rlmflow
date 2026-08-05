"""Compact terminal visualization for streamed Node trees."""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Iterable
from contextlib import suppress
from typing import Any

from rlmflow.consumers.base import StreamConsumer
from rlmflow.graph.nodes import (
    AgentStart,
    DoneOutput,
    ErrorOutput,
    ExecAction,
    ExecOutput,
    LLMOutput,
    Node,
    UserQuery,
)


class LiveTreeRenderer(StreamConsumer):
    """Redraw the live agent tree whenever a streamed Node is created."""

    def __init__(self, *, clear: bool = True) -> None:
        self.clear = clear
        self.roots: dict[int, AgentStart] = {}

    def handle(self, node: Node) -> None:
        root = node.root
        if root is None:
            return
        self.roots.setdefault(id(root), root)
        if self.clear and sys.stdout.isatty():
            print("\033[2J\033[H", end="")
        print(render_forest(self.roots.values()), flush=True)


class LiveGraphTree(StreamConsumer):
    """Rich live forest with per-agent status and optional spinners."""

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
        self.footer = footer
        self.roots: dict[str, AgentStart] = {}
        self.labels: dict[str, str] = {}
        self.last_paint = 0.0
        self.live: Any = None
        self.closed = False
        self.use_rich = _rich_importable() and (
            rich is True or (rich is None and sys.stdout.isatty())
        )

    def track(self, root: AgentStart, *, label: str | None = None) -> None:
        """Register a root before its first streamed Node."""
        key = root.id
        self.roots.setdefault(key, root)
        self.labels.setdefault(key, label or root.config.path)

    def label(self, root_id: str, label: str) -> None:
        """Change the heading for an already tracked root."""
        self.labels[root_id] = label

    def handle(self, node: Node) -> None:
        root = node.root
        if root is None or self.closed:
            return
        self.track(root)
        now = time.monotonic()
        if self.every_s and now - self.last_paint < self.every_s:
            return
        self._paint()

    def close(self) -> None:
        if self.closed:
            return
        self.last_paint = 0.0
        self._paint(final=True)
        if self.live is not None:
            with suppress(Exception):
                self.live.stop()
            self.live = None
        self.closed = True

    def render(self) -> str:
        blocks: list[str] = []
        if self.title:
            blocks.append(self.title)
        for key, root in self.roots.items():
            body = render_tree(root)
            heading = self.labels.get(key, root.config.path)
            blocks.append(f"[{heading}]\n{body}" if len(self.roots) > 1 else body)
        footer = self._footer_text()
        if footer:
            blocks.append(footer)
        return "\n\n".join(blocks)

    def _footer_text(self) -> str:
        if self.footer is None:
            return ""
        try:
            return (self.footer() or "").rstrip()
        except Exception:  # noqa: BLE001 - visualization cannot crash a run
            return ""

    def _paint(self, *, final: bool = False) -> None:
        self.last_paint = time.monotonic()
        if not self.roots:
            return
        if not self.use_rich:
            text = self.render()
            if self.clear and sys.stdout.isatty() and not final:
                print("\033[2J\033[H", end="")
            print(text, flush=True)
            return

        from rich.console import Group
        from rich.live import Live
        from rich.panel import Panel

        trees = [self._rich_tree(root, self.labels.get(key)) for key, root in self.roots.items()]
        body: Any = trees[0] if len(trees) == 1 else Group(*trees)
        footer = self._footer_text()
        if footer:
            from rich.text import Text

            body = Group(body, Text(""), Text(footer))
        if self.title:
            body = Panel(body, title=self.title, border_style="dim")
        if self.live is None:
            self.live = Live(
                body,
                refresh_per_second=12 if self.spinners else 4,
                vertical_overflow="visible",
                redirect_stdout=False,
                redirect_stderr=False,
            )
            self.live.start()
        else:
            self.live.update(body, refresh=True)
        if final:
            self.live.refresh()

    def _rich_tree(self, root: AgentStart, label: str | None):
        from rich.spinner import Spinner
        from rich.text import Text
        from rich.tree import Tree

        def render_label(agent: AgentStart):
            text = _status(agent)
            frontier = agent.frontier
            if isinstance(frontier, DoneOutput):
                return Text.assemble(("✓ ", "green"), (text, "green"))
            if isinstance(frontier, ErrorOutput):
                return Text.assemble(("✗ ", "red"), (text, "red"))
            if isinstance(frontier, ExecAction) and any(
                isinstance(child, AgentStart) for child in frontier.children
            ):
                return Text.assemble(("◌ ", "magenta"), (text, "magenta"))
            if self.spinners:
                return Spinner("dots", text=Text(text, style="cyan"))
            return Text.assemble(("• ", "cyan"), (text, "cyan"))

        tree = Tree(Text(label or root.config.path, style="bold"), guide_style="dim")
        root_branch = tree.add(render_label(root), guide_style="dim")

        def populate(branch, agent: AgentStart) -> None:
            for child in agent.sub_agents:
                child_branch = branch.add(render_label(child), guide_style="dim")
                populate(child_branch, child)

        populate(root_branch, root)
        return tree


def render_forest(roots: Iterable[AgentStart]) -> str:
    """Render one or more roots in first-seen order."""
    roots = list(roots)
    blocks = [render_tree(root) for root in roots]
    return "\n\n".join(blocks)


def render_tree(root: AgentStart) -> str:
    """Render the root query and recursive agent status tree."""
    lines = [_clip(root.content) or "root query"]
    _append_agent(lines, root, prefix="", last=True)
    return "\n".join(lines)


def _append_agent(
    lines: list[str],
    agent: AgentStart,
    *,
    prefix: str,
    last: bool,
) -> None:
    lines.append(f"{prefix}{'└── ' if last else '├── '}{_status(agent)}")
    child_prefix = prefix + ("    " if last else "│   ")
    for index, child in enumerate(agent.sub_agents):
        _append_agent(
            lines,
            child,
            prefix=child_prefix,
            last=index == len(agent.sub_agents) - 1,
        )


def _status(agent: AgentStart) -> str:
    frontier = agent.frontier
    turns = agent.llm_turns()
    suffix = f" ({turns} turns)" if turns else ""

    if isinstance(frontier, DoneOutput):
        return f"{agent.config.path}: done {_clip(frontier.result, limit=64)}{suffix}"
    if isinstance(frontier, ErrorOutput):
        return f"{agent.config.path}: error {_clip(frontier.content, limit=64)}{suffix}"
    if isinstance(frontier, ExecAction):
        children = [child for child in frontier.children if isinstance(child, AgentStart)]
        if children:
            active = sum(not child.terminal for child in children)
            return f"{agent.config.path}: children running {active}/{len(children)}{suffix}"
        return f"{agent.config.path}: running code{suffix}"
    if isinstance(frontier, LLMOutput):
        return f"{agent.config.path}: planning{suffix}"
    if isinstance(frontier, ExecOutput):
        return f"{agent.config.path}: observed output{suffix}"
    if isinstance(frontier, UserQuery):
        return f"{agent.config.path}: working{suffix}"
    return f"{agent.config.path}: pending{suffix}"


def _clip(value: Any, *, limit: int = 80) -> str:
    text = " ".join(str(value or "").strip().split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _rich_importable() -> bool:
    try:
        import rich  # noqa: F401

        return True
    except ImportError:
        return False


__all__ = [
    "LiveGraphTree",
    "LiveTreeRenderer",
    "render_forest",
    "render_tree",
]
