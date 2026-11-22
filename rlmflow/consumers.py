"""Consumers for ``Flow.run_streaming``'s stream: things that see each ``Step``.

Pass them as hooks, and the run calls them on every step and closes them when it ends:

    async for step in flow.run_streaming(g, hooks=[LiveTree(), Checkpoint("runs/x")]):
        ...

Or call them yourself in the loop, and ``close()`` them after it.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any

from rlmflow.graph.graph import Graph
from rlmflow.graph.nodes import Node, Step
from rlmflow.view.text import _status, render_tree


class StreamConsumer:
    """Something you call on each streamed ``Step``. Subclass it, or use any ``callable(step)``."""

    def __call__(self, step: Step) -> Any:
        raise NotImplementedError

    def close(self) -> None:
        """Flush or release; call it after the loop."""

    def on_step(self, flow: Any, step: Step) -> None:
        self(step)

    def end(self, flow: Any, g: Graph) -> None:
        self.close()


class Checkpoint(StreamConsumer):
    """Save one streamed graph every so often, on every terminal node, and on ``close``."""

    def __init__(
        self,
        path: str | Path,
        *,
        interval_s: float | None = 2.0,
        interval_nodes: int | None = 50,
    ) -> None:
        if interval_s is not None and interval_s <= 0:
            raise ValueError("interval_s must be positive or None")
        if interval_nodes is not None and interval_nodes <= 0:
            raise ValueError("interval_nodes must be positive or None")
        self.path = Path(path)
        self.interval_s = interval_s
        self.interval_nodes = interval_nodes
        self.last_flush_at = time.monotonic()
        self.latest: Graph | None = None
        self.dirty_nodes = 0

    def __call__(self, step: Step) -> None:
        if self.latest is not None and step.graph is not self.latest:
            raise ValueError("one Checkpoint cannot write multiple graphs")
        self.latest = step.graph
        self.dirty_nodes += 1
        if step.node.terminal or self._due():
            self.flush()

    def _due(self) -> bool:
        if self.interval_nodes is not None and self.dirty_nodes >= self.interval_nodes:
            return True
        return (
            self.interval_s is not None and time.monotonic() - self.last_flush_at >= self.interval_s
        )

    def flush(self) -> None:
        graph = self.latest
        if graph is None:
            return
        graph.save(self.path)
        self.dirty_nodes = 0
        self.last_flush_at = time.monotonic()

    def close(self) -> None:
        self.flush()


class LiveTree(StreamConsumer):
    """A live agent tree for every graph it sees: Rich with spinners in a terminal, plain text otherwise."""

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
        self.graphs: dict[str, Graph] = {}
        self.labels: dict[str, str] = {}
        self.last_paint = 0.0
        self.live: Any = None
        self.closed = False
        self.use_rich = _rich_importable() and (
            rich is True or (rich is None and sys.stdout.isatty())
        )

    def track(self, graph: Graph, *, label: str | None = None) -> None:
        """Register a graph before its first streamed item."""
        self.graphs.setdefault(graph.id, graph)
        self.labels.setdefault(graph.id, label or graph.id)

    def label(self, graph_id: str, label: str) -> None:
        """Change the heading for an already tracked graph."""
        self.labels[graph_id] = label

    def __call__(self, step: Step) -> None:
        if self.closed:
            return
        self.track(step.graph)
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
        for key, graph in self.graphs.items():
            body = render_tree(graph)
            heading = self.labels.get(key, key)
            blocks.append(f"[{heading}]\n{body}" if len(self.graphs) > 1 else body)
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
        if not self.graphs:
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

        trees = [self._rich_tree(graph, self.labels.get(key)) for key, graph in self.graphs.items()]
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

    def _rich_tree(self, graph: Graph, label: str | None):
        from rich.spinner import Spinner
        from rich.text import Text
        from rich.tree import Tree

        def render_label(agent: Node):
            text = _status(graph, agent)
            status = agent.status()
            if status == "done":
                if agent.last_turn().error:
                    return Text.assemble(("✗ ", "red"), (text, "red"))
                return Text.assemble(("✓ ", "green"), (text, "green"))
            if status == "idle" or not all(c.status() == "done" for c in agent.children()):
                return Text.assemble(("◌ ", "magenta"), (text, "magenta"))
            if self.spinners:
                return Spinner("dots", text=Text(text, style="cyan"))
            return Text.assemble(("• ", "cyan"), (text, "cyan"))

        tree = Tree(Text(label or graph.id, style="bold"), guide_style="dim")

        def populate(branch, agent: Node) -> None:
            for child in agent.children():
                child_branch = branch.add(render_label(child), guide_style="dim")
                populate(child_branch, child)

        for top in graph.roots():
            populate(tree.add(render_label(top), guide_style="dim"), top)
        return tree


def _rich_importable() -> bool:
    try:
        import rich  # noqa: F401

        return True
    except ImportError:
        return False


__all__ = [
    "Checkpoint",
    "LiveTree",
    "StreamConsumer",
]
