"""Composable consumers for ``Flow.run_streaming``'s ``Step`` stream.

``handle(item, graph)`` takes a streamed ``Step``, or a bare ``Node``.
"""

from __future__ import annotations

import shutil
import sys
import time
from collections.abc import Callable, Iterable
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rlmflow.graph.graph import Edge, Graph
from rlmflow.graph.nodes import DONE, WAITING, Node, Step
from rlmflow.view.text import _status, render_forest, render_tree, top_level

if TYPE_CHECKING:
    from rlmflow.flow import Flow
    from rlmflow.graph.graph import Graph


class StreamConsumer:
    """Base class for objects that react to streamed graph items."""

    def init(self, flow: Flow) -> None:
        """Hand this consumer the flow behind the stream.

        A consumer that only displays never needs this. It exists for the
        interactive ones, which have to start and step a run rather than watch
        one, and only a flow can do that.
        """

    def handle(self, item: Step | Node | Edge, graph: Graph) -> None:
        raise NotImplementedError

    def close(self) -> None:
        """Release resources or flush final state."""


class ConsumerGroup(StreamConsumer):
    """Fan out each item to a list of consumers."""

    def __init__(self, consumers: Iterable[StreamConsumer] = ()) -> None:
        self.consumers = list(consumers)

    def append(self, consumer: StreamConsumer) -> None:
        self.consumers.append(consumer)

    def init(self, flow: Flow) -> None:
        for consumer in self.consumers:
            consumer.init(flow)

    def handle(self, item: Step | Node | Edge, graph: Graph) -> None:
        for consumer in self.consumers:
            consumer.handle(item, graph)

    def close(self) -> None:
        for consumer in reversed(self.consumers):
            with suppress(Exception):
                consumer.close()


class GraphCheckpointer(StreamConsumer):
    """Periodically save one streamed run and always flush its final revision."""

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
        self.saved_revision = -1

    def handle(self, item: Step | Node | Edge, graph: Graph) -> None:
        if self.latest is not None and graph is not self.latest:
            raise ValueError("one GraphCheckpointer cannot write multiple runs")
        self.latest = graph
        self.dirty_nodes += 1
        node = item.node if isinstance(item, Step) else item
        if (isinstance(node, Node) and node.type == DONE) or self._due():
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
        revision = len(graph)
        if revision == self.saved_revision:
            return
        graph.save(self.path)
        self.saved_revision = revision
        self.dirty_nodes = 0
        self.last_flush_at = time.monotonic()

    def close(self) -> None:
        self.flush()


DEFAULT_IGNORE = (".venv", "__pycache__", ".pytest_cache", ".git")


class WorkspaceSync(StreamConsumer):
    """Mirror a run directory into a local workspace while items stream."""

    def __init__(
        self,
        source_dir: str | Path,
        target_dir: str | Path,
        *,
        every_s: float = 2.0,
        ignore: tuple[str, ...] = DEFAULT_IGNORE,
    ) -> None:
        self.source_dir = Path(source_dir).resolve()
        self.target_dir = Path(target_dir).resolve()
        self.every_s = every_s
        self.ignore = ignore
        self.last = 0.0
        if self.source_dir == self.target_dir:
            raise ValueError("source_dir and target_dir must be different")
        if _is_relative_to(self.target_dir, self.source_dir):
            raise ValueError("target_dir must not be inside source_dir")

    def handle(self, item: Step | Node | Edge, graph: Graph) -> None:
        now = time.monotonic()
        if self.every_s and now - self.last < self.every_s:
            return
        self.sync()

    def sync(self) -> None:
        if not self.source_dir.exists():
            return
        shutil.copytree(
            self.source_dir,
            self.target_dir,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(*self.ignore),
        )
        self.last = time.monotonic()

    def close(self) -> None:
        self.sync()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


class LiveTreeRenderer(StreamConsumer):
    """Redraw the live agent forest whenever a streamed node lands."""

    def __init__(self, *, clear: bool = True) -> None:
        self.clear = clear
        self.graphs: dict[str, Graph] = {}

    def handle(self, item: Step | Node | Edge, graph: Graph) -> None:
        if not isinstance(item, (Step, Node)):
            return
        self.graphs.setdefault(graph.id, graph)
        if self.clear and sys.stdout.isatty():
            print("\033[2J\033[H", end="")
        print(render_forest(self.graphs.values()), flush=True)


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

    def handle(self, item: Step | Node | Edge, graph: Graph) -> None:
        if self.closed:
            return
        self.track(graph)
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
            frontier = graph.frontier(agent)
            if frontier.type == DONE:
                if frontier.data.get("error"):
                    return Text.assemble(("✗ ", "red"), (text, "red"))
                return Text.assemble(("✓ ", "green"), (text, "green"))
            if frontier.type == WAITING or any(not graph.terminal(c) for c in graph.children(agent)):
                return Text.assemble(("◌ ", "magenta"), (text, "magenta"))
            if self.spinners:
                return Spinner("dots", text=Text(text, style="cyan"))
            return Text.assemble(("• ", "cyan"), (text, "cyan"))

        tree = Tree(Text(label or graph.id, style="bold"), guide_style="dim")

        def populate(branch, agent: Node) -> None:
            for child in graph.children(agent):
                child_branch = branch.add(render_label(child), guide_style="dim")
                populate(child_branch, child)

        for top in top_level(graph):
            populate(tree.add(render_label(top), guide_style="dim"), top)
        return tree


def _rich_importable() -> bool:
    try:
        import rich  # noqa: F401

        return True
    except ImportError:
        return False


__all__ = [
    "ConsumerGroup",
    "GraphCheckpointer",
    "LiveGraphTree",
    "LiveTreeRenderer",
    "StreamConsumer",
    "WorkspaceSync",
]
