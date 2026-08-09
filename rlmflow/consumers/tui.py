"""Textual dashboard for live Node streams."""

# Textual's compose DSL is clearer as nested container contexts.

from __future__ import annotations

from collections import Counter
from collections.abc import Awaitable, Callable
from typing import Any, ClassVar

from rlmflow.consumers.base import StreamConsumer
from rlmflow.consumers.ui import render_tree
from rlmflow.graph.nodes import (
    AgentStart,
    DoneOutput,
    ErrorOutput,
    ExecAction,
    LLMOutput,
    Node,
    UserQuery,
)

DriveFn = Callable[..., Awaitable[AgentStart | None]]


class FlowTUI(StreamConsumer):
    """Interactive chat plus overview, tree, agent, error, and activity panels."""

    def __init__(self, root: AgentStart | None = None) -> None:
        self.root = root
        self.app: Any = None
        self.latest: list[Node] = []
        self.busy = False

    def handle(self, node: Node) -> None:
        if node.root is not None:
            self.root = node.root
        self.latest.append(node)
        del self.latest[:-100]
        if self.app is not None:
            self.app.refresh_dashboard()

    def close(self) -> None:
        if self.app is not None:
            self.app.refresh_dashboard()

    def run(self, drive: DriveFn) -> AgentStart | None:
        """Open the dashboard; ``drive`` executes Send, Run, and Step actions.

        ``drive`` receives ``(root, query=..., inputs=..., until=...)`` and should
        call :meth:`handle` for each Node it streams.
        """
        try:
            from rich.panel import Panel
            from textual import work
            from textual.app import App, ComposeResult
            from textual.binding import Binding
            from textual.containers import Horizontal, Vertical, VerticalScroll
            from textual.widgets import (
                Footer,
                Header,
                RichLog,
                Static,
                TabbedContent,
                TabPane,
                TextArea,
            )
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError('FlowTUI requires `pip install "rlmflow[tui]"`.') from exc

        consumer = self

        class Dashboard(App):
            TITLE = "rlmflow"
            CSS = """
            Screen { layout: vertical; }
            #main { height: 1fr; }
            #chat-column { width: 3fr; min-width: 48; }
            #side-column { width: 2fr; min-width: 40; }
            #chat { height: 1fr; border: round $primary; padding: 0 1; }
            #prompt { height: 5; margin-top: 1; border: round $primary; }
            #context { height: 4; margin-top: 1; border: round $primary; }
            #tabs { height: 1fr; }
            TabPane { padding: 0 1; }
            VerticalScroll { height: 1fr; }
            Static { height: auto; width: 100%; }
            """
            BINDINGS: ClassVar[list[Any]] = [
                ("ctrl+c", "quit", "Quit"),
                Binding("ctrl+s", "submit_prompt", "Send", priority=True),
                Binding("ctrl+enter", "submit_prompt", "Send", priority=True, show=False),
                ("ctrl+r", "run_until_done", "Run"),
                ("ctrl+t", "step_once", "Step"),
            ]

            def __init__(self) -> None:
                super().__init__()
                self.seen: set[str] = set()

            def compose(self) -> ComposeResult:
                yield Header(show_clock=True)
                with Horizontal(id="main"):
                    with Vertical(id="chat-column"):
                        yield RichLog(id="chat", wrap=True, markup=False, highlight=True)
                        yield TextArea(
                            id="prompt",
                            soft_wrap=True,
                            placeholder="Query… (Ctrl+S to send)",
                        )
                        yield TextArea(
                            id="context",
                            soft_wrap=True,
                            placeholder="Optional context passed as INPUTS['context']",
                        )
                    with Vertical(id="side-column"):
                        with TabbedContent(initial="tree-tab", id="tabs"):
                            with TabPane("Overview", id="overview-tab"):
                                with VerticalScroll():
                                    yield Static(id="overview")
                            with TabPane("Tree", id="tree-tab"):
                                with VerticalScroll():
                                    yield Static(id="tree")
                            with TabPane("Agents", id="agents-tab"):
                                with VerticalScroll():
                                    yield Static(id="agents")
                            with TabPane("Counts", id="counts-tab"):
                                with VerticalScroll():
                                    yield Static(id="counts")
                            with TabPane("Waiting", id="waiting-tab"):
                                with VerticalScroll():
                                    yield Static(id="waiting")
                            with TabPane("Errors", id="errors-tab"):
                                with VerticalScroll():
                                    yield Static(id="errors")
                            with TabPane("Latest", id="latest-tab"):
                                with VerticalScroll():
                                    yield Static(id="latest")
                yield Footer()

            def on_mount(self) -> None:
                self.query_one("#chat", RichLog).write(
                    Panel(
                        "Enter a query, then Ctrl+S. Ctrl+R resumes to completion; "
                        "Ctrl+T advances to the next streamed Node.",
                        title="ready",
                        border_style="cyan",
                    )
                )
                self.refresh_dashboard()

            def refresh_dashboard(self) -> None:
                root = consumer.root
                if root is None:
                    for selector in (
                        "#overview",
                        "#tree",
                        "#agents",
                        "#counts",
                        "#waiting",
                        "#errors",
                        "#latest",
                    ):
                        self.query_one(selector, Static).update("No run yet.")
                    return

                self.query_one("#overview", Static).update(overview_table(root, consumer.busy))
                self.query_one("#tree", Static).update(Panel(render_tree(root), border_style="dim"))
                self.query_one("#agents", Static).update(agent_table(root))
                self.query_one("#counts", Static).update(node_counts_table(root))
                self.query_one("#waiting", Static).update(waiting_table(root))
                self.query_one("#errors", Static).update(error_table(root))
                self.query_one("#latest", Static).update(latest_table(consumer.latest))
                self._append_chat()

            def _append_chat(self) -> None:
                log = self.query_one("#chat", RichLog)
                for node in consumer.latest:
                    if node.id in self.seen:
                        continue
                    self.seen.add(node.id)
                    if isinstance(node, (UserQuery, LLMOutput, ErrorOutput, DoneOutput)):
                        log.write(node_panel(node))

            def action_submit_prompt(self) -> None:
                prompt = self.query_one("#prompt", TextArea)
                context = self.query_one("#context", TextArea)
                query = prompt.text.strip()
                if not query:
                    return
                prompt.text = ""
                inputs = {"context": context.text.strip()} if context.text.strip() else None
                self._drive(query=query, inputs=inputs, until="done")

            def action_run_until_done(self) -> None:
                self._drive(query=None, inputs=None, until="done")

            def action_step_once(self) -> None:
                self._drive(query=None, inputs=None, until="next")

            @work(exclusive=True)
            async def _drive(
                self,
                *,
                query: str | None,
                inputs: dict[str, str] | None,
                until: str,
            ) -> None:
                if consumer.busy:
                    return
                if consumer.root is None and query is None:
                    self.query_one("#chat", RichLog).write(
                        Panel("Enter a query first.", border_style="yellow")
                    )
                    return
                consumer.busy = True
                self.refresh_dashboard()
                try:
                    root = await drive(
                        consumer.root,
                        query=query,
                        inputs=inputs,
                        until=until,
                    )
                    if root is not None:
                        consumer.root = root
                except Exception as exc:  # noqa: BLE001 - report failures in the UI
                    self.query_one("#chat", RichLog).write(
                        Panel(f"{type(exc).__name__}: {exc}", border_style="red")
                    )
                finally:
                    consumer.busy = False
                    self.refresh_dashboard()

        self.app = Dashboard()
        try:
            self.app.run()
        finally:
            self.app = None
        return self.root


def overview_table(root: AgentStart, busy: bool = False):
    from rich.table import Table

    agents = _agents(root)
    usage = root.tokens()
    table = Table.grid(expand=True)
    table.add_column(style="dim")
    table.add_column(justify="right")
    table.add_row("status", "running" if busy else ("done" if root.terminal else "ready"))
    table.add_row("agents", str(len(agents)))
    table.add_row("nodes", str(sum(1 for _node in root.walk())))
    table.add_row("max depth", str(max(agent.config.depth for agent in agents)))
    table.add_row("tokens in", str(usage.input_tokens))
    table.add_row("tokens out", str(usage.output_tokens))
    return table


def agent_table(root: AgentStart):
    from rich.table import Table

    table = Table(show_header=True, header_style="bold dim", expand=True)
    table.add_column("agent", overflow="fold")
    table.add_column("status")
    table.add_column("frontier")
    table.add_column("turns", justify="right")
    for agent in _agents(root):
        table.add_row(
            agent.config.path,
            "done" if agent.terminal else "active",
            agent.frontier.type,
            str(agent.llm_turns()),
        )
    return table


def node_counts_table(root: AgentStart):
    from rich.table import Table

    counts = Counter(node.type for node in root.walk())
    table = Table(show_header=True, header_style="bold dim", expand=True)
    table.add_column("node")
    table.add_column("count", justify="right")
    for kind, count in sorted(counts.items()):
        table.add_row(kind, str(count))
    return table


def waiting_table(root: AgentStart):
    from rich.table import Table

    table = Table(show_header=True, header_style="bold dim", expand=True)
    table.add_column("agent")
    table.add_column("children")
    found = False
    for agent in _agents(root):
        frontier = agent.frontier
        if not isinstance(frontier, ExecAction):
            continue
        children = [child for child in frontier.children if isinstance(child, AgentStart)]
        if not children:
            continue
        active = sum(not child.terminal for child in children)
        table.add_row(agent.config.path, f"{active}/{len(children)} running")
        found = True
    if not found:
        table.add_row("-", "none")
    return table


def error_table(root: AgentStart):
    from rich.table import Table

    errors = [node for node in root.walk() if isinstance(node, ErrorOutput)]
    table = Table(show_header=True, header_style="bold dim", expand=True)
    table.add_column("agent")
    table.add_column("error", overflow="fold")
    for node in errors:
        table.add_row(node.parent_agent.config.path, _one_line(node.content, 80))
    if not errors:
        table.add_row("-", "none")
    return table


def latest_table(nodes: list[Node], limit: int = 12):
    from rich.table import Table

    table = Table(show_header=True, header_style="bold dim", expand=True)
    table.add_column("agent", overflow="fold")
    table.add_column("node")
    table.add_column("ms", justify="right")
    for node in nodes[-limit:]:
        duration = node.timing().get("duration_ms")
        table.add_row(
            node.parent_agent.config.path,
            node.type,
            "-" if duration is None else str(duration),
        )
    if not nodes:
        table.add_row("-", "-", "-")
    return table


def node_panel(node: Node):
    from rich.panel import Panel

    style = (
        "red"
        if isinstance(node, ErrorOutput)
        else "green"
        if isinstance(node, DoneOutput)
        else "cyan"
    )
    body = node.result if isinstance(node, DoneOutput) else node.content
    return Panel(
        str(body or f"({node.type})"),
        title=f"{node.parent_agent.config.path} · {node.type}",
        border_style=style,
    )


def _agents(root: AgentStart) -> list[AgentStart]:
    return [node for node in root.walk() if isinstance(node, AgentStart)]


def _one_line(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


__all__ = [
    "DriveFn",
    "FlowTUI",
    "agent_table",
    "error_table",
    "latest_table",
    "node_counts_table",
    "node_panel",
    "overview_table",
    "waiting_table",
]
