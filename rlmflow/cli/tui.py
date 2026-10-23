"""Textual dashboard for live Node streams."""

# Textual's compose DSL is clearer as nested container contexts.

from __future__ import annotations

from collections import Counter
from typing import Any, ClassVar

from rlmflow.consumers import StreamConsumer
from rlmflow.graph import ops, reads
from rlmflow.graph.graph import Edge, Graph
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
from rlmflow.view.text import _status, launched_by, top_level


class FlowTUI(StreamConsumer):
    """Interactive chat plus overview, tree, agent, error, and activity panels."""

    def __init__(
        self,
        graph: Graph | None = None,
        *,
        sink: StreamConsumer | None = None,
        refresh_interval_s: float = 0.10,
    ) -> None:
        if refresh_interval_s < 0:
            raise ValueError("refresh_interval_s must be non-negative")
        self.graph = graph
        self.sink = sink
        self.refresh_interval_s = refresh_interval_s
        self.flow: Any = None
        self.app: Any = None
        self.latest: list[Node] = []
        self._pending_chat: list[Node] = []
        self._refresh_pending = False
        self._dirty = False
        self.busy = False

    def init(self, flow: Any) -> None:
        """Adopt the flow whose turns the dashboard's keys will run."""
        self.flow = flow
        if self.sink is not None:
            self.sink.init(flow)

    async def turn(
        self,
        *,
        query: str | None = None,
        inputs: dict[str, str] | None = None,
        until: str = "done",
    ) -> Graph | None:
        """Run one dashboard action: start an agent or message the first one, and stream.

        Depth, iteration caps, and the rest come from the flow's own defaults, so
        there is one place to configure a run rather than two that can disagree.
        """
        if self.flow is None:
            raise RuntimeError("FlowTUI has no flow: call init(flow) before run().")
        if self.graph is None:
            if query is None:
                return None
            self.graph = Graph()
            self.graph.create_agent(query, config=self.flow.root_config, **(inputs or {}))
        elif query is not None:
            ops.send(self.graph, top_level(self.graph)[0], query)
        async for item in self.flow.run_streaming(self.graph, until=until):
            self.handle(item, self.graph)
        return self.graph

    def handle(self, item: Node | Edge, graph: Graph) -> None:
        self.graph = graph
        if not isinstance(item, Node):
            if self.sink is not None:
                self.sink.handle(item, graph)
            return
        node = item
        self.latest.append(node)
        del self.latest[:-100]
        self._pending_chat.append(node)
        self._dirty = True
        if self.app is not None:
            if self.refresh_interval_s == 0 or isinstance(node, (DoneOutput, ErrorOutput)):
                self.refresh()
            else:
                self._request_refresh()
        if self.sink is not None:
            self.sink.handle(node, graph)

    def _request_refresh(self) -> None:
        if self.app is None or self._refresh_pending:
            return
        self._refresh_pending = True
        self.app.set_timer(self.refresh_interval_s, self.refresh)

    def refresh(self) -> None:
        """Immediately render the latest accumulated state."""
        self._refresh_pending = False
        if self.app is None or not self._dirty:
            return
        self._dirty = False
        self.app.refresh_dashboard()

    def close(self) -> None:
        if self.app is not None:
            self._dirty = True
            self.refresh()
        if self.sink is not None:
            self.sink.close()

    def run(self, *, query: str | None = None) -> Graph | None:
        """Open the dashboard. Send, Run, and Step drive the flow from :meth:`init`.

        ``query`` starts a first turn as soon as the dashboard mounts, which is
        what ``rlmflow run "fix the tests"`` wants: the run underway, and the
        panels still there to watch it.
        """
        if self.flow is None:
            raise RuntimeError("FlowTUI needs init(flow) before run().")
        try:
            from rich.panel import Panel
            from textual import work
            from textual.app import App, ComposeResult
            from textual.binding import Binding
            from textual.containers import Horizontal, Vertical, VerticalScroll
            from textual.widgets import (
                Button,
                Footer,
                Header,
                RichLog,
                Static,
                Tab,
                TabbedContent,
                TabPane,
                Tabs,
                TextArea,
                Tree,
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
            #agent-tabs { height: 3; }
            #prompt { height: 5; margin-top: 1; border: round $primary; }
            #context { height: 4; margin-top: 1; border: round $primary; }
            #send-row { height: 3; margin-top: 1; align: right middle; }
            #send { min-width: 12; }
            #tree { height: 1fr; border: round $primary; }
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
                self.selected_id: str | None = None
                self._tab_ids: tuple[str, ...] = ()
                self._tree_ids: tuple[str, ...] = ()

            def compose(self) -> ComposeResult:
                yield Header(show_clock=True)
                with Horizontal(id="main"):
                    with Vertical(id="chat-column"):
                        yield Tabs(Tab("All", id="tab-all"), id="agent-tabs")
                        yield RichLog(id="chat", wrap=True, markup=False, highlight=True)
                        yield TextArea(
                            id="prompt",
                            soft_wrap=True,
                            placeholder="Query…",
                        )
                        yield TextArea(
                            id="context",
                            soft_wrap=True,
                            placeholder="Optional context bound as `context`",
                        )
                        with Horizontal(id="send-row"):
                            yield Button("Send", id="send", variant="primary")
                    with Vertical(id="side-column"):
                        yield Tree("run", id="tree")
                        with TabbedContent(initial="overview-tab", id="tabs"):
                            with TabPane("Overview", id="overview-tab"):
                                with VerticalScroll():
                                    yield Static(id="overview")
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
                        "Enter a query and optional context, then Send (or Ctrl+S). "
                        "Ctrl+R resumes to completion; Ctrl+T advances to the next "
                        "streamed Node.",
                        title="ready",
                        border_style="cyan",
                    )
                )
                self.refresh_dashboard()
                if query:
                    self._drive(query=query, inputs=None, until="done")

            def refresh_dashboard(self) -> None:
                graph = consumer.graph
                if graph is None:
                    for selector in (
                        "#overview",
                        "#agents",
                        "#counts",
                        "#waiting",
                        "#errors",
                        "#latest",
                    ):
                        self.query_one(selector, Static).update("No run yet.")
                    return

                agents = graph.nodes(AgentStart)
                self.query_one("#overview", Static).update(overview_table(graph, consumer.busy))
                self.query_one("#agents", Static).update(agent_table(graph))
                self.query_one("#counts", Static).update(node_counts_table(graph))
                self.query_one("#waiting", Static).update(waiting_table(graph))
                self.query_one("#errors", Static).update(error_table(graph))
                self.query_one("#latest", Static).update(latest_table(graph, consumer.latest))
                self._sync_tree(graph)
                self.call_later(self._sync_tabs, agents)
                self._append_chat()

            def _visible(self, node: Node) -> bool:
                if not _in_chat(node):
                    return False
                if self.selected_id is None:
                    return True
                return node.agent_id == self.selected_id

            def _append_chat(self) -> None:
                log = self.query_one("#chat", RichLog)
                graph = consumer.graph
                if graph is None:
                    return
                pending: list[Node] = [
                    agent for agent in top_level(graph) if agent.id not in self.seen
                ]
                pending.extend(consumer._pending_chat)
                consumer._pending_chat.clear()
                for node in pending:
                    if node.id in self.seen or not self._visible(node):
                        continue
                    self.seen.add(node.id)
                    log.write(node_panel(graph, node))

            def _rewrite_chat(self) -> None:
                log = self.query_one("#chat", RichLog)
                log.clear()
                self.seen.clear()
                graph = consumer.graph
                if graph is None:
                    return
                for node in graph.nodes():
                    if self._visible(node):
                        self.seen.add(node.id)
                        log.write(node_panel(graph, node))

            def _select(self, agent_id: str | None) -> None:
                if agent_id == self.selected_id:
                    return
                self.selected_id = agent_id
                tabs = self.query_one("#agent-tabs", Tabs)
                target = "tab-all" if agent_id is None else f"tab-{agent_id}"
                if tabs.active != target:
                    tabs.active = target
                self._rewrite_chat()

            def _sync_tree(self, graph: Graph) -> None:
                tree = self.query_one("#tree", Tree)
                ids = tuple(agent.id for agent in graph.nodes(AgentStart))
                if ids == self._tree_ids:

                    def relabel(node, agent: AgentStart) -> None:
                        node.set_label(_status(graph, agent))
                        node.data = agent
                        children = reads.children(graph, agent)
                        for child_node, child in zip(node.children, children, strict=False):
                            relabel(child_node, child)

                    for branch, top in zip(tree.root.children, top_level(graph), strict=False):
                        relabel(branch, top)
                    return
                tree.clear()
                tree.root.expand()

                def add(parent, agent: AgentStart) -> None:
                    node = parent.add(_status(graph, agent), data=agent)
                    node.expand()
                    for child in reads.children(graph, agent):
                        add(node, child)

                for top in top_level(graph):
                    add(tree.root, top)
                self._tree_ids = ids

            async def _sync_tabs(self, agents: list[AgentStart]) -> None:
                tabs = self.query_one("#agent-tabs", Tabs)
                ids = tuple(agent.id for agent in agents)
                if ids == self._tab_ids:
                    return
                have = {tab.id for tab in tabs.query(Tab)}
                want = {"tab-all", *(f"tab-{agent.id}" for agent in agents)}
                for tab_id in have - want:
                    await tabs.remove_tab(tab_id)
                for agent in agents:
                    tab_id = f"tab-{agent.id}"
                    if tab_id not in have:
                        label = agent.config.name
                        await tabs.add_tab(Tab(label, id=tab_id))
                self._tab_ids = ids

            def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
                agent = event.node.data
                if isinstance(agent, AgentStart):
                    self._select(agent.id)

            def on_tabs_tab_activated(self, event: Tabs.TabActivated) -> None:
                tab_id = event.tab.id if event.tab is not None else "tab-all"
                if tab_id == "tab-all" or tab_id is None:
                    self._select(None)
                elif tab_id.startswith("tab-"):
                    self._select(tab_id.removeprefix("tab-"))

            def on_button_pressed(self, event: Button.Pressed) -> None:
                if event.button.id == "send":
                    self.action_submit_prompt()

            def action_submit_prompt(self) -> None:
                prompt = self.query_one("#prompt", TextArea)
                context = self.query_one("#context", TextArea)
                query = prompt.text.strip()
                if not query:
                    return
                inputs = {"context": context.text.strip()} if context.text.strip() else None
                prompt.text = ""
                context.text = ""
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
                if consumer.graph is None and query is None:
                    self.query_one("#chat", RichLog).write(
                        Panel("Enter a query first.", border_style="yellow")
                    )
                    return
                consumer.busy = True
                self.query_one("#send", Button).disabled = True
                self.refresh_dashboard()
                try:
                    graph = await consumer.turn(query=query, inputs=inputs, until=until)
                    if graph is not None:
                        consumer.graph = graph
                except Exception as exc:  # noqa: BLE001 - report failures in the UI
                    self.query_one("#chat", RichLog).write(
                        Panel(f"{type(exc).__name__}: {exc}", border_style="red")
                    )
                finally:
                    consumer.busy = False
                    self.query_one("#send", Button).disabled = False
                    self.refresh_dashboard()

        self.app = Dashboard()
        try:
            self.app.run()
        finally:
            self.app = None
            if self.sink is not None:
                self.sink.close()
        return self.graph


def overview_table(graph: Graph, busy: bool = False):
    from rich.table import Table

    agents = graph.nodes(AgentStart)
    usage = reads.usage(graph)
    finished = all(reads.terminal(graph, agent) for agent in top_level(graph))
    table = Table.grid(expand=True)
    table.add_column(style="dim")
    table.add_column(justify="right")
    table.add_row("status", "running" if busy else ("done" if finished else "ready"))
    table.add_row("agents", str(len(agents)))
    table.add_row("nodes", str(len(graph.nodes())))
    table.add_row("max depth", str(max(reads.depth(graph, agent) for agent in agents)))
    table.add_row("tokens in", str(usage.input_tokens))
    table.add_row("tokens out", str(usage.output_tokens))
    return table


def agent_table(graph: Graph):
    from rich.table import Table

    table = Table(show_header=True, header_style="bold dim", expand=True)
    table.add_column("agent", overflow="fold")
    table.add_column("status")
    table.add_column("frontier")
    table.add_column("turns", justify="right")
    for agent in graph.nodes(AgentStart):
        history = reads.history(graph, agent)
        table.add_row(
            reads.path(graph, agent),
            "done" if reads.terminal(graph, agent) else "active",
            history[-1].type,
            str(sum(isinstance(node, LLMOutput) for node in history)),
        )
    return table


def node_counts_table(graph: Graph):
    from rich.table import Table

    counts = Counter(node.type for node in graph.nodes())
    table = Table(show_header=True, header_style="bold dim", expand=True)
    table.add_column("node")
    table.add_column("count", justify="right")
    for kind, count in sorted(counts.items()):
        table.add_row(kind, str(count))
    return table


def waiting_table(graph: Graph):
    from rich.table import Table

    table = Table(show_header=True, header_style="bold dim", expand=True)
    table.add_column("agent")
    table.add_column("children")
    found = False
    for agent in graph.nodes(AgentStart):
        frontier = reads.frontier(graph, agent)
        if not isinstance(frontier, ExecAction):
            continue
        children = launched_by(graph, frontier)
        if not children:
            continue
        active = sum(not reads.terminal(graph, child) for child in children)
        table.add_row(reads.path(graph, agent), f"{active}/{len(children)} running")
        found = True
    if not found:
        table.add_row("-", "none")
    return table


def error_table(graph: Graph):
    from rich.table import Table

    errors = graph.nodes(ErrorOutput)
    table = Table(show_header=True, header_style="bold dim", expand=True)
    table.add_column("agent")
    table.add_column("error", overflow="fold")
    for node in errors:
        table.add_row(_path(graph, node), _one_line(node.content, 80))
    if not errors:
        table.add_row("-", "none")
    return table


def latest_table(graph: Graph, nodes: list[Node], limit: int = 12):
    from rich.table import Table

    table = Table(show_header=True, header_style="bold dim", expand=True)
    table.add_column("agent", overflow="fold")
    table.add_column("node")
    table.add_column("ms", justify="right")
    for node in nodes[-limit:]:
        duration = node.timing().get("duration_ms")
        table.add_row(
            _path(graph, node),
            node.type,
            "-" if duration is None else str(duration),
        )
    if not nodes:
        table.add_row("-", "-", "-")
    return table


def _path(graph: Graph, node: Node) -> str:
    return reads.path(graph, graph.node(node.agent_id))


def _in_chat(node: Node) -> bool:
    """What the chat pane shows: the query, the model, what it printed, errors, the answer.

    The opening query lives on ``AgentStart``, not a ``UserQuery`` — the stream
    never yields a top-level agent, so chat reads it from the graph. ``ExecAction``
    stays out (that is the code, not the output). Empty ``ExecOutput`` stays out
    too.
    """
    if isinstance(node, ExecOutput):
        return bool(str(node.content or "").strip())
    return isinstance(node, (AgentStart, UserQuery, LLMOutput, ErrorOutput, DoneOutput))


def node_panel(graph: Graph, node: Node):
    from rich.panel import Panel

    if isinstance(node, ErrorOutput):
        style = "red"
    elif isinstance(node, DoneOutput):
        style = "green"
    elif isinstance(node, ExecOutput):
        style = "yellow"
    elif isinstance(node, (AgentStart, UserQuery)):
        style = "white"
    else:
        style = "cyan"

    if isinstance(node, DoneOutput):
        body = node.result
        kind = node.type
    elif isinstance(node, AgentStart):
        body = node.content
        args = node.args
        if args:
            listed = ", ".join(
                f"{name} ({len(value)} chars)" if isinstance(value, str) else name
                for name, value in args.items()
            )
            body = f"{body}\n\n[ARGS: {listed}]"
        kind = "query"
    elif isinstance(node, UserQuery):
        body = node.content
        kind = "query"
    else:
        body = node.content
        kind = node.type

    return Panel(
        str(body or f"({kind})"),
        title=f"{_path(graph, node)} · {kind}",
        border_style=style,
    )


def _one_line(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


__all__ = [
    "FlowTUI",
    "agent_table",
    "error_table",
    "latest_table",
    "node_counts_table",
    "node_panel",
    "overview_table",
    "waiting_table",
]
