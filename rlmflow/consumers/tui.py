"""Full-screen Textual dashboard as a :class:`~rlmflow.consumers.base.StreamConsumer`.

:class:`FlowTUI` only renders. Call :meth:`FlowTUI.handle` from your own
``async for event in flow.run_streaming(...)`` loop (optionally compose it with
:class:`~rlmflow.consumers.checkpoint.GraphCheckpointer` etc.). For the
interactive prompt UI, pass that loop as a ``drive`` callback to
:meth:`FlowTUI.run`.

Rich/Textual are imported lazily so ``import rlmflow`` stays light unless the
user opens the TUI.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Awaitable, Callable
from typing import Any

from rlmflow.consumers.base import StreamConsumer
from rlmflow.graph import (
    ActionNode,
    DoneOutput,
    ErrorOutput,
    ExecAction,
    ExecOutput,
    Graph,
    LLMOutput,
    Node,
    ResumeAction,
    SupervisingOutput,
    UserQuery,
)
from rlmflow.graph.events import Event
from rlmflow.utils.viewer import NODE_KINDS

#: ``await drive(graph, query=..., inputs=..., until=...) -> graph``
DriveFn = Callable[..., Awaitable[Graph | None]]


class FlowTUI(StreamConsumer):
    """Stream consumer that paints the interactive Flow dashboard.

    Does not own a :class:`~rlmflow.flow.Flow`. Feed it events via ``handle``,
    or open the prompt UI with ``run(drive)`` where ``drive`` is your streaming
    loop.
    """

    def __init__(self) -> None:
        self.graph: Graph | None = None
        self.app: Any = None
        self._seen_nodes: set[str] = set()
        self._busy = False

    def handle(self, event: Event, graph: Graph | None) -> None:
        if graph is not None:
            self.graph = graph
        self._refresh()

    def close(self) -> None:
        self._refresh()

    def _refresh(self) -> None:
        if self.app is not None:
            self.app.refresh_dashboard()

    def run(self, drive: DriveFn) -> Graph | None:
        """Open the interactive TUI; ``drive`` runs each Send / Run / Step.

        ``drive`` should stream the flow itself and call :meth:`handle` on every
        event. The TUI seeds a :class:`~rlmflow.graph.Graph` on the first Send
        (so the dashboard is live before the first event); later Sends pass
        ``query=`` for multi-turn. Example::

            async def drive(graph, *, query=None, inputs=None, until="done"):
                async for event in flow.run_streaming(
                    graph=graph, query=query, inputs=inputs, until=until
                ):
                    ui.handle(event, graph)
                return graph

            ui = FlowTUI()
            ui.run(drive)
        """
        try:
            from rich.console import Group
            from rich.panel import Panel
            from rich.text import Text
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
        except ImportError as exc:  # pragma: no cover - optional extra
            raise ImportError(
                'FlowTUI requires the TUI extra: `pip install -e ".[tui]"`.'
            ) from exc

        consumer = self

        class RichFlowTUI(App[None]):
            """Live chat + graph dashboard; execution is delegated to ``drive``."""

            TITLE = "rlmflow TUI"
            CSS = """
            Screen {
                layout: vertical;
            }
            #main {
                height: 1fr;
            }
            #chat-column {
                width: 3fr;
                min-width: 50;
            }
            #side-column {
                width: 2fr;
                min-width: 42;
            }
            #chat {
                height: 1fr;
                border: round $primary;
                padding: 0 1;
            }
            #tabs {
                height: 1fr;
            }
            #prompt {
                margin-top: 1;
                height: 5;
                border: round $primary;
            }
            #context {
                margin-top: 1;
                height: 5;
                border: round $primary;
            }
            TabPane {
                padding: 0 1;
            }
            #overview-scroll, #tree-scroll, #agents-scroll,
            #counts-scroll, #waiting-scroll, #errors-scroll, #latest-scroll {
                height: 1fr;
                overflow-y: auto;
            }
            #overview, #tree, #agents, #counts, #waiting, #errors, #latest {
                height: auto;
                width: 100%;
            }
            """
            BINDINGS = [
                ("ctrl+c", "quit", "Quit"),
                Binding("ctrl+s", "submit_prompt", "Send", priority=True),
                Binding(
                    "ctrl+enter",
                    "submit_prompt",
                    "Send",
                    priority=True,
                    show=False,
                ),
                ("ctrl+r", "run_until_done", "Run"),
                ("ctrl+t", "step_once", "Step"),
            ]

            def __init__(self) -> None:
                super().__init__()
                self._pending: list[tuple[str | None, dict[str, str] | None, Any]] = []

            def compose(self) -> ComposeResult:
                yield Header(show_clock=True)
                with Horizontal(id="main"):
                    with Vertical(id="chat-column"):
                        yield RichLog(
                            id="chat", wrap=True, markup=False, highlight=True
                        )
                        yield TextArea(
                            id="prompt",
                            soft_wrap=True,
                            placeholder=(
                                "Query... " "(what should the agent do? Ctrl+S to send)"
                            ),
                        )
                        yield TextArea(
                            id="context",
                            soft_wrap=True,
                            placeholder=(
                                "Context... "
                                "(optional supporting text, "
                                "passed as INPUTS['context'])"
                            ),
                        )
                    with Vertical(id="side-column"):
                        with TabbedContent(initial="tree-tab", id="tabs"):
                            with TabPane("Overview", id="overview-tab"):
                                with VerticalScroll(id="overview-scroll"):
                                    yield Static(id="overview")
                            with TabPane("Tree", id="tree-tab"):
                                with VerticalScroll(id="tree-scroll"):
                                    yield Static(id="tree")
                            with TabPane("Agents", id="agents-tab"):
                                with VerticalScroll(id="agents-scroll"):
                                    yield Static(id="agents")
                            with TabPane("Counts", id="counts-tab"):
                                with VerticalScroll(id="counts-scroll"):
                                    yield Static(id="counts")
                            with TabPane("Waiting", id="waiting-tab"):
                                with VerticalScroll(id="waiting-scroll"):
                                    yield Static(id="waiting")
                            with TabPane("Errors", id="errors-tab"):
                                with VerticalScroll(id="errors-scroll"):
                                    yield Static(id="errors")
                            with TabPane("Latest", id="latest-tab"):
                                with VerticalScroll(id="latest-scroll"):
                                    yield Static(id="latest")
                yield Footer()

            def on_mount(self) -> None:
                self.refresh_dashboard()
                self.query_one("#chat", RichLog).write(
                    Panel(
                        "Type a query below, add optional context, then press "
                        "Ctrl+S / Ctrl+Enter. Ctrl+R continues a paused run.",
                        title="ready",
                        border_style="cyan",
                    )
                )

            def action_submit_prompt(self) -> None:
                query_area = self.query_one("#prompt", TextArea)
                context_area = self.query_one("#context", TextArea)
                query = query_area.text.strip()
                if not query:
                    return
                context = context_area.text.strip()
                query_area.text = ""
                self._queue(query, _context_inputs(context), until="done")

            def action_run_until_done(self) -> None:
                if consumer._busy:
                    self._status("A turn is already running.", style="yellow")
                    return
                if consumer.graph is None:
                    self._status("No graph yet. Type a prompt first.", style="yellow")
                    return
                self._queue(None, None, until="done")

            def action_step_once(self) -> None:
                if consumer._busy:
                    self._status("A turn is already running.", style="yellow")
                    return
                if consumer.graph is None:
                    self._status("No graph yet. Type a prompt first.", style="yellow")
                    return
                self._queue(None, None, until="next")

            def _queue(
                self,
                prompt: str | None,
                turn_inputs: dict[str, str] | None,
                *,
                until: Any,
            ) -> None:
                self._pending.append((prompt, turn_inputs, until))
                if prompt:
                    self.query_one("#chat", RichLog).write(
                        Panel(prompt, title="you", border_style="magenta")
                    )
                if not consumer._busy:
                    consumer._busy = True
                    self._drain_queue()
                else:
                    self._status("Queued prompt.", style="cyan")

            @work(exclusive=True)
            async def _drain_queue(self) -> None:
                try:
                    while self._pending:
                        prompt, turn_inputs, until = self._pending.pop(0)
                        await self._advance(prompt, turn_inputs, until=until)
                finally:
                    consumer._busy = False
                    self.refresh_dashboard()

            async def _advance(
                self,
                prompt: str | None,
                turn_inputs: dict[str, str] | None,
                *,
                until: Any,
            ) -> None:
                try:
                    graph = consumer.graph
                    query = prompt
                    inputs = turn_inputs
                    # Seed the graph immediately on first Send so the dashboard
                    # (and Ctrl+R / Ctrl+T) don't think there's "no graph yet"
                    # while the first LLM event is still in flight.
                    if graph is None:
                        if not prompt:
                            return
                        graph = Graph(query=prompt, inputs=dict(turn_inputs or {}))
                        consumer.graph = graph
                        self.refresh_dashboard()
                        query = None
                        inputs = None
                    result = await drive(
                        graph,
                        query=query,
                        inputs=inputs,
                        until=until,
                    )
                    if result is not None:
                        consumer.graph = result
                    self.refresh_dashboard()
                except Exception as exc:  # noqa: BLE001 - surface in the chat pane
                    self._status(f"{type(exc).__name__}: {exc}", style="red")

            def _status(self, message: str, style: str = "cyan") -> None:
                self.query_one("#chat", RichLog).write(
                    Panel(message, title="status", border_style=style)
                )

            def refresh_dashboard(self) -> None:
                graph = consumer.graph
                if graph is None:
                    empty = Panel(
                        "No run yet. Type a prompt in the chat input.",
                        title="rlmflow",
                        border_style="dim",
                    )
                    for wid in (
                        "#overview",
                        "#tree",
                        "#agents",
                        "#counts",
                        "#waiting",
                        "#errors",
                        "#latest",
                    ):
                        self.query_one(wid, Static).update(empty)
                    return

                chat = self.query_one("#chat", RichLog)
                for node_id, bubble in chat_bubbles(graph, seen=consumer._seen_nodes):
                    consumer._seen_nodes.add(node_id)
                    chat.write(bubble)

                self.query_one("#overview", Static).update(
                    Panel(
                        Group(
                            run_stats_table(graph, busy=consumer._busy),
                            Text(""),
                            waiting_table(graph),
                        ),
                        title="overview",
                    )
                )
                self.query_one("#tree", Static).update(render_full_tree_panel(graph))
                self.query_one("#agents", Static).update(
                    Panel(agent_table(graph), title="agents")
                )
                self.query_one("#counts", Static).update(
                    Panel(node_counts_table(graph), title="node counts")
                )
                self.query_one("#waiting", Static).update(
                    Panel(waiting_table(graph), title="waiting")
                )
                self.query_one("#errors", Static).update(
                    Panel(error_table(graph), title="errors")
                )
                self.query_one("#latest", Static).update(
                    Panel(latest_table(graph), title="latest nodes")
                )

        self.app = RichFlowTUI()
        self.app.run()
        return self.graph


def chat_bubbles(
    graph: Graph, *, seen: set[str] | None = None
) -> list[tuple[str, Any]]:
    """Return Rich panels for graph nodes not present in ``seen``."""
    from rich.panel import Panel

    seen = seen or set()
    out: list[tuple[str, Any]] = []
    steps = _global_steps(graph)
    for node in _ordered_nodes(graph):
        if node.id in seen:
            continue
        kind = _kind(node)
        title = _node_title(node, kind, step=steps.get(node.id))
        body = _node_renderable(node)
        out.append(
            (
                node.id,
                Panel(
                    body,
                    title=title,
                    border_style=_node_style(node),
                    padding=(0, 1),
                ),
            )
        )
    return out


def _context_inputs(context: str) -> dict[str, str] | None:
    text = context.strip()
    return {"context": text} if text else None


def render_full_tree_panel(graph: Graph) -> Any:
    """Render the full graph as a scrollable Rich tree diagram."""
    from rich.panel import Panel

    return Panel(
        _render_execution_diagram(graph),
        title="execution tree",
        border_style="green",
    )


def _render_execution_diagram(graph: Graph) -> Any:
    from rich.text import Text
    from rich.tree import Tree

    steps = _global_steps(graph)

    def agent_label(sub: Graph) -> Text:
        label = Text()
        label.append(sub.agent_id or "root", style="bold")
        model = _model_label(sub)
        if model:
            label.append(f" ({model})", style="cyan")
        if sub.query:
            label.append(f" - {_clip_one_line(sub.query, 56)}", style="dim")
        return label

    def node_label(node: Node) -> Text:
        label = Text()
        step = steps.get(node.id)
        step_s = "-" if step is None else str(step)
        kind = _kind(node)
        label.append(f"[{node.seq:>2} | step {step_s}] ", style="dim")
        label.append(kind, style=_node_style(node))
        summary = _diagram_summary(node)
        if summary:
            label.append(f" - {summary}", style="dim")
        return label

    def add_agent(sub: Graph, parent: Tree | None = None) -> Tree:
        branch = (
            Tree(agent_label(sub), guide_style="green")
            if parent is None
            else parent.add(agent_label(sub), guide_style="green")
        )
        state_ids = {node.id for node in sub.nodes}
        sup_for_agent: dict[str, str] = {}
        for node in sub.nodes:
            if isinstance(node, SupervisingOutput):
                for child_id in node.waiting_on:
                    sup_for_agent[child_id] = node.id

        attach_at: dict[str, list[Graph]] = {}
        unplaced: list[Graph] = []
        for child in sub.children.values():
            key = sup_for_agent.get(child.agent_id)
            if key and key in state_ids:
                attach_at.setdefault(key, []).append(child)
            else:
                unplaced.append(child)

        if not sub.nodes:
            branch.add(Text("(no nodes)", style="dim"), guide_style="dim")

        for node in sub.nodes:
            node_branch = branch.add(node_label(node), guide_style="dim")
            for child in attach_at.get(node.id, []):
                add_agent(child, node_branch)
        for child in unplaced:
            add_agent(child, branch)
        return branch

    return add_agent(graph)


def _diagram_summary(node: Node) -> str:
    if isinstance(node, SupervisingOutput):
        return "waiting on " + (", ".join(node.waiting_on) or "-")
    if isinstance(node, DoneOutput):
        return _clip_one_line(node.result or node.output or node.content, 60)
    if isinstance(node, ErrorOutput):
        return _clip_one_line(node.error or node.content or node.output, 60)
    if isinstance(node, UserQuery):
        return _clip_one_line(node.content, 60)
    if isinstance(node, LLMOutput):
        return _clip_one_line(_assistant_body(node), 60)
    if isinstance(node, ExecOutput):
        return _clip_one_line(node.content or node.output, 60)
    if isinstance(node, ResumeAction):
        return "resumed from " + (", ".join(node.resumed_from) or "-")
    return ""


def run_stats_table(graph: Graph, *, busy: bool = False) -> Any:
    """Small table of whole-run counters."""
    from rich.table import Table

    inp, out = graph.tokens()
    nodes = _all_nodes(graph)
    table = Table.grid(expand=True)
    table.add_column(style="dim")
    table.add_column(justify="right")
    table.add_row(
        "status",
        "running" if busy and not graph.finished else _graph_status(graph),
    )
    table.add_row("agents", str(len(graph.agents)))
    table.add_row("nodes", str(len(nodes)))
    table.add_row("max depth", str(max((g.depth for g in graph.walk()), default=0)))
    table.add_row("runnable", ", ".join(_runnable_agents(graph)) or "-")
    table.add_row("tokens in", str(inp))
    table.add_row("tokens out", str(out))
    return table


def agent_table(graph: Graph) -> Any:
    """One row per agent with status, current node, and token counts."""
    from rich.table import Table

    table = Table(show_header=True, header_style="bold dim", expand=True)
    table.add_column("agent", overflow="fold")
    table.add_column("status", no_wrap=True)
    table.add_column("current", no_wrap=True)
    table.add_column("depth", justify="right")
    table.add_column("tokens", justify="right")
    for agent in graph.walk():
        cur = agent.current()
        table.add_row(
            agent.agent_id,
            _agent_status(agent),
            _kind(cur) if cur is not None else "-",
            str(agent.depth),
            str(agent.total_tokens(recursive=False)),
        )
    return table


def node_counts_table(graph: Graph) -> Any:
    """Counts by displayed node kind."""
    from rich.table import Table

    counts = Counter(_kind(node) for node in _all_nodes(graph))
    table = Table(show_header=True, header_style="bold dim", expand=True)
    table.add_column("kind")
    table.add_column("count", justify="right")
    for kind, count in sorted(counts.items()):
        table.add_row(kind, str(count))
    if not counts:
        table.add_row("-", "0")
    return table


def waiting_table(graph: Graph) -> Any:
    """Supervisors currently waiting on child agents."""
    from rich.table import Table

    table = Table(show_header=True, header_style="bold dim", expand=True)
    table.add_column("agent")
    table.add_column("waiting on", overflow="fold")
    rows = 0
    for agent in graph.walk():
        cur = agent.current()
        if isinstance(cur, SupervisingOutput):
            table.add_row(agent.agent_id, ", ".join(cur.waiting_on) or "-")
            rows += 1
    if rows == 0:
        table.add_row("-", "none")
    return table


def error_table(graph: Graph) -> Any:
    """Errors grouped by kind."""
    from rich.table import Table

    errors = [node for node in _all_nodes(graph) if isinstance(node, ErrorOutput)]
    counts = Counter(err.error or "error" for err in errors)
    table = Table(show_header=True, header_style="bold dim", expand=True)
    table.add_column("kind")
    table.add_column("count", justify="right")
    for kind, count in counts.most_common():
        table.add_row(kind, str(count))
    if not counts:
        table.add_row("-", "0")
    return table


def latest_table(graph: Graph, *, limit: int = 8) -> Any:
    """Most recent nodes across all agents."""
    from rich.table import Table

    steps = _global_steps(graph)
    nodes = sorted(
        _all_nodes(graph),
        key=lambda node: (
            steps.get(node.id, -1),
            node.agent_id,
            node.seq,
        ),
    )[-limit:]
    table = Table(show_header=True, header_style="bold dim", expand=True)
    table.add_column("step", justify="right", no_wrap=True)
    table.add_column("agent", overflow="fold")
    table.add_column("kind", no_wrap=True)
    for node in nodes:
        step = steps.get(node.id)
        table.add_row(
            "-" if step is None else str(step),
            node.agent_id,
            _kind(node),
        )
    if not nodes:
        table.add_row("-", "-", "-")
    return table


def _all_nodes(graph: Graph) -> list[Node]:
    nodes: list[Node] = []
    for agent in graph.walk():
        nodes.extend(agent.nodes)
    return nodes


def _ordered_nodes(graph: Graph) -> list[Node]:
    steps = _global_steps(graph)
    return sorted(
        _all_nodes(graph),
        key=lambda node: (
            steps.get(node.id, -1),
            node.agent_id,
            node.seq,
        ),
    )


def _global_steps(graph: Graph) -> dict[str, int]:
    return {node.id: i for i, node in enumerate(graph.execution_order())}


def _kind(node: Node | None) -> str:
    if node is None:
        return "-"
    return NODE_KINDS.get(node.type, node.type)


def _runnable_agents(graph: Graph) -> list[str]:
    ready: list[str] = []
    for agent in graph.walk():
        if agent.finished:
            continue
        cur = agent.current()
        if isinstance(cur, SupervisingOutput):
            continue
        ready.append(agent.agent_id)
    return ready


def _graph_status(graph: Graph) -> str:
    if graph.finished:
        return "finished"
    return "ready" if _runnable_agents(graph) else "waiting"


def _agent_status(agent: Graph) -> str:
    cur = agent.current()
    if cur is None:
        return "empty"
    if isinstance(cur, DoneOutput):
        return "done"
    if isinstance(cur, ErrorOutput):
        return "error"
    if isinstance(cur, SupervisingOutput):
        return "supervising"
    if cur.terminal:
        return "terminal"
    return "ready"


def _model_label(agent: Graph) -> str:
    if agent.model and agent.model != "default":
        return agent.model
    for node in reversed(agent.nodes):
        if isinstance(node, LLMOutput):
            model = node.metadata.get("model")
            if model:
                return str(model)
    return agent.model or "default"


def _node_style(node: Node) -> str:
    if isinstance(node, ErrorOutput):
        return "red"
    if isinstance(node, DoneOutput):
        return "green"
    if isinstance(node, SupervisingOutput):
        return "yellow"
    if isinstance(node, LLMOutput):
        return "magenta"
    if isinstance(node, (ExecAction, ExecOutput, ResumeAction)):
        return "cyan"
    if isinstance(node, UserQuery):
        return "magenta"
    return "dim"


def _node_title(node: Node, kind: str, *, step: int | None = None) -> str:
    step_s = "-" if step is None else str(step)
    return f"{node.agent_id} / {kind} · step {step_s}"


def _node_renderable(node: Node) -> Any:
    from rich.console import Group
    from rich.syntax import Syntax
    from rich.text import Text

    if isinstance(node, ExecAction):
        code = _clip(node.code, limit=6_000) or "# empty code block"
        return Syntax(
            code,
            "python",
            theme="ansi_dark",
            word_wrap=True,
            line_numbers=False,
        )
    if isinstance(node, LLMOutput):
        body = _assistant_body(node)
        if node.code:
            return Group(
                Text(body or "Emitted a REPL block.", style=""),
                Text(
                    f"repl block: {len(node.code)} chars (shown in execute bubble)",
                    style="dim",
                ),
            )
        return Text(body or "(empty assistant reply)")
    body = _node_body(node)
    return Text(body or f"({node.type})")


def _assistant_body(node: LLMOutput) -> str:
    reply = (node.content or "").strip()
    if not node.code:
        return _clip(reply)
    # Avoid showing the same generated code twice: the paired ExecAction gets a
    # dedicated syntax-highlighted bubble.
    if "```" in reply:
        before = reply.split("```", 1)[0].strip()
        after = reply.rsplit("```", 1)[-1].strip()
        reply = "\n\n".join(part for part in (before, after) if part)
    return _clip(reply)


def _node_body(node: Node) -> str:
    if isinstance(node, UserQuery):
        return _clip(node.content)
    if isinstance(node, ResumeAction):
        return "resumed from: " + (", ".join(node.resumed_from) or "-")
    if isinstance(node, SupervisingOutput):
        waiting = ", ".join(node.waiting_on) or "-"
        output = f"\n\n{node.output}" if node.output else ""
        return _clip(f"waiting on: {waiting}{output}")
    if isinstance(node, ErrorOutput):
        return _clip(node.content or node.output or node.error)
    if isinstance(node, DoneOutput):
        return _clip(node.result or node.output or node.content)
    if isinstance(node, ExecOutput):
        return _clip(node.output or node.content)
    if isinstance(node, ActionNode):
        return node.type
    return str(node)


def _clip(value: str, limit: int = 4000) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return text[:limit].rstrip() + f"\n...[truncated {omitted} chars]"


def _clip_one_line(value: str, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


__all__ = [
    "DriveFn",
    "FlowTUI",
    "agent_table",
    "chat_bubbles",
    "error_table",
    "latest_table",
    "node_counts_table",
    "render_full_tree_panel",
    "run_stats_table",
    "waiting_table",
]
