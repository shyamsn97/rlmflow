"""Interactive Textual stream UI for driving a :class:`Flow`.

:class:`FlowTUI` is a :class:`~rlmflow.consumers.base.StreamConsumer` like every
other consumer: its :meth:`~FlowTUI.handle` redraws the live tree for each
streamed event. It just *also* owns an interactive Textual app (via
:meth:`~FlowTUI.run`) that accepts queries and drives the flow, feeding its own
events back through ``handle``.
"""

from __future__ import annotations

from pathlib import Path

from rlmflow.consumers.base import StreamConsumer
from rlmflow.consumers.checkpoint import GraphCheckpointer
from rlmflow.consumers.ui import render_tree
from rlmflow.flow import Flow
from rlmflow.graph import Graph
from rlmflow.graph.events import Event


class FlowTUI(StreamConsumer):
    """A stream consumer that renders (and drives) an interactive Flow TUI.

    As a consumer it reacts to events by redrawing the tree panel; as an app
    (``run()``) it opens a compact prompt-and-tree Textual interface, streams
    each submitted query through the flow, and routes those events back into its
    own ``handle`` so the display stays live.
    """

    def __init__(
        self,
        flow: Flow,
        *,
        max_steps_per_turn: int | None = None,
        out_dir: str | Path | None = None,
    ) -> None:
        self.flow = flow
        self.max_steps_per_turn = max_steps_per_turn
        self.save_dir = Path(out_dir).resolve() if out_dir is not None else None
        if self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        self.graph: Graph | None = None
        self.app = None  # the Textual app, set while running

    # -- StreamConsumer API: refresh the live display for each event. --------
    def handle(self, event: Event, graph: Graph | None) -> None:
        if graph is not None:
            self.graph = graph
        self._refresh(event.type)

    def close(self) -> None:
        self._refresh("idle")

    def _refresh(self, status: str) -> None:
        if self.app is None or self.graph is None:
            return
        from textual.widgets import Static

        self.app.query_one("#tree", Static).update(
            f"status={status}\n\n{render_tree(self.graph)}"
        )

    def _save_latest(self) -> None:
        if self.save_dir is not None and self.graph is not None:
            self.graph.save(self.save_dir / "graph")

    # -- Interactive driver. --------------------------------------------------
    def run(self) -> Graph | None:
        """Open the interactive TUI, blocking until quit; return the last graph."""
        try:
            from textual import work
            from textual.app import App, ComposeResult
            from textual.binding import Binding
            from textual.containers import Horizontal, Vertical
            from textual.widgets import Footer, Header, RichLog, Static, TextArea
        except ImportError as exc:  # pragma: no cover - optional extra.
            raise ImportError(
                'FlowTUI requires the TUI extra: `pip install -e ".[tui]"`.'
            ) from exc

        consumer = self

        class MinimalFlowTUI(App[None]):
            TITLE = "rlmflow TUI"
            CSS = """
            Screen { layout: vertical; }
            #main { height: 1fr; }
            #chat { width: 2fr; border: round $primary; }
            #side { width: 1fr; border: round $accent; padding: 0 1; }
            #prompt { height: 5; border: round $primary; }
            """
            BINDINGS = [
                ("ctrl+c", "quit", "Quit"),
                Binding("ctrl+enter", "submit_prompt", "Send", priority=True),
            ]

            def __init__(self) -> None:
                super().__init__()
                self.busy = False

            def compose(self) -> ComposeResult:
                yield Header(show_clock=True)
                with Horizontal(id="main"):
                    yield RichLog(id="chat", wrap=True, markup=False)
                    with Vertical(id="side"):
                        yield Static("No graph yet.", id="tree")
                yield TextArea(
                    id="prompt",
                    soft_wrap=True,
                    placeholder="Query... Ctrl+Enter to send",
                )
                yield Footer()

            def on_mount(self) -> None:
                self.query_one("#chat", RichLog).write(
                    "Type a query below and press Ctrl+Enter."
                )

            def action_submit_prompt(self) -> None:
                if self.busy:
                    self.query_one("#chat", RichLog).write("A run is already active.")
                    return
                area = self.query_one("#prompt", TextArea)
                query = area.text.strip()
                if not query:
                    return
                area.text = ""
                self.busy = True
                self.query_one("#chat", RichLog).write(f"> {query}")
                self._run_query(query)

            @work(exclusive=True)
            async def _run_query(self, query: str) -> None:
                chat = self.query_one("#chat", RichLog)
                checkpointer = (
                    GraphCheckpointer(consumer.save_dir / "graph")
                    if consumer.save_dir is not None
                    else None
                )
                try:
                    consumer.graph = Graph(query=query)
                    steps = 0
                    async for event in consumer.flow.run_streaming(
                        graph=consumer.graph
                    ):
                        steps += 1
                        if checkpointer is not None:
                            checkpointer.handle(event, consumer.graph)
                        consumer.handle(event, consumer.graph)
                        cap = consumer.max_steps_per_turn
                        if cap is not None and steps >= cap:
                            chat.write(f"Stopped at step cap ({cap}).")
                            break
                    if consumer.graph.result():
                        chat.write(consumer.graph.result())
                    consumer._save_latest()
                finally:
                    if checkpointer is not None:
                        checkpointer.close()
                    self.busy = False
                    consumer._refresh("idle")

        self.app = MinimalFlowTUI()
        self.app.run()
        return self.graph


def tui(
    flow: Flow,
    *,
    max_steps_per_turn: int | None = None,
    out_dir: str | Path | None = None,
) -> Graph | None:
    """Convenience wrapper: build a :class:`FlowTUI` and run it."""
    return FlowTUI(flow, max_steps_per_turn=max_steps_per_turn, out_dir=out_dir).run()


__all__ = ["FlowTUI", "tui"]
