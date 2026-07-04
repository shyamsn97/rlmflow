"""Tiny Textual UI for driving a minimal :class:`Flow`."""

from __future__ import annotations

from pathlib import Path

from rflow.minimal.flow import Flow
from rflow.minimal.graph import Graph
from rflow.minimal.rendering import render_tree


def tui(
    flow: Flow,
    *,
    max_steps_per_turn: int | None = None,
    out_dir: str | Path | None = None,
) -> Graph | None:
    """Open a compact prompt-and-tree TUI for a minimal Flow."""

    try:
        from textual import work
        from textual.app import App, ComposeResult
        from textual.binding import Binding
        from textual.containers import Horizontal, Vertical
        from textual.widgets import Footer, Header, RichLog, Static, TextArea
    except ImportError as exc:  # pragma: no cover - optional extra.
        raise ImportError(
            "minimal.tui() requires the TUI extra: `pip install -e \".[tui]\"`."
        ) from exc

    save_dir = Path(out_dir).resolve() if out_dir is not None else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    class MinimalFlowTUI(App[None]):
        TITLE = "rflow.minimal TUI"
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
            self.graph: Graph | None = None
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
            try:
                self.graph = Graph(query=query)
                steps = 0
                async for event in flow.run_streaming(self.graph):
                    steps += 1
                    self._refresh(event.type)
                    if (
                        max_steps_per_turn is not None
                        and steps >= max_steps_per_turn
                    ):
                        self.query_one("#chat", RichLog).write(
                            f"Stopped at step cap ({max_steps_per_turn})."
                        )
                        break
                if self.graph.result():
                    self.query_one("#chat", RichLog).write(self.graph.result())
                self._save_latest()
            finally:
                self.busy = False
                self._refresh("idle")

        def _refresh(self, status: str) -> None:
            if self.graph is None:
                return
            self.query_one("#tree", Static).update(
                f"status={status}\n\n{render_tree(self.graph)}"
            )

        def _save_latest(self) -> None:
            if save_dir is not None and self.graph is not None:
                self.graph.save(save_dir / "graph")

    app = MinimalFlowTUI()
    app.run()
    return app.graph


__all__ = ["tui"]
