"""A live Gradio viewer for the Shepherd backtrack-and-branch example.

The shepherd loop runs on a background asyncio thread and writes its progress
into a thread-safe :class:`Dashboard`; the Gradio UI polls that snapshot a few
times a second and redraws. You watch the story unfold: the worker plays forward
and jams a box into a wall, the shepherd proposes N routes, the branches fan out
side by side, and the winner is picked.

It reuses the example's ``PanelViewer`` aggregation verbatim: that viewer already
tracks one live board per lane (worker + each branch), so we just point its
``sink`` at :meth:`Dashboard.set_panels` instead of the terminal. Nothing here is
required by the core example — ``shepherd.py`` runs headless without it.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Dashboard:
    """Thread-safe view model shared between the run thread and the Gradio poll.

    The shepherd loop calls the mutating hooks; the UI only ever reads
    :meth:`snapshot`. ``panels`` mirrors the terminal ``PanelViewer`` (a
    ``(label, board)`` per lane); the rest are phase annotations.
    """

    status: str = "starting…"
    panels: list[tuple[str, str]] = field(default_factory=list)
    proposals: list[tuple[int, str]] = field(default_factory=list)
    picked: str = ""
    log_lines: list[str] = field(default_factory=list)
    done: bool = False
    error: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def set_status(self, status: str) -> None:
        with self._lock:
            self.status = status

    def set_panels(self, panels: list[tuple[str, str]]) -> None:
        with self._lock:
            self.panels = list(panels)

    def set_proposals(self, proposals: list[tuple[int, str]]) -> None:
        with self._lock:
            self.proposals = list(proposals)

    def set_picked(self, label: str) -> None:
        with self._lock:
            self.picked = label

    def log(self, line: str) -> None:
        with self._lock:
            self.log_lines.append(line)

    def finish(self, status: str, error: str | None = None) -> None:
        with self._lock:
            self.status = status
            self.error = error
            self.done = True

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "status": self.status,
                "panels": list(self.panels),
                "proposals": list(self.proposals),
                "picked": self.picked,
                "log": list(self.log_lines),
                "done": self.done,
                "error": self.error,
            }


_PRE = (
    "<pre style='font-family:ui-monospace,Menlo,Consolas,monospace;"
    "font-size:15px;line-height:1.15;letter-spacing:3px;margin:0;"
    "background:#0d1117;color:#e6edf3;padding:10px 12px;border-radius:8px;"
    "display:inline-block'>{body}</pre>"
)


def _board_html(board: str) -> str:
    if not board:
        return "<em>waiting…</em>"
    # Prefer the tiled sprite image; fall back to a monospace board if Pillow or
    # the vendored tiles are unavailable.
    try:
        import sprites

        uri = sprites.data_uri(board, tile=28)
    except Exception:  # noqa: BLE001 - a viewer must never crash the run
        uri = None
    if uri:
        return (
            f"<img src='{uri}' style='image-rendering:pixelated;"
            "border-radius:8px;display:inline-block' alt='sokoban board'>"
        )
    safe = board.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return _PRE.format(body=safe)


def _panels_html(panels: list[tuple[str, str]], picked: str) -> str:
    if not panels:
        return "<em>waiting for the worker…</em>"
    cells = []
    for label, board in panels:
        is_pick = bool(picked) and label.split()[:1] == [picked]
        solved = "SOLVED" in label or "solved" in label
        color = "#d29922" if is_pick else ("#2ea043" if solved else "#8b949e")
        ring = "outline:2px solid #d29922;outline-offset:3px;" if is_pick else ""
        badge = " ◀ picked" if is_pick else ""
        cells.append(
            f"<div style='display:inline-block;vertical-align:top;margin:8px;{ring}'>"
            f"<div style='color:{color};font-weight:600;margin-bottom:4px'>{label}{badge}</div>"
            f"{_board_html(board)}</div>"
        )
    return "<div style='white-space:nowrap;overflow-x:auto'>" + "".join(cells) + "</div>"


def _proposals_md(proposals: list[tuple[int, str]]) -> str:
    if not proposals:
        return "_the shepherd has not proposed yet…_"
    blocks = [
        f"**branch {i}** · rewind {rewind}\n\n{order}"
        for i, (rewind, order) in enumerate(proposals)
    ]
    return "\n\n---\n\n".join(blocks)


def launch(
    run_factory: Callable[[Dashboard], Awaitable[None]],
    *,
    host: str = "127.0.0.1",
    port: int = 7860,
    share: bool = False,
) -> None:
    """Launch the live viewer, running ``run_factory(dashboard)`` in the background.

    ``run_factory`` is an async callable that drives the shepherd run and updates
    the passed :class:`Dashboard`. Blocks on the Gradio server until the tab is
    closed / Ctrl+C.
    """
    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover - optional extra.
        raise ImportError(
            "The --gradio viewer needs gradio: `pip install gradio`. No extra ships it — "
            "this example is the only thing in the repo that uses it."
        ) from exc

    dashboard = Dashboard()

    def _run() -> None:
        try:
            asyncio.run(run_factory(dashboard))
        except Exception as exc:  # noqa: BLE001 - surface crashes into the UI, don't die silently
            dashboard.finish("crashed", error=f"{type(exc).__name__}: {exc}")

    threading.Thread(target=_run, daemon=True).start()

    def poll() -> tuple[str, str, str, str]:
        s = dashboard.snapshot()
        header = f"### {s['status']}"
        if s["error"]:
            header += f"\n\n**error:** `{s['error']}`"
        log = "\n".join(s["log"][-30:])
        return (
            header,
            _panels_html(s["panels"], s["picked"]),
            _proposals_md(s["proposals"]),
            f"```\n{log}\n```" if log else "",
        )

    with gr.Blocks(title="Shepherd · backtrack & branch") as demo:
        gr.Markdown("# 🐑 Shepherd — rewind a stuck worker, then branch in parallel")
        status = gr.Markdown("### starting…")
        gr.Markdown("#### Worker & recovery branches (live, push-by-push)")
        panels = gr.HTML()
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("#### Shepherd proposals")
                proposals = gr.Markdown()
            with gr.Column(scale=1):
                gr.Markdown("#### Log")
                log = gr.Markdown()

        timer = gr.Timer(0.5)
        timer.tick(poll, outputs=[status, panels, proposals, log])

    demo.launch(server_name=host, server_port=port, share=share)


__all__ = ["Dashboard", "launch"]
