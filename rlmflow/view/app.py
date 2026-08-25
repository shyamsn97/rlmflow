"""The browser viewer: a run, scrubbable, with the transcript beside it.

``open_viewer`` puts a step slider over the figure, an agent picker beside it, and
the picked agent's transcript underneath, all as of the step you are on. It needs
Gradio (``pip install rlmflow[viewer]``); nothing else here does, which is why it
sits in its own module and imports it inside the call.

For a file to send someone rather than a server to sit in front of, use
``save_html`` — same run, same figure, no dependencies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from rlmflow.graph.nodes import AgentStart, Node
from rlmflow.view.figure import BG, DIM, INK, graph_svg
from rlmflow.view.replay import as_root
from rlmflow.view.steps import Step, node_detail, steps

_MISSING = (
    "open_viewer needs Gradio: pip install 'rlmflow[viewer]'. "
    "For a single self-contained file instead, use rlmflow.view.save_html(root, 'run.html')."
)

_FRAME = (
    f"overflow:auto;max-height:68vh;border:1px solid #21262d;border-radius:10px;background:{BG}"
)
# Fit the column rather than clip it: the figure is wider than Gradio's grid.
_FIT = "max-width:100%;height:auto;display:block"
_MONO = "font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px"


def figure_html(
    ordered: list[Node],
    at: int,
    *,
    title: str = "",
    height: int = 420,
) -> str:
    """The figure as of step ``at``, boxed and scaled to whatever column holds it."""
    svg = graph_svg(
        ordered,
        title=title,
        height=height,
        dim_after=at + 1,
        highlight=ordered[at].id,
    )
    # Styled inline rather than through Gradio's CSS, which moved between major
    # versions; this renders the same on all of them.
    svg = svg.replace("<svg ", f'<svg style="{_FIT}" ', 1)
    return f'<div style="{_FRAME}">{svg}</div>'


def _panel(step: Step) -> str:
    """The node reached, as a titled block of its own content."""
    ran = ""
    if step.node.started_at and step.node.finished_at:
        ran = f" · ran {(step.node.finished_at - step.node.started_at) * 1000:.0f}ms"
    body = escape(step.detail or step.summary or "(no content)")
    return (
        f'<div style="{_MONO}">'
        f'<div style="font-weight:600;font-size:13px">{escape(step.kind)}</div>'
        f'<div style="color:{DIM};font-size:11.5px;margin:2px 0 8px">'
        f"{escape(step.agent)} · +{step.elapsed:.2f}s{ran}</div>"
        f'<pre style="white-space:pre-wrap;word-break:break-word;color:{INK};margin:0">'
        f"{body}</pre></div>"
    )


def _transcript(nodes: list[Node], agent_path: str) -> str:
    """One agent's own nodes, in order, as of the step being shown."""
    own = [node for node in nodes if _path_of(node) == agent_path]
    if not own:
        return f"_{escape(agent_path)} has not started yet._"
    blocks = [f"**{agent_path}** · {len(own)} nodes so far"]
    for node in own:
        detail = node_detail(node, limit=1200)
        blocks.append(f"`{node.seq}` **{node.type}**\n\n```\n{detail or '(no content)'}\n```")
    return "\n\n".join(blocks)


def _path_of(node: Node) -> str:
    agent = node.parent_agent
    return agent.config.path if agent is not None else ""


def open_viewer(
    source: AgentStart | str | Path,
    *,
    height: int = 420,
    **launch_kwargs: Any,
) -> Any:
    """Open a run in the browser. Extra keywords go to ``gradio.Blocks.launch``."""
    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover - depends on the extra
        raise ImportError(_MISSING) from exc

    root = as_root(source)
    walked = steps(root)
    if not walked:
        raise ValueError("nothing to view: that graph has no nodes")
    ordered = [step.node for step in walked]
    agents = [node.config.path for node in ordered if isinstance(node, AgentStart)]
    tokens = root.usage

    def figure(at: int) -> str:
        return figure_html(ordered, at, title=root.config.path, height=height)

    def render(at: float, agent_path: str) -> tuple[str, str, str]:
        index = max(0, min(int(at) - 1, len(walked) - 1))
        return (
            figure(index),
            _panel(walked[index]),
            _transcript(ordered[: index + 1], agent_path),
        )

    last = len(walked)
    with gr.Blocks(title=f"{root.config.path} · rlmflow", fill_height=True) as app:
        gr.Markdown(
            f"### {root.config.path} · {last} steps\n"
            f"{len(agents) - 1} sub-agents · "
            f"{sum(1 for n in ordered if n.type == 'llm_output')} model turns · "
            f"{tokens.input_tokens + tokens.output_tokens:,} tokens — drag the slider to "
            "scrub the run; the ringed node is the step you are on."
        )
        with gr.Row():
            with gr.Column(scale=3):
                figure_out = gr.HTML(figure(last - 1))
            with gr.Column(scale=2):
                detail_out = gr.HTML(_panel(walked[-1]))
        with gr.Row():
            back = gr.Button("← prev", scale=0)
            forward = gr.Button("next →", scale=0)
            slider = gr.Slider(
                minimum=1,
                maximum=last,
                value=last,
                step=1,
                label="step",
                interactive=last > 1,
            )
        picker = gr.Dropdown(
            choices=agents,
            value=root.config.path,
            label="agent transcript",
            interactive=True,
        )
        transcript_out = gr.Markdown(_transcript(ordered, root.config.path))

        outputs = [figure_out, detail_out, transcript_out]
        slider.change(render, [slider, picker], outputs)
        picker.change(render, [slider, picker], outputs)
        back.click(lambda at: max(1, int(at) - 1), slider, slider)
        forward.click(lambda at: min(last, int(at) + 1), slider, slider)

    app.launch(**launch_kwargs)
    return app
