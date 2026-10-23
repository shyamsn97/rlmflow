"""One self-contained HTML file that steps through a run.

The figure is drawn once and revealed a node at a time: the node reached is
ringed, everything after it is faded back, and its content sits beside it. The
markers never move between steps, so the graph reads as one thing being built
rather than a flipbook of separate pictures — and a 500-node run stays a file you
can mail instead of half a gigabyte of duplicated drawings.

Everything is inlined — figure, text, and the few lines of script for the arrows
and keyboard nav — so it works with no server and no network.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from rlmflow.graph import reads
from rlmflow.graph.graph import Graph
from rlmflow.graph.nodes import AgentStart, Node
from rlmflow.view.figure import BG, DIM, INK, graph_svg, node_color, run_title, step_svg
from rlmflow.view.timeline import Step, as_graph, node_detail, steps, timeline

_CSS = (
    """
* { box-sizing: border-box; }
body { margin: 0; background: $bg; color: $ink;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif; }
header { padding: 18px 22px 6px; }
h1 { margin: 0; font-size: 15px; font-weight: 600; }
.sub { color: $dim; font-size: 12px; margin-top: 4px; }
main { display: grid; grid-template-columns: minmax(0, 1fr) 380px; gap: 18px;
  padding: 10px 22px 22px; align-items: start; }
.figure { overflow: auto; max-height: 74vh; border: 1px solid #21262d; border-radius: 10px; }
.figure svg { display: block; }
.rlm-n, .rlm-e { transition: opacity 90ms linear; }
.panel { border: 1px solid #21262d; border-radius: 10px; padding: 14px;
  max-height: 74vh; overflow: auto; }
.kind { font-size: 13px; font-weight: 600; }
.meta { color: $dim; font-size: 11.5px; margin: 4px 0 10px; }
pre { margin: 0; white-space: pre-wrap; word-break: break-word; font-size: 11.5px;
  line-height: 1.5; color: #c9d1d9;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
nav { display: flex; align-items: center; gap: 12px; padding: 0 22px 26px; }
button { background: #21262d; color: $ink; border: 1px solid #30363d; border-radius: 7px;
  padding: 6px 13px; font-size: 13px; cursor: pointer; }
button:hover:not(:disabled) { background: #30363d; }
button:disabled { opacity: 0.4; cursor: default; }
.count { color: $dim; font-size: 12px; white-space: nowrap; }
.dots { display: flex; flex-wrap: wrap; gap: 5px; margin-left: auto; max-width: 62%; }
.dot { width: 9px; height: 9px; border-radius: 50%; background: #30363d; border: 0;
  padding: 0; cursor: pointer; }
.dot.on { transform: scale(1.35); }
input[type=range] { margin-left: auto; width: 46%; accent-color: #58a6ff; }
""".replace("$bg", BG)
    .replace("$ink", INK)
    .replace("$dim", DIM)
)

_JS = """
const frames = FRAMES;
const svg = document.querySelector('#fig svg');
const marks = [...svg.querySelectorAll('.rlm-n')];
const wires = [...svg.querySelectorAll('.rlm-e')];
const ring = svg.querySelector('#rlm-ring');
const counts = svg.querySelector('#rlm-counts');
const panel = document.getElementById('panel');
const count = document.getElementById('count');
const dots = document.getElementById('dots');
const slider = document.getElementById('slider');
const prev = document.getElementById('prev');
const next = document.getElementById('next');

// #12 in the address bar is step 12, so a link can point at one.
const fromHash = () => {
  const m = /^#(\\d+)$/.exec(location.hash);
  return m ? +m[1] - 1 : frames.length - 1;
};
let at = fromHash();

function show(i) {
  at = Math.max(0, Math.min(frames.length - 1, i));
  const f = frames[at];
  let edges = 0;
  for (const g of marks) g.style.opacity = (+g.dataset.i <= at) ? 1 : 0.17;
  for (const e of wires) {
    const on = +e.dataset.i <= at;
    e.style.opacity = on ? 1 : 0.2;
    if (on) edges++;
  }
  const here = marks[at];
  if (here) {
    ring.setAttribute('cx', here.dataset.x);
    ring.setAttribute('cy', here.dataset.y);
    ring.setAttribute('r', here.dataset.r);
    const box = svg.parentElement;
    const y = +here.dataset.y;
    if (y < box.scrollTop + 40 || y > box.scrollTop + box.clientHeight - 40) {
      box.scrollTop = Math.max(0, y - box.clientHeight / 2);
    }
  }
  counts.textContent = ' \\u00b7 ' + (at + 1) + ' nodes \\u00b7 ' + edges + ' edges';
  panel.innerHTML = '<div class="kind" style="color:' + f.color + '">' + f.kind + '</div>'
    + '<div class="meta">' + f.meta + '</div><pre>' + f.detail + '</pre>';
  count.textContent = 'step ' + (at + 1) + ' of ' + frames.length;
  prev.disabled = at === 0;
  next.disabled = at === frames.length - 1;
  if (slider) slider.value = at;
  history.replaceState(null, '', '#' + (at + 1));
  if (dots) {
    for (const [j, d] of [...dots.children].entries()) {
      d.className = 'dot' + (j === at ? ' on' : '');
    }
  }
}
prev.onclick = () => show(at - 1);
next.onclick = () => show(at + 1);
if (slider) slider.oninput = () => show(+slider.value);
window.addEventListener('hashchange', () => show(fromHash()));
document.addEventListener('keydown', (e) => {
  if (e.key === 'ArrowLeft') show(at - 1);
  if (e.key === 'ArrowRight') show(at + 1);
  if (e.key === 'Home') show(0);
  if (e.key === 'End') show(frames.length - 1);
});
if (dots) {
  frames.forEach((f, j) => {
    const d = document.createElement('button');
    d.className = 'dot';
    d.style.background = f.color;
    d.title = (j + 1) + ' \\u00b7 ' + f.kind;
    d.onclick = () => show(j);
    dots.appendChild(d);
  });
}
show(at);
"""

# Past this many steps the dot strip is more wall than navigation.
DOT_LIMIT = 120


def _meta(step: Step) -> str:
    bits = [step.agent, f"+{step.elapsed:.2f}s"]
    if step.node.started_at and step.node.finished_at:
        bits.append(f"ran {(step.node.finished_at - step.node.started_at) * 1000:.0f}ms")
    return escape(" · ".join(bits))


def render_html(g: Graph, *, title: str = "") -> str:
    """The whole run as a single-file stepper."""
    ordered = timeline(g)
    walked = steps(g)
    frames = [
        {
            "kind": escape(step.kind),
            "color": node_color(step.node),
            "meta": _meta(step),
            "detail": escape(step.detail or step.summary or "(no content)"),
        }
        for step in walked
    ]
    # The figure is drawn once for every step, so its title names the run rather
    # than a node type that would go stale the moment you press prev.
    name = run_title(g)
    figure = graph_svg(g, title=name, interactive=True)
    heading = title or f"{name} · {len(ordered)} nodes"
    tokens = reads.usage(g)
    agents = sum(1 for node in ordered if isinstance(node, AgentStart) and reads.launcher(g, node))
    turns = sum(1 for node in ordered if node.type == "llm_output")
    sub = (
        f"{agents} sub-agents · {turns} model turns · "
        f"{tokens.input_tokens + tokens.output_tokens:,} tokens"
    )
    scrub = (
        '<div class="dots" id="dots"></div>'
        if len(frames) <= DOT_LIMIT
        else f'<input type="range" id="slider" min="0" max="{max(len(frames) - 1, 0)}" step="1"/>'
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{escape(heading)}</title><style>{_CSS}</style></head>
<body>
<header><h1>{escape(heading)}</h1><div class="sub">{escape(sub)}</div></header>
<main><div class="figure" id="fig">{figure}</div><div class="panel" id="panel"></div></main>
<nav><button id="prev">&larr; prev</button><button id="next">next &rarr;</button>
<span class="count" id="count"></span>{scrub}</nav>
<script>{_JS.replace("FRAMES", json.dumps(frames))}</script>
</body></html>
"""


def save_html(g: Graph, path: str | Path, *, title: str = "") -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(g, title=title), encoding="utf-8")
    return out


def save_svg(g: Graph, path: str | Path, *, step: int | None = None) -> Path:
    """The figure as SVG — the whole run, or the graph as of one step."""
    from rlmflow.view.figure import figure_title

    ordered = timeline(g)
    if step is None:
        svg = graph_svg(g, title=figure_title(g))
    else:
        svg = step_svg(g, max(0, min(len(ordered) - 1, step)))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(svg, encoding="utf-8")
    return out


_MISSING = (
    "open_viewer needs Gradio: pip install 'rlmflow[viewer]'. "
    "For a single self-contained file instead, use rlmflow.view.save_html(graph, 'run.html')."
)

_FRAME = (
    f"overflow:auto;max-height:68vh;border:1px solid #21262d;border-radius:10px;background:{BG}"
)
# Fit the column rather than clip it: the figure is wider than Gradio's grid.
_FIT = "max-width:100%;height:auto;display:block"
_MONO = "font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px"


def figure_html(
    g: Graph,
    at: int,
    *,
    title: str = "",
    height: int = 420,
) -> str:
    """The figure as of step ``at``, boxed and scaled to whatever column holds it."""
    svg = graph_svg(
        g,
        title=title,
        height=height,
        dim_after=at + 1,
        highlight=g.nodes()[at].id,
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


def _transcript(g: Graph, nodes: list[Node], agent_path: str) -> str:
    """One agent's own nodes, in order, as of the step being shown."""
    own = [node for node in nodes if reads.path(g, g.node(node.agent_id)) == agent_path]
    if not own:
        return f"_{escape(agent_path)} has not started yet._"
    blocks = [f"**{agent_path}** · {len(own)} nodes so far"]
    for node in own:
        detail = node_detail(node, limit=1200)
        blocks.append(f"`{node.seq}` **{node.type}**\n\n```\n{detail or '(no content)'}\n```")
    return "\n\n".join(blocks)


def open_viewer(
    source: Graph | str | Path,
    *,
    height: int = 420,
    **launch_kwargs: Any,
) -> Any:
    """Open a run in the browser. Extra keywords go to ``gradio.Blocks.launch``."""
    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover - depends on the extra
        raise ImportError(_MISSING) from exc

    g = as_graph(source)
    walked = steps(g)
    if not walked:
        raise ValueError("nothing to view: that graph has no nodes")
    ordered = [step.node for step in walked]
    agents = [reads.path(g, node) for node in ordered if isinstance(node, AgentStart)]
    tokens = reads.usage(g)
    name = run_title(g)

    def figure(at: int) -> str:
        return figure_html(g, at, title=name, height=height)

    def render(at: float, agent_path: str) -> tuple[str, str, str]:
        index = max(0, min(int(at) - 1, len(walked) - 1))
        return (
            figure(index),
            _panel(walked[index]),
            _transcript(g, ordered[: index + 1], agent_path),
        )

    last = len(walked)
    with gr.Blocks(title=f"{name} · rlmflow", fill_height=True) as app:
        gr.Markdown(
            f"### {name} · {last} steps\n"
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
            value=agents[0],
            label="agent transcript",
            interactive=True,
        )
        transcript_out = gr.Markdown(_transcript(g, ordered, agents[0]))

        outputs = [figure_out, detail_out, transcript_out]
        slider.change(render, [slider, picker], outputs)
        picker.change(render, [slider, picker], outputs)
        back.click(lambda at: max(1, int(at) - 1), slider, slider)
        forward.click(lambda at: min(last, int(at) + 1), slider, slider)

    app.launch(**launch_kwargs)
    return app
