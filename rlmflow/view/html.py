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
from xml.sax.saxutils import escape

from rlmflow.graph.nodes import AgentStart, Node
from rlmflow.view.figure import BG, DIM, INK, figure_title, graph_svg, node_color
from rlmflow.view.steps import Step, steps, timeline

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


def step_svg(root: AgentStart, ordered: list[Node], upto: int, **kwargs: object) -> str:
    """The graph drawn in full, with everything after ``upto`` faded back."""
    return graph_svg(
        ordered,
        title=figure_title(root, ordered[: upto + 1]),
        dim_after=upto + 1,
        highlight=ordered[upto].id,
        **kwargs,  # type: ignore[arg-type]
    )


def _meta(step: Step) -> str:
    bits = [step.agent, f"+{step.elapsed:.2f}s"]
    if step.node.started_at and step.node.finished_at:
        bits.append(f"ran {(step.node.finished_at - step.node.started_at) * 1000:.0f}ms")
    return escape(" · ".join(bits))


def render_html(root: AgentStart, *, title: str = "") -> str:
    """The whole run as a single-file stepper."""
    ordered = timeline(root)
    walked = steps(root)
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
    figure = graph_svg(ordered, title=root.config.path, interactive=True)
    heading = title or f"{root.config.path} · {len(ordered)} nodes"
    tokens = root.tokens()
    agents = sum(1 for node in ordered if isinstance(node, AgentStart)) - 1
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


def save_html(root: AgentStart, path: str | Path, *, title: str = "") -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(root, title=title), encoding="utf-8")
    return out


def save_svg(root: AgentStart, path: str | Path, *, step: int | None = None) -> Path:
    """The figure as SVG — the whole run, or the graph as of one step."""
    ordered = timeline(root)
    if step is None:
        svg = graph_svg(ordered, title=figure_title(root, ordered))
    else:
        index = max(0, min(len(ordered) - 1, step))
        svg = step_svg(root, ordered, index)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(svg, encoding="utf-8")
    return out
