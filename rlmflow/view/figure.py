"""Draw a saved graph as SVG: one marker per node, coloured and shaped by type.

This is the figure the old Plotly viewer drew, rebuilt on the current node model
and without the dependency. The palette, the marker shapes, the tidy-tree layout
and the dark ground are all carried over, so a figure here is recognisably the
same figure. SVG is written by hand because it keeps the output reproducible from
a bare checkout and embeds straight into the HTML stepper.

Layout is computed in abstract units — x is a leaf slot, y is depth — then scaled
into a pixel box, which is what stopped Plotly from turning a hundred-deep chain
into a hundred-screen-tall image.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from xml.sax.saxutils import escape

from rlmflow.graph.nodes import AgentStart, LLMOutput, Node

BG = "#0d1117"
INK = "#e6edf3"
DIM = "#8b949e"
EDGE = "#30363d"
AGENT_LABEL = "#3fb950"

# Hue per node type, from the original viewer.
NODE_COLORS: dict[str, str] = {
    "agent_start": "#58a6ff",  # sky blue   — an agent's entry point
    "user_query": "#79c0ff",  # pale blue  — input handed to it
    "llm_output": "#bc8cff",  # lavender   — a model turn
    "exec_action": "#ff9e64",  # peach      — code about to run
    "exec_output": "#56d4dd",  # cyan       — what came back
    "append_child": "#ffd33d",  # gold       — spawns children, then waits
    "done_output": "#56d364",  # emerald    — terminal success
    "error_output": "#ff7b72",  # coral      — failure
}

# Shape pairs with colour so types stay separable when small, in greyscale, or
# to someone who cannot tell gold from peach.
NODE_SHAPES: dict[str, str] = {
    "agent_start": "circle",
    "user_query": "circle-open",
    "llm_output": "diamond",
    "exec_action": "square",
    "exec_output": "triangle-right",
    "append_child": "star",
    "done_output": "hexagon",
    "error_output": "x",
}

FALLBACK_COLOR = "#8b949e"
FALLBACK_SHAPE = "circle"
SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, monospace"

MARGIN_X, MARGIN_TOP, MARGIN_BOTTOM = 16.0, 44.0, 16.0
LABEL_SIZE = 10.5
LABEL_ADVANCE = 0.6  # monospace glyph width as a share of font size
BASE_DIAMETER = 14.0
MAX_TOKEN_GROWTH = 22.0


@dataclass
class Placed:
    """A node and where it landed."""

    node: Node
    x: float
    y: float
    r: float
    label: str = ""


@dataclass
class Layout:
    placed: list[Placed] = field(default_factory=list)
    edges: list[tuple[float, float, float, float]] = field(default_factory=list)
    width: float = 0.0
    height: float = 0.0
    scale: float = 1.0


def node_color(node: Node) -> str:
    return NODE_COLORS.get(node.type, FALLBACK_COLOR)


def node_shape(node: Node) -> str:
    return NODE_SHAPES.get(node.type, FALLBACK_SHAPE)


def agent_of(node: Node) -> AgentStart | None:
    return node if isinstance(node, AgentStart) else node.parent_agent


def node_label(node: Node) -> str:
    """Agents get named; the rest are identified by colour and shape."""
    if not isinstance(node, AgentStart):
        return ""
    path = node.config.path or node.config.name
    return path.rsplit(".", 1)[-1] if node.config.depth else path


def node_tokens(node: Node) -> int:
    """What this node cost, which is what sizes its marker."""
    if not isinstance(node, LLMOutput):
        return 0
    return int(node.usage.input_tokens) + int(node.usage.output_tokens)


def marker_diameter(node: Node) -> float:
    """``14 + min(22, sqrt(tokens) / 3.5)``, as the Plotly viewer sized them."""
    return BASE_DIAMETER + min(MAX_TOKEN_GROWTH, math.sqrt(node_tokens(node)) / 3.5)


def tidy_tree(nodes: list[Node]) -> dict[str, tuple[float, float]]:
    """Give every subtree a horizontal slot by leaf count; centre parents over it."""
    present = {node.id for node in nodes}
    children: dict[str, list[Node]] = {node.id: [] for node in nodes}
    for node in nodes:
        parent = node.parent
        if parent is not None and parent.id in present:
            children[parent.id].append(node)

    leaves: dict[str, int] = {}
    for origin in nodes:
        if origin.id in leaves:
            continue
        active: set[str] = set()
        stack: list[tuple[Node, bool]] = [(origin, False)]
        while stack:
            node, expanded = stack.pop()
            if node.id in leaves:
                continue
            if expanded:
                active.discard(node.id)
                leaves[node.id] = (
                    sum(leaves.get(kid.id, 1) for kid in children[node.id])
                    if children[node.id]
                    else 1
                )
                continue
            if node.id in active:
                leaves[node.id] = 1
                continue
            active.add(node.id)
            stack.append((node, True))
            stack.extend(
                (kid, False) for kid in reversed(children[node.id]) if kid.id not in leaves
            )

    pos: dict[str, tuple[float, float]] = {}

    cursor = 0.0
    for node in nodes:
        parent = node.parent
        if parent is None or parent.id not in present:
            width = float(max(leaves[node.id], 1))
            pending: list[tuple[Node, float, float, int]] = [(node, cursor, cursor + width, 0)]
            while pending:
                current, left, right, depth = pending.pop()
                if current.id in pos:
                    continue
                pos[current.id] = ((left + right) / 2, float(depth))
                kids = children[current.id]
                total = sum(leaves[kid.id] for kid in kids) or 1
                child_left = left
                placements: list[tuple[Node, float, float, int]] = []
                for kid in kids:
                    child_width = (right - left) * (leaves[kid.id] / total)
                    placements.append((kid, child_left, child_left + child_width, depth + 1))
                    child_left += child_width
                pending.extend(reversed(placements))
            cursor += width + 1.0
    for node in nodes:  # cycles or broken edges, parked rather than dropped
        if node.id not in pos:
            cursor += 1.0
            pos[node.id] = (cursor, 0.0)
    return pos


def legend_width() -> float:
    """How wide the key needs to be, which is the floor on a figure's width."""
    names = list(NODE_COLORS)
    return sum(15 + len(name) * 10 * 0.56 for name in names) + 16 * (len(names) - 1)


def layout_graph(
    nodes: list[Node],
    *,
    width: float = 1024.0,
    height: float = 360.0,
    legend: bool = True,
) -> Layout:
    """Scale the abstract tree into a pixel box that stays a sane size."""
    layout = Layout()
    if not nodes:
        return layout

    pos = tidy_tree(nodes)
    xs = [pos[node.id][0] for node in nodes]
    ys = [pos[node.id][1] for node in nodes]
    span_x = max(max(xs) - min(xs), 0.001)
    span_y = max(max(ys) - min(ys), 0.0)

    legend_h = 30.0 if legend else 0.0
    usable_w = max(width - MARGIN_X * 2, 120.0)
    usable_h = max(height - MARGIN_TOP - MARGIN_BOTTOM - legend_h, 80.0)

    # Clamped so a two-node graph is not stretched across the page and a
    # hundred-deep chain is dense instead of enormous.
    col = min(max(usable_w / span_x, 26.0), 128.0)
    row = min(max(usable_h / span_y, 9.0), 34.0) if span_y else 0.0
    layout.scale = min(1.0, max(0.42, row / 26.0)) if span_y else 1.0

    left, top = min(xs), min(ys)
    for node in nodes:
        ax, ay = pos[node.id]
        layout.placed.append(
            Placed(
                node=node,
                x=MARGIN_X + (ax - left) * col,
                y=MARGIN_TOP + (ay - top) * row,
                r=marker_diameter(node) / 2 * layout.scale,
                label=node_label(node),
            )
        )

    # An agent's name is wider than its marker, so the extents have to include it
    # or the outermost column loses its label off the edge.
    def half(placed: Placed) -> float:
        label = len(placed.label) * LABEL_SIZE * layout.scale * LABEL_ADVANCE / 2
        return max(placed.r, label)

    overflow = MARGIN_X - min(p.x - half(p) for p in layout.placed)
    if overflow > 0:
        for placed in layout.placed:
            placed.x += overflow

    # A single chain has no width of its own, so the canvas floor is whatever the
    # title and key need; content narrower than that gets centred in it.
    content_w = max(p.x + half(p) for p in layout.placed) + MARGIN_X
    floor = (legend_width() + MARGIN_X * 2) if legend else 260.0
    layout.width = max(content_w, floor)
    if layout.width > content_w:
        shift = (layout.width - content_w) / 2
        for placed in layout.placed:
            placed.x += shift

    at = {placed.node.id: placed for placed in layout.placed}
    for placed in layout.placed:
        parent = placed.node.parent
        if parent is not None and parent.id in at:
            source = at[parent.id]
            layout.edges.append((source.x, source.y, placed.x, placed.y))

    layout.height = max(p.y + p.r for p in layout.placed) + MARGIN_BOTTOM + legend_h
    return layout


def marker(x: float, y: float, shape: str, color: str, r: float, *, ring: bool = True) -> str:
    """One marker symbol, outlined in the background colour so overlaps read."""
    stroke = f' stroke="{BG}" stroke-width="{1.5 if r > 4 else 1:.1f}"' if ring else ""
    if shape == "circle":
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{color}"{stroke}/>'
    if shape == "circle-open":
        return (
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{max(r - 1.2, 1.4):.1f}" fill="{BG}" '
            f'stroke="{color}" stroke-width="{max(1.6, r * 0.42):.1f}"/>'
        )
    if shape == "square":
        return (
            f'<rect x="{x - r:.1f}" y="{y - r:.1f}" width="{r * 2:.1f}" '
            f'height="{r * 2:.1f}" rx="1.2" fill="{color}"{stroke}/>'
        )
    if shape == "diamond":
        pts = f"{x:.1f},{y - r:.1f} {x + r:.1f},{y:.1f} {x:.1f},{y + r:.1f} {x - r:.1f},{y:.1f}"
        return f'<polygon points="{pts}" fill="{color}"{stroke}/>'
    if shape == "triangle-right":
        pts = f"{x - r * 0.85:.1f},{y - r:.1f} {x + r:.1f},{y:.1f} {x - r * 0.85:.1f},{y + r:.1f}"
        return f'<polygon points="{pts}" fill="{color}"{stroke}/>'
    if shape == "star":
        pts = []
        for step in range(10):
            radius = r * 1.35 if step % 2 == 0 else r * 0.54
            angle = -math.pi / 2 + step * math.pi / 5
            pts.append(f"{x + radius * math.cos(angle):.1f},{y + radius * math.sin(angle):.1f}")
        return f'<polygon points="{" ".join(pts)}" fill="{color}"{stroke}/>'
    if shape == "hexagon":
        pts = []
        for step in range(6):
            angle = step * math.pi / 3
            pts.append(f"{x + r * math.cos(angle):.1f},{y + r * math.sin(angle):.1f}")
        return f'<polygon points="{" ".join(pts)}" fill="{color}"{stroke}/>'
    if shape == "x":
        d = r * 0.82
        return (
            f'<path d="M {x - d:.1f} {y - d:.1f} L {x + d:.1f} {y + d:.1f} '
            f'M {x + d:.1f} {y - d:.1f} L {x - d:.1f} {y + d:.1f}" stroke="{color}" '
            f'stroke-width="{max(1.8, r * 0.5):.1f}" fill="none" stroke-linecap="round"/>'
        )
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{color}"{stroke}/>'


def hover_text(node: Node) -> str:
    """Identity only — the detail panel carries the content."""
    agent = agent_of(node)
    rows = [f"{agent.config.path if agent else 'root'} · {node.type}"]
    tokens = node_tokens(node)
    if tokens:
        usage = node.usage  # type: ignore[attr-defined]
        rows.append(f"tokens {tokens:,} (in {usage.input_tokens:,} / out {usage.output_tokens:,})")
    return "\n".join(rows)


def graph_svg(
    nodes: list[Node],
    *,
    title: str = "execution graph",
    width: float = 1024.0,
    height: float = 360.0,
    dim_after: int | None = None,
    highlight: str = "",
    legend: bool = True,
    interactive: bool = False,
) -> str:
    """Render ``nodes`` as an SVG figure.

    ``dim_after`` fades everything from that index on, which is how the stepper
    shows a partly-built graph without markers moving between slides.
    ``highlight`` rings one node id: the step you are on.

    With ``interactive`` the markers carry their step index and geometry, and a
    spare ring is parked in the drawing. That lets one figure be stepped through
    by script instead of shipping a separate copy of it per step.
    """
    layout = layout_graph(nodes, width=width, height=height, legend=legend)
    if not layout.placed:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="420" height="80">'
            f'<rect width="420" height="80" fill="{BG}"/>'
            f'<text x="16" y="46" font-family="{SANS}" font-size="13" fill="{DIM}">'
            f"(empty graph)</text></svg>"
        )

    shown = nodes[:dim_after] if dim_after is not None else nodes
    ids = {node.id for node in shown}
    edges_shown = sum(1 for node in shown if node.parent is not None and node.parent.id in ids)
    w, h = layout.width, layout.height
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:.0f}" height="{h:.0f}" '
        f'viewBox="0 0 {w:.0f} {h:.0f}">',
        f'<rect width="{w:.0f}" height="{h:.0f}" fill="{BG}"/>',
        f'<text x="{MARGIN_X:.0f}" y="22" font-family="{SANS}" font-size="12" fill="{INK}">'
        f'<tspan font-weight="700">{escape(title)}</tspan>'
        f'<tspan fill="{DIM}" id="rlm-counts"> · {len(shown)} nodes · '
        f"{edges_shown} edges</tspan></text>",
    ]

    index_of = {placed.node.id: i for i, placed in enumerate(layout.placed)}
    for x0, y0, x1, y1, child in _edges_with_child(layout, index_of):
        faded = dim_after is not None and child >= dim_after
        tag = f' class="rlm-e" data-i="{child}"' if interactive else ""
        parts.append(
            f'<path d="M {x0:.1f} {y0:.1f} L {x1:.1f} {y1:.1f}" stroke="{EDGE}" '
            f'stroke-width="1" fill="none" opacity="{0.2 if faded else 1}"{tag}/>'
        )

    for placed in layout.placed:
        node = placed.node
        index = index_of[node.id]
        faded = dim_after is not None and index >= dim_after
        if interactive:
            parts.append(
                f'<g class="rlm-n" data-i="{index}" data-x="{placed.x:.1f}" '
                f'data-y="{placed.y:.1f}" data-r="{placed.r + 5.5:.1f}">'
            )
        else:
            parts.append(f'<g opacity="{0.17 if faded else 1}">')
        if node.id == highlight:
            parts.append(
                f'<circle cx="{placed.x:.1f}" cy="{placed.y:.1f}" r="{placed.r + 5.5:.1f}" '
                f'fill="none" stroke="{INK}" stroke-width="1.5" opacity="0.85"/>'
            )
        parts.append(f"<title>{escape(hover_text(node))}</title>")
        parts.append(marker(placed.x, placed.y, node_shape(node), node_color(node), placed.r))
        if placed.label:
            size = max(8.5, LABEL_SIZE * layout.scale)
            parts.append(
                f'<text x="{placed.x:.1f}" y="{placed.y - placed.r - 5:.1f}" '
                f'text-anchor="middle" font-family="{MONO}" font-size="{size:.1f}" '
                f'fill="{AGENT_LABEL}">{escape(placed.label)}</text>'
            )
        parts.append("</g>")

    if interactive:
        parts.append(
            f'<circle id="rlm-ring" cx="-99" cy="-99" r="0" fill="none" stroke="{INK}" '
            f'stroke-width="1.5" opacity="0.85"/>'
        )
    if legend:
        parts.append(_legend(w, h - 10))
    parts.append("</svg>")
    return "\n".join(parts)


def _edges_with_child(
    layout: Layout, index_of: dict[str, int]
) -> list[tuple[float, float, float, float, int]]:
    """Edges tagged with the step their child arrives on, so they can appear with it."""
    at = {placed.node.id: placed for placed in layout.placed}
    out = []
    for placed in layout.placed:
        parent = placed.node.parent
        if parent is not None and parent.id in at:
            source = at[parent.id]
            out.append((source.x, source.y, placed.x, placed.y, index_of[placed.node.id]))
    return out


def _legend(width: float, y: float) -> str:
    """Horizontal key, centred under the figure."""
    entries = [(t, c, NODE_SHAPES[t]) for t, c in NODE_COLORS.items()]
    gap, text_size = 16.0, 10.0
    widths = [11 + 4 + len(name) * text_size * 0.56 for name, _, _ in entries]
    x = max((width - (sum(widths) + gap * (len(entries) - 1))) / 2, MARGIN_X)
    out = []
    for (name, color, shape), w in zip(entries, widths):
        out.append(marker(x + 5, y - 3.5, shape, color, 5.0, ring=False))
        out.append(
            f'<text x="{x + 15:.1f}" y="{y:.1f}" font-family="{SANS}" '
            f'font-size="{text_size:.1f}" fill="{DIM}">{escape(name)}</text>'
        )
        x += w + gap
    return "".join(out)


def figure_title(root: AgentStart, nodes: list[Node]) -> str:
    """The bold half of the title: ``root · done_output``."""
    last = nodes[-1].type if nodes else "empty"
    return f"{root.config.path} · {last}"
