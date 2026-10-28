"""Build a deep agent tree by hand and plot it.

No Flow, no LLM — just graph ops:

    python examples/graph/deep_tree.py
    python examples/graph/deep_tree.py --depth 3 --branch 2 --show
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rlmflow import AGENT_START, CODE, DONE, LLM_OUTPUT, Graph, LLMUsage, Node
from rlmflow.graph.graph import FROM, NEXT

EXAMPLES_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = EXAMPLES_DIR / "_runs" / "deep-tree"


def grow(g: Graph, agent: Node, *, depth: int, branch: int) -> None:
    """Grow ``agent`` into a depth-``depth`` tree with ``branch`` children per fanout.

    Children are launched by the model turn whose cell called ``launch_agent``,
    the way a real run records it.
    """
    path = g.path(agent)
    turn = g.append(
        agent,
        Node(
            LLM_OUTPUT,
            text=f"plan at {path}",
            usage=LLMUsage(input_tokens=10, output_tokens=5).as_dict(),
        ),
    )

    if depth <= 0:
        g.append(agent, Node(CODE, code=f"print({path!r})", output=f"ran {path}"))
        g.append(agent, Node(DONE, result=path))
        return

    for index in range(branch):
        child = g.spawn(f"subtask of {path}", by=turn, name=f"c{index}")
        grow(g, child, depth=depth - 1, branch=branch)

    g.append(agent, Node(CODE, code="print(await handles)", output=f"fanout from {path}"))
    g.append(agent, Node(DONE, result=f"merged:{path}"))


def tree_children(g: Graph) -> dict[str, list[Node]]:
    """Each node's successors: its ``next`` node plus any agents its action launched."""
    kids: dict[str, list[Node]] = {node.id: [] for node in g.nodes()}
    for edge in g.edges(kind=NEXT):
        kids[edge.source].append(g.node(edge.target))
    for edge in g.edges(kind=FROM):
        if edge.target in kids and g.node(edge.source).type == AGENT_START:
            kids[edge.target].append(g.node(edge.source))
    return kids


def render_ascii(
    node: Node,
    kids: dict[str, list[Node]],
    *,
    prefix: str = "",
    is_last: bool = True,
) -> list[str]:
    """Render the whole graph below ``node`` as an ASCII tree."""
    connector = "└── " if is_last else "├── "
    label = _label(node)
    lines = [f"{prefix}{connector}{label}" if prefix or not is_last else label]
    child_prefix = prefix + ("    " if is_last else "│   ")
    children = kids[node.id]
    for index, child in enumerate(children):
        lines.extend(
            render_ascii(child, kids, prefix=child_prefix, is_last=index == len(children) - 1)
        )
    return lines


def render_agents(g: Graph, agent: Node, *, prefix: str = "", is_last: bool = True) -> list[str]:
    """Render only the agent tree."""
    connector = "└── " if is_last else "├── "
    label = f"{g.path(agent)}  [{g.frontier(agent).type}]"
    lines = [f"{prefix}{connector}{label}" if prefix or not is_last else label]
    child_prefix = prefix + ("    " if is_last else "│   ")
    children = g.children(agent)
    for index, child in enumerate(children):
        lines.extend(
            render_agents(g, child, prefix=child_prefix, is_last=index == len(children) - 1)
        )
    return lines


def _label(node: Node) -> str:
    if node.type == AGENT_START:
        return f"agent:{node.name}"
    text = str(node.get("text") or node.get("output") or node.get("result") or "").replace(
        "\n", " "
    )
    if len(text) > 40:
        text = text[:37] + "..."
    suffix = f" {text!r}" if text else ""
    return f"{node.type}{suffix}"


def _layout(root: Node, kids: dict[str, list[Node]]) -> dict[str, tuple[float, float]]:
    """Assign (x, y) positions keyed by node id."""
    positions: dict[str, tuple[float, float]] = {}
    counter = {"x": 0.0}

    def walk(node: Node, depth: int) -> None:
        children = kids[node.id]
        if not children:
            positions[node.id] = (counter["x"], -float(depth))
            counter["x"] += 1.0
            return
        for child in children:
            walk(child, depth + 1)
        xs = [positions[child.id][0] for child in children]
        positions[node.id] = (sum(xs) / len(xs), -float(depth))

    walk(root, 0)
    return positions


def plot_tree(g: Graph, root: Node, out: Path, *, title: str, show: bool) -> Path:
    """Save a PNG (matplotlib) or SVG fallback of the node tree."""
    kids = tree_children(g)
    nodes = {node.id: node for node in g.nodes()}
    edges = [(parent, child.id) for parent, children in kids.items() for child in children]
    positions = _layout(root, kids)

    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch
    except ImportError:
        return _plot_svg(nodes, positions, edges, out.with_suffix(".svg"), title=title)

    fig, ax = plt.subplots(figsize=(12, 8))
    for parent_id, child_id in edges:
        x0, y0 = positions[parent_id]
        x1, y1 = positions[child_id]
        ax.plot([x0, x1], [y0, y1], color="#888888", linewidth=1.0, zorder=1)

    for node_id, (x, y) in positions.items():
        node = nodes[node_id]
        box = FancyBboxPatch(
            (x - 0.35, y - 0.18),
            0.7,
            0.36,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.0,
            edgecolor="#222222",
            facecolor=_color(node),
            zorder=2,
        )
        ax.add_patch(box)
        ax.text(x, y, _short(node), ha="center", va="center", fontsize=7, zorder=3)

    ax.set_title(title)
    ax.set_axis_off()
    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    pad = 0.8
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    if show:
        plt.show()
    plt.close(fig)
    return out


def _plot_svg(
    nodes: dict[str, Node],
    positions: dict[str, tuple[float, float]],
    edges: list[tuple[str, str]],
    out: Path,
    *,
    title: str,
) -> Path:
    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    scale = 80
    width = (max_x - min_x + 2) * scale
    height = (max_y - min_y + 2) * scale

    def xy(node_id: str) -> tuple[float, float]:
        x, y = positions[node_id]
        return ((x - min_x + 1) * scale, (max_y - y + 1) * scale)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}">',
        f'<text x="16" y="24" font-family="monospace" font-size="16">{title}</text>',
    ]
    for parent_id, child_id in edges:
        x0, y0 = xy(parent_id)
        x1, y1 = xy(child_id)
        parts.append(
            f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" '
            f'stroke="#888" stroke-width="1.5"/>'
        )
    for node_id, node in nodes.items():
        x, y = xy(node_id)
        parts.append(
            f'<rect x="{x - 36:.1f}" y="{y - 14:.1f}" width="72" height="28" '
            f'rx="6" fill="{_color(node)}" stroke="#222"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{y + 4:.1f}" text-anchor="middle" '
            f'font-family="monospace" font-size="10">{_short(node)}</text>'
        )
    parts.append("</svg>")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(parts), encoding="utf-8")
    return out


def _color(node: Node) -> str:
    return {
        AGENT_START: "#cfe8ff",
        CODE: "#ffe6a8",
        DONE: "#c8f7c5",
        LLM_OUTPUT: "#e8d5ff",
    }.get(node.type, "#f0f0f0")


def _short(node: Node) -> str:
    if node.type == AGENT_START:
        return node.name
    return node.type.replace("_", "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", type=int, default=3, help="fanout depth below root")
    parser.add_argument("--branch", type=int, default=2, help="children per fanout")
    parser.add_argument("--show", action="store_true", help="open matplotlib window")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    g = Graph()
    root = g.create_agent("deep tree root")
    grow(g, root, depth=args.depth, branch=args.branch)

    print(f"agents={len(g.nodes(AGENT_START))}  nodes={len(g.nodes())}")
    print()
    print("=== agent tree ===")
    print("\n".join(render_agents(g, root)))
    print()
    print("=== full node tree ===")
    print("\n".join(render_ascii(root, tree_children(g))))

    plot_path = plot_tree(
        g,
        root,
        args.out_dir / "deep_tree.png",
        title=f"deep tree (depth={args.depth}, branch={args.branch})",
        show=args.show,
    )
    print()
    print(f"plot: {plot_path}")


if __name__ == "__main__":
    main()
