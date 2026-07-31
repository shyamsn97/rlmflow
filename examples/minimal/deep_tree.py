"""Build a deep agent tree with ``rlmflow.minimal.nodes`` and plot it.

No Flow, no LLM — just the graph API:

    python examples/minimal/deep_tree.py
    python examples/minimal/deep_tree.py --depth 3 --branch 2 --show
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rlmflow.minimal.nodes import (
    AgentStart,
    DoneOutput,
    ExecAction,
    ExecOutput,
    LLMOutput,
    LLMUsage,
    Node,
    start,
)

EXAMPLES_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = EXAMPLES_DIR / "_runs" / "minimal-deep-tree"


def grow(agent: AgentStart, *, depth: int, branch: int) -> None:
    """Grow ``agent`` into a depth-``depth`` tree with ``branch`` children per fanout.

    Each append goes to the agent's frontier, which is the node the last one
    returned. Children branch off the action that launched them, the way a real
    ``launch_subagents`` turn records it.
    """
    turn = agent.append(
        LLMOutput(
            content=f"plan at {agent.config.path}",
            code=f"# depth={agent.config.depth}",
            usage=LLMUsage(input_tokens=10, output_tokens=5),
        )
    )
    action = turn.append(ExecAction(code=f"print({agent.config.path!r})"))

    if depth <= 0:
        output = action.append(ExecOutput(content=f"ran {agent.config.path}"))
        output.append(DoneOutput(result=agent.config.path))
        return

    for index in range(branch):
        child = action.append(
            AgentStart(
                content=f"subtask of {agent.config.path}",
                config=agent.config.child(f"c{index}"),
            )
        )
        grow(child, depth=depth - 1, branch=branch)

    output = action.append(ExecOutput(content=f"fanout from {agent.config.path}"))
    output.append(DoneOutput(result=f"merged:{agent.config.path}"))


def render_ascii(node: Node, *, prefix: str = "", is_last: bool = True) -> list[str]:
    """Render the full ``children`` graph as an ASCII tree."""
    connector = "└── " if is_last else "├── "
    label = _label(node)
    lines = [f"{prefix}{connector}{label}" if prefix or not is_last else label]
    child_prefix = prefix + ("    " if is_last else "│   ")
    for index, child in enumerate(node.children):
        lines.extend(
            render_ascii(
                child,
                prefix=child_prefix,
                is_last=index == len(node.children) - 1,
            )
        )
    return lines


def render_agents(agent: AgentStart, *, prefix: str = "", is_last: bool = True) -> list[str]:
    """Render only the agent ``sub_agents`` tree."""
    connector = "└── " if is_last else "├── "
    tip = agent.frontier.type
    label = f"{agent.config.path}  [{tip}]  leaves={len(agent.leaves())}"
    lines = [f"{prefix}{connector}{label}" if prefix or not is_last else label]
    child_prefix = prefix + ("    " if is_last else "│   ")
    for index, child in enumerate(agent.sub_agents):
        lines.extend(
            render_agents(
                child,
                prefix=child_prefix,
                is_last=index == len(agent.sub_agents) - 1,
            )
        )
    return lines


def _label(node: Node) -> str:
    if isinstance(node, AgentStart):
        return f"agent:{node.config.path}"
    text = (node.content or "").replace("\n", " ")
    if len(text) > 40:
        text = text[:37] + "..."
    suffix = f" {text!r}" if text else ""
    return f"{node.type}{suffix}"


def _collect(root: Node) -> tuple[dict[str, Node], list[tuple[str, str]]]:
    nodes: dict[str, Node] = {}
    edges: list[tuple[str, str]] = []

    def walk(node: Node) -> None:
        nodes[node.id] = node
        for child in node.children:
            edges.append((node.id, child.id))
            walk(child)

    walk(root)
    return nodes, edges


def _layout(
    root: Node,
    nodes: dict[str, Node],
) -> dict[str, tuple[float, float]]:
    """Assign (x, y) positions keyed by node id."""
    positions: dict[str, tuple[float, float]] = {}
    counter = {"x": 0.0}

    def walk(node: Node, depth: int) -> None:
        kids = node.children
        if not kids:
            positions[node.id] = (counter["x"], -float(depth))
            counter["x"] += 1.0
            return
        for child in kids:
            walk(child, depth + 1)
        xs = [positions[child.id][0] for child in kids]
        positions[node.id] = (sum(xs) / len(xs), -float(depth))

    walk(root, 0)
    assert set(positions) == set(nodes)
    return positions


def plot_tree(root: Node, out: Path, *, title: str, show: bool) -> Path:
    """Save a PNG (matplotlib) or SVG fallback of the node tree."""
    nodes, edges = _collect(root)
    positions = _layout(root, nodes)

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
    if isinstance(node, AgentStart):
        return "#cfe8ff"
    if isinstance(node, ExecAction):
        return "#ffe6a8"
    if isinstance(node, DoneOutput):
        return "#c8f7c5"
    if isinstance(node, LLMOutput):
        return "#e8d5ff"
    return "#f0f0f0"


def _short(node: Node) -> str:
    if isinstance(node, AgentStart):
        return node.config.name
    return node.type.replace("_", "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", type=int, default=3, help="fanout depth below root")
    parser.add_argument("--branch", type=int, default=2, help="children per supervise")
    parser.add_argument("--show", action="store_true", help="open matplotlib window")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    root = start("deep tree root", max_depth=args.depth + 1)
    grow(root, depth=args.depth, branch=args.branch)

    agents = 1 + sum(1 for _ in _iter_agents(root))
    nodes = sum(1 for _ in _iter_nodes(root))
    print(f"agents={agents}  nodes={nodes}  leaves={len(root.leaves())}")
    print()
    print("=== agent tree ===")
    print("\n".join(render_agents(root)))
    print()
    print("=== full node tree ===")
    print("\n".join(render_ascii(root)))

    plot_path = plot_tree(
        root,
        args.out_dir / "deep_tree.png",
        title=f"minimal deep tree (depth={args.depth}, branch={args.branch})",
        show=args.show,
    )
    print()
    print(f"plot: {plot_path}")


def _iter_agents(agent: AgentStart):
    for child in agent.sub_agents:
        yield child
        yield from _iter_agents(child)


def _iter_nodes(node: Node):
    yield node
    for child in node.children:
        yield from _iter_nodes(child)


if __name__ == "__main__":
    main()
