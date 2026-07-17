"""Concise interactive/static graph viewer utilities.

One ``_figure`` (a Plotly swimlane: x = execution step, y = agent lane) powers
both the Gradio stepper (``open_viewer``) and the static exports
(``save_image`` / ``save_steps``) used for docs and blog frames. Plotly + Gradio
come from the ``viewer`` extra; static PNG export also needs ``kaleido`` (the
``image`` extra).
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

from rflow.consumers.ui import _clip, render_tree
from rflow.graph import Graph, SupervisingOutput

ViewSource = Graph | Iterable[Graph] | str | Path

NODE_COLORS = {
    "user_query": "#64748b",
    "llm_output": "#3b82f6",
    "exec_action": "#8b5cf6",
    "exec_output": "#10b981",
    "supervising_output": "#f59e0b",
    "resume_action": "#eab308",
    "error_output": "#ef4444",
    "done_output": "#22c55e",
}


def _graphs_from(source: ViewSource) -> list[Graph]:
    if isinstance(source, Graph):
        return [source]
    if isinstance(source, (str, Path)):
        return [Graph.load(source)]
    graphs = list(source)
    if not graphs:
        raise ValueError("viewer needs at least one Graph")
    return graphs


def _node_text(node: Any) -> str:
    for attr in ("result", "error", "output", "content", "code"):
        value = getattr(node, attr, "")
        if value:
            return _clip(value, limit=60)
    return ""


def _timeline(graph: Graph) -> list[Any]:
    """Nodes in execution order: preorder walk through supervising children."""
    order: list[Any] = []

    def walk(agent: Graph) -> None:
        by_id = {child.agent_id: child for child in agent.children.values()}
        for node in agent.nodes:
            order.append(node)
            if isinstance(node, SupervisingOutput):
                for child_id in node.waiting_on:
                    if child_id in by_id:
                        walk(by_id[child_id])

    walk(graph)
    return order


def _positions(graph: Graph) -> dict[str, tuple[int, int]]:
    lanes = {agent.agent_id: i for i, agent in enumerate(graph.walk())}
    return {
        node.id: (x, -lanes.get(node.agent_id, 0))
        for x, node in enumerate(_timeline(graph))
    }


def replay(graph: Graph) -> list[Graph]:
    """Progressive snapshots: one per execution step, each a truncated copy."""
    order = _timeline(graph)
    snapshots: list[Graph] = []
    for step in range(1, len(order) + 1):
        keep = {node.id for node in order[:step]}
        snapshots.append(_truncate(graph, keep))
    return snapshots or [deepcopy(graph)]


def _truncate(graph: Graph, keep: set[str]) -> Graph:
    clone = deepcopy(graph)

    def prune(agent: Graph) -> None:
        agent.nodes = [node for node in agent.nodes if node.id in keep]
        for child_id, child in list(agent.children.items()):
            prune(child)
            if not child.nodes:
                del agent.children[child_id]

    prune(clone)
    return clone


def _figure(
    graph: Graph,
    positions: dict[str, tuple[int, int]] | None = None,
    *,
    marker_mult: float = 1.0,
    text_mult: float = 1.0,
) -> Any:
    import plotly.graph_objects as go

    pos = positions or _positions(graph)
    agents = list(graph.walk())
    present = {node.id: node for agent in agents for node in agent.nodes}

    edge_x: list[float | None] = []
    edge_y: list[float | None] = []

    def link(a_id: str, b_id: str) -> None:
        if a_id in pos and b_id in pos and a_id in present and b_id in present:
            ax, ay = pos[a_id]
            bx, by = pos[b_id]
            edge_x.extend([ax, bx, None])
            edge_y.extend([ay, by, None])

    for agent in agents:
        for earlier, later in zip(agent.nodes, agent.nodes[1:]):
            link(earlier.id, later.id)
        by_id = {child.agent_id: child for child in agent.children.values()}
        for node in agent.nodes:
            if isinstance(node, SupervisingOutput):
                for child_id in node.waiting_on:
                    child = by_id.get(child_id)
                    if child and child.nodes:
                        link(node.id, child.nodes[0].id)

    xs, ys, colors, hover = [], [], [], []
    for agent in agents:
        for node in agent.nodes:
            if node.id not in pos:
                continue
            x, y = pos[node.id]
            xs.append(x)
            ys.append(y)
            colors.append(NODE_COLORS.get(node.type, "#94a3b8"))
            detail = _node_text(node)
            hover.append(
                f"<b>{node.agent_id}</b> · {node.type}"
                + (f"<br>{detail}" if detail else "")
            )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(color="#cbd5e1", width=1.5),
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="markers",
            hoverinfo="text",
            hovertext=hover,
            marker=dict(
                size=16 * marker_mult, color=colors, line=dict(color="#1e293b", width=1)
            ),
        )
    )
    lane_labels = {-i: agent.agent_id for i, agent in enumerate(agents)}
    fig.update_layout(
        showlegend=False,
        margin=dict(l=10, r=10, t=30, b=10),
        title=f"{graph.agent_id} · {len(present)} steps · {graph.total_tokens()} tokens",
        font=dict(size=int(12 * text_mult)),
        plot_bgcolor="white",
        xaxis=dict(title="step", showgrid=True, gridcolor="#f1f5f9", zeroline=False),
        yaxis=dict(
            tickmode="array",
            tickvals=list(lane_labels),
            ticktext=list(lane_labels.values()),
            zeroline=False,
        ),
    )
    return fig


def _transcript(agent: Graph) -> str:
    header = f"# {agent.agent_id}  (model={agent.model}, depth={agent.depth})"
    lines = [header]
    if agent.query:
        lines.append(f"[query] {_clip(agent.query, limit=200)}")
    if not agent.nodes:
        lines.append("(not started at this step)")
    for node in agent.nodes:
        text = _node_text(node)
        lines.append(f"[{node.type}] {text}" if text else f"[{node.type}]")
    return "\n".join(lines)


def open_viewer(source: ViewSource, **launch_kwargs: object) -> object:
    """Open a Gradio stepper, or print trees if Gradio is unavailable."""
    snapshots = _snapshots(source)
    should_launch = bool(launch_kwargs.pop("launch", True))

    try:
        import gradio as gr
    except ImportError:
        for index, graph in enumerate(snapshots):
            print(f"\n=== step {index + 1}/{len(snapshots)} ===")
            print(render_tree(graph))
        return snapshots

    positions = _positions(snapshots[-1])
    agent_ids = [agent.agent_id for agent in snapshots[-1].walk()]
    last = len(snapshots)

    def render(step: int, agent_id: str):
        graph = snapshots[min(step, last) - 1]
        try:
            agent = graph[agent_id]
        except KeyError:
            agent = graph
        return _figure(graph, positions), _transcript(agent)

    with gr.Blocks(title="rflow viewer") as demo:
        gr.Markdown("# rflow viewer")
        with gr.Row():
            step = gr.Slider(1, last, value=last, step=1, label="Step")
            agent = gr.Dropdown(agent_ids, value=agent_ids[0], label="Agent")
        plot = gr.Plot()
        transcript = gr.Textbox(lines=18, label="Transcript")
        for control in (step, agent):
            control.change(render, [step, agent], [plot, transcript])
        demo.load(render, [step, agent], [plot, transcript])

    if not should_launch:
        return demo
    return demo.launch(**launch_kwargs)


def _snapshots(source: ViewSource) -> list[Graph]:
    graphs = _graphs_from(source)
    return graphs if len(graphs) > 1 else replay(graphs[0])


def save_image(
    source: ViewSource,
    path: str | Path,
    *,
    width: int = 1200,
    height: int = 800,
    scale: float = 2,
    marker_mult: float = 1.0,
    text_mult: float = 1.0,
) -> Path:
    """Write the full graph as one PNG/SVG/PDF (needs the ``image`` extra)."""
    graphs = _graphs_from(source)
    graph = graphs[-1]
    fig = _figure(graph, marker_mult=marker_mult, text_mult=text_mult)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(path), width=width, height=height, scale=scale)
    return path


def save_steps(
    source: ViewSource,
    out_dir: str | Path,
    *,
    width: int = 1200,
    height: int = 800,
    scale: float = 2,
    marker_mult: float = 1.0,
    text_mult: float = 1.0,
) -> list[Path]:
    """Write one frame per execution step (stable layout) for GIFs/blog strips."""
    snapshots = _snapshots(source)
    positions = _positions(snapshots[-1])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for index, graph in enumerate(snapshots):
        fig = _figure(graph, positions, marker_mult=marker_mult, text_mult=text_mult)
        frame = out_dir / f"step_{index:03d}.png"
        fig.write_image(str(frame), width=width, height=height, scale=scale)
        paths.append(frame)
    return paths


__all__ = ["ViewSource", "open_viewer", "replay", "save_image", "save_steps"]
