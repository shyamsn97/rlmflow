"""Interactive and static semantic graph visualization.

One Plotly topology powers both the Gradio stepper (``open_viewer``) and the
static exports (``save_image`` / ``save_steps``): each agent's trajectory is a
vertical spine and delegated children fan out from the ``SupervisingOutput``
that owns their lifecycle. Node color + shape encode the node kind. Plotly +
Gradio come from the ``viewer`` extra; static PNG export also needs ``kaleido``
(the ``image`` extra).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from rflow.consumers.ui import _clip, render_tree
from rflow.graph import Graph, SupervisingOutput

ViewSource = Graph | Iterable[Graph] | str | Path

#: node ``type`` -> display kind (several types collapse onto one visual kind).
NODE_KINDS = {
    "user_query": "query",
    "llm_output": "llm",
    "exec_action": "exec",
    "exec_output": "exec",
    "supervising_output": "supervising",
    "resume_action": "resume",
    "error_output": "errored",
    "done_output": "done",
}

#: display kind -> marker color (GitHub-dark palette).
NODE_COLORS = {
    "query": "#58a6ff",
    "llm": "#bc8cff",
    "exec": "#ff9e64",
    "supervising": "#ffd33d",
    "resume": "#56d4dd",
    "done": "#56d364",
    "errored": "#ff7b72",
}

#: display kind -> marker symbol.
NODE_SYMBOLS = {
    "query": "circle",
    "llm": "diamond",
    "exec": "square",
    "supervising": "star",
    "resume": "triangle-right",
    "done": "hexagon",
    "errored": "x",
}

#: An action node is hidden when its paired observation follows it, so the
#: figure reads as one node per obs-to-obs transition. ``resume_action`` stays
#: visible so a resumed parent shows a distinct glyph.
HIDDEN_ACTION_PAIRS = {
    "exec_action": {"exec_output", "supervising_output"},
}

_Y_SPACING = 1.4
_MIN_COLUMN_GAP = 0.75
_BG = "#0d1117"
_AGENT_LABEL_COLOR = "#3fb950"


def _graphs_from(source: ViewSource) -> list[Graph]:
    if isinstance(source, Graph):
        return [source]
    if isinstance(source, (str, Path)):
        return [Graph.load(source)]
    graphs = list(source)
    if not graphs:
        raise ValueError("viewer needs at least one Graph")
    return graphs


def _kind(node: Any) -> str:
    return NODE_KINDS.get(node.type, node.type)


def _node_text(node: Any) -> str:
    for attr in ("result", "error", "output", "content", "code"):
        value = getattr(node, attr, "")
        if value:
            return _clip(value, limit=60)
    return ""


def _is_bookkeeping(node: Any, successor: Any | None) -> bool:
    return bool(
        successor is not None
        and successor.type in HIDDEN_ACTION_PAIRS.get(node.type, ())
    )


def _visible_nodes(graph: Graph) -> list[Any]:
    """Every node the figure draws (bookkeeping action nodes collapsed away)."""
    visible: list[Any] = []
    for agent in graph.walk():
        successor = {n.id: s for n, s in zip(agent.nodes, agent.nodes[1:])}
        visible.extend(
            node
            for node in agent.nodes
            if not _is_bookkeeping(node, successor.get(node.id))
        )
    return visible


def _visible_signature(graph: Graph) -> tuple:
    """Fingerprint of what the figure shows; used to dedupe identical frames."""
    return tuple(
        (agent.agent_id, tuple(n.id for n in _visible_nodes(agent)))
        for agent in graph.walk()
    )


def _topology(
    graph: Graph,
    fixed_positions: dict[str, tuple[float, float]] | None = None,
) -> tuple[
    dict[str, tuple[float, float]],
    list[tuple[str, str]],
    list[tuple[str, str]],
    list[Any],
]:
    """Tidy-tree layout: agent spines vertical, children fanned out from the
    supervising node that awaited them. Returns positions, chain edges, spawn
    edges, and the drawn nodes.
    """
    nodes = _visible_nodes(graph)
    by_id = {node.id: node for node in nodes}
    agents = graph.agents

    chain_child: dict[str, str] = {}
    spawn_children: dict[str, list[str]] = {node.id: [] for node in nodes}
    parent_of: dict[str, str] = {}

    for agent in graph.walk():
        spine = [node for node in agent.nodes if node.id in by_id]
        for earlier, later in zip(spine, spine[1:]):
            chain_child[earlier.id] = later.id
            parent_of[later.id] = earlier.id

    for node in nodes:
        if not isinstance(node, SupervisingOutput):
            continue
        for child_id in node.waiting_on:
            child = agents.get(child_id)
            first = (
                next((n for n in child.nodes if n.id in by_id), None) if child else None
            )
            if first is None or first.id in parent_of:
                continue
            spawn_children[node.id].append(first.id)
            parent_of[first.id] = node.id

    def children(node_id: str) -> list[str]:
        kids = list(spawn_children.get(node_id, []))
        chain = chain_child.get(node_id)
        if chain is not None:
            kids.insert(len(kids) // 2, chain)  # keep the agent's own spine centered
        return kids

    def leaf_count(node_id: str, seen: set[str]) -> int:
        if node_id in seen:
            return 1
        seen.add(node_id)
        kids = children(node_id)
        return sum(leaf_count(k, seen.copy()) for k in kids) if kids else 1

    positions: dict[str, tuple[float, float]] = {}

    def place(
        node_id: str, left: float, right: float, depth: int, seen: set[str]
    ) -> None:
        if node_id in seen:
            return
        seen.add(node_id)
        positions[node_id] = ((left + right) / 2, -depth * _Y_SPACING)
        kids = children(node_id)
        widths = [leaf_count(k, set()) for k in kids]
        total = sum(widths) or 1
        cursor = left
        for kid, width in zip(kids, widths):
            span = (right - left) * width / total
            place(kid, cursor, cursor + span, depth + 1, seen)
            cursor += span

    roots = sorted(
        (node.id for node in nodes if node.id not in parent_of),
        key=lambda nid: (agents[by_id[nid].agent_id].depth, nid),
    )
    cursor = 0.0
    placed: set[str] = set()
    for root in roots:
        width = max(leaf_count(root, set()), 1)
        place(root, cursor, cursor + width, 0, placed)
        cursor += width + 1.0
    for node in nodes:
        if node.id not in positions:
            cursor += 1.0
            positions[node.id] = (cursor, 0.0)

    def shift_subtree(agent_id: str, dx: float) -> None:
        prefix = f"{agent_id}."
        for nid, node in by_id.items():
            if node.agent_id == agent_id or node.agent_id.startswith(prefix):
                x, y = positions[nid]
                positions[nid] = (x + dx, y)

    for _ in range(6):  # nudge apart columns from different agents that collide
        moved = False
        rows: dict[float, list[str]] = {}
        for nid, (_x, y) in positions.items():
            rows.setdefault(round(y, 6), []).append(nid)
        for row in rows.values():
            row.sort(key=lambda nid: positions[nid][0])
            for left_id, right_id in zip(row, row[1:]):
                gap = positions[right_id][0] - positions[left_id][0]
                if gap >= _MIN_COLUMN_GAP:
                    continue
                if by_id[left_id].agent_id == by_id[right_id].agent_id:
                    continue
                shift_subtree(by_id[right_id].agent_id, _MIN_COLUMN_GAP - gap)
                moved = True
        if not moved:
            break

    for node in nodes:  # recenter supervisors over their fanned-out children
        spawned = spawn_children.get(node.id, [])
        if isinstance(node, SupervisingOutput) and spawned:
            xs = [positions[c][0] for c in spawned]
            _x, y = positions[node.id]
            positions[node.id] = ((min(xs) + max(xs)) / 2, y)

    if fixed_positions is not None:
        positions.update(
            {nid: fixed_positions[nid] for nid in positions if nid in fixed_positions}
        )

    chain_edges = [
        (a, b) for a, b in chain_child.items() if a in positions and b in positions
    ]
    spawn_edges = [
        (parent, child)
        for parent, kids in spawn_children.items()
        for child in kids
        if parent in positions and child in positions
    ]
    return positions, chain_edges, spawn_edges, nodes


def _positions(graph: Graph) -> dict[str, tuple[float, float]]:
    return _topology(graph)[0]


def replay(graph: Graph) -> list[Graph]:
    """Progressive snapshots: one per execution step, each a truncated copy.

    Thin wrapper over :meth:`Graph.get_timeline`.
    """
    return graph.get_timeline()


def _agent_label(agent: Graph, *, limit: int = 22) -> str:
    label = agent.agent_id.removeprefix("root.")
    if agent.depth >= 2 and "." in label:
        label = label.rsplit(".", 1)[-1]
    return f"{label[: limit - 1]}…" if len(label) > limit else label


def _figure(
    graph: Graph,
    positions: dict[str, tuple[float, float]] | None = None,
    *,
    marker_mult: float = 1.0,
    text_mult: float = 1.0,
) -> Any:
    import plotly.graph_objects as go

    pos, chain_edges, spawn_edges, nodes = _topology(graph, positions)

    def segments(edges: list[tuple[str, str]]):
        xs: list[float | None] = []
        ys: list[float | None] = []
        for a, b in edges:
            (ax, ay), (bx, by) = pos[a], pos[b]
            xs += [ax, bx, None]
            ys += [ay, by, None]
        return xs, ys

    spawn_x, spawn_y = segments(spawn_edges)
    chain_x, chain_y = segments(chain_edges)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=spawn_x,
            y=spawn_y,
            mode="lines",
            line=dict(color="rgba(48,54,61,0.55)", width=0.8),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=chain_x,
            y=chain_y,
            mode="lines",
            line=dict(color="#30363d", width=1.4),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[pos[n.id][0] for n in nodes],
            y=[pos[n.id][1] for n in nodes],
            mode="markers",
            hoverinfo="text",
            hovertext=[
                f"<b>{n.agent_id}</b> · {_kind(n)} · seq {n.seq}"
                + (f"<br>{_node_text(n)}" if _node_text(n) else "")
                for n in nodes
            ],
            customdata=[n.id for n in nodes],
            marker=dict(
                size=24 * marker_mult,
                color=[NODE_COLORS.get(_kind(n), "#8b949e") for n in nodes],
                symbol=[NODE_SYMBOLS.get(_kind(n), "circle") for n in nodes],
                line=dict(color=_BG, width=2 * marker_mult),
            ),
            cliponaxis=False,
            showlegend=False,
        )
    )

    seen_agents: set[str] = set()
    for node in nodes:
        if node.agent_id in seen_agents:
            continue
        seen_agents.add(node.agent_id)
        x, y = pos[node.id]
        fig.add_annotation(
            x=x,
            y=y,
            text=_agent_label(graph.agents[node.agent_id]),
            showarrow=False,
            xanchor="center",
            yanchor="bottom",
            yshift=12,
            font=dict(
                color=_AGENT_LABEL_COLOR,
                family="ui-monospace, SFMono-Regular, Menlo, monospace",
                size=max(8, int(11 * text_mult)),
            ),
        )

    visible_kinds = {_kind(n) for n in nodes}
    for kind, color in NODE_COLORS.items():
        if kind not in visible_kinds:
            continue
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(
                    color=color,
                    symbol=NODE_SYMBOLS.get(kind, "circle"),
                    size=14 * marker_mult,
                    line=dict(color=_BG, width=1),
                ),
                name=kind,
                hoverinfo="skip",
                showlegend=True,
            )
        )

    xs = [pos[n.id][0] for n in nodes] or [0.0]
    ys = [pos[n.id][1] for n in nodes] or [0.0]
    x_pad = max(1.0, (max(xs) - min(xs)) * 0.18)
    y_pad = max(0.75, (max(ys) - min(ys)) * 0.16)
    current = graph.current()
    n_edges = len(chain_edges) + len(spawn_edges)
    fig.update_layout(
        title=dict(
            text=(
                f"{graph.agent_id} · {current.type if current else 'empty'} · "
                f"{len(nodes)} states · {n_edges} edges"
            ),
            font=dict(color="#e6edf3", size=max(10, int(12 * text_mult))),
            x=0.0,
            xanchor="left",
        ),
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        font=dict(color="#e6edf3"),
        hoverlabel=dict(
            bgcolor="#161b22",
            bordercolor="#30363d",
            font=dict(
                color="#e6edf3",
                family="ui-monospace, SFMono-Regular, Menlo, monospace",
                size=11,
            ),
            align="left",
        ),
        xaxis=dict(visible=False, range=[min(xs) - x_pad, max(xs) + x_pad]),
        yaxis=dict(visible=False, range=[min(ys) - y_pad, max(ys) + y_pad]),
        margin=dict(l=44, r=44, t=52, b=92),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.22,
            xanchor="center",
            x=0.5,
            font=dict(color="#8b949e", size=10),
            bgcolor="rgba(0,0,0,0)",
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

    # Freeze the final layout so scrubbing keeps node positions stable.
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
    snapshots = graphs if len(graphs) > 1 else replay(graphs[0])
    # Collapse consecutive frames that render identically (e.g. a tick that only
    # added a bookkeeping action) so step counts track visible progress.
    deduped: list[Graph] = []
    for graph in snapshots:
        if deduped and _visible_signature(deduped[-1]) == _visible_signature(graph):
            deduped[-1] = graph
        else:
            deduped.append(graph)
    return deduped


def save_image(
    source: ViewSource,
    path: str | Path,
    *,
    width: int = 1600,
    height: int = 1200,
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
    width: int = 1600,
    height: int = 1200,
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


def save_gif(
    source: ViewSource,
    path: str | Path,
    *,
    frames_dir: str | Path | None = None,
    duration_ms: int = 350,
    loop: int = 0,
    width: int = 1200,
    height: int = 900,
    scale: float = 1,
    marker_mult: float = 1.0,
    text_mult: float = 1.0,
) -> Path:
    """Animate a run as a GIF (needs the ``image`` extra + Pillow).

    Writes step PNGs via :func:`save_steps` (to ``frames_dir``, or
    ``<stem>_frames`` next to ``path``), then stitches them with Pillow.
    """
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError(
            "save_gif needs Pillow; install with `pip install pillow` "
            '(or `pip install -e ".[image]"` once pillow is in that extra).'
        ) from exc

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out_frames = (
        Path(frames_dir)
        if frames_dir is not None
        else path.parent / f"{path.stem}_frames"
    )
    frames = save_steps(
        source,
        out_frames,
        width=width,
        height=height,
        scale=scale,
        marker_mult=marker_mult,
        text_mult=text_mult,
    )
    if not frames:
        raise ValueError("save_gif needs at least one step frame")
    images = [Image.open(frame).convert("RGBA") for frame in frames]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=loop,
        disposal=2,
    )
    for image in images:
        image.close()
    return path


__all__ = [
    "ViewSource",
    "open_viewer",
    "replay",
    "save_gif",
    "save_image",
    "save_steps",
]
