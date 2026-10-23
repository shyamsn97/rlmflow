"""Render Needle as parallel graph waves rather than serialized events.

The normal graph replay advances one node at a time because the graph log has
one total order. For the Needle showcase that obscures the
important structure: sibling workers advance independently. This renderer keeps
the normal graph layout but reveals every worker's same local sequence level in
one frame, so parallel branches light together.
"""

from __future__ import annotations

import argparse
import io
from collections.abc import Sequence
from pathlib import Path

from rlmflow import AgentStart, ExecAction, Graph, Node, reads
from rlmflow.view.figure import graph_svg, parent_ids


def _launch_action(root_nodes: Sequence[Node]) -> ExecAction:
    launches = [
        node for node in root_nodes if isinstance(node, ExecAction) and node.launch_agent_ids()
    ]
    if len(launches) != 1:
        raise ValueError(f"expected one fan-out action, found {len(launches)}")
    return launches[0]


def _parallel_order(g: Graph, root: AgentStart) -> tuple[list[Node], list[tuple[int, str]]]:
    """Return a parent-safe node order and reveal boundaries with frame titles."""
    root_nodes = g.nodes(agent=root)
    launch = _launch_action(root_nodes)
    launch_index = root_nodes.index(launch)
    prelude = root_nodes[: launch_index + 1]
    tail = root_nodes[launch_index + 1 :]
    workers = reads.children(g, root)
    worker_nodes = [g.nodes(agent=worker) for worker in workers]

    ordered: list[Node] = []
    frames: list[tuple[int, str]] = []

    for index, node in enumerate(prelude):
        ordered.append(node)
        label = "root partitions the million-line haystack"
        if index == len(prelude) - 1:
            label = f"root launches {len(workers)} independent workers"
        frames.append((len(ordered), label))

    # A worker's nth own node is its level: siblings at one level light together.
    for level in range(max(map(len, worker_nodes))):
        wave = [nodes[level] for nodes in worker_nodes if level < len(nodes)]
        ordered.extend(wave)
        if level == 0:
            label = f"{len(wave)} worker branches light together"
        else:
            noun = "branch" if len(wave) == 1 else "branches"
            label = f"parallel worker level {level} · {len(wave)} {noun} advance together"
        frames.append((len(ordered), label))

    for node in tail:
        ordered.append(node)
        label = (
            f"root combines worker results · answer {reads.result(g, root)}"
            if node.type == "done_output"
            else "root combines completed worker results"
        )
        frames.append((len(ordered), label))

    if {node.id for node in ordered} != {node.id for node in g.nodes()}:
        raise ValueError("parallel ordering did not preserve the complete graph")

    parents = parent_ids(g)
    present: set[str] = set()
    for node in ordered:
        parent = parents.get(node.id)
        if parent is not None and parent not in present:
            raise ValueError(f"node {node.id} appears before parent {parent}")
        present.add(node.id)
    return ordered, frames


def render(
    run_dir: Path,
    out: Path,
    *,
    ms_per_frame: int = 240,
    hold_last_ms: int = 2200,
) -> Path:
    try:
        import cairosvg
        from PIL import Image
    except ImportError as exc:
        raise SystemExit("rendering needs CairoSVG and Pillow: install rlmflow[image]") from exc

    g = Graph.load(run_dir)
    root = next(agent for agent in g.nodes(AgentStart) if reads.launcher(g, agent) is None)
    ordered, reveal_frames = _parallel_order(g, root)
    frames = []
    for revealed, label in reveal_frames:
        svg = graph_svg(
            g,
            order=ordered,
            title=f"Needle · {label}",
            width=1040,
            height=370,
            dim_after=revealed,
        )
        png = cairosvg.svg2png(bytestring=svg.encode("utf-8"))
        frames.append(Image.open(io.BytesIO(png)).convert("P", palette=Image.Palette.ADAPTIVE))

    out.parent.mkdir(parents=True, exist_ok=True)
    durations = [ms_per_frame] * len(frames)
    durations[-1] = max(hold_last_ms, ms_per_frame)
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a saved Needle graph with same-level worker nodes revealed together."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("examples/_runs/needle-haystack"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/static/needle-haystack.gif"),
    )
    parser.add_argument("--ms-per-frame", type=int, default=240)
    parser.add_argument("--hold-last-ms", type=int, default=2200)
    args = parser.parse_args()
    written = render(
        args.run_dir,
        args.out,
        ms_per_frame=args.ms_per_frame,
        hold_last_ms=args.hold_last_ms,
    )
    print(f"wrote {written}")


if __name__ == "__main__":
    main()
