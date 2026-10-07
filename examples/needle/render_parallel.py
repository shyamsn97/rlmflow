"""Render Needle as parallel graph waves rather than serialized events.

The normal graph replay advances one node at a time because a persisted event
log has one total creation order. For the Needle showcase that obscures the
important structure: sibling workers advance independently. This renderer keeps
the normal graph layout but reveals every worker's same local sequence level in
one frame, so parallel branches light together.
"""

from __future__ import annotations

import argparse
import io
from collections.abc import Sequence
from pathlib import Path

from rlmflow import persistence
from rlmflow.graph.nodes import AgentStart, ExecAction, Node
from rlmflow.view.figure import graph_svg
from rlmflow.view.steps import timeline


def _owned_nodes(agent: AgentStart) -> list[Node]:
    return sorted(
        (node for node in agent.walk() if node.parent_agent is agent),
        key=lambda node: node.seq,
    )


def _launch_action(root_nodes: Sequence[Node]) -> ExecAction:
    launches = [
        node
        for node in root_nodes
        if isinstance(node, ExecAction) and node.launch_agent_ids()
    ]
    if len(launches) != 1:
        raise ValueError(f"expected one fan-out action, found {len(launches)}")
    return launches[0]


def _parallel_order(root: AgentStart) -> tuple[list[Node], list[tuple[int, str]]]:
    """Return a parent-safe node order and reveal boundaries with frame titles."""
    root_nodes = _owned_nodes(root)
    launch = _launch_action(root_nodes)
    launch_index = root_nodes.index(launch)
    prelude = root_nodes[: launch_index + 1]
    tail = root_nodes[launch_index + 1 :]
    workers = sorted(root.sub_agents, key=lambda agent: agent.created_at)
    worker_nodes = [_owned_nodes(worker) for worker in workers]

    ordered: list[Node] = []
    frames: list[tuple[int, str]] = []

    for index, node in enumerate(prelude):
        ordered.append(node)
        label = "root partitions the million-line haystack"
        if index == len(prelude) - 1:
            label = f"root launches {len(workers)} independent workers"
        frames.append((len(ordered), label))

    max_seq = max(node.seq for nodes in worker_nodes for node in nodes)
    for seq in range(max_seq + 1):
        wave = [node for nodes in worker_nodes for node in nodes if node.seq == seq]
        if not wave:
            continue
        ordered.extend(wave)
        if seq == 0:
            label = f"{len(wave)} worker branches light together"
        else:
            noun = "branch" if len(wave) == 1 else "branches"
            label = f"parallel worker level {seq} · {len(wave)} {noun} advance together"
        frames.append((len(ordered), label))

    for node in tail:
        ordered.append(node)
        label = (
            f"root combines worker results · answer {root.result()}"
            if node.type == "done_output"
            else "root combines completed worker results"
        )
        frames.append((len(ordered), label))

    actual = timeline(root)
    if len(ordered) != len(actual) or {node.id for node in ordered} != {
        node.id for node in actual
    }:
        raise ValueError("parallel ordering did not preserve the complete graph")

    present: set[str] = set()
    for node in ordered:
        if node.parent is not None and node.parent.id not in present:
            raise ValueError(f"node {node.id} appears before parent {node.parent.id}")
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

    root = persistence.load(run_dir)
    ordered, reveal_frames = _parallel_order(root)
    frames = []
    for revealed, label in reveal_frames:
        svg = graph_svg(
            ordered,
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
