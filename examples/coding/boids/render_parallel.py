"""Render a Boids coding run as level-synchronized graph waves."""

from __future__ import annotations

import argparse
import io
from pathlib import Path

from rlmflow import persistence
from rlmflow.graph.nodes import AgentStart, Node
from rlmflow.view.figure import graph_svg
from rlmflow.view.steps import timeline


def _depth(node: Node, cache: dict[str, int]) -> int:
    if node.id in cache:
        return cache[node.id]
    cache[node.id] = 0 if node.parent is None else _depth(node.parent, cache) + 1
    return cache[node.id]


def _parallel_order(root: AgentStart) -> tuple[list[Node], list[tuple[int, str]]]:
    """Order the complete graph by structural level, preserving parent safety."""
    chronological = timeline(root)
    chronology = {node.id: index for index, node in enumerate(chronological)}
    depths: dict[str, int] = {}
    ordered = sorted(chronological, key=lambda node: (_depth(node, depths), chronology[node.id]))

    frames = []
    cursor = 0
    for level in sorted(set(depths.values())):
        wave = [node for node in ordered if depths[node.id] == level]
        cursor += len(wave)
        agents = sum(isinstance(node, AgentStart) and node is not root for node in wave)
        if agents == 1:
            label = "root launches one follow-up child"
        elif agents:
            label = f"{agents} child agents launch together"
        elif level == 0:
            label = "root receives the application specification"
        elif any(node.type == "done_output" and node.parent_agent is root for node in wave):
            label = "root completes integration and verification"
        else:
            noun = "node" if len(wave) == 1 else "nodes"
            label = f"parallel graph level {level} · {len(wave)} {noun}"
        frames.append((cursor, label))

    present: set[str] = set()
    for node in ordered:
        if node.parent is not None and node.parent.id not in present:
            raise ValueError(f"node {node.id} appears before parent {node.parent.id}")
        present.add(node.id)
    if {node.id for node in ordered} != {node.id for node in chronological}:
        raise ValueError("parallel ordering did not preserve the complete graph")
    return ordered, frames


def render(
    run_dir: Path,
    out: Path,
    *,
    ms_per_frame: int = 300,
    hold_last_ms: int = 2400,
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
            title=f"Boids coding · {label}",
            width=1040,
            height=420,
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
        description="Render a Boids graph with same-level nodes revealed together."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("examples/_runs/coding/boids-rlmflow/graph"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/static/boids/graph.gif"),
    )
    parser.add_argument("--ms-per-frame", type=int, default=300)
    parser.add_argument("--hold-last-ms", type=int, default=2400)
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
