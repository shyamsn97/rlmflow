"""Runner-neutral graph and environment artifact bundle."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from benchmarks.eval.loggers.jsonl import write_json
from benchmarks.eval.types import Example
from rlmflow import DEPENDS_ON, AgentStart, Graph, reads


def render_problem_artifacts(
    example: Example,
    answer: str,
    graph: Graph,
    root: AgentStart,
    artifact_dir: Path,
    *,
    environment: dict[str, Any] | None = None,
    initial_svg: str | None = None,
) -> dict[str, str]:
    """Write only artifacts common to the retained suite."""
    del answer, initial_svg
    artifact_dir.mkdir(parents=True, exist_ok=True)
    graph_path = artifact_dir / "agent_graph.svg"
    graph_path.write_text(_agent_graph_svg(graph, root), encoding="utf-8")
    paths = {"agent_graph": graph_path}
    if environment:
        replay_path = artifact_dir / "environment_replay.json"
        write_json(
            replay_path,
            environment.get(
                "replay",
                {
                    "schema_version": 1,
                    "kind": environment.get("kind"),
                    "events": environment.get("events", []),
                    "instances": environment.get("instances", []),
                },
            ),
        )
        paths["environment_replay"] = replay_path
    manifest_path = artifact_dir / "artifacts.json"
    write_json(
        manifest_path,
        {
            "schema_version": 2,
            "example_id": example.id,
            "domain": example.metadata.get("domain"),
            "synchronization": {
                "environment_event_key": "exec_action_id",
                "agent_key": "agent_id",
            },
            "artifacts": {key: path.name for key, path in paths.items()},
        },
    )
    paths["manifest"] = manifest_path
    return {key: str(path) for key, path in paths.items()}


def _agent_graph_svg(g: Graph, root: AgentStart) -> str:
    agents = reads.tree(g, root)
    nodes = [
        {
            "id": agent.id,
            "label": reads.path(g, agent),
            "column": reads.depth(g, agent),
            "row": index,
        }
        for index, agent in enumerate(agents)
    ]
    edges: list[tuple[str, str, str]] = []
    for agent in agents[1:]:
        owner = reads.launcher(g, agent)
        if owner is not None:
            edges.append((owner.id, agent.id, "launch"))
        edges.extend(
            (edge.target, agent.id, f"arg:{edge.data['as']}")
            for edge in g.edges(kind=DEPENDS_ON, source=agent.id)
            if edge.data.get("as")
        )
    return _graph_svg(nodes, edges)


def _graph_svg(
    nodes: list[dict[str, Any]],
    edges: list[tuple[str, str, str]],
) -> str:
    width = 980
    height = max(130, 86 + 58 * len(nodes))
    positions = {
        node["id"]: (40 + int(node["column"]) * 190, 75 + int(node["row"]) * 58) for node in nodes
    }
    body = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="24" y="30" font-size="19">Executed agent graph</text>',
    ]
    for source, target, kind in edges:
        if source not in positions or target not in positions:
            continue
        x1, y1 = positions[source]
        x2, y2 = positions[target]
        color = "#d33" if kind.startswith("input:") else "#777"
        body.append(
            f'<line x1="{x1 + 8}" y1="{y1}" x2="{x2 + 8}" y2="{y2}" '
            f'stroke="{color}" stroke-width="2" '
            f'data-source-agent-id="{html.escape(str(source), quote=True)}" '
            f'data-target-agent-id="{html.escape(str(target), quote=True)}" '
            f'data-edge-kind="{html.escape(kind, quote=True)}"/>'
        )
        if kind != "launch":
            body.append(
                f'<text x="{(x1 + x2) / 2 + 8}" y="{(y1 + y2) / 2 - 5}" '
                f'text-anchor="middle" font-size="10" fill="{color}">'
                f"{html.escape(kind)}</text>"
            )
    for node in nodes:
        x, y = positions[node["id"]]
        body.extend(
            [
                f'<g data-agent-id="{html.escape(str(node["id"]), quote=True)}">',
                f'<circle cx="{x + 8}" cy="{y}" r="8" fill="#315efb"/>',
                f'<text x="{x + 24}" y="{y + 4}" font-size="12">'
                f"{html.escape(str(node['label']))}</text>",
                "</g>",
            ]
        )
    body.append("</svg>")
    return "".join(body)


__all__ = ["render_problem_artifacts"]
