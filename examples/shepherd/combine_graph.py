"""Merge a shepherd run's separate graphs into one, for `rlmflow render`.

A shepherd run saves the jammed worker (``graph/``), the shepherd (``shepherd/``)
and each fork (``branches/branchN/``) as separate graphs. A fork repeats the
worker's nodes up to the cut, so this keeps only the nodes each fork added and
makes them an agent named after the branch. Each branch hangs off the worker
node it was forked at by a ``spawn`` edge, so its rewind depth shows, and its
last node has a ``return`` edge to the shepherd node that read the results. Nodes and edges are
renumbered by time.

    python examples/shepherd/combine_graph.py examples/_runs/shepherd
    rlmflow render gif examples/_runs/shepherd/combined shepherd.gif
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from rlmflow import Graph


def load(path: Path) -> dict:
    return json.loads((path / "graph.json").read_text())


def combine(run_dir: Path) -> dict:
    jam = load(run_dir / "graph")
    shepherd = load(run_dir / "shepherd")
    branches = sorted(
        (run_dir / "branches").iterdir(), key=lambda p: int(p.name.removeprefix("branch"))
    )

    jam_ids = {n["id"] for n in jam["nodes"]}
    worker = next(n["agent"] for n in jam["nodes"] if "agent" in n)
    turns = [n for n in shepherd["nodes"] if n["role"] == "assistant"]

    nodes = jam["nodes"] + shepherd["nodes"]
    edges = []
    for path in branches:
        fork = load(path)
        added = [copy.deepcopy(n) for n in fork["nodes"] if n["id"] not in jam_ids]
        if not added:
            continue
        root = added[0]
        for node in added:
            node["owner"] = root["id"]
        root["agent"] = {
            **worker,
            "name": path.name,
            "task": root.get("text") or worker["task"],
            "repl": root["id"],
        }
        cut = fork.get("metadata", {}).get("forked_from", {}).get("node")
        launcher = (
            {"id": cut}
            if cut in jam_ids
            else max(
                (t for t in turns if t["created_at"] <= root["created_at"]),
                key=lambda t: t["created_at"],
            )
        )
        reader = min(
            (t for t in turns if t["created_at"] >= added[-1]["created_at"]),
            key=lambda t: t["created_at"],
            default=turns[-1],
        )
        edges.append(("spawn", launcher["id"], root["id"], root["created_at"], 0))
        edges.append(("return", added[-1]["id"], reader["id"], added[-1]["created_at"], 1))
        nodes += added

    events = [(n["created_at"], 0, n) for n in nodes]
    events += [(at, order, e) for (*e, at, order) in edges]
    events.sort(key=lambda event: (event[0], event[1]))
    out_edges = []
    for seq, (_, _, item) in enumerate(events):
        if isinstance(item, dict):
            item["seq"] = seq
        else:
            kind, source, target = item
            out_edges.append(
                {
                    "id": f"edge_{kind}_{source[-8:]}_{target[-8:]}",
                    "kind": kind,
                    "source": source,
                    "target": target,
                    "data": {},
                    "seq": seq,
                }
            )
    nodes.sort(key=lambda n: n["seq"])
    return {
        "format": jam["format"],
        "id": shepherd["id"],
        "metadata": {"combined_from": str(run_dir)},
        "nodes": nodes,
        "edges": out_edges,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", type=Path, help="default: RUN_DIR/combined")
    args = parser.parse_args()
    out = args.out or args.run_dir / "combined"
    out.mkdir(parents=True, exist_ok=True)
    (out / "graph.json").write_text(json.dumps(combine(args.run_dir)))
    graph = Graph.load(out)
    print(f"wrote {out}: {len(list(graph.nodes()))} nodes, {len(list(graph.edges()))} edges")


if __name__ == "__main__":
    main()
