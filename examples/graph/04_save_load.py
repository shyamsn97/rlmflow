"""Saving and loading a graph."""

from __future__ import annotations

import argparse
import importlib.util
import shutil
from pathlib import Path

from rlmflow import AGENT_START, Graph


def build_graph():
    spec = importlib.util.spec_from_file_location(
        "graph_01_query", Path(__file__).with_name("01_query.py")
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load 01_query.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_graph()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parents[1] / "_runs" / "graph-save-load"),
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)

    g, root = build_graph()
    run_dir = g.save(out_dir / "run")
    loaded = Graph.load(run_dir)

    print(f"saved: {run_dir}")
    print("agents:", [loaded.path(agent) for agent in loaded.nodes(AGENT_START)])
    print("result:", loaded.result(loaded.node(root.id)))
    print("roundtrip:", [item.id for item in loaded.items()] == [item.id for item in g.items()])


if __name__ == "__main__":
    main()
