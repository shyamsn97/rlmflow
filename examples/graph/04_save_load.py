"""Saving and loading a minimal Node trajectory."""

from __future__ import annotations

import argparse
import importlib.util
import shutil
from pathlib import Path

from rlmflow import Node, persistence


def build_graph() -> Node:
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

    graph = build_graph()
    run_dir = persistence.save(graph, out_dir / "run")
    loaded = persistence.load(run_dir)

    print(f"saved: {run_dir}")
    print("agents:", list(loaded.agent_ids()))
    print("result:", loaded.agent_result())
    print("roundtrip:", persistence.to_dict(graph) == persistence.to_dict(loaded))


if __name__ == "__main__":
    main()
