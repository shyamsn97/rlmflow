"""Inject alternate prompts into a minimal word-search supervisor trace.

Prerequisite:
    python examples/control/injection/word_search.py

That produces ``examples/_runs/word-search/baseline/`` — a run directory with
``graph.json`` (manifest) and per-agent logs under ``agents/``. This example
loads that run and creates two edited copies, replacing real supervising nodes:

1. replace ``root.cols`` so it scans columns directly instead of delegating each
   column;
2. replace the root supervising node so the parent writes one direct
   all-direction scanner instead of reconciling direction children.

The replacements are operator prompts, not pre-written solution code or mocked
results. Each edited graph is an isolated fork continued by its own minimal
:class:`Flow`.

Both finished variants are saved as run directories beside the baseline, at
``examples/_runs/word-search/variant-cols/`` and ``.../variant-root/``.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from rlmflow import (
    ExecOutput,
    Flow,
    Graph,
    GraphCheckpointer,
    SupervisingOutput,
    parallel_stream,
)

examples_dir = next(p for p in Path(__file__).resolve().parents if p.name == "examples")
if str(examples_dir) not in sys.path:
    sys.path.insert(0, str(examples_dir))

from common import build_client, example_run_dir, save_example_graph  # noqa: E402


class WordHit(BaseModel):
    """One found word and its inclusive coordinates."""

    word: str
    start_row: int
    start_col: int
    end_row: int
    end_col: int
    direction: Literal["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


class WordSearchResult(BaseModel):
    """All target words found in the grid."""

    found: list[WordHit]
    missing: list[str]


EXPECTED_HITS = {("AGENT", 1, 8, 5, 8, "S")}
EXPECTED_MISSING: set[str] = set()

COLS_FUNCTION_PROMPT = """\
Actually, change the column-agent route.

Instead of delegating each column, search the columns directly with a helper
function.

In the next REPL block:

1. define `find_column_hits(grid: list[str], target_words: list[str]) -> list[dict]`;
2. scan every column in both S and N directions;
2. run the helper and verify each returned coordinate range spells the claimed word;
"""

ROOT_DIRECT_SCAN_PROMPT = """\
Actually, change the root route.

Instead of delegating to sub-agents, write a backtracking algorithm to find the target word yourself.
"""




def word_search_runs_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "_runs" / "word-search"


def default_source() -> Path:
    return word_search_runs_dir() / "baseline"


def summarize(label: str, graph: Graph) -> None:
    current = graph.current()
    print(f"\n{label}")
    print("-" * len(label))
    print(f"root current: {current.type if current else '<empty>'}")
    if isinstance(current, SupervisingOutput):
        print(f"waiting_on: {', '.join(current.waiting_on)}")
    print(f"children: {', '.join(graph.children) or '<none>'}")
    if graph.finished:
        print("result:")
        print(graph.result())


def _hit_key(hit: WordHit) -> tuple[str, int, int, int, int, str]:
    return (
        hit.word,
        hit.start_row,
        hit.start_col,
        hit.end_row,
        hit.end_col,
        hit.direction,
    )


def validate_result(label: str, graph: Graph) -> None:
    if not graph.finished:
        print(f"\n{label} validation: skipped (graph is not finished)")
        return

    result = WordSearchResult.model_validate_json(graph.result())
    actual = {_hit_key(hit) for hit in result.found}
    missing = set(result.missing)
    ok = actual == EXPECTED_HITS and missing == EXPECTED_MISSING
    print(f"\n{label} validation: {'PASS' if ok else 'FAIL'}")
    print("found:")
    for hit in sorted(result.found, key=lambda h: (h.word, h.direction)):
        print(
            f"- {hit.word}: ({hit.start_row},{hit.start_col}) -> "
            f"({hit.end_row},{hit.end_col}) {hit.direction}"
        )
    if result.missing:
        print(f"missing: {', '.join(result.missing)}")
    if not ok:
        print("expected:")
        for word, sr, sc, er, ec, direction in sorted(EXPECTED_HITS):
            print(f"- {word}: ({sr},{sc}) -> ({er},{ec}) {direction}")
    assert ok


def supervising_node(graph: Graph, agent_id: str) -> SupervisingOutput:
    matches = [
        node
        for node in graph[agent_id].nodes
        if isinstance(node, SupervisingOutput)
    ]
    return matches[-1]


def replace_supervisor_with_prompt(
    graph: Graph,
    agent_id: str,
    prompt: str,
) -> Graph:
    variant = graph.fork(session="isolated")
    variant.replace(
        supervising_node(variant, agent_id),
        ExecOutput(
            output=prompt,
            content=f"REPL output for previous block:\n{prompt}",
        ),
        truncate="descendants",
    )
    return variant


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=default_source())
    parser.add_argument(
        "--out",
        type=Path,
        default=word_search_runs_dir(),
        help="directory to save the variant runs beside the baseline",
    )
    parser.add_argument("--model", default="gpt-5-mini")
    args = parser.parse_args()

    graph = Graph.load(args.source.resolve())
    summarize("Loaded real word-search run", graph)

    # Each edit starts from a fresh, independent graph value.
    cols_graph = replace_supervisor_with_prompt(
        graph,
        "root.cols",
        COLS_FUNCTION_PROMPT,
    )
    root_graph = replace_supervisor_with_prompt(
        graph,
        "root",
        ROOT_DIRECT_SCAN_PROMPT,
    )

    # One Flow is the engine; it drives both variant graphs at once and merges
    # their event streams (each event carries graph_id).
    flow = Flow(
        build_client(args.model),
        max_depth=2,
        max_iters=30,
    )

    out = args.out.resolve()
    # One checkpointer per variant, keyed by graph_id, since parallel_stream
    # interleaves events for both graphs (each event carries graph_id).
    graphs = {cols_graph.graph_id: cols_graph, root_graph.graph_id: root_graph}
    checkpointers = {
        cols_graph.graph_id: GraphCheckpointer(out / "variant-cols"),
        root_graph.graph_id: GraphCheckpointer(out / "variant-root"),
    }

    async def run_variants() -> None:
        try:
            async for event in parallel_stream(flow, cols_graph, root_graph):
                gid = event.graph_id
                checkpointers[gid].handle(event, graphs[gid])
                node_type = getattr(event, "node_type", event.type)
                print(f"step: {gid} -> {node_type}")
        finally:
            for checkpointer in checkpointers.values():
                checkpointer.close()

    asyncio.run(run_variants())

    flow.close_repls(cols_graph.graph_id)
    flow.close_repls(root_graph.graph_id)

    cols_dir = cols_graph.save(out / "variant-cols")
    root_dir = root_graph.save(out / "variant-root")

    summarize("Variation A: prompt root.cols to scan columns directly", cols_graph)
    validate_result("Variation A", cols_graph)
    print(f"saved -> {cols_dir}")
    save_example_graph(
        cols_graph,
        "injection-variants",
        out_dir=example_run_dir("injection-variants") / "variant-cols",
    )

    summarize("Variation B: prompt root to write a direct scanner", root_graph)
    validate_result("Variation B", root_graph)
    print(f"saved -> {root_dir}")
    save_example_graph(
        root_graph,
        "injection-variants",
        out_dir=example_run_dir("injection-variants") / "variant-root",
    )


if __name__ == "__main__":
    main()
