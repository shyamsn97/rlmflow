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
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from rflow.clients import AnthropicClient, OpenAIClient
from rflow.minimal import ExecOutput, Flow, Graph, SupervisingOutput


def _example_run_dir(source_file: str | Path, name: str) -> Path:
    source = Path(source_file).resolve()
    for parent in (source.parent, *source.parents):
        if parent.name == "examples":
            return parent / "_runs" / name
    return source.parent / "_runs" / name


def _save_example_graph(
    graph,
    source_file: str | Path,
    name: str,
    *,
    out_dir: str | Path | None = None,
    label: str = "Graph saved to",
) -> Path:
    path = graph.save(
        Path(out_dir) if out_dir is not None else _example_run_dir(source_file, name)
    )
    print(f"{label} {path}")
    return path



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


def client_for_model(model: str):
    return (
        AnthropicClient(model)
        if model.startswith("claude")
        else OpenAIClient(model)
    )


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
    old = supervising_node(variant, agent_id)
    target = variant[agent_id]
    target.nodes = target.nodes[: old.seq]
    target.commit(
        ExecOutput(
            output=prompt,
            content=f"REPL output for previous block:\n{prompt}",
        )
    )
    for child_id in old.waiting_on:
        if child_id in target.children:
            variant.remove_child(child_id, parent_agent_id=agent_id)
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

    def new_flow() -> Flow:
        return Flow(
            client_for_model(args.model),
            max_depth=2,
            max_iters=30,
        )

    # One Flow per variant; step active variants through one shared parallel batch.
    cols_flow = new_flow()
    root_flow = new_flow()

    async def step_variants() -> None:
        async def step(flow: Flow, graph: Graph) -> None:
            try:
                await flow.step(graph, until="node")
            except RuntimeError as exc:
                if "run is already active" not in str(exc):
                    raise
                await flow.step(until="node")

        while not (cols_graph.finished and root_graph.finished):
            active: list[tuple[Flow, Graph]] = []
            if not cols_graph.finished:
                active.append((cols_flow, cols_graph))
            if not root_graph.finished:
                active.append((root_flow, root_graph))

            await asyncio.gather(*(step(flow, graph) for flow, graph in active))
            cols_state = cols_graph.current().type if cols_graph.current() else "<empty>"
            root_state = root_graph.current().type if root_graph.current() else "<empty>"
            print(f"step: Variation A={cols_state}, Variation B={root_state}")

    asyncio.run(step_variants())

    cols_flow.close_repls(cols_graph.graph_id)
    root_flow.close_repls(root_graph.graph_id)

    out = args.out.resolve()
    cols_dir = cols_graph.save(out / "variant-cols")
    root_dir = root_graph.save(out / "variant-root")

    summarize("Variation A: prompt root.cols to scan columns directly", cols_graph)
    validate_result("Variation A", cols_graph)
    print(f"saved -> {cols_dir}")
    _save_example_graph(
        cols_graph,
        __file__,
        "injection-variants",
        out_dir=_example_run_dir(__file__, "injection-variants") / "variant-cols",
    )

    summarize("Variation B: prompt root to write a direct scanner", root_graph)
    validate_result("Variation B", root_graph)
    print(f"saved -> {root_dir}")
    _save_example_graph(
        root_graph,
        __file__,
        "injection-variants",
        out_dir=_example_run_dir(__file__, "injection-variants") / "variant-root",
    )


if __name__ == "__main__":
    main()
