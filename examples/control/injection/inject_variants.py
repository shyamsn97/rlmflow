"""Steer a saved word-search run down two different routes.

Prerequisite:
    python examples/control/injection/word_search.py

That produces ``examples/_runs/word-search/baseline/``, a saved graph. This
example loads it, forks the root twice just before the turn where it decided
to delegate, and steers each fork with a different operator instruction:

1. scan the columns directly with a helper function, instead of one agent per
   column;
2. write one direct all-direction scanner, instead of reconciling children.

A fork shares the history up to the node it is taken at and nothing after it,
so the delegated children of that turn are not part of either fork and the
model plans again from the same history with the new instruction in front of
it. The instructions are prompts, not pre-written solution code or mocked
results.

The baseline and both forks live in one graph; one ``flow.run`` drives the two
forks in parallel, and the graph is saved beside the baseline at
``examples/_runs/word-search/variants/``.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from rlmflow import AGENT_START, LLM_OUTPUT, AgentConfig, Flow, Graph, Node, json_schema_for
from rlmflow.graph.graph import FROM

examples_dir = next(p for p in Path(__file__).resolve().parents if p.name == "examples")
if str(examples_dir) not in sys.path:
    sys.path.insert(0, str(examples_dir))

from common import checkpoint_stream, example_run_dir, save_example_graph  # noqa: E402


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
Actually, change the column route.

Instead of delegating each column to its own agent, search the columns directly
with a helper function.

In the next REPL block:

1. define `find_column_hits(grid: list[str], target_words: list[str]) -> list[dict]`;
2. scan every column in both S and N directions;
3. run the helper and verify each returned coordinate range spells the claimed word;
"""

ROOT_DIRECT_SCAN_PROMPT = """\
Actually, change the root route.

Instead of delegating to sub-agents, write a backtracking algorithm to find the target word yourself.
"""


def word_search_runs_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "_runs" / "word-search"


def default_source() -> Path:
    return word_search_runs_dir() / "baseline"


def delegating_turn(g: Graph, agent: Node) -> Node:
    """The most recent model turn of this agent whose block launched sub-agents."""
    launchers = {
        edge.target for child in g.children(agent) for edge in g.edges(kind=FROM, source=child.id)
    }
    for node in reversed(g.history(agent)):
        if node.type == LLM_OUTPUT and node.id in launchers:
            return node
    raise SystemExit("the baseline run never delegated; regenerate it with word_search.py")


def steer(g: Graph, root: Node, prompt: str, name: str) -> Node:
    """Fork the run back to before it delegated, then hand it a new instruction."""
    variant = g.fork(g.prev(delegating_turn(g, root)), name=name)
    g.send(variant, prompt)
    return variant


def summarize(label: str, g: Graph, root: Node) -> None:
    print(f"\n{label}")
    print("-" * len(label))
    print(f"root frontier: {g.frontier(root).type}")
    children = ", ".join(child.name for child in g.children(root))
    print(f"children: {children or '<none>'}")
    if g.terminal(root):
        print("result:")
        print(g.result(root))


def _hit_key(hit: WordHit) -> tuple[str, int, int, int, int, str]:
    return (
        hit.word,
        hit.start_row,
        hit.start_col,
        hit.end_row,
        hit.end_col,
        hit.direction,
    )


def validate_result(label: str, g: Graph, root: Node) -> None:
    if not g.terminal(root):
        print(f"\n{label} validation: skipped (the run did not finish)")
        return

    answer = g.result(root)
    # A structured answer comes back parsed; an unstructured one is still text.
    result = (
        WordSearchResult.model_validate(answer)
        if isinstance(answer, dict)
        else WordSearchResult.model_validate_json(answer)
    )
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

    g = Graph.load(args.source.resolve())
    baseline = next(agent for agent in g.nodes(AGENT_START) if g.launcher(agent) is None)
    summarize("Loaded real word-search run", g, baseline)

    # Both forks share the baseline's history; the baseline itself is untouched.
    cols = steer(g, baseline, COLS_FUNCTION_PROMPT, "variant_cols")
    direct = steer(g, baseline, ROOT_DIRECT_SCAN_PROMPT, "variant_root")

    # One run drives every unfinished agent in the graph: both forks, in parallel.
    flow = Flow(configs=[AgentConfig(model=args.model, max_depth=2)])
    # The roots run with the baseline's root config, so they answer in its schema.
    root_config = AgentConfig(
        model=args.model, output_schema=json_schema_for(WordSearchResult), max_depth=2
    )
    out_dir = args.out.resolve() / "variants"

    async def run_variants() -> None:
        stream = checkpoint_stream(flow.run_streaming(g, root_config=root_config), g, out_dir)
        async for step in stream:
            print(f"step: {g.path(step.agent)} -> {step.node.type}")

    asyncio.run(run_variants())
    flow.runtime.close_repls()

    summarize("Variation A: scan the columns directly", g, cols)
    validate_result("Variation A", g, cols)
    summarize("Variation B: write a direct scanner", g, direct)
    validate_result("Variation B", g, direct)
    print(f"saved -> {g.save(out_dir)}")
    save_example_graph(g, "injection-variants", out_dir=example_run_dir("injection-variants"))


if __name__ == "__main__":
    main()
