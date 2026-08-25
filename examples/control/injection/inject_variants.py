"""Steer a saved word-search run down two different routes.

Prerequisite:
    python examples/control/injection/word_search.py

That produces ``examples/_runs/word-search/baseline/`` — a run directory with
``graph.json`` (manifest) and per-agent logs under ``agents/``. This example
loads that run, forks it twice at the turn where the root decided to delegate,
and steers each copy with a different operator instruction:

1. scan the columns directly with a helper function, instead of one agent per
   column;
2. write one direct all-direction scanner, instead of reconciling children.

A fork cuts everything after the node it is taken at, so the delegated children
of that turn are gone from both copies and the model plans again from the same
history with the new instruction in front of it. The instructions are prompts,
not pre-written solution code or mocked results.

Both variants run in parallel on one :class:`Flow` and are saved beside the
baseline, at ``examples/_runs/word-search/variant-cols/`` and
``.../variant-root/``.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from rlmflow import (
    AgentConfig,
    AgentStart,
    ExecAction,
    Flow,
    GraphCheckpointer,
    UserQuery,
    parallel_stream,
    persistence,
)
from rlmflow.llm import client_for

examples_dir = next(p for p in Path(__file__).resolve().parents if p.name == "examples")
if str(examples_dir) not in sys.path:
    sys.path.insert(0, str(examples_dir))

from common import example_run_dir, save_example_graph  # noqa: E402


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


def delegating_action(agent: AgentStart) -> ExecAction:
    """The most recent block this agent ran that launched sub-agents."""
    for node in agent.frontier.iter_backwards():
        launched = isinstance(node, ExecAction) and any(
            isinstance(child, AgentStart) for child in node.children
        )
        if launched:
            return node
    raise SystemExit("the baseline run never delegated; regenerate it with word_search.py")


def steer(root: AgentStart, prompt: str) -> AgentStart:
    """Fork the run back to before it delegated, then hand it a new instruction."""
    turn = delegating_action(root).prev  # the model turn that wrote that block
    variant = (turn.prev or root).fork()
    variant.frontier.append(UserQuery(content=prompt))
    return variant


def summarize(label: str, root: AgentStart) -> None:
    print(f"\n{label}")
    print("-" * len(label))
    print(f"root frontier: {root.frontier.type}")
    print(f"children: {', '.join(child.config.name for child in root.sub_agents) or '<none>'}")
    if root.terminal:
        print("result:")
        print(root.result())


def _hit_key(hit: WordHit) -> tuple[str, int, int, int, int, str]:
    return (
        hit.word,
        hit.start_row,
        hit.start_col,
        hit.end_row,
        hit.end_col,
        hit.direction,
    )


def validate_result(label: str, root: AgentStart) -> None:
    if not root.terminal:
        print(f"\n{label} validation: skipped (the run did not finish)")
        return

    answer = root.result()
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

    baseline = persistence.load(args.source.resolve())
    summarize("Loaded real word-search run", baseline)

    # Each fork is an independent copy with its own ids; the baseline is untouched.
    cols_graph = steer(baseline, COLS_FUNCTION_PROMPT)
    root_graph = steer(baseline, ROOT_DIRECT_SCAN_PROMPT)

    # One Flow drives both variants and merges their Node streams.
    flow = Flow(
        client_for(args.model),
        root_config=AgentConfig(max_depth=2, max_iters=30),
    )

    out = args.out.resolve()
    dirs = {id(cols_graph): out / "variant-cols", id(root_graph): out / "variant-root"}

    async def run_variants() -> None:
        checkpoints = {identity: GraphCheckpointer(path) for identity, path in dirs.items()}
        try:
            async for node in parallel_stream(flow, cols_graph, root_graph):
                root = node.root
                checkpoints[id(root)].handle(node)
                print(f"step: {dirs[id(root)].name} -> {node.type}")
        finally:
            for checkpoint in checkpoints.values():
                checkpoint.close()

    asyncio.run(run_variants())
    flow.runtime.close_repls()

    summarize("Variation A: scan the columns directly", cols_graph)
    validate_result("Variation A", cols_graph)
    print(f"saved -> {cols_graph.save(dirs[id(cols_graph)])}")
    save_example_graph(
        cols_graph,
        "injection-variants",
        out_dir=example_run_dir("injection-variants") / "variant-cols",
    )

    summarize("Variation B: write a direct scanner", root_graph)
    validate_result("Variation B", root_graph)
    print(f"saved -> {root_graph.save(dirs[id(root_graph)])}")
    save_example_graph(
        root_graph,
        "injection-variants",
        out_dir=example_run_dir("injection-variants") / "variant-root",
    )


if __name__ == "__main__":
    main()
