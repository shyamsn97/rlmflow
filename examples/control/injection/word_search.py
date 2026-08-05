"""Generate the baseline word-search run used by the injection example.

This is the original route: ask a real Flow agent to find ``AGENT`` by
delegating direction-specific search to three child agents:

- ``rows`` searches rows east/west;
- ``cols`` searches columns north/south;
- ``diagonals`` searches diagonals in all four diagonal directions.

That route is intentionally plausible but more complicated than necessary. It
creates a real delegating turn at the root, with each child agent hanging off
the ``ExecAction`` that launched it. The finished run is saved as a run
directory (``graph.json`` manifest plus per-agent logs nested under
``agents/``) that ``inject_variants.py`` loads with ``persistence.load``.

Run:
    export OPENAI_API_KEY=...
    python examples/control/injection/word_search.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from rlmflow import AgentStart, Flow, Node, json_schema_for, start

examples_dir = next(p for p in Path(__file__).resolve().parents if p.name == "examples")
if str(examples_dir) not in sys.path:
    sys.path.insert(0, str(examples_dir))

from common import build_client, save_example_graph  # noqa: E402

TARGET_WORD = "AGENT"

GRID_ROWS = [
    "TYPHONQWER",
    "LMNOPQRSAT",
    "ZGXYZLMNGO",
    "ABDDEFGHEI",
    "QRSATUVWNY",
    "JKLMPQRSTF",
    "ABCDEPFGOI",
    "UVWXYZARBC",
    "MNOPQRKSTU",
    "ABCDECARTZ",
]

# One self-contained input: the root forwards it to the children verbatim, so the
# word and the grid have to travel together. Data only — indices or instructions
# mixed in here come back as coordinate errors or as invented target words.
PUZZLE = json.dumps({"target_word": TARGET_WORD, "grid": GRID_ROWS}, indent=2)


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


EXPECTED_HITS = {
    ("AGENT", 1, 8, 5, 8, "S"),
}
EXPECTED_MISSING: set[str] = set()

QUERY = """\
Solve the word search puzzle in `INPUTS["puzzle"]`.

`json.loads(INPUTS["puzzle"])` gives `{"target_word": str, "grid": [str, ...]}`,
one string per grid row, so `grid[row][col]` is the letter at that coordinate.
Pass `INPUTS["puzzle"]` through unchanged to every sub-agent you create — a
sub-agent that does not get it cannot see the grid or the word to look for.

Rules for the coordinates you report:
- rows and columns are 0-indexed, counting from the top-left cell;
- `start_row`/`start_col` and `end_row`/`end_col` are inclusive, so an N-letter
  word covers N cells and its start and end are N-1 apart along its direction;
- before answering, slice the grid along the coordinates you are about to
  report and confirm they spell the target word.

`missing` lists target words you could not find anywhere in the grid — nothing
else, so leave it empty if you found the target word.

You can approach this problem with the following strategy:
1. The root should first delegate to three child agents: `rows`, `cols`,
    and `diagonals`.
2. Inside row and column agents, subdelegate the actual line searches:
    - `rows` should search rows for the target word and delegate each row in parallel to its own sub-agent (rows.<row_number>);
    - `cols` should search columns for the target word and delegate each column in parallel to its own sub-agent (cols.<column_number>).
3. `diagonals` should search diagonals by itself directly without delegating.
4. Each child agent should return a list of tuples, each containing the word found and its inclusive coordinates.
"""


def _hit_key(hit: WordHit) -> tuple[str, int, int, int, int, str]:
    return (
        hit.word,
        hit.start_row,
        hit.start_col,
        hit.end_row,
        hit.end_col,
        hit.direction,
    )


def print_tree(node: Node, depth: int = 0) -> None:
    """One indented line per node, from ``node`` down."""
    label = f"{node.type} [{node.config.path}]" if isinstance(node, AgentStart) else node.type
    print(f"{'  ' * depth}{label}")
    for child in node.children:
        print_tree(child, depth + 1)


def run(model: str, out_dir: Path) -> None:
    flow = Flow(build_client(model))

    graph = start(
        query=QUERY,
        inputs={"puzzle": PUZZLE},
        output_schema=json_schema_for(WordSearchResult),
        max_depth=2,
    )
    print_tree(graph)

    async def run_to_done() -> None:
        async for node in flow.run_streaming(graph):
            # Checkpointing as nodes land keeps the run directory current.
            graph.save(out_dir)
            print(f"- {node.parent_agent.config.path}: {node.type}")

    asyncio.run(run_to_done())
    flow.runtime.close_repls()

    print("=== TREE ===")
    print_tree(graph)

    # With an output_schema, done(...) records the parsed value, not raw text.
    result = WordSearchResult.model_validate(graph.result())
    actual = {_hit_key(hit) for hit in result.found}
    missing = set(result.missing)

    print("=== RESULT ===")
    print(result.model_dump_json(indent=2))
    if actual != EXPECTED_HITS or missing != EXPECTED_MISSING:
        raise SystemExit("agent returned word-search hits that do not match EXPECTED")

    print("agent returned the expected word-search hits!")
    path = graph.save(out_dir)
    agents = sum(isinstance(node, AgentStart) for node in graph.walk())
    print(f"\nwrote baseline run: {path}")
    print(f"  manifest: {path / 'graph.json'}")
    print(f"  agents:   {path / 'agents'} ({agents} agents)")
    save_example_graph(graph, "injection-word-search")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "_runs" / "word-search" / "baseline",
    )
    args = parser.parse_args()

    run(args.model, args.out_dir.resolve())


if __name__ == "__main__":
    main()
