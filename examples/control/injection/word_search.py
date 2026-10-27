"""Generate the baseline word-search run used by the injection example.

This is the original route: ask a real Flow agent to find ``AGENT`` by
delegating direction-specific search to three child agents:

- ``rows`` searches rows east/west;
- ``cols`` searches columns north/south;
- ``diagonals`` searches diagonals in all four diagonal directions.

That route is intentionally plausible but more complicated than necessary. It
creates a real delegating turn at the root, with each child agent launched by
the model turn whose cell called ``launch_agent``. The finished graph is saved
as a run directory that ``inject_variants.py`` loads with ``Graph.load``.

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

from rlmflow import AGENT_START, AgentConfig, Flow, Graph, json_schema_for
from rlmflow.llm import client_for

examples_dir = next(p for p in Path(__file__).resolve().parents if p.name == "examples")
if str(examples_dir) not in sys.path:
    sys.path.insert(0, str(examples_dir))

from common import print_agent_tree, print_run, save_example_graph  # noqa: E402

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
Solve the word search puzzle in `puzzle`.

`json.loads(puzzle)` gives `{"target_word": str, "grid": [str, ...]}`,
one string per grid row, so `grid[row][col]` is the letter at that coordinate.
Pass `puzzle` through unchanged to every sub-agent you create — a
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


def run(model: str, out_dir: Path) -> None:
    flow = Flow(client_for(model))

    g = Graph()
    root = g.create_agent(
        QUERY,
        config=AgentConfig(
            output_schema=json_schema_for(WordSearchResult),
            max_depth=2,
        ),
        puzzle=PUZZLE,
    )
    print_agent_tree(g, root)

    asyncio.run(print_run(flow, g, out_dir))
    flow.runtime.close_repls()

    print("=== TREE ===")
    print_agent_tree(g, root)

    # With an output_schema, submit(...) records the parsed value, not raw text.
    result = WordSearchResult.model_validate(g.result(root))
    actual = {_hit_key(hit) for hit in result.found}
    missing = set(result.missing)

    print("=== RESULT ===")
    print(result.model_dump_json(indent=2))
    if actual != EXPECTED_HITS or missing != EXPECTED_MISSING:
        raise SystemExit("agent returned word-search hits that do not match EXPECTED")

    print("agent returned the expected word-search hits!")
    path = g.save(out_dir)
    print(f"\nwrote baseline run: {path}")
    print(f"  graph:  {path / 'graph.json'} ({len(g.nodes(AGENT_START))} agents)")
    save_example_graph(g, "injection-word-search")


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
