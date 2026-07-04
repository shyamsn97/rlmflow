"""Generate the baseline word-search run used by the injection example.

This is the original route: ask a real Flow agent to find ``AGENT`` by
delegating direction-specific search to three child agents:

- ``rows`` searches rows east/west;
- ``cols`` searches columns north/south;
- ``diagonals`` searches diagonals in all four diagonal directions.

That route is intentionally plausible but more complicated than necessary. It
creates a real root ``SupervisingOutput`` that ``inject_variants.py`` can later
replace with a direct scanner route. The finished run is saved as a run
directory (``graph.json`` manifest plus per-agent logs nested under
``agents/``) that ``inject_variants.py`` loads with minimal ``Graph.load``.

Run:
    export OPENAI_API_KEY=...
    python examples/control/injection/word_search.py
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from rflow.clients import AnthropicClient, OpenAIClient
from rflow.minimal import Flow, render_tree


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

TARGET_WORD = "AGENT"

GRID = f"""Word Search Grid:

TYPHONQWER
LMNOPQRSAT
ZGXYZLMNGO
ABDDEFGHEI
QRSATUVWNY
JKLMPQRSTF
ABCDEPFGOI
UVWXYZARBC
MNOPQRKSTU
ABCDECARTZ

Target Word:
{TARGET_WORD}

Row and Column indices start at 0.
"""


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
Solve the word search puzzle in `INPUTS["grid"]`.

You can aproach this problem with the following strategy:
1. The root should first delegate to three child agents: `rows`, `cols`,
    and `diagonals`.
2. Inside row and column agents, subdelegate the actual line searches:
    - `rows` should search rows for the target words and delegate each row in parallel to its own sub-agent (rows.<row_number>);
    - `cols` should search columns for the target words and delegate each column in parallel to its own sub-agent (cols.<column_number>).
3. `diagonals` should search diagonals by itself directly without delegating.
4. Each child agent should return a list of tuples, each containing the word found and its inclusive coordinates.
"""


def client_for_model(model: str):
    return (
        AnthropicClient(model)
        if model.startswith("claude")
        else OpenAIClient(model)
    )


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
    flow = Flow(
        client_for_model(model),
        max_depth=2,
    )

    graph = flow.start(QUERY, {"grid": GRID}, output_schema=WordSearchResult)
    print(render_tree(graph))

    async def run_to_done() -> None:
        async for _event in flow.run_streaming(graph):
            print(render_tree(graph))

    asyncio.run(run_to_done())
    flow.close_repls(graph.graph_id)

    result = WordSearchResult.model_validate_json(graph.result())
    actual = {_hit_key(hit) for hit in result.found}
    missing = set(result.missing)

    print("=== RESULT ===")
    print(result.model_dump_json(indent=2))
    if actual != EXPECTED_HITS or missing != EXPECTED_MISSING:
        raise SystemExit("agent returned word-search hits that do not match EXPECTED")

    print("agent returned the expected word-search hits!")
    path = graph.save(out_dir)
    print(f"\nwrote baseline run: {path}")
    print(f"  manifest: {path / 'graph.json'}")
    print(f"  agents:   {path / 'agents'} ({len(list(graph.agents))} agents)")
    _save_example_graph(graph, __file__, "injection-word-search")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "_runs"
        / "word-search"
        / "baseline",
    )
    args = parser.parse_args()

    run(args.model, args.out_dir.resolve())


if __name__ == "__main__":
    main()
