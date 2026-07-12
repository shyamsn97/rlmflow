"""Recursive map-reduce summarization over a long document.

The canonical RLM pattern: a document too long to summarize well in one shot
is passed as an input (`document`), and the root agent splits it into chunks,
launches a summary of each chunk on a cheap `fast` child in parallel with
`await launch_subagents([...])` (the *map* step), then synthesizes the child
summaries into one final summary (the *reduce* step).

Usage:
    python examples/summarizer.py
    python examples/summarizer.py --sections 40 --no-viz
    python examples/summarizer.py --input-file path/to/doc.txt
    python examples/summarizer.py --docker-image rlmflow:local --viewer
"""

from __future__ import annotations

import argparse
import asyncio
import random
from pathlib import Path

from rflow.clients import AnthropicClient, OpenAIClient
from rflow import DockerRuntime, Flow, Graph, LiveTreeRenderer

_TOPICS = [
    "the migration to the new billing system",
    "Q3 infrastructure reliability incidents",
    "the hiring plan for the platform team",
    "customer feedback on the onboarding flow",
    "the cost of the data warehouse",
    "the security audit findings",
    "the roadmap for the mobile app",
    "latency regressions in the search service",
]

_FILLER = [
    "The team reviewed the relevant dashboards and agreed on next steps.",
    "Several stakeholders raised concerns that were noted for follow-up.",
    "A decision was deferred pending more data from the analytics group.",
    "Action items were assigned with owners and due dates.",
    "The discussion referenced last quarter's results for context.",
]


def generate_long_document(sections: int, *, seed: int = 7) -> str:
    """Build a synthetic multi-section report with one planted key fact.

    The fact ("the launch date was moved to ...") is buried in a random
    section so you can eyeball whether the final summary surfaced it.
    """

    rng = random.Random(seed)
    planted_section = rng.randint(1, sections)
    launch_date = "March 14, 2027"

    parts: list[str] = []
    for i in range(1, sections + 1):
        topic = _TOPICS[i % len(_TOPICS)]
        body = [f"## Section {i}: {topic.title()}", ""]
        for _ in range(rng.randint(6, 12)):
            body.append(rng.choice(_FILLER))
        if i == planted_section:
            body.append(
                f"Critically, the team confirmed the public launch date was "
                f"moved to {launch_date}."
            )
        parts.append("\n".join(body))

    print(
        f"Generated {sections}-section document "
        f"(key fact planted in section {planted_section})."
    )
    return "\n\n".join(parts)


def build_llm(model: str):
    return (
        AnthropicClient(model)
        if model.startswith("claude")
        else OpenAIClient(model)
    )


SUMMARIZE_QUERY = """\
The full document is in `INPUTS["document"]` (a str). It is long, so summarize
it with a map-reduce strategy instead of reading it all at once:

1. Split `INPUTS["document"]` into a handful of contiguous chunks (aim for ~4-8
   chunks). For example: `lines = INPUTS["document"].splitlines()`, then slice
   into ranges.
2. Summarize all chunks in parallel with one call, using the "fast" model.
   Pass each chunk to its child via `inputs` (a dict of str -> str):
   summaries = await launch_subagents([
       {"name": f"chunk-{i}",
        "query": "Summarize `INPUTS['passage']` in 3-4 sentences, preserving "
                 "any concrete facts, dates, and decisions.",
        "inputs": {"passage": chunk_text},
        "model": "fast"}
       for i, chunk_text in enumerate(chunks)
   ])
   launch_subagents runs the children concurrently and returns their summaries
   in order.
3. Combine the child summaries into a single coherent summary of the whole
   document (a short intro paragraph plus bullet points), then call
   done(final_summary).
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Recursive map-reduce summarizer")
    parser.add_argument("--sections", type=int, default=30)
    parser.add_argument(
        "--input-file",
        default=None,
        help="Summarize this file instead of a synthetic doc.",
    )
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--fast-model", default="gpt-5-nano")
    parser.add_argument(
        "--docker-image",
        default=None,
        help="If set, run agent code inside this Docker image (e.g. rlmflow:local).",
    )
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--max-iters", type=int, default=15)
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parents[1] / "_runs" / "summarizer"),
        help="Save the final run here (default: examples/_runs/summarizer/).",
    )
    parser.add_argument(
        "--viewer", action="store_true", help="Open the viewer after finishing."
    )
    args = parser.parse_args()

    print(f">>> {'DOCKER' if args.docker_image else 'LOCAL'} RUNTIME")

    if args.input_file:
        document = Path(args.input_file).read_text()
        print(f"Loaded {len(document):,} chars from {args.input_file}")
    else:
        document = generate_long_document(args.sections)

    runtime = DockerRuntime(args.docker_image) if args.docker_image else None

    llm_clients = None
    if args.fast_model:
        llm_clients = {"fast": build_llm(args.fast_model)}

    flow = Flow(
        build_llm(args.model),
        llm_clients=llm_clients,
        runtime=runtime,
        max_depth=args.max_depth,
        max_iters=args.max_iters,
    )

    graph = Graph(query=SUMMARIZE_QUERY)

    async def drive() -> None:
        renderer = LiveTreeRenderer(clear=not args.no_viz)
        async for event in flow.run_streaming(graph, inputs={"document": document}):
            renderer.handle(event, graph)

    asyncio.run(drive())

    print(f"\n{'=' * 60}\nFINAL SUMMARY\n{'=' * 60}")
    print(graph.result())

    if args.out_dir:
        path = graph.save(Path(args.out_dir))
        print(f"\nGraph saved to {path}")

    if args.viewer:
        print("Viewer support is not part of the minimal example path.")

    flow.close_repls(graph.graph_id)


if __name__ == "__main__":
    main()
