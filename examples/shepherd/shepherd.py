"""Shepherd-style explore / score / commit loop over a needle-in-a-haystack.

A Python *coordinator* drives a single coordinator ``Graph`` through the minimal
reversibility API — the same primitives Shepherd is built on, adapted to rflow's
"code is the action" model (state is rebuilt by replay, not a typed effect
algebra). The task is the needle-in-a-haystack from ``examples/needle/haystack.py``:
a magic number is buried in ``INPUTS["haystack"]`` and must be found.

The coordinator:

1. seeds shared base state — splits the haystack into chunks in its REPL,
2. **checkpoint -> speculate -> revert**: cheaply scan the first chunk; when the
   needle is not there, roll the speculation back out,
3. **fork -> run-N -> score -> select**: fork one branch per chunk (each inherits
   the ``chunks`` list via replay); each branch is a *real agent* that writes its
   own search code and calls ``done(...)``; score by whether it found the needle,
4. **merge** the winning branch back into the coordinator (its findings land in
   the prompt as one summary node),
5. **discard** the losing branches, then let the coordinator finalize the answer.

Unlike ``examples/control/branching/best_of_n.py`` (independent graphs), every
branch here is a real ``flow.fork`` of a shared base and the winner is folded back
with ``flow.merge`` — so the coordinator keeps running on one evolving graph.

This calls a real model, so it needs an API key (e.g. ``OPENAI_API_KEY``).

Usage:
    python examples/shepherd/shepherd.py
    python examples/shepherd/shepherd.py --model gpt-5-mini --num-lines 20000
"""

from __future__ import annotations

import argparse
import asyncio
import random
import re
import shutil
from pathlib import Path

from rflow.minimal.clients import AnthropicClient, OpenAIClient
from rflow.minimal import ExecAction, Flow, Graph, LLMOutput

ANSWER = "8675309"
NUM_CHUNKS = 5
MISS = "NONE"

QUERY = (
    "A magic number is buried in INPUTS['haystack']. Find it. The haystack has "
    "been split into the REPL list `chunks`; search the pieces."
)

# Coordinator scaffolding (deterministic): split the haystack (from INPUTS) into
# NUM_CHUNKS pieces that every forked branch inherits via replay.
BASE_CODE = (
    "import re\n"
    "lines = INPUTS['haystack'].split('\\n')\n"
    f"size = (len(lines) + {NUM_CHUNKS} - 1) // {NUM_CHUNKS}\n"
    f"chunks = ['\\n'.join(lines[i * size:(i + 1) * size]) for i in range({NUM_CHUNKS})]\n"
    "print(f'{len(lines)} lines -> {len(chunks)} chunks')"
)

# One cheap idea the coordinator tries first: scan only chunk 0. The needle sits
# in the middle of the haystack, so this misses and gets rolled back.
SPECULATION_CODE = (
    "spec_hit = re.search(r'The magic number is (\\d+)', chunks[0])\n"
    "spec_answer = spec_hit.group(1) if spec_hit else 'NONE'\n"
    "print('speculative scan of chunk 0 ->', spec_answer)"
)

BRANCH_INSTRUCTION = (
    "Search chunk {idx} for the magic number. The REPL already holds the list "
    "`chunks`; scan `chunks[{idx}]` for a line like 'The magic number is <digits>'. "
    "Call done with just the digits if found, otherwise done('NONE'). Do not print "
    "the chunk itself."
)

FINALIZE_INSTRUCTION = (
    "The search branches reported their findings above. Report the magic number by "
    "calling done(...) with just the digits."
)


def build_haystack(num_lines: int, answer: str, *, seed: int = 0) -> tuple[str, int]:
    rng = random.Random(seed)
    words = ["blah", "random", "text", "data", "content", "information", "sample"]
    lines = [
        " ".join(rng.choice(words) for _ in range(rng.randint(3, 8)))
        for _ in range(num_lines)
    ]
    needle_line = rng.randint(int(num_lines * 0.4), int(num_lines * 0.6))
    lines[needle_line] = f"The magic number is {answer}"
    return "\n".join(lines), needle_line


def make_client(model: str):
    return AnthropicClient(model) if model.startswith("claude") else OpenAIClient(model)


def extract_number(result: str | None) -> str | None:
    match = re.search(r"\d{5,}", result or "")
    return match.group(0) if match else None


async def coordinator_turn(flow: Flow, graph: Graph, code: str, *, label: str):
    """Scripted coordinator move: record one (llm_output, exec_action) + run it."""
    graph.commit(LLMOutput(content=label, code=code))
    graph.commit(ExecAction(code=code))
    return await flow.exec_turn(graph, code)


async def run_shepherd(root: Path, *, model: str, num_lines: int, max_iters: int) -> None:
    haystack, needle_line = build_haystack(num_lines, ANSWER)
    print(f"[setup] {num_lines} lines, needle on line {needle_line}, model={model}")

    flow = Flow(make_client(model), max_depth=1, max_iters=max_iters)

    # 1. Seed shared base state (chunks of the haystack) into the coordinator REPL.
    coordinator = Graph(query=QUERY, inputs={"haystack": haystack})
    flow.start(coordinator)
    await coordinator_turn(flow, coordinator, BASE_CODE, label="seed base state")

    # 2. checkpoint -> speculate -> revert. Scan chunk 0 cheaply; on a miss, roll
    #    the speculation back out so the coordinator continues from clean base state.
    checkpoint = coordinator.checkpoint()
    nodes_before = len(coordinator.nodes)
    await coordinator_turn(flow, coordinator, SPECULATION_CODE, label="speculate")
    spec_answer = flow.repl_for(coordinator).namespace["spec_answer"]
    print(f"[speculate] chunk 0 -> {spec_answer}")
    if spec_answer == MISS:
        coordinator.revert(checkpoint)
        await flow.rebuild_repl(coordinator)  # keep the REPL in sync with the graph
        print(
            f"[revert] speculation missed: {nodes_before + 3} -> "
            f"{len(coordinator.nodes)} nodes (fanning out instead)"
        )

    # 3. fork -> run-N -> score. One branch per chunk; each inherits `chunks` via
    #    replay and is a real agent that writes its own search and returns a number.
    branches: list[tuple[int, Graph, str]] = []
    for idx in range(NUM_CHUNKS):
        child = await flow.fork(coordinator)
        child.inject(BRANCH_INSTRUCTION.format(idx=idx))
        await flow.run_agent(child)
        result = child.result()
        branches.append((idx, child, result))
        child.save(root / f"branch_{idx}")
        print(f"[branch {idx}] found={result!r}")

    # score: prefer the branch that returned a magic-number-looking answer.
    best_idx, winner, best_result = max(
        branches, key=lambda item: extract_number(item[2]) is not None
    )
    best_number = extract_number(best_result) or MISS

    # 4. merge the winner back. The finding lands in the coordinator's prompt as one
    #    summary node (the coordinator reads it to finalize).
    await flow.merge(
        coordinator,
        winner,
        summary=f"needle found in chunk {best_idx}: {best_number}",
    )
    print(f"[merge] folded chunk {best_idx}; summary -> {coordinator.nodes[-1].content!r}")

    # 5. discard the losing branches (frees remote REPLs; a no-op on LocalRuntime).
    losers = [child for idx, child, _ in branches if idx != best_idx]
    flow.discard(*losers)

    # Finalize: the coordinator reads the merged summary and reports the number.
    coordinator.inject(FINALIZE_INSTRUCTION)
    await flow.run_agent(coordinator)
    flow.close_repls(coordinator.graph_id)
    coordinator.save(root / "coordinator")

    result = coordinator.result()
    print(f"\n[done] coordinator result: {result!r}")
    print(f"[check] actual={ANSWER} correct={ANSWER in result}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Shepherd-style needle-in-a-haystack")
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--num-lines", type=int, default=10_000)
    parser.add_argument("--max-iters", type=int, default=6)
    parser.add_argument(
        "--root-dir",
        default=str(Path(__file__).resolve().parents[1] / "_runs" / "shepherd"),
        help="where to drop saved graphs (default: examples/_runs/shepherd/)",
    )
    args = parser.parse_args()

    root = Path(args.root_dir).resolve()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    asyncio.run(
        run_shepherd(
            root, model=args.model, num_lines=args.num_lines, max_iters=args.max_iters
        )
    )


if __name__ == "__main__":
    main()
