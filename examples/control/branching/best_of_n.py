"""Best-of-N with independent Flow branches.

Each branch is a fresh ``flow.start(...)`` run saved to its own ``graph.json``,
demonstrating the Node-only surface.

Usage:
    python examples/control/branching/best_of_n.py
    python examples/control/branching/best_of_n.py --n 8
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from rlmflow import (
    Flow,
    LLMUsage,
)

FRUITS = [
    ("lemon", "citrus"),
    ("orange", "citrus"),
    ("apple", "not_citrus"),
    ("lime", "citrus"),
]
TRUTH = dict(FRUITS)

QUERY = (
    "Classify each fruit: "
    + ", ".join(name for name, _ in FRUITS)
    + ". Delegate one classification per fruit, then return '<fruit>=<label>, ...'."
)

ROOT_REPLY = (
    "```repl\n"
    f"fruits = {[name for name, _ in FRUITS]!r}\n"
    "handles = [await launch_subagent("
    "f\"Classify {name} as 'citrus' or 'not_citrus'.\", "
    "name=f'classify_{name}') for name in fruits]\n"
    "results = [await handle.wait_for_result() for handle in handles]\n"
    "done(', '.join(f'{name}={r}' for name, r in zip(fruits, results)))\n"
    "```"
)


class MockLLM:
    APPLE_CANDIDATES = ["citrus", "not_citrus", "citrus", "not_citrus"]

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.call_count = 0

    def chat(self, messages, *args, **kwargs):
        self.call_count += 1
        self.last_usage = LLMUsage(input_tokens=80, output_tokens=20)
        text = messages[-1]["content"]
        if "as 'citrus' or 'not_citrus'" not in text:
            return ROOT_REPLY
        for name, correct in FRUITS:
            if name in text:
                answer = correct
                if name == "apple":
                    answer = self.APPLE_CANDIDATES[self.seed % len(self.APPLE_CANDIDATES)]
                return f'```repl\ndone("{answer}")\n```'
        return ROOT_REPLY


def score(result: str) -> tuple[int, dict[str, str]]:
    preds = dict(chunk.strip().split("=", 1) for chunk in result.split(",") if "=" in chunk)
    preds = {key.strip(): value.strip() for key, value in preds.items()}
    return sum(1 for key, value in TRUTH.items() if preds.get(key) == value), preds


def run_branch(root: Path, idx: int) -> tuple[str, int, dict[str, str], int]:
    llm = MockLLM(seed=idx)
    flow = Flow(llm)
    graph = flow.start(QUERY, max_depth=1, max_iters=10)
    flow.run(graph)
    # Each branch persists to its own directory (graph.json).
    graph.save(root / f"branch_{idx}")
    flow.runtime.close_repls()
    result = graph.result()
    correct, preds = score(result)
    return result, correct, preds, llm.call_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument(
        "--root-dir",
        default=str(Path(__file__).resolve().parents[2] / "_runs" / "best-of-n"),
        help="where to drop per-branch run directories (default: examples/_runs/best-of-n/)",
    )
    args = parser.parse_args()

    root = Path(args.root_dir).resolve()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    branches = [run_branch(root, i) for i in range(args.n)]
    for i, (_result, correct, preds, calls) in enumerate(branches):
        print(f"branch_{i}: score={correct}/{len(TRUTH)} calls={calls} {preds}")

    best_result, best_correct, best_preds, _calls = max(branches, key=lambda item: item[1])
    print(f"\n[best] score={best_correct}/{len(TRUTH)} {best_preds}")
    print(best_result)


if __name__ == "__main__":
    main()
