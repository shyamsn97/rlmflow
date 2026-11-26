"""OOLONG-Pairs: the RLM paper's 20 pair-aggregation questions at 32K tokens.

Each question asks for every pair of users that satisfies a condition, so the
work grows with the square of the document, not its length. Gold pairs come from
``mit-oasys/oolong-pairs``; the document is the matching unlabeled
``trec_coarse`` window. Scored by F1 over the pairs in ``answer.txt``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from benchmarks.table.systems import Limits, Outcome

NAME = "oolong_pairs"
CONTEXT_LEN = 32_768
PAIR_DATASET = "mit-oasys/oolong-pairs"
PAIR_REVISION = "d1e1522b86ac0c169bbc890b0471408aaa29e8fa"
CONTEXT_DATASET = "lsteno/RLM-Evals"
CONTEXT_CONFIG = "oolong_pairs_contexts"
CONTEXT_REVISION = "a6aea6d06da9f08d701038b64195049cf71e1997"
CACHE = Path.home() / ".cache" / "rlmflow-bench" / f"oolong-pairs-{CONTEXT_LEN}.json"

TASK = """\
{question}

The source text is in `context.txt` in this directory. Write your final answer
to `answer.txt`: every pair as `(id1, id2)`, one per line, and nothing else."""

PARENTHESIZED = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)")
BARE = re.compile(r"(?<!\d)(\d+)\s*,\s*(\d+)(?!\d)")


def data() -> dict:
    if not CACHE.exists():
        from datasets import load_dataset
        from huggingface_hub import hf_hub_download

        rows = json.loads(
            Path(
                hf_hub_download(
                    repo_id=PAIR_DATASET,
                    filename=f"data/oolong-pairs-{CONTEXT_LEN}.json",
                    repo_type="dataset",
                    revision=PAIR_REVISION,
                )
            ).read_text()
        )
        contexts = load_dataset(
            CONTEXT_DATASET, CONTEXT_CONFIG, split="eval", revision=CONTEXT_REVISION
        )
        (context,) = [
            row["context_window_text"]
            for row in contexts
            if int(row.get("context_len") or 0) == CONTEXT_LEN
        ]
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(
            json.dumps(
                {
                    "context": context,
                    "questions": {
                        f"pairs_{int(row['id']):02d}": {
                            "question": str(row["question"]).strip(),
                            "expected": list(row.get("answer") or []),
                        }
                        for row in rows
                    },
                }
            )
        )
    return json.loads(CACHE.read_text())


def ids() -> list[str]:
    return sorted(data()["questions"])


def prepare(task_id: str, workdir: Path, limits: Limits) -> str:
    d = data()
    (workdir / "context.txt").write_text(d["context"])
    return TASK.format(question=d["questions"][task_id]["question"])


def score(task_id: str, workdir: Path, outcome: Outcome) -> dict:
    expected = pairs(data()["questions"][task_id]["expected"])
    path = workdir / "answer.txt"
    answer = path.read_text() if path.exists() else ""
    predicted = pairs(answer)
    hits = len(expected & predicted)
    precision = hits / len(predicted) if predicted else float(not expected)
    recall = hits / len(expected) if expected else float(not predicted)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "score": f1,
        "precision": precision,
        "recall": recall,
        "answer": answer[:2000],
        "expected_pairs": len(expected),
        "predicted_pairs": len(predicted),
    }


def pairs(value) -> set[tuple[int, int]]:
    text = "\n".join(map(str, value)) if isinstance(value, list) else str(value)
    found = PARENTHESIZED.findall(text) or BARE.findall(text)
    return {(min(int(a), int(b)), max(int(a), int(b))) for a, b in found}
