"""One GRS-QA mixed dependency-graph problem."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlopen

from benchmarks.eval import dataset
from benchmarks.eval.tasks._delegation_utils import (
    exact_or_alias,
    parse_json_answer,
    set_f1,
    token_f1,
)
from benchmarks.eval.types import Dataset, Example, Prediction, Score

REVISION = "a7dee05c1792b552355354fc310c273256bcdc3b"
SOURCE_ID = "sample-comparison-1"


@dataset("delegation_grsqa", tags=["delegation", "task-graph", "qa"])
class GRSQATaskGraphDataset(Dataset):
    def __init__(
        self,
        data_dir: str = "evals/data",
    ) -> None:
        self.data_dir = Path(data_dir)
        self._rows: list[dict] | None = None

    def examples(self, *, split: str, limit: int | None, seed: int) -> list[Example]:
        del split, seed
        row = self._load()[0]
        return [] if limit == 0 else [self._example(row)]

    def score(self, example: Example, prediction: Prediction) -> Score:
        expected = dict(example.expected)
        try:
            parsed = parse_json_answer(prediction.answer)
        except (TypeError, ValueError):
            parsed = {"answer": prediction.answer, "evidence": []}
        if not isinstance(parsed, dict):
            parsed = {"answer": str(parsed), "evidence": []}
        answer = str(parsed.get("answer", ""))
        answer_f1 = token_f1(answer, expected["answer"])
        answer_exact = exact_or_alias(answer, expected["answer"])
        evidence_f1 = set_f1(parsed.get("evidence", []), expected["evidence"])
        value = answer_f1 * evidence_f1
        return Score(
            value=value,
            correct=answer_exact and evidence_f1 == 1.0,
            details={
                "answer_exact": answer_exact,
                "answer_f1": answer_f1,
                "evidence_f1": evidence_f1,
            },
        )

    def _load(self) -> list[dict]:
        if self._rows is not None:
            return self._rows
        path = self.data_dir / "delegation" / "grsqa" / "example.json"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            url = (
                "https://raw.githubusercontent.com/kyone138/grs-qa/"
                f"{REVISION}/sample_data/example.json"
            )
            with urlopen(url, timeout=30) as response:
                path.write_bytes(response.read())
        self._rows = [json.loads(path.read_text())]
        return self._rows

    def _example(self, row: dict) -> Example:
        positive = list(row["pos_graph"]["nodes"])
        negative = [
            node
            for graph in row.get("neg_graph", [])
            for node in graph.get("nodes", [])
            if node.get("type") == "neg_para"
        ]
        context = "\n\n".join(
            f"[{node_id}] {node['evidence']}"
            for node_id, node in [
                *((str(node["node_id"]), node) for node in positive),
                *((f"neg:{node['node_id']}", node) for node in negative),
            ]
        )
        answer = row.get("answer", "")
        if isinstance(answer, list):
            answer = answer[0]
        return Example(
            id=f"delegation_grsqa_09_{SOURCE_ID}",
            prompt=(
                f"{row.get('question', '')}\n\n"
                "Return JSON with `answer` and a list of supporting evidence node IDs."
            ),
            context={"context": context},
            expected={
                "answer": str(answer),
                "evidence": [str(node["node_id"]) for node in positive],
            },
            metadata={
                "source_id": SOURCE_ID,
                "problem": 9,
                "native_scorer": "answer_evidence_joint_f1",
                "graph_type": "comparison-with-hard-negative",
            },
        )


__all__ = ["GRSQATaskGraphDataset", "SOURCE_ID"]
