"""LiveCodeBench dataset adapter."""

from __future__ import annotations

import html
import json
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from benchmarks.eval import dataset
from benchmarks.eval.types import Dataset, Example, Prediction, Score


@dataset("official_livecodebench", tags=["official", "code"])
class LiveCodeBenchDataset(Dataset):
    """LiveCodeBench code generation lite."""

    dataset_name = "livecodebench/code_generation_lite"
    _hf_base = "https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main"
    _jsonl_files = [
        f"{_hf_base}/test.jsonl",
        f"{_hf_base}/test2.jsonl",
        f"{_hf_base}/test3.jsonl",
        f"{_hf_base}/test4.jsonl",
        f"{_hf_base}/test5.jsonl",
    ]

    def __init__(
        self,
        data_dir: str = "evals/data",
        max_samples: int | None = None,
        test_timeout: int = 10,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.max_samples = max_samples
        self.test_timeout = test_timeout
        self._rows: list[dict[str, Any]] | None = None

    def examples(self, *, split: str, limit: int | None, seed: int) -> list[Example]:
        rows = _select_rows(self._load(split), limit=limit, seed=seed)
        return [self._example(row) for row in rows]

    def score(self, example: Example, prediction: Prediction) -> Score:
        try:
            info = json.loads(str(example.expected or "{}"))
        except json.JSONDecodeError:
            info = {}
        test_cases = info.get("test_cases", [])
        code = _extract_code_block(prediction.answer)
        passed_all, passed, total = _execute_against_tests(
            code,
            test_cases if isinstance(test_cases, list) else [],
            timeout=self.test_timeout,
        )
        value = passed / total if total else 0.0
        return Score(
            value=value,
            correct=passed_all if total else False,
            details={
                "passed": passed,
                "total": total,
                "expected": f"{passed}/{total} public tests",
            },
        )

    def _load(self, split: str) -> list[dict[str, Any]]:
        del split
        if self._rows is not None:
            return self._rows
        local = self.data_dir / "livecodebench"
        if local.exists():
            rows = _load_jsonl_dir(local)
        else:
            try:
                from datasets import load_dataset
                from datasets.utils.logging import disable_progress_bar
            except ImportError as exc:
                raise RuntimeError(
                    "LiveCodeBench requires the eval extra: pip install -e '.[eval]'"
                ) from exc
            disable_progress_bar()
            rows = [
                dict(row)
                for row in load_dataset("json", data_files=self._jsonl_files, split="train")
            ]
        if self.max_samples:
            rows = rows[: self.max_samples]
        if not rows:
            raise ValueError("No LiveCodeBench examples were loaded.")
        self._rows = [{**row, "_source_index": index} for index, row in enumerate(rows)]
        return self._rows

    def _example(self, row: dict[str, Any]) -> Example:
        public_tests = _parse_public_tests(row.get("public_test_cases", "[]"))
        examples_text = _format_public_tests(public_tests)
        title = str(row.get("question_title", "")).strip()
        content = _strip_html(str(row.get("question_content", "")))
        starter = str(row.get("starter_code", "")).strip()
        parts = [f"Solve this programming problem in Python.\n\n## {title}\n\n{content}"]
        if examples_text:
            parts.append(f"\n\n## Examples\n\n{examples_text}")
        if starter:
            parts.append(f"\n\n## Starter Code\n```python\n{starter}\n```")
        parts.append(
            "\n\nWrite a complete Python solution. For stdin/stdout problems, read from "
            "stdin and print to stdout. For function problems, implement the given "
            "function signature."
        )
        index = int(row.get("_source_index") or 0)
        return Example(
            id=f"official_livecodebench_{index:05d}",
            prompt="".join(parts),
            expected=json.dumps({"test_cases": public_tests, "starter_code": starter}),
            metadata={
                "question_id": row.get("question_id"),
                "platform": row.get("platform"),
                "difficulty": row.get("difficulty"),
                "num_public_tests": len(public_tests),
            },
        )


def _select_rows(
    rows: list[dict[str, Any]], *, limit: int | None, seed: int
) -> list[dict[str, Any]]:
    count = limit or 1
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    return [rows[index] for index in indices[: min(count, len(indices))]]


def _load_jsonl_dir(path: Path) -> list[dict[str, Any]]:
    files = sorted(item for item in path.iterdir() if item.suffix in {".json", ".jsonl"})
    rows: list[dict[str, Any]] = []
    for file in files:
        if file.suffix == ".json":
            payload = json.loads(file.read_text(encoding="utf-8"))
            rows.extend(payload if isinstance(payload, list) else payload.get("data", []))
            continue
        for line in file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _parse_public_tests(raw: Any) -> list[dict[str, Any]]:
    try:
        tests = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError):
        return []
    return tests if isinstance(tests, list) else []


def _format_public_tests(tests: list[dict[str, Any]]) -> str:
    examples = []
    for index, test in enumerate(tests[:3]):
        input_text = str(test.get("input", "")).strip()
        output_text = str(test.get("output") or test.get("expected_output", "")).strip()
        examples.append(f"Example {index + 1}:\nInput:\n{input_text}\nOutput:\n{output_text}")
    return "\n\n".join(examples)


def _strip_html(text: str) -> str:
    text = re.sub(r"<pre[^>]*>", "\n```\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</pre>", "\n```\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<code[^>]*>", "`", text, flags=re.IGNORECASE)
    text = re.sub(r"</code>", "`", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    return html.unescape(text).strip()


def _extract_code_block(response: str) -> str:
    for pattern in (r"```python\s*\n(.*?)```", r"```\s*\n(.*?)```"):
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()
    return response.strip()


def _execute_against_tests(
    code: str,
    test_cases: list[dict[str, Any]],
    *,
    timeout: int,
) -> tuple[bool, int, int]:
    passed = 0
    total = len(test_cases)
    if not code or total == 0:
        return False, 0, total
    for test in test_cases:
        expected = str(test.get("output") or test.get("expected_output", "")).strip()
        try:
            result = subprocess.run(
                [sys.executable, "-c", code],
                input=str(test.get("input", "")),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except (subprocess.TimeoutExpired, OSError):
            continue
        if result.stdout.strip() == expected:
            passed += 1
    return passed == total, passed, total


__all__ = ["LiveCodeBenchDataset"]
