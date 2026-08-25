"""RULER single needle-in-a-haystack benchmark adapter."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from benchmarks.eval import dataset
from benchmarks.eval.types import Dataset, Example, Prediction, Score


@dataset("official_sniah", tags=["rlm-comparison", "long-context"])
class RulerSNIAHDataset(Dataset):
    """RULER single-needle tasks using complete, untruncated prompts."""

    dataset_name = "tonychenxyz/ruler-full"

    def __init__(
        self,
        data_dir: str = "evals/data",
        split: str = "validation",
        max_samples: int | None = 50,
        category_filter: str = "niah_single",
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_samples = max_samples
        self.category_filter = category_filter
        self._dataset: Any | None = None
        self._indices: list[int] | None = None

    def examples(self, *, split: str, limit: int | None, seed: int) -> list[Example]:
        del split
        dataset, eligible = self._load()
        positions = list(range(len(eligible)))
        random.Random(seed).shuffle(positions)
        if limit is not None:
            positions = positions[: min(limit, len(positions))]
        return [
            self._example(dict(dataset[eligible[position]]), eligible[position])
            for position in positions
        ]

    def score(self, example: Example, prediction: Prediction) -> Score:
        candidates = [
            str(value).strip() for value in (example.expected or []) if str(value).strip()
        ]
        answer = prediction.answer.strip().lower()
        matched = next((candidate for candidate in candidates if candidate.lower() in answer), None)
        return Score(
            value=1.0 if matched is not None else 0.0,
            correct=matched is not None,
            details={"expected": candidates, "matched": matched},
        )

    def _load(self) -> tuple[Any, list[int]]:
        if self._dataset is not None and self._indices is not None:
            return self._dataset, self._indices
        try:
            from datasets import load_dataset, load_from_disk
            from datasets.utils.logging import disable_progress_bar
        except ImportError as exc:
            raise RuntimeError(
                "RULER S-NIAH requires the eval extra: pip install -e '.[eval]'"
            ) from exc

        disable_progress_bar()
        local = next(
            (
                path
                for path in (
                    self.data_dir / "ruler_full_mirror",
                    self.data_dir / "ruler-full",
                )
                if path.exists()
            ),
            None,
        )
        if local is not None:
            loaded = load_from_disk(str(local))
            dataset = (
                loaded[self.split] if hasattr(loaded, "keys") and self.split in loaded else loaded
            )
        else:
            dataset = load_dataset(self.dataset_name, "plain", split=self.split)

        categories = (
            dataset["category"] if "category" in dataset.column_names else [""] * len(dataset)
        )
        eligible = [
            index
            for index, category in enumerate(categories)
            if self.category_filter in str(category)
        ]
        if not eligible:
            eligible = list(range(len(dataset)))
        if self.max_samples is not None:
            eligible = eligible[: self.max_samples]
        if not eligible:
            raise ValueError("No RULER S-NIAH examples fit the configured filters.")

        self._dataset = dataset
        self._indices = eligible
        return dataset, eligible

    def _example(self, row: dict[str, Any], index: int) -> Example:
        prompt = _strip_ruler_prompt(str(row.get("prompt", "")))
        extra = _as_mapping(row.get("extra_info"))
        ground_truth = _as_mapping(extra.get("ground_truth"))
        raw_answers = ground_truth.get("answers") or []
        if isinstance(raw_answers, str):
            raw_answers = [raw_answers]
        answers = [str(answer) for answer in raw_answers]
        return Example(
            id=f"official_sniah_{index:05d}",
            prompt="Answer the final question in the text. Return ONLY the answer.",
            context={"context": prompt},
            expected=answers,
            metadata={
                "category": row.get("category"),
                "source_index": index,
                "context_chars": len(prompt),
            },
        )


def _strip_ruler_prompt(prompt: str) -> str:
    if "<|im_start|>user" in prompt:
        prompt = prompt.split("<|im_start|>user", 1)[1].split("<|im_end|>", 1)[0]
    return prompt.replace("<|im_start|>assistant", "").strip()


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


__all__ = ["RulerSNIAHDataset"]
