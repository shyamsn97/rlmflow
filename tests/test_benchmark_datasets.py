from __future__ import annotations

from benchmarks.eval.run import build_parser, config_from_args
from benchmarks.eval.tasks.aime import AIME2025Dataset
from benchmarks.eval.tasks.longbench import CodeQADataset
from benchmarks.eval.tasks.sniah import RulerSNIAHDataset
from benchmarks.eval.types import Prediction


def test_rlm_core_alias_expands_to_minimal_comparison_suite():
    args = build_parser().parse_args(
        [
            "--dataset",
            "rlm-core",
            "--runner",
            "fake",
            "--model",
            "fake",
            "--limit",
            "2",
        ]
    )

    config = config_from_args(args)

    assert [spec.name for spec in config.datasets] == [
        "official_sniah",
        "official_aime_2025",
        "official_sudoku_extreme",
        "oolong",
        "official_codeqa",
    ]
    assert config.limit == 2
    assert config.seeds == [0]


def test_full_flag_disables_example_limit():
    args = build_parser().parse_args(
        ["--dataset", "rlm-core", "--runner", "fake", "--model", "fake", "--full"]
    )

    assert config_from_args(args).limit is None


def test_aime_uses_complete_rows_and_exact_integer_scoring():
    dataset = AIME2025Dataset()
    dataset._rows = [
        {"problem": "First", "answer": "17", "_source_index": 0},
        {"problem": "Second", "answer": "23", "_source_index": 1},
        {"problem": "Third", "answer": "42", "_source_index": 2},
    ]

    all_examples = dataset.examples(split="test", limit=None, seed=0)
    short_examples = dataset.examples(split="test", limit=2, seed=0)

    assert len(all_examples) == 3
    assert len(short_examples) == 2
    assert dataset.score(
        all_examples[0],
        Prediction(answer=f"The answer is {all_examples[0].expected}."),
    ).correct
    assert not dataset.score(all_examples[0], Prediction(answer="999")).correct


def test_codeqa_filters_longbench_without_truncating_context(monkeypatch):
    rows = [
        {
            "domain": "Code Repository Understanding",
            "sub_domain": "Code repo QA",
            "question": "Which function parses the config?",
            "choice_A": "load",
            "choice_B": "parse",
            "choice_C": "save",
            "choice_D": "render",
            "answer": "B",
            "context": "complete repository context",
        },
        {
            "domain": "Single-Document QA",
            "sub_domain": "Paper QA",
            "question": "Not code",
            "choice_A": "A",
            "choice_B": "B",
            "choice_C": "C",
            "choice_D": "D",
            "answer": "A",
            "context": "paper",
        },
    ]
    monkeypatch.setattr(
        "benchmarks.eval.tasks.longbench._load_hf_rows",
        lambda *args, **kwargs: rows,
    )
    dataset = CodeQADataset()

    examples = dataset.examples(split="test", limit=None, seed=0)

    assert len(examples) == 1
    assert examples[0].id == "official_codeqa_00000"
    assert examples[0].context == {"context": "complete repository context"}
    assert dataset.score(examples[0], Prediction(answer="B")).correct


def test_sniah_extracts_ground_truth_from_complete_ruler_prompt():
    dataset = RulerSNIAHDataset()
    dataset._dataset = [
        {
            "category": "niah_single_1",
            "prompt": (
                "<|im_start|>user\nA long context.\nWhat is the code?"
                "<|im_end|>\n<|im_start|>assistant"
            ),
            "extra_info": {"ground_truth": {"answers": ["purple-17"]}},
        }
    ]
    dataset._indices = [0]

    example = dataset.examples(split="test", limit=None, seed=0)[0]

    assert example.context == {"context": "A long context.\nWhat is the code?"}
    assert example.expected == ["purple-17"]
    assert dataset.score(example, Prediction(answer="purple-17")).correct
