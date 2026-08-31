"""Small stable suites for delegation-policy iteration."""

from __future__ import annotations

from dataclasses import replace

from benchmarks.eval import dataset
from benchmarks.eval.tasks.arc_agi import ArcAgiTaskGraphDataset
from benchmarks.eval.tasks.delegation_codeqa import DelegationCodeQADataset
from benchmarks.eval.tasks.delegation_sudoku import DelegationSudokuDataset
from benchmarks.eval.tasks.musique import MuSiQueTaskGraphDataset
from benchmarks.eval.tasks.parallelqa import ParallelQATaskGraphDataset
from benchmarks.eval.tasks.planbench import PlanBenchTaskGraphDataset
from benchmarks.eval.tasks.twowiki import TwoWikiTaskGraphDataset
from benchmarks.eval.types import Dataset, Example, Prediction, Score

FIVE_TASKS = (
    ("delegation_parallelqa_22", "local_lookup_control"),
    ("delegation_parallelqa_94", "parallel_lookup"),
    ("delegation_planbench_16_logistics_instance_20", "candidate_and_verification"),
    ("delegation_sudoku_18_cross-product", "local_constraint_control"),
    ("delegation_arc_agi_20_2ba387bc", "parallel_analysis_and_synthesis"),
)
TEN_TASKS = (
    ("delegation_parallelqa_11", "local_numeric_control"),
    ("delegation_parallelqa_22", "local_lookup_control"),
    ("delegation_parallelqa_63", "local_arithmetic_control"),
    ("delegation_parallelqa_94", "parallel_lookup"),
    ("delegation_musique_05_4hop1__38130_8966_31714_79432", "multi_hop_retrieval"),
    ("delegation_twowiki_08_d594f50208c111ebbd8bac1f6bf848b6", "parallel_evidence"),
    ("delegation_planbench_16_logistics_instance_20", "candidate_and_verification"),
    ("delegation_codeqa_17_official_codeqa_00072", "long_context_isolation"),
    ("delegation_sudoku_18_cross-product", "local_constraint_control"),
    ("delegation_arc_agi_20_2ba387bc", "parallel_analysis_and_synthesis"),
)
REGRESSION_TASKS = (
    ("delegation_parallelqa_63", "authoritative_tool_selection"),
    ("delegation_musique_05_4hop1__38130_8966_31714_79432", "child_source_verification"),
    ("delegation_twowiki_08_d594f50208c111ebbd8bac1f6bf848b6", "structured_result"),
    ("delegation_planbench_16_logistics_instance_20", "candidate_and_verification"),
    ("delegation_arc_agi_20_2ba387bc", "recursive_budget_control"),
)
SELECTED_TASKS = FIVE_TASKS


class _DelegationIterationDataset(Dataset):
    selected_tasks: tuple[tuple[str, str], ...] = ()

    def __init__(self, sources: tuple[Dataset, ...]) -> None:
        self._datasets = sources

    def examples(self, *, split: str, limit: int | None, seed: int) -> list[Example]:
        available: dict[str, Example] = {}
        for source in self._datasets:
            for example in source.examples(split=split, limit=None, seed=seed):
                available[example.id] = example

        missing = [
            example_id for example_id, _role in self.selected_tasks if example_id not in available
        ]
        if missing:
            raise ValueError(f"delegation iteration suite is missing tasks: {missing}")

        selected = [
            replace(
                available[example_id],
                metadata={**available[example_id].metadata, "iteration_role": role},
            )
            for example_id, role in self.selected_tasks
        ]
        return selected if limit is None else selected[:limit]

    def score(self, example: Example, prediction: Prediction) -> Score:
        prefixes = (
            ("delegation_parallelqa_", ParallelQATaskGraphDataset),
            ("delegation_musique_", MuSiQueTaskGraphDataset),
            ("delegation_twowiki_", TwoWikiTaskGraphDataset),
            ("delegation_planbench_", PlanBenchTaskGraphDataset),
            ("delegation_codeqa_", DelegationCodeQADataset),
            ("delegation_sudoku_", DelegationSudokuDataset),
            ("delegation_arc_agi_", ArcAgiTaskGraphDataset),
        )
        for prefix, source_type in prefixes:
            if example.id.startswith(prefix):
                source = next(
                    (item for item in self._datasets if isinstance(item, source_type)),
                    None,
                )
                if source is not None:
                    return source.score(example, prediction)
        raise ValueError(f"no scorer for iteration task {example.id}")


@dataset("delegation-iteration-five", tags=["delegation", "iteration", "canary"])
class DelegationIterationDataset(_DelegationIterationDataset):
    """A stable mix of delegation opportunities and local controls."""

    selected_tasks = FIVE_TASKS

    def __init__(self, data_dir: str = "evals/data") -> None:
        super().__init__(
            (
                ParallelQATaskGraphDataset(data_dir=data_dir),
                PlanBenchTaskGraphDataset(data_dir=data_dir),
                DelegationSudokuDataset(),
                ArcAgiTaskGraphDataset(data_dir=data_dir),
            )
        )


@dataset("delegation-iteration-ten", tags=["delegation", "iteration"])
class DelegationIterationTenDataset(_DelegationIterationDataset):
    """A broader ten-task suite that remains cheap enough for prompt iteration."""

    selected_tasks = TEN_TASKS

    def __init__(self, data_dir: str = "evals/data") -> None:
        super().__init__(
            (
                ParallelQATaskGraphDataset(data_dir=data_dir),
                MuSiQueTaskGraphDataset(data_dir=data_dir),
                TwoWikiTaskGraphDataset(data_dir=data_dir),
                PlanBenchTaskGraphDataset(data_dir=data_dir),
                DelegationCodeQADataset(data_dir=data_dir),
                DelegationSudokuDataset(),
                ArcAgiTaskGraphDataset(data_dir=data_dir),
            )
        )


@dataset("delegation-regression-five", tags=["delegation", "iteration", "regression"])
class DelegationRegressionDataset(_DelegationIterationDataset):
    """Five failures targeted by the current delegation fixes."""

    selected_tasks = REGRESSION_TASKS

    def __init__(self, data_dir: str = "evals/data") -> None:
        super().__init__(
            (
                ParallelQATaskGraphDataset(data_dir=data_dir),
                MuSiQueTaskGraphDataset(data_dir=data_dir),
                TwoWikiTaskGraphDataset(data_dir=data_dir),
                PlanBenchTaskGraphDataset(data_dir=data_dir),
                ArcAgiTaskGraphDataset(data_dir=data_dir),
            )
        )


__all__ = [
    "DelegationIterationDataset",
    "DelegationIterationTenDataset",
    "DelegationRegressionDataset",
    "FIVE_TASKS",
    "REGRESSION_TASKS",
    "SELECTED_TASKS",
    "TEN_TASKS",
]
