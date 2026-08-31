"""Frozen task allocation for the first task-graph delegation suite."""

from __future__ import annotations

from dataclasses import dataclass

SUITE_VERSION = "task-graph-v1"
ADAPTATION_VERSION = 2
FAULT_SEEDS = (101, 211, 307, 401, 503, 601)


@dataclass(frozen=True)
class TaskGraphSlot:
    dataset: str
    count: int
    problems: tuple[int, ...]
    source: str
    revision: str
    license: str


SLOTS = (
    TaskGraphSlot(
        "delegation_parallelqa",
        4,
        (1, 2, 3, 4),
        "SqueezeAILab/LLMCompiler:datasets/parallelqa_dataset.json",
        "dca77b2197f89a20e505545a1458ae4a2ff04571",
        "Apache-2.0",
    ),
    TaskGraphSlot(
        "delegation_musique",
        2,
        (5, 6),
        "voidful/MuSiQue",
        "a7d9f9adf6191604fc67cde318ee1a86fcf7babc",
        "CC-BY-4.0",
    ),
    TaskGraphSlot(
        "delegation_twowiki",
        2,
        (7, 8),
        "framolfese/2WikiMultihopQA",
        "fe713bfbd1afbca1a65246741a75890405d56a3a",
        "Apache-2.0",
    ),
    TaskGraphSlot(
        "delegation_grsqa",
        1,
        (9,),
        "kyone138/grs-qa:sample_data/example.json",
        "a7dee05c1792b552355354fc310c273256bcdc3b",
        "NOASSERTION",
    ),
    TaskGraphSlot(
        "delegation_entailmentbank",
        1,
        (10,),
        "sxiong/entailmentbank:task2",
        "8c6c148f7a21c21ff037a42d9a22446c9d42debc",
        "Apache-2.0",
    ),
    TaskGraphSlot(
        "delegation_dabstep",
        2,
        (11, 12),
        "adyen/DABstep",
        "9cef9a2976ccce4d306bf220604597788b090d43",
        "CC-BY-4.0",
    ),
    TaskGraphSlot(
        "delegation_natural_plan",
        3,
        (13, 14, 15),
        "google-deepmind/natural-plan",
        "9b79bc4d52ee1c5bfd6eb4d4bb24d88e828f8b84",
        "CC-BY-4.0",
    ),
    TaskGraphSlot(
        "delegation_planbench",
        1,
        (16,),
        "karthikv792/LLMs-Planning",
        "fc638a1aff7df3fe7a1a1d289fa2c04cc24dc284",
        "MIT",
    ),
    TaskGraphSlot(
        "delegation_codeqa",
        1,
        (17,),
        "zai-org/LongBench-v2",
        "2b48e494f2c7a2f0af81aae178e05c7e1dde0fe9",
        "Apache-2.0",
    ),
    TaskGraphSlot(
        "delegation_sudoku",
        1,
        (18,),
        "SakanaAI/Sudoku-Bench:challenge_100",
        "48fd26702ecedbecfe1ffdd3edc91a346c1137e8",
        "MIT",
    ),
    TaskGraphSlot(
        "delegation_arc_agi",
        2,
        (19, 20),
        "arcprize/ARC-AGI-2",
        "f3283f727488ad98fe575ea6a5ac981e4a188e49",
        "Apache-2.0",
    ),
)

PROBLEM_IDS = {
    1: "11",
    2: "22",
    3: "63",
    4: "94",
    5: "4hop1__38130_8966_31714_79432:answerable",
    6: "4hop1__38130_8966_31714_79432:unanswerable",
    7: "948c33ea0baf11ebab90acde48001122",
    8: "d594f50208c111ebbd8bac1f6bf848b6",
    9: "sample-comparison-1",
    10: "Mercury_SC_416126",
    11: "2536",
    12: "2769",
    13: "trip_planning_example_593",
    14: "meeting_planning_example_594",
    15: "calendar_scheduling_example_976",
    16: "logistics/generated_basic/instance-20.pddl",
    17: "train:72",
    18: "cross-product",
    19: "1ae2feb7",
    20: "2ba387bc",
}


def validate_manifest() -> None:
    problems = [problem for slot in SLOTS for problem in slot.problems]
    if problems != list(range(1, 21)):
        raise ValueError(f"task-graph problems must be exactly 1..20, got {problems}")
    for slot in SLOTS:
        if slot.count != len(slot.problems):
            raise ValueError(f"{slot.dataset} count does not match its problem numbers")
        if len(slot.revision) != 40:
            raise ValueError(f"{slot.dataset} source revision is not a full commit hash")
    if sorted(PROBLEM_IDS) != list(range(1, 21)):
        raise ValueError("problem IDs must cover exactly 1..20")


validate_manifest()

__all__ = [
    "ADAPTATION_VERSION",
    "FAULT_SEEDS",
    "PROBLEM_IDS",
    "SLOTS",
    "SUITE_VERSION",
    "TaskGraphSlot",
    "validate_manifest",
]
