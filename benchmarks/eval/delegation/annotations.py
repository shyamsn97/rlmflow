"""Private obligation graphs used to evaluate delegation semantics."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Obligation:
    id: str
    description: str
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskAnnotation:
    problem: int
    obligations: tuple[Obligation, ...]
    preferred_local: bool = False


def _chain(problem: int, *descriptions: str, preferred_local: bool = False) -> TaskAnnotation:
    obligations = tuple(
        Obligation(
            id=f"o{index}",
            description=description,
            depends_on=(() if index == 1 else (f"o{index - 1}",)),
        )
        for index, description in enumerate(descriptions, start=1)
    )
    return TaskAnnotation(problem, obligations, preferred_local)


ANNOTATIONS = {
    1: _chain(
        1, "resolve each factual operand", "combine the operands with the requested operation"
    ),
    2: _chain(2, "resolve branch facts", "aggregate branches", "apply the final operation"),
    3: _chain(3, "resolve independent branch facts", "join branch results", "compute the answer"),
    4: _chain(
        4, "resolve all leaf facts", "evaluate intermediate expressions", "evaluate the root"
    ),
    5: _chain(
        5, "find four supporting hops", "compose the hop answers", "cite supporting paragraphs"
    ),
    6: _chain(
        6, "test every required hop for support", "detect the missing hop", "abstain with evidence"
    ),
    7: _chain(
        7,
        "extract supporting sentences",
        "follow the inference bridge",
        "return evidence and answer",
    ),
    8: _chain(
        8, "resolve both comparison branches", "join the bridge", "return evidence and answer"
    ),
    9: _chain(9, "separate positive evidence from the hard negative", "compare the positive facts"),
    10: _chain(
        10, "identify proof leaves", "derive intermediate conclusions", "derive the hypothesis"
    ),
    11: _chain(
        11, "locate relevant transactions", "join reference data", "compute the requested fact"
    ),
    12: _chain(
        12, "locate affected transactions", "apply the counterfactual", "compute the fee delta"
    ),
    13: _chain(
        13, "extract trip constraints", "construct a connected itinerary", "validate all durations"
    ),
    14: _chain(
        14,
        "extract meeting windows and travel times",
        "construct a feasible route",
        "validate meetings",
    ),
    15: _chain(15, "intersect participant availability", "select a valid meeting slot"),
    16: _chain(
        16,
        "parse the initial and goal state",
        "construct a causal action sequence",
        "validate the goal",
    ),
    17: _chain(
        17,
        "locate relevant repository components",
        "trace cross-file behavior",
        "select the answer",
    ),
    18: _chain(
        18,
        "combine interacting Sudoku constraints",
        "solve and verify the grid",
        preferred_local=True,
    ),
    19: _chain(
        19, "infer component transformations", "compose the transformations", "produce test grids"
    ),
    20: _chain(
        20, "identify corresponding objects", "derive the object mapping", "produce test grids"
    ),
}


def annotation_for(problem: int) -> TaskAnnotation:
    try:
        return ANNOTATIONS[problem]
    except KeyError as exc:
        raise ValueError(f"no delegation annotation for problem {problem}") from exc


def validate_annotations() -> None:
    if sorted(ANNOTATIONS) != list(range(1, 21)):
        raise ValueError("delegation annotations must cover exactly problems 1..20")
    for annotation in ANNOTATIONS.values():
        seen = set()
        for obligation in annotation.obligations:
            if not set(obligation.depends_on) <= seen:
                raise ValueError(
                    f"problem {annotation.problem} obligation {obligation.id} has a forward dependency"
                )
            seen.add(obligation.id)


validate_annotations()

__all__ = ["ANNOTATIONS", "Obligation", "TaskAnnotation", "annotation_for"]
