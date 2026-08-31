"""A frozen PlanBench Logistics problem with deterministic validation."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.request import urlopen

from benchmarks.eval import dataset
from benchmarks.eval.types import Dataset, Example, Prediction, Score

REVISION = "fc638a1aff7df3fe7a1a1d289fa2c04cc24dc284"
INSTANCE = "instance-20.pddl"
OPTIMAL_LENGTH = 11


@dataset("delegation_planbench", tags=["delegation", "task-graph", "planning"])
class PlanBenchTaskGraphDataset(Dataset):
    def __init__(self, data_dir: str = "evals/data") -> None:
        self.data_dir = Path(data_dir) / "delegation" / "planbench"

    def examples(self, *, split: str, limit: int | None, seed: int) -> list[Example]:
        del split, seed
        if limit == 0:
            return []
        domain = self._download("generated_domain.pddl")
        problem = self._download(INSTANCE)
        return [
            Example(
                id="delegation_planbench_16_logistics_instance_20",
                prompt=(
                    "Find a valid plan for the supplied PDDL Logistics problem. Return "
                    "only one grounded action per line, each enclosed in parentheses."
                ),
                context={"domain": domain, "problem": problem},
                expected={"problem": problem, "optimal_length": OPTIMAL_LENGTH},
                metadata={
                    "source_id": "logistics/generated_basic/instance-20.pddl",
                    "problem": 16,
                    "native_scorer": "pddl_state_validation",
                },
            )
        ]

    def score(self, example: Example, prediction: Prediction) -> Score:
        expected = dict(example.expected)
        valid, actions, error = _validate(prediction.answer, expected["problem"])
        gap = len(actions) - int(expected["optimal_length"]) if valid else None
        return Score(
            value=float(valid),
            correct=valid,
            details={
                "valid": valid,
                "actions": len(actions),
                "optimal_length": expected["optimal_length"],
                "optimality_gap": gap,
                "error": error,
            },
        )

    def _download(self, filename: str) -> str:
        path = self.data_dir / filename
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            subdir = (
                "plan-bench/instances/logistics"
                if filename == "generated_domain.pddl"
                else "plan-bench/instances/logistics/generated_basic"
            )
            url = (
                "https://raw.githubusercontent.com/karthikv792/LLMs-Planning/"
                f"{REVISION}/{subdir}/{filename}"
            )
            with urlopen(url, timeout=30) as response:
                path.write_bytes(response.read())
        return path.read_text()


def _validate(plan: str, problem: str) -> tuple[bool, list[tuple[str, ...]], str | None]:
    actions = [
        (match.group(1).lower(), *match.group(2).lower().split())
        for match in re.finditer(
            r"\((load-truck|load-airplane|unload-truck|unload-airplane|"
            r"drive-truck|fly-airplane)\s+([^()]*)\)",
            plan,
            flags=re.IGNORECASE,
        )
    ]
    state = _initial_state(problem)
    goals = _goals(problem)
    for index, action in enumerate(actions):
        ok, error = _apply(state, action)
        if not ok:
            return False, actions, f"action {index + 1}: {error}"
    if not goals <= state:
        return False, actions, "goal not reached"
    return True, actions, None


def _initial_state(problem: str) -> set[tuple[str, ...]]:
    section = problem.split("(:init", 1)[1].split("(:goal", 1)[0]
    return {
        tuple(match.group(1).lower().split()) for match in re.finditer(r"\(([^()]+)\)", section)
    }


def _goals(problem: str) -> set[tuple[str, ...]]:
    section = problem.split("(:goal", 1)[1]
    return {
        tuple(match.group(1).lower().split())
        for match in re.finditer(r"\((at\s+[^()]+)\)", section)
    }


def _apply(state: set[tuple[str, ...]], action: tuple[str, ...]) -> tuple[bool, str | None]:
    name, *args = action
    if name in {"load-truck", "load-airplane"} and len(args) == 3:
        obj, vehicle, location = args
        required = {("at", obj, location), ("at", vehicle, location)}
        if not required <= state:
            return False, "load precondition failed"
        state.remove(("at", obj, location))
        state.add(("in", obj, vehicle))
    elif name in {"unload-truck", "unload-airplane"} and len(args) == 3:
        obj, vehicle, location = args
        required = {("in", obj, vehicle), ("at", vehicle, location)}
        if not required <= state:
            return False, "unload precondition failed"
        state.remove(("in", obj, vehicle))
        state.add(("at", obj, location))
    elif name == "drive-truck" and len(args) == 4:
        truck, source, target, city = args
        required = {
            ("at", truck, source),
            ("in-city", source, city),
            ("in-city", target, city),
        }
        if not required <= state:
            return False, "drive precondition failed"
        state.remove(("at", truck, source))
        state.add(("at", truck, target))
    elif name == "fly-airplane" and len(args) == 3:
        airplane, source, target = args
        required = {
            ("at", airplane, source),
            ("airport", source),
            ("airport", target),
        }
        if not required <= state:
            return False, "flight precondition failed"
        state.remove(("at", airplane, source))
        state.add(("at", airplane, target))
    else:
        return False, f"unknown or malformed action {action}"
    return True, None


__all__ = ["PlanBenchTaskGraphDataset"]
