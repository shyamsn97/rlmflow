"""Three NATURAL PLAN tasks with model-free native validation."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from benchmarks.eval import dataset
from benchmarks.eval.types import Dataset, Example, Prediction, Score

REVISION = "9b79bc4d52ee1c5bfd6eb4d4bb24d88e828f8b84"
FILES = {
    "trip": "trip_planning.json",
    "meeting": "meeting_planning.json",
    "calendar": "calendar_scheduling.json",
}
FROZEN_IDS = {
    "trip": "trip_planning_example_593",
    "meeting": "meeting_planning_example_594",
    "calendar": "calendar_scheduling_example_976",
}


@dataset("delegation_natural_plan", tags=["delegation", "task-graph", "planning"])
class NaturalPlanTaskGraphDataset(Dataset):
    def __init__(self, data_dir: str = "evals/data") -> None:
        self.data_dir = Path(data_dir) / "delegation" / "natural_plan"
        self._rows: dict[str, list[dict[str, Any]]] = {}

    def examples(self, *, split: str, limit: int | None, seed: int) -> list[Example]:
        del split, seed
        trip = self._select("trip")
        meeting = self._select("meeting")
        calendar = self._select("calendar")
        examples = [
            self._example("trip", trip, 13),
            self._example("meeting", meeting, 14),
            self._example("calendar", calendar, 15),
        ]
        return examples if limit is None else examples[:limit]

    def score(self, example: Example, prediction: Prediction) -> Score:
        expected = dict(example.expected)
        kind = expected["kind"]
        if kind == "trip":
            actual = _parse_trip(prediction.answer)
            gold = list(zip(expected["cities"], expected["durations"]))
            correct = actual == gold
            details = {"parsed_plan": actual, "expected_plan": gold}
        elif kind == "meeting":
            actual_score = _meeting_score(prediction.answer, expected)
            gold_score = _meeting_score(expected["golden_plan"], expected, parsed=True)
            correct = actual_score == gold_score
            details = {"valid_meetings": actual_score, "gold_valid_meetings": gold_score}
        else:
            actual = _parse_calendar(prediction.answer)
            gold = _parse_calendar(expected["golden_plan"])
            correct = actual == gold
            details = {"parsed_slot": actual, "expected_slot": gold}
        return Score(value=float(correct), correct=correct, details=details)

    def _select(self, kind: str) -> dict[str, Any]:
        source_id = FROZEN_IDS[kind]
        rows = [row for row in self._load(kind) if row["_source_id"] == source_id]
        if not rows:
            raise ValueError(f"NATURAL PLAN {kind} source is missing frozen ID {source_id!r}")
        return rows[0]

    def _load(self, kind: str) -> list[dict[str, Any]]:
        if kind in self._rows:
            return self._rows[kind]
        filename = FILES[kind]
        path = self.data_dir / filename
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            url = (
                "https://raw.githubusercontent.com/google-deepmind/natural-plan/"
                f"{REVISION}/data/{filename}"
            )
            with urlopen(url, timeout=60) as response:
                path.write_bytes(response.read())
        raw = json.loads(path.read_text())
        rows = [{**dict(value), "_source_id": str(key)} for key, value in raw.items()]
        self._rows[kind] = rows
        return rows

    def _example(self, kind: str, row: dict[str, Any], problem: int) -> Example:
        expected: dict[str, Any] = {"kind": kind, "golden_plan": row.get("golden_plan")}
        if kind == "trip":
            expected["cities"] = [item for item in str(row.get("cities", "")).split("**") if item]
            expected["durations"] = [
                int(item) for item in str(row.get("durations", "")).split("**") if item
            ]
        elif kind == "meeting":
            expected["constraints"] = row.get("constraints", [])
            expected["dist_matrix"] = row.get("dist_matrix", {})
        return Example(
            id=f"delegation_natural_plan_{problem:02d}_{row['_source_id']}",
            prompt=str(row.get("prompt_0shot", "")).strip(),
            expected=expected,
            metadata={
                "source_id": row["_source_id"],
                "problem": problem,
                "task_type": kind,
                "native_scorer": f"natural_plan_{kind}",
            },
        )


def _parse_trip(response: str) -> list[tuple[str, int]]:
    total = re.search(r"European cities for (\d+) days", response)
    total_days = int(total.group(1)) if total else None
    day_ranges = []
    flights = []
    for line in response.splitlines():
        visit = re.search(r"\d+-\d+", line)
        if visit:
            day_ranges.append(visit.group(0))
            if total_days and int(visit.group(0).split("-")[1]) == total_days:
                break
        flight = re.search(r".*Day (\d+).*from (\w+) to (\w+)", line)
        if flight:
            flights.append((int(flight.group(1)), flight.group(2), flight.group(3)))
    if not day_ranges or not flights:
        return []
    cities = [flights[0][1], *(flight[2] for flight in flights)]
    boundaries = [1, *(flight[0] for flight in flights), int(day_ranges[-1].split("-")[1])]
    return [
        (city, boundaries[index + 1] - boundaries[index] + 1) for index, city in enumerate(cities)
    ]


def _to_time(value: str) -> datetime:
    return datetime.strptime(value, "%I:%M%p")


def _meeting_score(response: Any, expected: dict[str, Any], *, parsed: bool = False) -> int:
    constraints = expected["constraints"]
    start_location, initial_time = constraints[0]
    people = defaultdict(dict)
    for name, location, times, duration in constraints[1:]:
        people[name] = {
            "location": location,
            "start": _to_time(times.split("to")[0].strip()),
            "end": _to_time(times.split("to")[1].strip()),
            "duration": int(duration),
        }
    steps = list(response) if parsed else _parse_meeting(str(response))
    location = start_location
    current = _to_time(initial_time)
    met = set()
    score = 0
    for step in steps:
        try:
            if step.startswith("You start"):
                continue
            if step.startswith("You travel"):
                destination = step.split("travel to ", 1)[1].split(" in", 1)[0].strip()
                current += timedelta(minutes=expected["dist_matrix"][location][destination])
                location = destination
            elif step.startswith("You wait"):
                end = _to_time(step.split("wait until ", 1)[1].split(".", 1)[0].strip())
                if end <= current:
                    break
                current = end
            elif step.startswith("You meet"):
                person = step.split("meet ", 1)[1].split(" for", 1)[0].strip()
                if person in met or person not in people:
                    break
                details = people[person]
                end = current + timedelta(minutes=details["duration"])
                if (
                    location != details["location"]
                    or current < details["start"]
                    or end > details["end"]
                ):
                    break
                met.add(person)
                current = end
                score += 1
            else:
                break
        except (KeyError, ValueError):
            break
    return score


def _parse_meeting(response: str) -> list[str]:
    if "SOLUTION:" in response:
        response = response.split("SOLUTION:", 1)[1]
    return [step.strip() for step in response.split(".") if step.strip()]


def _parse_calendar(response: str) -> tuple[str, float, float]:
    match = re.search(r"([A-Za-z]+), (\d+):(\d+) - (\d+):(\d+)", response)
    if not match:
        return "", -1, -1
    day, start_h, start_m, end_h, end_m = match.groups()
    return day, int(start_h) + int(start_m) / 60, int(end_h) + int(end_m) / 60


__all__ = ["FROZEN_IDS", "NaturalPlanTaskGraphDataset"]
