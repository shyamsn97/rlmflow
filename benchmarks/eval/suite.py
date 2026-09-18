"""The complete public evaluation surface.

A suite is a named scientific question backed by datasets and one runner
matrix. Dataset ``split`` selects pilot versus full manifests; it is not encoded
in duplicate suite names.
"""

from __future__ import annotations

from dataclasses import dataclass

from benchmarks.eval.registry import Registry

DEPTH_COMPARISON = (
    "official-rlm",
    "rlmflow-depth-0",
    "rlmflow-depth-1",
    "rlmflow-depth-2",
)


@dataclass(frozen=True, slots=True)
class SuiteSpec:
    question: str
    datasets: tuple[str, ...]
    runners: tuple[str, ...] = DEPTH_COMPARISON


SUITES: dict[str, SuiteSpec] = {
    "smoke": SuiteSpec(
        "Does the evaluation pipeline run end to end without network access?",
        ("synthetic_needle",),
        ("fake",),
    ),
    "conformance": SuiteSpec(
        "Do all runtimes obey the same offline environment and selection contract?",
        ("minimaze", "sudoku_search"),
    ),
    "core": SuiteSpec(
        "How do the harnesses manage paper-aligned long-context information tasks?",
        ("oolong-pairs-32k", "oolong-yahoo-128k"),
    ),
    "interactive": SuiteSpec(
        "How do the harnesses scale on persistent visual environments?",
        ("arc-agi-3", "mazebench"),
    ),
}


def register_suites(registry: Registry) -> None:
    for name, spec in SUITES.items():
        registry.alias(name, spec.datasets)
        registry.expand([name])


def runners_for(names: list[str]) -> tuple[str, ...]:
    """Return suite runners only when every requested token is a suite."""
    specs = [SUITES[name] for name in names if name in SUITES]
    if not specs or len(specs) != len(names):
        raise ValueError("explicit --runner is required when running datasets directly")
    runners: list[str] = []
    for spec in specs:
        for runner in spec.runners:
            if runner not in runners:
                runners.append(runner)
    return tuple(runners)


def format_suites_help() -> str:
    width = max(len(name) for name in SUITES)
    lines = [
        "Evaluation suites. Pass one suite name to --dataset.",
        "Use --split pilot for protocol bring-up or --split full for evidence.",
        "",
    ]
    for name, spec in SUITES.items():
        pad = " " * width
        lines.append(f"{name:<{width}}  {spec.question}")
        lines.append(f"{pad}  datasets: {' '.join(spec.datasets)}")
        lines.append(f"{pad}  runners:  {' '.join(spec.runners)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "DEPTH_COMPARISON",
    "SUITES",
    "SuiteSpec",
    "format_suites_help",
    "register_suites",
    "runners_for",
]
