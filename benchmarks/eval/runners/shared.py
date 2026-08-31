"""Shared runner-neutral benchmark behavior."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from benchmarks.eval.types import Example

FACT_LOOKUP_DESCRIPTION = (
    "Required task-data source: call once for each exact entity named in the "
    "question and preserve the returned numeric precision."
)


def example_inputs(example: Example) -> dict[str, str]:
    """Return the exact public input mapping supplied to every runner."""
    return example.inputs()


def fact_lookup_data(example: Example) -> dict[str, Any] | None:
    """Return canonical lookup data when the task declares that tool."""
    if example.metadata.get("tool") != "fact_lookup":
        return None
    return {str(key).casefold(): value for key, value in example.metadata.get("facts", {}).items()}


def materialize_fixtures(example: Example, work_dir: Path) -> None:
    """Copy the same declared filesystem fixtures into one runner workdir."""
    raw = example.metadata.get("fixture_paths", [])
    paths = [raw] if isinstance(raw, str) else list(raw)
    for value in paths:
        source = Path(value).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"benchmark fixture does not exist: {source}")
        target = work_dir / source.name
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            shutil.copy2(source, target)


__all__ = [
    "FACT_LOOKUP_DESCRIPTION",
    "example_inputs",
    "fact_lookup_data",
    "materialize_fixtures",
]
