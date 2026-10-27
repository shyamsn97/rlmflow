"""Runtime bindings for concrete environment instances."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from benchmarks.eval.environments.environment import Environment
from rlmflow.harnesses.repl.tools import tool

Perform = Callable[[str, Mapping[str, Any] | None], dict[str, Any]]


def build_environment(
    configuration: Mapping[str, Any],
    *,
    artifact_dir: str | Path,
) -> Environment:
    """Read old environment identifiers while artifacts migrate to the new schema."""
    from benchmarks.eval.environments.catalog import environment_from_id

    kind = str(configuration.get("kind", ""))
    environment_id = str(
        configuration.get("task_id")
        or configuration.get("puzzle_id")
        or configuration.get("environment_id")
        or ("mazebench_hxi" if kind == "mazebench" else "")
    )
    return environment_from_id(
        kind=kind,
        environment_id=environment_id,
        artifact_dir=Path(artifact_dir),
        configuration=configuration,
    )


def bind_tools(kind: str, perform: Perform, *, proxy: bool) -> dict[str, Any]:
    from benchmarks.eval.environments.catalog import environment_class

    environment_type = environment_class(kind)
    bound: dict[str, Any] = {}
    for tool_name, operation, description, parameters in environment_type.tool_operations:
        signature = inspect.Signature(
            [
                inspect.Parameter(
                    name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=annotation,
                )
                for name, annotation in parameters
            ],
            return_annotation=dict[str, Any],
        )

        def make_invoke(selected: str, selected_signature: inspect.Signature):
            def invoke(*args: Any, **kwargs: Any) -> dict[str, Any]:
                payload = dict(selected_signature.bind(*args, **kwargs).arguments)
                return perform(selected, payload or None)

            invoke.__signature__ = selected_signature  # type: ignore[attr-defined]
            return invoke

        invoke = make_invoke(operation, signature)
        invoke.__name__ = tool_name
        invoke.__qualname__ = tool_name
        if proxy:
            bound[tool_name] = tool(description, proxy=True)(invoke)
        else:
            bound[tool_name] = {"tool": invoke, "description": description}
    return bound


def action_budget(environment: Environment) -> int:
    value = getattr(environment, "action_budget", 256)
    return int(value)


def candidate_budget(environment: Environment) -> int:
    value = getattr(environment, "candidate_budget", 0)
    return int(value)


__all__ = ["action_budget", "bind_tools", "build_environment", "candidate_budget"]
