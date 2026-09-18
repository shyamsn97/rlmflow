"""Portable, offline HTML report for benchmark rows.

The report contains its data, styles, and renderer in one HTML file.  Artifact
links are deliberately relative, while small JSON replay artifacts are copied
into the payload after their paths have been checked against the run root.
"""

from __future__ import annotations

import argparse
import html
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmarks.eval.loggers.jsonl import load_rows
from benchmarks.eval.matrix import build_matrix
from benchmarks.eval.telemetry import RowTelemetry, expected_from_row, is_infra_error
from benchmarks.eval.types import Row
from benchmarks.eval.visual_viewer import (
    externalize_large_replays,
    write_static_assets,
)

REPORT_VERSION = 1
_JSON_ARTIFACT_LIMIT = 8 * 1024 * 1024
_DISCOVERY_CANDIDATE_LIMIT = 2_048
_DISCOVERY_VALUE_LIMIT = 32 * 1024


def build_visual_payload(
    rows: Sequence[Row],
    *,
    artifact_root: str | Path | None = None,
    report_dir: str | Path | None = None,
    title: str = "Benchmark visual report",
) -> dict[str, Any]:
    """Build a JSON-safe report payload from benchmark ``Row`` objects.

    ``artifact_root`` is the security boundary for every path recorded in a
    prediction.  Paths outside it (including paths reached through symlinks) are
    never linked or read.  ``report_dir`` controls how accepted links are made
    relative; it should be the directory that will contain ``report.html``.
    """
    root = Path(artifact_root or report_dir or Path.cwd()).expanduser().resolve()
    link_from = Path(report_dir or root).expanduser().resolve()
    matrix = (
        build_matrix(rows)
        if rows
        else {
            "runners": [],
            "seeds": [],
            "summary": {},
            "by_route": {},
            "tasks": [],
        }
    )
    matrix_tasks = {task["example_id"]: task for task in matrix["tasks"]}
    grouped: dict[tuple[str, str], list[Row]] = {}
    for row in rows:
        grouped.setdefault((row.dataset, row.example_id), []).append(row)

    tasks: list[dict[str, Any]] = []
    for (dataset, example_id), task_rows in sorted(grouped.items()):
        first = task_rows[0]
        matrix_task = matrix_tasks.get(example_id, {})
        lanes = [
            _lane_payload(
                row,
                root=root,
                link_from=link_from,
                matrix_cell=_matrix_cell(matrix_task, row),
            )
            for row in sorted(
                task_rows,
                key=lambda item: (item.runner, item.seed is None, item.seed),
            )
        ]
        tasks.append(
            {
                "dataset": dataset,
                "example_id": example_id,
                "domain": str(first.metadata.get("domain") or dataset),
                "role": str(
                    first.metadata.get(
                        "route_hypothesis",
                        first.metadata.get("iteration_role", ""),
                    )
                ),
                "expected": expected_from_row(first),
                "lanes": lanes,
                "timeline": _synchronize(lanes),
            }
        )

    infra_statuses = sorted({lane["infra_status"] for task in tasks for lane in task["lanes"]})
    all_lanes = [lane for task in tasks for lane in task["lanes"]]
    return {
        "version": REPORT_VERSION,
        "title": str(title),
        "filters": {
            "domains": sorted({task["domain"] for task in tasks}),
            "runners": list(matrix["runners"]),
            "seeds": list(matrix["seeds"]),
            "infra_statuses": infra_statuses,
            "selected_depths": sorted(
                {
                    str(lane["candidate_search"]["selected_depth"])
                    for lane in all_lanes
                    if lane["candidate_search"]["selected_depth"] is not None
                }
            ),
            "candidate_counts": sorted(
                {str(lane["candidate_search"]["count"]) for lane in all_lanes},
                key=int,
            ),
            "delegation_topologies": sorted({lane["delegation_topology"] for lane in all_lanes}),
        },
        # Keep matrix-level aggregates, but not its raw cells: cells contain
        # unsanitized absolute artifact paths and duplicate lane data.
        "matrix": {
            "runners": list(matrix["runners"]),
            "seeds": list(matrix["seeds"]),
            "summary": matrix["summary"],
            "by_route": matrix["by_route"],
            "by_domain": matrix.get("by_domain", {}),
            "paired_deltas_by_domain": matrix.get("paired_deltas_by_domain", {}),
        },
        "tasks": tasks,
    }


def _matrix_cell(task: Mapping[str, Any], row: Row) -> Mapping[str, Any] | None:
    cells = task.get("cells", {})
    if not isinstance(cells, Mapping):
        return None
    runner_cells = cells.get(row.runner, {})
    return runner_cells.get(str(row.seed)) if isinstance(runner_cells, Mapping) else None


def _lane_payload(
    row: Row,
    *,
    root: Path,
    link_from: Path,
    matrix_cell: Mapping[str, Any] | None,
) -> dict[str, Any]:
    telemetry = RowTelemetry.from_row(row)
    artifacts, artifact_paths = _artifacts(row, root=root, link_from=link_from)
    graph = _graph_payload(artifact_paths)
    environment = row.prediction.metrics.get("environment", {})
    events = _environment_events(environment, graph)
    input_tokens = telemetry.input_tokens
    output_tokens = telemetry.output_tokens
    reasoning_tokens = _number(row.prediction.usage.get("reasoning_tokens", 0))
    graph_metrics = telemetry.graph
    delegation = telemetry.delegation
    comparison = row.prediction.metrics.get("comparison_contract", {})
    topology = {
        "configured_depth": _number(comparison.get("recursive_levels", 0)),
        "agents": _number(graph_metrics.get("agents", 0)),
        "nodes": _number(graph_metrics.get("nodes", len(graph["nodes"]))),
        "llm_turns": _number(graph_metrics.get("llm_turns", 0)),
        "max_depth": _number(graph_metrics.get("max_depth", 0)),
        "max_branching": _number(graph_metrics.get("max_branching", 0)),
        "children": _number(delegation.get("children", 0)),
        "dependency_edges": _number(delegation.get("dependency_edge_count", 0)),
        "dependency_chain_depth": _number(delegation.get("dependency_chain_depth", 0)),
        "recursive_launches": _number(delegation.get("recursive_launch_count", 0)),
        "retrieved_results": _number(delegation.get("retrieved_child_results", 0)),
        "retrieval_ratio": _number(delegation.get("retrieval_ratio", 0)),
        "launch_batches": _number(delegation.get("launch_batches", 0)),
        "max_launch_batch": _number(delegation.get("max_launch_batch", 0)),
        "max_ready_launch_batch": _number(delegation.get("max_ready_launch_batch", 0)),
        "gold_graph_alignment": delegation.get("gold_graph_alignment"),
    }
    environment_actions = {
        "total": _number(environment.get("total_actions", len(events)))
        if isinstance(environment, Mapping)
        else len(events),
        "primary": _number(environment.get("primary_actions", 0))
        if isinstance(environment, Mapping)
        else 0,
        "probe": _number(environment.get("probe_actions", 0))
        if isinstance(environment, Mapping)
        else 0,
        "instances_forked": _number(environment.get("instances_forked", 0))
        if isinstance(environment, Mapping)
        else 0,
    }
    total_tokens = input_tokens + output_tokens
    score_value = _number(row.score.value)
    wall_seconds = telemetry.time_seconds
    error = telemetry.error
    infra_status = (
        "infra_error"
        if bool(matrix_cell and matrix_cell.get("infra_error")) or is_infra_error(error)
        else "error"
        if error
        else "ok"
    )
    seed = str(row.seed)
    lane_id = f"{row.runner}::s{seed}"
    lane = {
        "id": lane_id,
        "runner": row.runner,
        "seed": seed,
        "model": row.model,
        "infra_status": infra_status,
        "error": error,
        "answer": row.prediction.answer,
        "score": {
            "value": row.score.value,
            "correct": row.score.correct,
            "label": _score_label(row),
            "details": dict(row.score.details),
        },
        "tokens": {
            "input": input_tokens,
            "output": output_tokens,
            "reasoning": reasoning_tokens,
            "total": total_tokens,
        },
        "time": {
            "wall_seconds": wall_seconds,
            "compute_seconds": _number(
                row.metadata.get(
                    "aggregate_compute_seconds",
                    row.prediction.metrics.get("time_seconds", 0.0),
                )
            ),
        },
        "efficiency": {
            "score_per_1k_tokens": (score_value * 1_000 / total_tokens if total_tokens else None),
            "score_per_environment_action": (
                score_value / environment_actions["total"] if environment_actions["total"] else None
            ),
            "score_per_second": score_value / wall_seconds if wall_seconds else None,
        },
        "actions": {**environment_actions, "count": environment_actions["total"], "events": events},
        "topology": topology,
        "delegation_topology": (
            "dependency-dag"
            if topology["dependency_edges"]
            else "delegation-tree"
            if topology["children"]
            else "single-agent"
        ),
        "candidate_search": {
            "count": 0,
            "selected_depth": None,
            "selection_compliance": (
                environment.get("selection_compliance")
                if isinstance(environment, Mapping)
                else None
            ),
        },
        "graph": graph,
        "artifacts": artifacts,
    }
    discovery_tree = _discovery_payload(environment, artifact_paths)
    if discovery_tree is not None:
        lane["discovery_tree"] = discovery_tree
        selected_id = discovery_tree.get("selected_candidate_id")
        selected = next(
            (
                candidate
                for candidate in discovery_tree.get("candidates", [])
                if candidate.get("id") == selected_id
            ),
            None,
        )
        lane["candidate_search"] = {
            "count": len(discovery_tree.get("candidates", [])),
            "selected_depth": selected.get("owner_depth") if selected else None,
            "selection_compliance": (
                environment.get("selection_compliance")
                if isinstance(environment, Mapping)
                else None
            ),
        }
    replay = _replay_payload(row, environment, artifact_paths)
    if replay is not None:
        lane["replay"] = replay
    return lane


def _score_label(row: Row) -> str:
    details = row.score.details
    domain = str(row.metadata.get("domain", ""))
    if domain == "fanoutqa":
        coverage = details.get("coverage", details.get("loose_coverage"))
        return f"coverage {coverage}" if coverage is not None else f"{row.score.value:g}"
    if domain == "planbench":
        return "goal reached" if row.score.correct else "goal not reached"
    if domain in {"arc", "arc_agi", "arc_agi_2"}:
        return "exact grid" if row.score.correct else "grid mismatch"
    if domain in {"pencil_puzzle", "maze", "mazebench", "mazebench-derived"}:
        reward = row.prediction.metrics.get("environment", {}).get("reward")
        if reward is not None:
            return f"reward {reward}"
    if domain == "sudoku":
        fraction = details.get("placement_fraction")
        if fraction is not None:
            return f"{float(fraction):.1%} placed"
    return f"{row.score.value:g}"


def _artifacts(
    row: Row,
    *,
    root: Path,
    link_from: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Path]]:
    rendered: dict[str, dict[str, Any]] = {}
    accepted: dict[str, Path] = {}
    for name, raw_value in sorted(row.prediction.artifacts.items()):
        raw = str(raw_value)
        path, status = _contained_path(raw, root)
        item: dict[str, Any] = {"status": status}
        if path is not None:
            relative = Path(os.path.relpath(path, link_from)).as_posix()
            # A bare colon can turn a DOM-assigned href into a URL scheme
            # (notably ``javascript:``).  Explicit dot-relative paths cannot.
            first_segment = relative.split("/", 1)[0]
            item["href"] = (
                f"./{relative}"
                if ":" in first_segment and not relative.startswith((".", "/"))
                else relative
            )
            item["kind"] = "directory" if path.is_dir() else path.suffix.lower().lstrip(".")
            accepted[name] = path
        rendered[str(name)] = item
    return rendered, accepted


def _contained_path(raw: str, root: Path) -> tuple[Path | None, str]:
    if not raw or "\x00" in raw:
        return None, "blocked"
    source = Path(raw).expanduser()
    candidates = [source] if source.is_absolute() else [root / source, Path.cwd() / source]
    safe_missing: Path | None = None
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        if not _is_relative_to(resolved, root):
            continue
        if resolved.exists():
            return resolved, "ok"
        safe_missing = safe_missing or resolved
    # A missing in-root path is still safe to expose as a relative link.
    if safe_missing is not None:
        return safe_missing, "missing"
    return None, "blocked"


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _graph_payload(paths: Mapping[str, Path]) -> dict[str, Any]:
    graph_path = next(
        (
            path / "graph.json" if path.is_dir() else path
            for name, path in paths.items()
            if name in {"graph", "graph_path", "graph_json"}
            and (path.is_dir() or path.name == "graph.json")
        ),
        None,
    )
    if graph_path is None or not graph_path.is_file():
        return {"status": "missing", "nodes": [], "exec_actions": []}
    data = _read_json(graph_path)
    if not isinstance(data, Mapping) or not isinstance(data.get("nodes"), list):
        return {"status": "invalid", "nodes": [], "exec_actions": []}

    agent_paths = {
        str(node.get("id")): str(node.get("agent_id") or node.get("id"))
        for node in data["nodes"]
        if isinstance(node, Mapping) and node.get("type") == "agent_start"
    }
    output_by_parent = {
        str(node.get("parent_id")): _node_text(node)
        for node in data["nodes"]
        if isinstance(node, Mapping)
        and node.get("type") in {"exec_output", "error_output", "repl_dead"}
    }
    exec_actions = []
    nodes = []
    for position, raw in enumerate(data["nodes"]):
        if not isinstance(raw, Mapping):
            continue
        agent = str(raw.get("agent_id") or "")
        node = {
            "id": str(raw.get("id") or ""),
            "type": str(raw.get("type") or ""),
            "agent": agent,
            "depth": _agent_depth(agent),
            "order": raw.get("order"),
            "position": position,
            "created_at": _nested(raw, "metadata", "created_at"),
            "timing": _nested(raw, "metadata", "timing") or {},
            "text": _node_text(raw),
        }
        nodes.append(node)
        if node["type"] in {"exec_action", "append_child"}:
            exec_actions.append(
                {
                    **node,
                    "output": output_by_parent.get(node["id"], ""),
                }
            )
    return {
        "status": "ok",
        "node_count": len(nodes),
        "nodes": nodes,
        "exec_actions": exec_actions,
        "agent_paths": agent_paths,
    }


def _node_text(node: Mapping[str, Any]) -> Any:
    payload = node.get("payload", {})
    if not isinstance(payload, Mapping):
        return ""
    for key in ("code", "output", "content", "result"):
        if key in payload:
            return payload[key]
    return ""


def _environment_events(environment: Any, graph: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(environment, Mapping):
        return []
    raw_events = environment.get("events")
    if not isinstance(raw_events, list):
        raw_events = environment.get("steps")
    raw_actions = environment.get("actions")
    actions = raw_actions if isinstance(raw_actions, list) else []
    if not isinstance(raw_events, list):
        raw_events = actions
    exec_by_id = {
        str(event["id"]): event
        for event in graph.get("exec_actions", [])
        if isinstance(event, Mapping) and event.get("id")
    }
    agent_paths = graph.get("agent_paths", {})
    events: list[dict[str, Any]] = []
    for position, raw in enumerate(raw_events):
        event = dict(raw) if isinstance(raw, Mapping) else {"observation": raw}
        if "action" not in event and position < len(actions):
            event["action"] = actions[position]
        exec_action_id = _exec_action_id(event)
        graph_event = exec_by_id.get(exec_action_id or "")
        raw_agent = str(event.get("agent_id") or event.get("actor") or "")
        agent = (
            str(graph_event.get("agent"))
            if graph_event is not None
            else str(agent_paths.get(raw_agent, raw_agent))
        )
        index = event.get("index", event.get("step", event.get("order", position)))
        normalized = {
            "index": index,
            "position": position,
            "exec_action_id": exec_action_id,
            "started_at": event.get("started_at"),
            "finished_at": event.get("finished_at"),
            "duration_ms": event.get("duration_ms"),
            "instance_id": event.get("instance_id"),
            "agent_id": event.get("agent_id"),
            "agent": agent,
            "depth": (
                graph_event.get("depth")
                if graph_event is not None
                else event.get("depth", _agent_depth(agent))
            ),
            "operation": event.get("operation"),
            "action": event.get(
                "action",
                event.get("accepted_actions", event.get("operation")),
            ),
            "observation": event.get(
                "observation",
                event.get("state", event.get("observation_summary")),
            ),
            "reward": event.get("reward"),
            "done": event.get("done", False),
            "primary": event.get("primary"),
            "probe": event.get("probe"),
        }
        if graph_event is not None:
            normalized["graph_event"] = graph_event
        events.append(normalized)
    return events


def _exec_action_id(event: Mapping[str, Any]) -> str | None:
    for source in (
        event,
        event.get("action"),
        event.get("metadata"),
    ):
        if not isinstance(source, Mapping):
            continue
        for key in ("exec_action_id", "action_node_id", "graph_action_id"):
            value = source.get(key)
            if value is not None:
                return str(value)
    return None


def _synchronize(lanes: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Align lane events by native environment index, not wall-clock time."""
    indexes: list[Any] = []
    seen: set[str] = set()
    for lane in lanes:
        for event in lane.get("actions", {}).get("events", []):
            index = event.get("index")
            key = json.dumps(index, sort_keys=True, default=str)
            if key not in seen:
                seen.add(key)
                indexes.append(index)
    indexes.sort(key=_index_sort_key)
    frames = []
    for index in indexes:
        paired = {}
        for lane in lanes:
            paired[lane["id"]] = next(
                (
                    event
                    for event in lane.get("actions", {}).get("events", [])
                    if event.get("index") == index
                ),
                None,
            )
        frames.append({"index": index, "lanes": paired})
    return frames


def _index_sort_key(value: Any) -> tuple[int, Any]:
    if isinstance(value, (int, float)):
        return (0, value)
    return (1, str(value))


def _replay_payload(
    row: Row,
    environment: Any,
    artifact_paths: Mapping[str, Path],
) -> dict[str, Any] | None:
    for name, path in artifact_paths.items():
        if "replay" not in name.casefold() or path.suffix.casefold() != ".json":
            continue
        value = _read_json(path)
        if isinstance(value, Mapping):
            replay = dict(value)
            replay.setdefault("kind", _replay_kind(row))
            return replay

    domain = str(row.metadata.get("domain", "")).casefold()
    if domain in {"maze", "minimaze", "mazebench", "mazebench-derived"} and isinstance(
        environment, Mapping
    ):
        steps = environment.get("events", environment.get("steps", []))
        return {
            "kind": "maze",
            "initial": environment.get(
                "initial",
                environment.get("initial_state", row.metadata.get("initial_state")),
            ),
            "frames": steps if isinstance(steps, list) else [],
        }
    if domain in {"arc", "arc_agi", "arc_agi_2"}:
        task = _jsonish(
            row.metadata.get(
                "arc_task",
                row.metadata.get("task", row.score.details.get("task")),
            )
        )
        prediction = _jsonish(row.prediction.answer)
        expected = row.score.details.get("expected", row.metadata.get("expected"))
        if task is not None or prediction is not None or expected is not None:
            return {
                "kind": "arc",
                "task": task,
                "prediction": prediction,
                "expected": expected,
            }
    if domain == "arc_agi_3" and isinstance(environment, Mapping):
        events = environment.get("events", environment.get("steps", []))
        instances = environment.get("instances", [])
        return {
            "kind": "arc",
            "primary_instance_id": environment.get("primary_instance_id"),
            "instances": instances if isinstance(instances, (list, Mapping)) else [],
            "events": events if isinstance(events, list) else [],
        }
    if domain == "sudoku" and isinstance(environment, Mapping):
        steps = environment.get("events", environment.get("steps", []))
        return {
            "kind": "sudoku",
            "initial": environment.get(
                "initial",
                environment.get("initial_state", row.metadata.get("initial_state")),
            ),
            "frames": steps if isinstance(steps, list) else [],
            "prediction": _jsonish(row.prediction.answer),
            "expected": row.score.details.get("expected", row.metadata.get("expected")),
        }
    return None


def _discovery_payload(
    environment: Any,
    artifact_paths: Mapping[str, Path],
) -> dict[str, Any] | None:
    """Return a bounded, renderer-friendly candidate discovery tree."""
    metric_value = environment.get("candidate_search") if isinstance(environment, Mapping) else None
    artifact_value: Any = None
    for name, path in artifact_paths.items():
        folded = name.casefold()
        if ("discovery" in folded or "candidate" in folded) and path.suffix.casefold() == ".json":
            value = _read_json(path)
            if isinstance(value, Mapping) and isinstance(value.get("candidates"), list):
                artifact_value = value
                break
    raw = artifact_value if isinstance(artifact_value, Mapping) else metric_value
    if not isinstance(raw, Mapping):
        return None

    raw_candidates = raw.get("candidates")
    if not isinstance(raw_candidates, list):
        return {
            "status": "invalid",
            "candidates": [],
            "timeline": [],
            "selected_candidate_id": None,
            "replay_policies": {},
        }

    selected_id = _optional_text(raw.get("selected_candidate_id"))
    candidates: list[dict[str, Any]] = []
    parent_by_id: dict[str, str | None] = {}
    for position, item in enumerate(raw_candidates[:_DISCOVERY_CANDIDATE_LIMIT]):
        if not isinstance(item, Mapping):
            continue
        candidate_id = _optional_text(item.get("id")) or f"candidate_{position}"
        parent_id = _optional_text(item.get("parent_candidate_id"))
        candidate = {
            "id": candidate_id,
            "index": _number_or(item.get("index"), position),
            "owner_id": _optional_text(item.get("owner_id")) or "",
            "owner_path": _optional_text(item.get("owner_path")) or "",
            "owner_depth": _number_or(item.get("owner_depth"), 0),
            "instance_id": _optional_text(item.get("instance_id")) or "",
            "parent_candidate_id": parent_id,
            "label": _optional_text(item.get("label")) or candidate_id,
            "proposal": _bounded_json(item.get("proposal")),
            "state_hash": _optional_text(item.get("state_hash")) or "",
            "evaluation_cost": _number_or(item.get("evaluation_cost"), 0),
            "score": _optional_number(item.get("score")),
            "public_summary": _bounded_json(item.get("public_summary", {})),
            "created_at": _bounded_json(item.get("created_at")),
            "selected": bool(item.get("selected")) or candidate_id == selected_id,
        }
        candidates.append(candidate)
        parent_by_id[candidate_id] = parent_id

    if selected_id is None:
        selected = next((item for item in candidates if item["selected"]), None)
        selected_id = selected["id"] if selected is not None else None
    selected_path: set[str] = set()
    current = selected_id
    while current and current not in selected_path:
        selected_path.add(current)
        current = parent_by_id.get(current)

    oracle_best_score = _optional_number(raw.get("oracle_best_score"))
    if oracle_best_score is None:
        scores = [item["score"] for item in candidates if item["score"] is not None]
        oracle_best_score = max(scores, default=None)
    cumulative_cost: int | float = 0
    best_score: int | float | None = None
    timeline: list[dict[str, Any]] = []
    for item in sorted(candidates, key=lambda value: _index_sort_key(value["index"])):
        candidate_id = item["id"]
        if candidate_id == selected_id:
            state = "selected"
        elif selected_id is not None and candidate_id in selected_path:
            state = "selected_path"
        elif selected_id is not None:
            state = "pruned"
        else:
            state = "candidate"
        item["state"] = state
        item["selected"] = state == "selected"
        item["pruned"] = state == "pruned"
        item["on_selected_path"] = candidate_id in selected_path
        cumulative_cost += item["evaluation_cost"]
        score = item["score"]
        if score is not None:
            best_score = score if best_score is None else max(best_score, score)
        timeline.append(
            {
                "index": item["index"],
                "candidate_id": candidate_id,
                "parent_candidate_id": item["parent_candidate_id"],
                "owner_id": item["owner_id"],
                "owner_path": item["owner_path"],
                "owner_depth": item["owner_depth"],
                "label": item["label"],
                "score": score,
                "evaluation_cost": item["evaluation_cost"],
                "cumulative_cost": cumulative_cost,
                "best_score": best_score,
                "regret": (
                    oracle_best_score - score
                    if oracle_best_score is not None and score is not None
                    else None
                ),
                "state": state,
                "created_at": item["created_at"],
            }
        )

    policies: dict[str, Any] = {}
    raw_policies = raw.get("replay_policies")
    if isinstance(raw_policies, Mapping):
        for name, policy in list(raw_policies.items())[:256]:
            if not isinstance(policy, Mapping):
                continue
            revealed = policy.get("revealed_candidate_ids", [])
            policies[_optional_text(name) or "policy"] = {
                "revealed_candidate_ids": [
                    str(value)
                    for value in (
                        revealed[:_DISCOVERY_CANDIDATE_LIMIT] if isinstance(revealed, list) else []
                    )
                ],
                "selected_candidate_id": _optional_text(policy.get("selected_candidate_id")),
                "score": _optional_number(policy.get("score")),
                "evaluation_cost": _number_or(policy.get("evaluation_cost"), 0),
            }

    selected_candidate = next(
        (item for item in candidates if item["id"] == selected_id),
        None,
    )
    selected_score = _optional_number(raw.get("selected_score"))
    if selected_score is None and selected_candidate is not None:
        selected_score = selected_candidate["score"]
    selection_regret = _optional_number(raw.get("selection_regret"))
    if selection_regret is None and oracle_best_score is not None and selected_score is not None:
        selection_regret = oracle_best_score - selected_score

    return {
        "status": "ok",
        "schema_version": _bounded_json(raw.get("schema_version")),
        "episode_id": _optional_text(raw.get("episode_id")),
        "evaluation_budget": _number_or(raw.get("evaluation_budget"), 0),
        "evaluations_used": _number_or(raw.get("evaluations_used"), len(candidates)),
        "evaluations_remaining": _number_or(raw.get("evaluations_remaining"), 0),
        "selected_candidate_id": selected_id,
        "selected_score": selected_score,
        "oracle_best_candidate_id": _optional_text(raw.get("oracle_best_candidate_id")),
        "oracle_best_score": oracle_best_score,
        "selection_regret": selection_regret,
        "truncated": len(raw_candidates) > _DISCOVERY_CANDIDATE_LIMIT,
        "candidates": candidates,
        "timeline": timeline,
        "replay_policies": policies,
    }


def _replay_kind(row: Row) -> str:
    domain = str(row.metadata.get("domain", "")).casefold()
    if "maze" in domain:
        return "maze"
    if domain.startswith("arc"):
        return "arc"
    if domain == "sudoku":
        return "sudoku"
    return domain or "generic"


def _read_json(path: Path) -> Any:
    try:
        if path.stat().st_size > _JSON_ARTIFACT_LIMIT:
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def _jsonish(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _bounded_json(value: Any) -> Any:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
            allow_nan=False,
        )
    except (TypeError, ValueError):
        encoded = json.dumps(str(value), ensure_ascii=False)
    if len(encoded) <= _DISCOVERY_VALUE_LIMIT:
        return json.loads(encoded)
    return {
        "truncated": True,
        "preview": encoded[:_DISCOVERY_VALUE_LIMIT],
    }


def _optional_text(value: Any) -> str | None:
    return None if value is None else str(value)[:_DISCOVERY_VALUE_LIMIT]


def _optional_number(value: Any) -> int | float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    if isinstance(value, float) and (value != value or value in {float("inf"), float("-inf")}):
        return None
    return value


def _number_or(value: Any, default: int | float) -> int | float:
    number = _optional_number(value)
    return default if number is None else number


def _nested(value: Mapping[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _agent_depth(agent: str) -> int:
    return max(agent.count("."), 0) if agent else 0


def _number(value: Any) -> int | float:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


_CSS = r"""
:root { color-scheme: dark; --bg:#0d1117; --panel:#161b22; --line:#30363d;
  --ink:#e6edf3; --muted:#8b949e; --blue:#58a6ff; --bad:#ff7b72; --ok:#56d364; }
* { box-sizing:border-box } body { margin:0; background:var(--bg); color:var(--ink);
  font:13px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif }
header { padding:20px 24px 10px } h1 { margin:0 0 4px; font-size:20px }
.sub,.muted { color:var(--muted) }.filters { display:flex; flex-wrap:wrap; gap:10px;
  padding:10px 24px 18px; position:sticky; top:0; background:var(--bg); z-index:2 }
label { color:var(--muted); font-size:11px } select { display:block; margin-top:3px;
  background:var(--panel); color:var(--ink); border:1px solid var(--line); padding:5px }
main { padding:0 24px 30px }.task { border:1px solid var(--line); border-radius:10px;
  margin:0 0 18px; overflow:hidden }.task>h2 { font-size:15px; margin:0; padding:12px 14px;
  background:var(--panel) }.lanes { display:grid; gap:1px; background:var(--line);
  grid-template-columns:repeat(auto-fit,minmax(300px,1fr)) }.lane { background:var(--bg);
  padding:12px; min-width:0 }.lane h3 { margin:0 0 6px; font-size:14px }
.badge { border:1px solid var(--line); border-radius:99px; padding:1px 7px; font-size:11px }
.ok { color:var(--ok) }.infra_error,.error { color:var(--bad) }.metrics { display:grid;
  grid-template-columns:repeat(auto-fit,minmax(95px,1fr)); gap:5px; margin:8px 0 }.metric { background:var(--panel);
  border-radius:5px; padding:5px }.metric b { display:block; font-size:12px }
.timeline { display:flex; gap:5px; overflow:auto; padding:5px 0 }.event { min-width:34px;
  min-height:28px; border:1px solid var(--line); border-radius:4px; background:var(--panel);
  color:var(--ink); cursor:pointer }.event.linked { border-color:var(--blue) }
details { margin-top:7px } summary { cursor:pointer; color:var(--muted) }
pre { white-space:pre-wrap; overflow-wrap:anywhere; max-height:280px; overflow:auto;
  background:var(--panel); border-radius:5px; padding:8px; color:var(--ink) }
.links { display:flex; flex-wrap:wrap; gap:8px }.links a { color:var(--blue) }
.missing,.blocked { color:var(--muted) }.hidden { display:none!important }
.replay-grid { display:inline-grid; gap:1px; margin:4px; vertical-align:top }
.cell { width:13px; height:13px; background:#222 }.empty { padding:20px; color:var(--muted) }
.overview-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr));
  gap:10px; margin-bottom:18px }.domain-summary { border:1px solid var(--line);
  border-radius:8px; padding:10px; background:var(--panel) }.domain-summary h2 {
  margin:0 0 8px; font-size:14px }.runner-bar { margin:6px 0 }.runner-bar meter {
  width:100%; accent-color:var(--blue) }.delta { font-size:11px; color:var(--muted) }
.discovery-head,.replay-controls,.tabs { display:flex; gap:6px; flex-wrap:wrap;
  align-items:center; margin:7px 0 }.tree { border-left:1px solid var(--line); margin:8px 0 }
.candidate { margin:4px 0; padding:5px 7px; border:1px solid var(--line);
  border-radius:5px; background:var(--panel) }.candidate.selected { border-color:var(--ok) }
.candidate.selected_path { border-color:var(--blue) }.candidate.pruned { opacity:.58 }
.tree-meta { font-size:11px; color:var(--muted) }.candidate-state { margin-left:6px;
  text-transform:uppercase; font-size:10px }.score-timeline { display:flex; gap:5px;
  overflow:auto; padding:5px 0 }.score-point { min-width:125px; text-align:left;
  border:1px solid var(--line); border-radius:5px; color:var(--ink); background:var(--panel);
  padding:5px }.score-point.selected { border-color:var(--ok) }
.score-point.pruned { opacity:.58 }.tabs button,.replay-controls button { background:var(--panel);
  color:var(--ink); border:1px solid var(--line); border-radius:5px; padding:4px 7px;
  cursor:pointer }.tabs button.active { border-color:var(--blue); color:var(--blue) }
.tabs button.selected { border-color:var(--ok) }.replay-controls input[type=range] {
  min-width:180px; accent-color:var(--blue) }.viewer { border:1px solid var(--line);
  border-radius:6px; padding:8px; margin-top:6px }.maze-ascii { max-height:420px;
  white-space:pre; overflow:auto; font:12px/1.2 ui-monospace,SFMono-Regular,monospace }
.sudoku-grid { display:inline-grid; gap:1px; padding:3px; background:var(--line);
  margin:5px 0 }.sudoku-cell { width:27px; height:27px; display:grid; place-items:center;
  color:var(--ink); background:var(--panel); font-weight:600 }.sudoku-cell.violation {
  color:var(--bad); outline:2px solid var(--bad); outline-offset:-2px }
.policy { padding:3px 0; border-top:1px solid var(--line) }
.route-overlay { color:var(--blue); overflow:auto; white-space:nowrap; margin-top:5px }
"""


_JS = r"""
"use strict";
const DATA = JSON.parse(document.getElementById("report-data").textContent);
const MAX_FRAMES = 5000, MAX_GRID_SIDE = 64, MAX_ASCII = 100000;
const $ = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
};
const fmt = value => typeof value === "number" ? value.toLocaleString(undefined,
  {maximumFractionDigits:2}) : "—";
const dump = value => value === undefined || value === null ? "—" :
  (typeof value === "string" ? value : JSON.stringify(value, null, 2));
const main = document.getElementById("tasks");
const overview = document.getElementById("overview");

function metric(name, value) {
  const box = $("div", "metric"); box.append($("span", "muted", name), $("b", "", value));
  return box;
}
function details(label, value) {
  const box = $("details"); box.append($("summary", "", label), $("pre", "", dump(value)));
  return box;
}
function arcGrid(grid) {
  const wrap = $("span", "replay-grid");
  if (!Array.isArray(grid) || !grid.length || !Array.isArray(grid[0])) return wrap;
  const rows=grid.slice(0,MAX_GRID_SIDE), width=Math.min(MAX_GRID_SIDE,grid[0].length);
  if (!width) return wrap;
  wrap.style.gridTemplateColumns = `repeat(${width},13px)`;
  const colors=["#000","#0074d9","#ff4136","#2ecc40","#ffdc00","#aaa",
    "#f012be","#ff851b","#7fdbff","#870c25"];
  rows.forEach(row => row.slice(0,width).forEach(value => {
    const cell=$("i","cell"); cell.style.background=colors[+value]||"#fff";
    cell.title=String(value); wrap.append(cell);
  }));
  return wrap;
}
function discoveryState(discovery, instanceId) {
  const item=(discovery?.candidates||[]).find(candidate=>candidate.instance_id===instanceId);
  return item ? item.state : "";
}
function discoveryTreeView(tree) {
  const box=$("details"); box.open=true;
  box.append($("summary","",`discovery tree · ${(tree.candidates||[]).length} candidates`));
  if (tree.status!=="ok") {
    box.append($("div","empty","Discovery artifact is unavailable or invalid.")); return box;
  }
  const head=$("div","discovery-head");
  head.append(metric("selected",tree.selected_candidate_id||"none"),
    metric("oracle best",fmt(tree.oracle_best_score)),
    metric("selection regret",fmt(tree.selection_regret)),
    metric("budget",`${fmt(tree.evaluations_used)} / ${fmt(tree.evaluation_budget)}`));
  box.append(head);
  const timeline=$("div","score-timeline");
  (tree.timeline||[]).forEach(point => {
    const button=$("button",`score-point ${point.state}`,
      `#${point.index} ${point.label}\nscore ${fmt(point.score)} · cost ${fmt(point.cumulative_cost)} · regret ${fmt(point.regret)}`);
    button.title=`owner ${point.owner_path||point.owner_id||"unknown"} · depth ${point.owner_depth}`;
    button.onclick=()=> {
      const old=box.querySelector(".candidate-detail"); if (old) old.remove();
      const item=(tree.candidates||[]).find(candidate=>candidate.id===point.candidate_id);
      const view=details(`candidate ${point.candidate_id}`,item||point);
      view.className="candidate-detail"; box.append(view);
    };
    timeline.append(button);
  });
  if (timeline.childNodes.length) box.append(timeline);
  const byId=new Map((tree.candidates||[]).map(item=>[item.id,item]));
  const candidateDepth=item => {
    let depth=0,current=item,seen=new Set();
    while (current?.parent_candidate_id && !seen.has(current.id) && depth<32) {
      seen.add(current.id); current=byId.get(current.parent_candidate_id); depth++;
    }
    return depth;
  };
  const branch=$("div","tree");
  (tree.candidates||[]).forEach(item => {
    const row=$("div",`candidate ${item.state}`);
    row.style.marginLeft=`${Math.min(32,candidateDepth(item))*16}px`;
    const title=$("div","",`#${item.index} ${item.label}`);
    title.append($("span",`candidate-state ${item.state}`,item.state.replace("_"," ")));
    row.append(title,$("div","tree-meta",
      `parent ${item.parent_candidate_id||"root"} · owner ${item.owner_path||item.owner_id||"unknown"} · depth ${fmt(item.owner_depth)} · instance ${item.instance_id||"—"} · score ${fmt(item.score)} · cost ${fmt(item.evaluation_cost)}`));
    branch.append(row);
  });
  box.append(branch);
  const policies=Object.entries(tree.replay_policies||{});
  if (policies.length) {
    const policyBox=$("details"); policyBox.append($("summary","","zero-execution policy replay"));
    policies.forEach(([name,policy])=>policyBox.append($("div","policy",
      `${name}: selected ${policy.selected_candidate_id||"none"} · score ${fmt(policy.score)} · cost ${fmt(policy.evaluation_cost)} · revealed ${(policy.revealed_candidate_ids||[]).length}`)));
    box.append(policyBox);
  }
  if (tree.truncated) box.append($("div","muted","Candidate list was capped for this report."));
  return box;
}
function replayInstances(replay) {
  const result=[];
  if (Array.isArray(replay.instances)) {
    replay.instances.slice(0,256).forEach((value,index)=>{
      if (value && typeof value==="object") result.push({
        id:String(value.instance_id||value.id||`instance ${index}`), data:value, frames:[]
      });
    });
  } else if (replay.instances && typeof replay.instances==="object") {
    Object.entries(replay.instances).slice(0,256).forEach(([id,value])=>result.push({
      id, data:value&&typeof value==="object"?value:{}, frames:[]
    }));
  }
  const allEvents=[...(Array.isArray(replay.events)?replay.events:[]),
    ...(Array.isArray(replay.steps)?replay.steps:[])].slice(0,MAX_FRAMES);
  allEvents.forEach(event => {
    const id=String(event?.instance_id||replay.primary_instance_id||"primary");
    let instance=result.find(item=>item.id===id);
    if (!instance) { instance={id,data:{},frames:[]}; result.push(instance); }
    instance.frames.push(event);
  });
  result.forEach(instance => {
    const local=instance.data.events||instance.data.frames||instance.data.observations||
      instance.data.grid_evolution||instance.data.actions;
    if (Array.isArray(local)) instance.frames.push(...local.slice(0,MAX_FRAMES-instance.frames.length));
  });
  if (!result.length) {
    const local=replay.frames||replay.actions||replay.events||replay.steps||[];
    result.push({id:String(replay.primary_instance_id||"primary"),data:replay,
      frames:Array.isArray(local)?local.slice(0,MAX_FRAMES):[]});
  }
  return result.sort((left,right)=>Number(Boolean(right.data.primary))-Number(Boolean(left.data.primary)));
}
function frameValue(frame) {
  if (!frame || typeof frame!=="object") return frame;
  return frame.frame ?? frame.observation ?? frame.state ?? frame.status ?? frame;
}
function asciiOf(value, seen=new Set()) {
  if (typeof value==="string") return value.slice(0,MAX_ASCII);
  if (Array.isArray(value) && value.every(item=>typeof item==="string"))
    return value.join("\n").slice(0,MAX_ASCII);
  if (!value || typeof value!=="object" || seen.has(value)) return "";
  seen.add(value);
  for (const key of ["ascii","ascii_map","current_view","view","map","maze","render","observation"]) {
    if (key in value) { const found=asciiOf(value[key],seen); if (found) return found; }
  }
  return "";
}
function scrubber(parent, frames, renderFrame) {
  const controls=$("div","replay-controls"), slider=$("input"), position=$("span","muted");
  slider.type="range"; slider.min="0"; slider.max=String(Math.max(0,frames.length-1));
  slider.value="0"; slider.disabled=!frames.length;
  const previous=$("button","","←"), next=$("button","","→");
  const show=()=> {
    const index=Number(slider.value)||0;
    position.textContent=frames.length?`${index+1} / ${frames.length}`:"no frames";
    renderFrame(frames[index],index);
  };
  previous.onclick=()=>{ slider.value=String(Math.max(0,Number(slider.value)-1)); show(); };
  next.onclick=()=>{ slider.value=String(Math.min(Math.max(0,frames.length-1),Number(slider.value)+1)); show(); };
  slider.oninput=show; controls.append(previous,slider,next,position); parent.append(controls); show();
}
function mazeView(replay, discovery) {
  const box=$("div","viewer"), tabs=$("div","tabs"), stage=$("div");
  const instances=replayInstances(replay);
  const select=instance => {
    [...tabs.children].forEach(button=>button.classList.toggle("active",button.dataset.id===instance.id));
    stage.replaceChildren();
    const frames=[];
    if (instance.data.initial!==undefined) frames.push({label:"initial",observation:instance.data.initial});
    frames.push(...instance.frames);
    scrubber(stage,frames,(frame,index)=>{
      const old=stage.querySelector(".maze-frame"); if (old) old.remove();
      const panel=$("div","maze-frame"), value=frameValue(frame);
      const ascii=asciiOf(value)||asciiOf(frame);
      const command=frame?.command ?? frame?.action ?? frame?.accepted_actions;
      panel.append($("div","muted",`frame ${index} · ${dump(command)} · ${frame?.instance_id||instance.id}`));
      panel.append($("pre","maze-ascii",ascii||dump(value)));
      const route=(instance.data.actions||[]).slice(0,index+1)
        .map(step=>Array.isArray(step?.to)?`(${step.to.join(",")})`:step?.action)
        .filter(Boolean).join(" → ");
      if (route) panel.append($("div","route-overlay",`route overlay: ${route}`));
      stage.append(panel);
    });
  };
  instances.forEach((instance,index)=>{
    const state=discoveryState(discovery,instance.id);
    const button=$("button",state,`${instance.id}${state?` · ${state}`:""}`);
    button.dataset.id=instance.id; button.onclick=()=>select(instance); tabs.append(button);
    if (!index) select(instance);
  });
  box.append(tabs,stage); return box;
}
function gridFrom(value, depth=0) {
  if (depth>5 || value===null || value===undefined) return null;
  if (typeof value==="string") {
    try { return gridFrom(JSON.parse(value),depth+1); } catch (_) {
      const cells=(value.match(/[0-9.]/g)||[]), side=Math.sqrt(cells.length);
      return Number.isInteger(side)&&side>=2&&side<=MAX_GRID_SIDE
        ? Array.from({length:side},(_,i)=>cells.slice(i*side,(i+1)*side).map(v=>v==="."?0:+v))
        : null;
    }
  }
  if (Array.isArray(value) && value.length && value.every(row=>Array.isArray(row)))
    return value.slice(0,MAX_GRID_SIDE).map(row=>row.slice(0,MAX_GRID_SIDE));
  if (typeof value==="object") {
    for (const key of ["grid","board","frame","observation","state","prediction","proposal"]) {
      if (key in value) { const found=gridFrom(value[key],depth+1); if (found) return found; }
    }
  }
  return null;
}
function arcView(replay, discovery) {
  const box=$("div","viewer"), tabs=$("div","tabs"), stage=$("div");
  const instances=replayInstances(replay);
  const select=instance => {
    [...tabs.children].forEach(button=>button.classList.toggle("active",button.dataset.id===instance.id));
    stage.replaceChildren();
    let frames=instance.frames.filter(event=>gridFrom(event?.frame??event?.observation??event));
    if (!frames.length && replay.task) {
      frames=[
        ...(replay.task.train||[]).flatMap((pair,index)=>[
          {action:`train ${index} input`,frame:pair.input},
          {action:`train ${index} output`,frame:pair.output},
        ]),
        ...(replay.task.test||[]).flatMap((pair,index)=>[
          {action:`test ${index} input`,frame:pair.input},
          ...(gridFrom(replay.prediction?.[index]??replay.prediction)
            ? [{action:`test ${index} prediction`,frame:replay.prediction?.[index]??replay.prediction}]
            : []),
        ]),
      ].slice(0,MAX_FRAMES);
    }
    scrubber(stage,frames,(event,index)=>{
      const old=stage.querySelector(".arc-frame"); if (old) old.remove();
      const panel=$("div","arc-frame");
      panel.append($("div","muted",
        `frame ${index} · ${event?.action?.name||event?.action||"observe"} · ${event?.state||"—"} · ${event?.replay?"replay":"live"}`));
      panel.append(arcGrid(gridFrom(event?.frame??event?.observation??event))); stage.append(panel);
    });
  };
  instances.forEach((instance,index)=>{
    const state=discoveryState(discovery,instance.id);
    const button=$("button",state,`${instance.id}${instance.data.primary?" · primary":""}${state?` · ${state}`:""}`);
    button.dataset.id=instance.id; button.onclick=()=>select(instance); tabs.append(button);
    if (!index) select(instance);
  });
  box.append(tabs,stage); return box;
}
function sudokuViolations(grid) {
  const bad=new Set(), side=grid.length;
  const mark=positions=>{
    const seen=new Map();
    positions.forEach(([r,c])=>{
      const value=Number(grid[r]?.[c]); if (!value) return;
      if (seen.has(value)) { bad.add(`${r}:${c}`); bad.add(seen.get(value)); }
      else seen.set(value,`${r}:${c}`);
    });
  };
  for (let i=0;i<side;i++) {
    mark(Array.from({length:side},(_,j)=>[i,j]));
    mark(Array.from({length:side},(_,j)=>[j,i]));
  }
  const root=Math.sqrt(side);
  const boxRows=side===6?2:(Number.isInteger(root)?root:0), boxColumns=boxRows?side/boxRows:0;
  if (boxRows && Number.isInteger(boxColumns)) for(let br=0;br<side;br+=boxRows)
    for(let bc=0;bc<side;bc+=boxColumns)
      mark(Array.from({length:side},(_,n)=>[br+Math.floor(n/boxColumns),bc+n%boxColumns]));
  return bad;
}
function sudokuGrid(grid) {
  const wrap=$("div","sudoku-grid"), bad=sudokuViolations(grid), side=grid.length;
  wrap.style.gridTemplateColumns=`repeat(${side},27px)`;
  grid.forEach((row,r)=>row.slice(0,side).forEach((value,c)=>
    wrap.append($("span",`sudoku-cell ${bad.has(`${r}:${c}`)?"violation":""}`,value||""))));
  return {wrap,bad};
}
function sudokuView(replay, discovery) {
  const box=$("div","viewer"), tabs=$("div","tabs"), stage=$("div");
  const instances=replayInstances(replay);
  const select=instance=>{
    [...tabs.children].forEach(button=>button.classList.toggle("active",button.dataset.id===instance.id));
    stage.replaceChildren();
    const frames=[];
    const add=(label,value,state="")=>{
      const grid=gridFrom(value); if(grid) frames.push({label,grid,state,value});
    };
    add("initial",replay.initial??replay.initial_grid);
    (instance.data.grid_evolution||instance.frames||[]).slice(0,MAX_FRAMES)
      .forEach((value,index)=>add(`step ${value?.index??index}`,value));
    add("final",instance.data.final_grid??replay.final_grid??replay.prediction,
      discoveryState(discovery,instance.id));
    scrubber(stage,frames,(frame)=>{
      const old=stage.querySelector(".sudoku-frame"); if(old) old.remove();
      const panel=$("div","sudoku-frame"), rendered=frame?sudokuGrid(frame.grid):null;
      const provided=frame?.value?.violations ?? frame?.value?.public_summary?.violations;
      const badCount=rendered?rendered.bad.size:0;
      panel.append($("div",`muted ${frame?.state||""}`,
        frame?`${frame.label} · ${frame.state||"candidate"} · ${badCount} violating cells${provided!==undefined?` · reported ${dump(provided)}`:""}`:"No grid frames available."));
      if(rendered) panel.append(rendered.wrap);
      stage.append(panel);
    });
  };
  instances.forEach((instance,index)=>{
    const state=discoveryState(discovery,instance.id);
    const button=$("button",state,`${instance.id}${state?` · ${state}`:""}`);
    button.dataset.id=instance.id; button.onclick=()=>select(instance); tabs.append(button);
    if(!index) select(instance);
  });
  box.append(tabs,stage);
  return box;
}
function replayView(replay, discovery) {
  const kind=String(replay.kind||"native").toLowerCase();
  const box=$("details"); box.open=true; box.append($("summary", "", `${kind} replay`));
  if (replay.status==="external" && replay.external_href) {
    const link=$("a","","Open portable replay JSON");
    link.href=replay.external_href; link.download="";
    box.append($("div","muted",`${fmt(replay.bytes)} bytes · kept outside the bounded HTML payload`),
      link,details("replay summary",replay.summary||{}));
  }
  else if (kind.startsWith("arc")) box.append(arcView(replay,discovery));
  else if (kind.includes("maze")) box.append(mazeView(replay,discovery));
  else if (kind.includes("sudoku")) box.append(sudokuView(replay,discovery));
  else box.append($("pre", "", dump(replay)));
  return box;
}
function laneView(lane) {
  const box=$("section","lane"); const heading=$("h3");
  heading.append(document.createTextNode(`${lane.runner} · seed ${lane.seed} `),
    $("span",`badge ${lane.infra_status}`,lane.infra_status));
  box.append(heading,$("div","muted",lane.model));
  const m=$("div","metrics");
  m.append(metric("score",lane.score.label),metric("tokens",fmt(lane.tokens.total)),
    metric("seconds",fmt(lane.time.wall_seconds)),
    metric("actions",`${lane.actions.count} (${lane.actions.primary} primary / ${lane.actions.probe} probe)`),
    metric("topology",`${lane.topology.agents}a / observed d${lane.topology.max_depth} / cap d${lane.topology.configured_depth}`),
    metric("selection compliance",
      lane.candidate_search.selection_compliance===null?"n/a":
      (lane.candidate_search.selection_compliance?"yes":"no")));
  box.append(m);
  if (lane.actions.events.length) {
    const strip=$("div","timeline");
    lane.actions.events.forEach(event => {
      const button=$("button",`event ${event.graph_event ? "linked" : ""}`,event.index);
      if (event.exec_action_id) button.dataset.execActionId=event.exec_action_id;
      if (event.agent_id) button.dataset.agentId=event.agent_id;
      if (event.instance_id) button.dataset.instanceId=event.instance_id;
      button.title=`${event.agent || "unknown"} · depth ${event.depth}`;
      button.onclick=()=> {
        const old=box.querySelector(".event-detail"); if (old) old.remove();
        const view=details(`action ${event.index}`,event); view.className="event-detail"; box.append(view);
      };
      strip.append(button);
    });
    box.append(strip);
  }
  box.append(details("answer",lane.answer),details("score details",lane.score.details));
  box.append(details("score efficiency",lane.efficiency));
  box.append(details("agent graph", {
    topology: lane.topology,
    exec_actions: lane.graph.exec_actions || [],
  }));
  if (lane.error) box.append(details("error",lane.error));
  if (lane.discovery_tree) box.append(discoveryTreeView(lane.discovery_tree));
  if (lane.replay) box.append(replayView(lane.replay,lane.discovery_tree));
  const links=$("div","links");
  Object.entries(lane.artifacts).forEach(([name,item]) => {
    if (item.href) { const a=$("a",item.status,name); a.href=item.href; links.append(a); }
    else links.append($("span",item.status,`${name}: ${item.status}`));
  });
  if (links.childNodes.length) box.append(links);
  return box;
}
function renderOverview() {
  overview.replaceChildren();
  const grid=$("section","overview-grid");
  Object.entries(DATA.matrix.by_domain || {}).forEach(([domain,runners]) => {
    const card=$("article","domain-summary"); card.append($("h2","",domain));
    const values=Object.values(runners).map(item=>Number(item.mean_score)||0);
    const max=Math.max(1,...values);
    Object.entries(runners).forEach(([runner,item]) => {
      const row=$("div","runner-bar");
      row.append($("div","",`${runner}: ${fmt(item.mean_score)} · ${fmt(item.tokens)} tok · ${fmt(item.environment_actions)} env`));
      const meter=$("meter"); meter.min=0; meter.max=max; meter.value=Number(item.mean_score)||0;
      row.append(meter);
      const paired=(DATA.matrix.paired_deltas_by_domain?.[domain]||{})[runner];
      if (paired) row.append($("div","delta",
        `Δ ${fmt(paired.mean_delta)} · 95% CI ${fmt(paired.ci95[0])}…${fmt(paired.ci95[1])} · ${paired.paired_examples} tasks / ${paired.paired_rows} rows`));
      card.append(row);
    });
    grid.append(card);
  });
  if (grid.childNodes.length) overview.append(grid);
}
function render() {
  main.replaceChildren();
  const selected = id => document.getElementById(id).value;
  const taskScore = task => Math.max(...task.lanes.map(lane=>Number(lane.score.value)||0),0);
  const ordered=[...DATA.tasks].sort((left,right)=>{
    const mode=selected("sort");
    if(mode==="best-score") return taskScore(right)-taskScore(left);
    if(mode==="domain") return String(left.domain).localeCompare(String(right.domain)) ||
      String(left.example_id).localeCompare(String(right.example_id));
    return String(left.example_id).localeCompare(String(right.example_id));
  });
  let shown=0;
  ordered.forEach(task => {
    const lanes=task.lanes.filter(lane =>
      (!selected("runner") || lane.runner===selected("runner")) &&
      (!selected("seed") || lane.seed===selected("seed")) &&
      (!selected("infra") || lane.infra_status===selected("infra")) &&
      (!selected("selected-depth") ||
        String(lane.candidate_search.selected_depth)===selected("selected-depth")) &&
      (!selected("candidate-count") ||
        String(lane.candidate_search.count)===selected("candidate-count")) &&
      (!selected("delegation-topology") ||
        lane.delegation_topology===selected("delegation-topology")));
    if ((!selected("domain") || task.domain===selected("domain")) && lanes.length) {
      const card=$("article","task"); card.append($("h2","",
        `${task.example_id} · ${task.domain}${task.role ? " · "+task.role : ""}`));
      const headline=lanes.filter(lane=>
        lane.runner==="official-rlm"||lane.runner==="rlmflow-depth-2");
      const diagnostic=lanes.filter(lane=>!headline.includes(lane));
      const laneBox=$("div","lanes");
      (headline.length?headline:lanes).forEach(lane=>laneBox.append(laneView(lane)));
      card.append(laneBox);
      if(headline.length&&diagnostic.length) {
        const extra=$("details");
        extra.append($("summary","",`diagnostic controls · ${diagnostic.length} lanes`));
        const extraLanes=$("div","lanes");
        diagnostic.forEach(lane=>extraLanes.append(laneView(lane)));
        extra.append(extraLanes); card.append(extra);
      }
      main.append(card); shown++;
    }
  });
  if (!shown) main.append($("div","empty","No lanes match these filters."));
}
function filter(id, label, values) {
  const wrap=$("label","",label), select=$("select"); select.id=id;
  select.append($("option","","all")); select.firstChild.value="";
  values.forEach(value => { const option=$("option","",value); option.value=value; select.append(option); });
  select.onchange=render; wrap.append(select); document.getElementById("filters").append(wrap);
}
filter("domain","domain",DATA.filters.domains); filter("runner","runner",DATA.filters.runners);
filter("seed","seed",DATA.filters.seeds); filter("infra","infra status",DATA.filters.infra_statuses);
filter("selected-depth","selected depth",DATA.filters.selected_depths);
filter("candidate-count","candidate count",DATA.filters.candidate_counts);
filter("delegation-topology","delegation topology",DATA.filters.delegation_topologies);
filter("sort","sort",["task","domain","best-score"]);
document.getElementById("subtitle").textContent =
  `${DATA.tasks.length} tasks · ${DATA.tasks.reduce((n,t)=>n+t.lanes.length,0)} lanes · offline`;
renderOverview();
render();
"""


def render_visual_html(
    payload: Mapping[str, Any],
    *,
    title: str | None = None,
    asset_paths: Mapping[str, str] | None = None,
) -> str:
    """Render ``payload`` as an offline HTML application.

    JSON characters significant to the HTML parser are emitted as unicode
    escapes.  The script renders every value through ``textContent``.
    """
    heading = str(title if title is not None else payload.get("title", "Benchmark report"))
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    encoded = (
        encoded.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    if asset_paths is None:
        policy = (
            "default-src 'none'; style-src 'unsafe-inline'; "
            "script-src 'unsafe-inline'; img-src data:;"
        )
        style = f"<style>{_CSS}</style>"
        data = f'<script type="application/json" id="report-data">{encoded}</script>'
        script = f"<script>{_JS}</script>"
    else:
        css_href = html.escape(str(asset_paths["css"]), quote=True)
        javascript_src = html.escape(str(asset_paths["javascript"]), quote=True)
        policy = "default-src 'none'; style-src 'self'; script-src 'self'; img-src data:;"
        style = f'<link rel="stylesheet" href="{css_href}"/>'
        data = f'<div hidden id="report-data">{encoded}</div>'
        script = f'<script src="{javascript_src}"></script>'
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<meta http-equiv="Content-Security-Policy" content="{policy}"/>
<title>{html.escape(heading)}</title>{style}</head>
<body><header><h1>{html.escape(heading)}</h1><div class="sub" id="subtitle"></div></header>
<div class="filters" id="filters"></div><main id="overview"></main><main id="tasks"></main>
{data}
{script}</body></html>
"""


def write_visual_report(
    rows: Sequence[Row],
    output: str | Path,
    *,
    artifact_root: str | Path | None = None,
    title: str = "Benchmark visual report",
) -> Path:
    """Write ``report.html`` and return its path.

    ``output`` may name the HTML file or a directory.
    """
    path = Path(output).expanduser()
    if path.suffix.casefold() != ".html":
        path = path / "report.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_visual_payload(
        rows,
        artifact_root=artifact_root or path.parent,
        report_dir=path.parent,
        title=title,
    )
    externalize_large_replays(payload, output_dir=path.parent)
    payload_path = path.parent / "visual_report.json"
    payload_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    asset_paths = write_static_assets(
        path.parent,
        css=_CSS,
        javascript=_JS,
    )
    path.write_text(
        render_visual_html(payload, asset_paths=asset_paths),
        encoding="utf-8",
    )
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Run directory or rows.jsonl path")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output HTML path (default: RUN/report.html)",
    )
    parser.add_argument("--title", default="Benchmark visual report")
    args = parser.parse_args(argv)

    source = args.run.expanduser()
    rows_path = source if source.suffix == ".jsonl" else source / "rows.jsonl"
    run_root = rows_path.parent.resolve()
    rows = load_rows(rows_path)
    if not rows:
        parser.error(f"run has no rows: {rows_path}")
    output = args.output or run_root / "report.html"
    result = write_visual_report(
        rows,
        output,
        artifact_root=run_root,
        title=args.title,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_visual_payload",
    "main",
    "render_visual_html",
    "write_visual_report",
]
