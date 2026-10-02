"""Runner-neutral evidence contract for the swarm-fanout benchmark."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any

SCHEMA_VERSION = 1


def shard_for_text(shards: list[dict[str, Any]], text: str) -> str | None:
    """Return the one shard unambiguously named by text."""
    normalized = text.replace("\\", "/")
    hits = [str(shard["file"]) for shard in shards if str(shard["file"]) in normalized]
    return hits[0] if len(hits) == 1 else None


def trace(runtime: str, children: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "runtime": runtime,
        "children": children,
    }


def rlmflow_trace(shards: list[dict[str, Any]], child_runs: list[dict[str, Any]]) -> dict[str, Any]:
    worker_to_shard = {str(shard["worker"]): str(shard["file"]) for shard in shards}
    children = []
    for run in child_runs:
        if run.get("depth") != 1:
            continue
        name = str(run.get("name") or "")
        children.append(
            {
                "id": str(run.get("agent_id") or ""),
                "parent_id": str(run.get("parent_agent_id") or "root"),
                "depth": 1,
                "worker": name,
                "shard": worker_to_shard.get(name),
                "terminal": bool(run.get("terminal")),
                "failed": bool(run.get("failed")),
                "persisted": True,
                "receipt": bool(run.get("result_retrieved")),
                "batch_id": run.get("launch_action_id"),
                "overlap_duplicate": False,
                "prompt_sha256": run.get("prompt_sha256"),
            }
        )
    _mark_batch_duplicates(children)
    return trace("rlmflow", children)


def official_trace(shards: list[dict[str, Any]], calls: list[dict[str, Any]]) -> dict[str, Any]:
    worker_for_shard = {str(shard["file"]): str(shard["worker"]) for shard in shards}
    children = []
    for index, call in enumerate(calls):
        prompt = _text(call.get("prompt"))
        shard = shard_for_text(shards, prompt)
        response = str(call.get("response") or "")
        error = call.get("error")
        children.append(
            {
                "id": f"official-call-{index:04d}",
                "parent_id": "root",
                "depth": 1,
                "worker": worker_for_shard.get(shard or ""),
                "shard": shard,
                "terminal": True,
                "failed": bool(error) or response.startswith("Error:"),
                "persisted": True,
                "receipt": True,
                "batch_id": call.get("batch_id"),
                "overlap_duplicate": False,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            }
        )
    _mark_batch_duplicates(children)
    return trace("official-rlm", children)


def jaz_trace(shards: list[dict[str, Any]], invokes: list[dict[str, Any]]) -> dict[str, Any]:
    """Adapt JAZ's nested invoke records to the shared fan-out contract."""
    worker_to_shard = {str(shard["worker"]): str(shard["file"]) for shard in shards}
    children = []
    for invoke in invokes:
        if invoke.get("depth") != 2:
            continue
        worker = str(invoke.get("worker") or "")
        shard = str(invoke.get("filename") or "") or worker_to_shard.get(worker)
        if shard not in worker_to_shard.values():
            shard = worker_to_shard.get(worker)
        children.append(
            {
                "id": str(invoke.get("invoke_id") or ""),
                "parent_id": str(invoke.get("parent_invoke_id") or "root"),
                "depth": 1,
                "worker": worker,
                "shard": shard,
                "terminal": bool(invoke.get("terminal")),
                "failed": bool(invoke.get("failed")),
                "persisted": bool(invoke.get("persisted")),
                "receipt": bool(invoke.get("receipt")),
                "batch_id": invoke.get("parent_repl_iteration"),
                "overlap_duplicate": False,
                "prompt_sha256": invoke.get("prompt_sha256"),
            }
        )
    _mark_batch_duplicates(children)
    return trace("jaz", children)


def prime_trace(
    shards: list[dict[str, Any]],
    records: list[dict[str, Any]],
    *,
    persisted_child_ids: set[str],
    reply_session_ids: set[str],
    noticed_child_ids: set[str],
) -> dict[str, Any]:
    worker_to_shard = {str(shard["worker"]): str(shard["file"]) for shard in shards}
    live_by_worker: dict[str, set[str]] = {}
    children_by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        child_id = record.get("childId")
        if not isinstance(child_id, str):
            continue
        if record.get("op") == "spawn" and record.get("depth") == 1:
            worker = str(record.get("name") or "")
            live = live_by_worker.setdefault(worker, set())
            child_file = str(record.get("child") or "")
            session_id = child_file.replace("\\", "/").rsplit("/", 1)[-1].removesuffix(".jsonl")
            child = {
                "id": child_id,
                "parent_id": str(record.get("parent") or "root"),
                "depth": 1,
                "worker": worker,
                "shard": worker_to_shard.get(worker),
                "terminal": False,
                "failed": False,
                "persisted": child_id in persisted_child_ids,
                "receipt": session_id in reply_session_ids or child_id in noticed_child_ids,
                "batch_id": record.get("at"),
                "overlap_duplicate": bool(live),
                "prompt_sha256": _prompt_hash(record),
            }
            children_by_id[child_id] = child
            live.add(child_id)
        elif record.get("op") == "delete":
            child = children_by_id.get(child_id)
            if child is not None:
                child["terminal"] = True
                child["failed"] = str(record.get("reason") or "").casefold() not in {
                    "completed",
                    "parent-teardown",
                }
                live_by_worker.get(str(child.get("worker") or ""), set()).discard(child_id)
    for child in children_by_id.values():
        child["terminal"] = bool(child["terminal"] or child["persisted"])
    return trace("prime-agent", list(children_by_id.values()))


def score_orchestration(
    shards: list[dict[str, Any]],
    evidence: dict[str, Any],
    *,
    expected_prompt_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    required = {str(shard["file"]) for shard in shards}
    children = [
        child
        for child in evidence.get("children", [])
        if isinstance(child, dict) and child.get("depth") == 1
    ]
    by_shard = Counter(str(child["shard"]) for child in children if child.get("shard") in required)
    missing = sorted(required - set(by_shard))
    unpersisted = sorted(str(child.get("id")) for child in children if not child.get("persisted"))
    unterminated = sorted(str(child.get("id")) for child in children if not child.get("terminal"))
    overlap_duplicates = sorted(
        str(child.get("id")) for child in children if child.get("overlap_duplicate")
    )
    spawn_budget = 2 * len(shards)
    missing_receipts = sorted(
        str(child.get("id"))
        for child in children
        if not child.get("receipt") and not child.get("failed")
    )
    prompt_mismatches = sorted(
        str(child.get("id"))
        for child in children
        if expected_prompt_hashes is not None
        and child.get("shard") in expected_prompt_hashes
        and child.get("prompt_sha256") != expected_prompt_hashes[child["shard"]]
    )
    return {
        "delegation_evidence": not missing and not unpersisted and not unterminated,
        "dedup": not overlap_duplicates and len(children) <= spawn_budget,
        "receipts": not missing_receipts and not missing,
        "prompt_parity": expected_prompt_hashes is None
        or (not missing and not prompt_mismatches),
        "missing_delegated_shards": missing,
        "unpersisted_children": unpersisted,
        "unterminated_children": unterminated,
        "overlap_duplicates": overlap_duplicates,
        "missing_receipts": missing_receipts,
        "prompt_mismatches": prompt_mismatches,
        "direct_children": len(children),
        "children": len(children),
        "launch_batches": len(
            {child.get("batch_id") for child in children if child.get("batch_id") is not None}
        ),
        "retrieved_child_results": sum(bool(child.get("receipt")) for child in children),
        "duplicate_launches": len(overlap_duplicates),
        "verified_shards": len(required & set(by_shard)),
        "spawn_budget": spawn_budget,
        "runtime": evidence.get("runtime"),
        "trace_schema_version": evidence.get("schema_version"),
    }


def _mark_batch_duplicates(children: list[dict[str, Any]]) -> None:
    seen: set[tuple[object, object]] = set()
    for child in children:
        key = (child.get("batch_id"), child.get("shard"))
        if child.get("shard") is not None and key in seen:
            child["overlap_duplicate"] = True
        seen.add(key)


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _prompt_hash(record: dict[str, Any]) -> str | None:
    prompt = record.get("task")
    if not isinstance(prompt, str):
        prompt = record.get("prompt")
    return hashlib.sha256(prompt.encode()).hexdigest() if isinstance(prompt, str) else None


__all__ = [
    "SCHEMA_VERSION",
    "jaz_trace",
    "official_trace",
    "prime_trace",
    "rlmflow_trace",
    "score_orchestration",
    "shard_for_text",
    "trace",
]
