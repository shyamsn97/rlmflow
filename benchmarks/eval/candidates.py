"""Shared candidate ledger and zero-execution discovery-tree replay."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _json_value(value: Any, *, max_chars: int = 8_000) -> Any:
    """Return a bounded JSON value suitable for durable benchmark artifacts."""
    try:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        encoded = json.dumps(str(value), ensure_ascii=False)
    if len(encoded) <= max_chars:
        return json.loads(encoded)
    return {"truncated": True, "preview": encoded[:max_chars]}


@dataclass(slots=True)
class CandidateRecord:
    id: str
    index: int
    owner_id: str
    owner_path: str
    owner_depth: int
    instance_id: str
    parent_candidate_id: str | None
    label: str
    proposal: Any
    environment_prefix: Any
    state_hash: str
    evaluation_cost: int
    score: float
    public_summary: dict[str, Any]
    replay_artifacts: dict[str, str]
    created_at: float = field(default_factory=time.time)
    selected: bool = False

    def to_dict(self, *, public: bool = False) -> dict[str, Any]:
        value = asdict(self)
        value["proposal"] = _json_value(value["proposal"])
        if public:
            value.pop("score", None)
            value.pop("state_hash", None)
        return value


class CandidateLedger:
    """Thread-safe row-level candidate registry with explicit root selection."""

    def __init__(self, *, max_evaluations: int, episode_id: str) -> None:
        if (
            isinstance(max_evaluations, bool)
            or not isinstance(max_evaluations, int)
            or max_evaluations < 1
        ):
            raise ValueError("max_evaluations must be a positive integer")
        self.max_evaluations = max_evaluations
        self.episode_id = episode_id
        self._records: list[CandidateRecord] = []
        self._by_id: dict[str, CandidateRecord] = {}
        self._selected_id: str | None = None
        self._lock = threading.RLock()

    @property
    def evaluations_remaining(self) -> int:
        return max(
            0,
            self.max_evaluations - sum(record.evaluation_cost for record in self._records),
        )

    @property
    def selected(self) -> CandidateRecord | None:
        return self._by_id.get(self._selected_id) if self._selected_id else None

    def validate_submission(
        self,
        *,
        owner_path: str,
        label: str,
        parent_candidate_id: str | None,
        evaluation_cost: int = 1,
    ) -> None:
        """Validate a candidate request before an environment snapshot is created."""
        with self._lock:
            if (
                isinstance(evaluation_cost, bool)
                or not isinstance(evaluation_cost, int)
                or evaluation_cost < 1
            ):
                raise ValueError("evaluation_cost must be a positive integer")
            if self.evaluations_remaining < evaluation_cost:
                raise RuntimeError("candidate evaluation budget exhausted")
            if not isinstance(label, str) or not label.strip():
                raise ValueError("candidate label must be non-empty")
            if len(label) > 256:
                raise ValueError("candidate label must not exceed 256 characters")
            if parent_candidate_id is not None and parent_candidate_id not in self._by_id:
                raise ValueError(f"unknown parent candidate {parent_candidate_id!r}")
            if parent_candidate_id is not None:
                parent = self._by_id[parent_candidate_id]
                if not (
                    owner_path == parent.owner_path
                    or owner_path.startswith(parent.owner_path.rstrip("/") + "/")
                ):
                    raise PermissionError(
                        "candidate parent must belong to the owner or an ancestor branch"
                    )

    def submit(
        self,
        *,
        owner_id: str,
        owner_path: str,
        owner_depth: int,
        instance_id: str,
        label: str,
        proposal: Any,
        parent_candidate_id: str | None,
        environment_prefix: Any = None,
        state_material: Any,
        score: float,
        public_summary: Mapping[str, Any],
        evaluation_cost: int = 1,
        replay_artifacts: Mapping[str, str] | None = None,
    ) -> CandidateRecord:
        with self._lock:
            self.validate_submission(
                owner_path=owner_path,
                label=label,
                parent_candidate_id=parent_candidate_id,
                evaluation_cost=evaluation_cost,
            )
            index = len(self._records)
            canonical = json.dumps(
                _json_value(state_material),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            state_hash = hashlib.sha256(canonical.encode()).hexdigest()
            candidate_id = (
                "candidate_"
                + hashlib.sha256(
                    (
                        f"{self.episode_id}\0{index}\0{owner_id}\0{label.strip()}\0{state_hash}"
                    ).encode()
                ).hexdigest()[:16]
            )
            record = CandidateRecord(
                id=candidate_id,
                index=index,
                owner_id=owner_id,
                owner_path=owner_path,
                owner_depth=owner_depth,
                instance_id=instance_id,
                parent_candidate_id=parent_candidate_id,
                label=label.strip(),
                proposal=_json_value(proposal),
                environment_prefix=_json_value(environment_prefix),
                state_hash=state_hash,
                evaluation_cost=evaluation_cost,
                score=float(score),
                public_summary=dict(_json_value(dict(public_summary))),
                replay_artifacts=dict(replay_artifacts or {}),
            )
            self._records.append(record)
            self._by_id[record.id] = record
            return record

    def select(self, candidate_id: str, *, actor_is_root: bool) -> CandidateRecord:
        with self._lock:
            if not actor_is_root:
                raise PermissionError("only the root agent may select the scored candidate")
            try:
                record = self._by_id[candidate_id]
            except KeyError as exc:
                raise ValueError(f"unknown candidate {candidate_id!r}") from exc
            if self._selected_id is not None:
                self._by_id[self._selected_id].selected = False
            record.selected = True
            self._selected_id = candidate_id
            return record

    def public_records(self) -> list[dict[str, Any]]:
        with self._lock:
            return [record.to_dict(public=True) for record in self._records]

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            selected = self.selected
            best = max(self._records, key=lambda record: record.score, default=None)
            count = len(self._records)
            proposal_fingerprints = {
                json.dumps(
                    {"label": record.label, "proposal": record.proposal},
                    ensure_ascii=False,
                    sort_keys=True,
                )
                for record in self._records
            }
            outcome_fingerprints = {
                (round(record.score, 8), json.dumps(record.public_summary, sort_keys=True))
                for record in self._records
            }
            return {
                "schema_version": 1,
                "episode_id": self.episode_id,
                "evaluation_budget": self.max_evaluations,
                "evaluations_used": sum(record.evaluation_cost for record in self._records),
                "evaluations_remaining": self.evaluations_remaining,
                "selected_candidate_id": selected.id if selected else None,
                "selected_score": selected.score if selected else None,
                "oracle_best_candidate_id": best.id if best else None,
                "oracle_best_score": best.score if best else None,
                "selection_regret": (
                    best.score - selected.score
                    if best is not None and selected is not None
                    else None
                ),
                "candidate_diversity": (len(proposal_fingerprints) / count if count else 0.0),
                "outcome_diversity": (len(outcome_fingerprints) / count if count else 0.0),
                "candidates": [record.to_dict() for record in self._records],
                "replay_policies": replay_discovery_policies(self._records),
            }

    def write(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return output


def replay_discovery_policies(
    records: Sequence[CandidateRecord],
    *,
    reveal_budget: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Score deterministic policies over already-recorded outcomes."""
    budget = len(records) if reveal_budget is None else max(0, min(reveal_budget, len(records)))
    policies = {
        "fixed_breadth": sorted(records, key=lambda item: (item.owner_depth, item.index)),
        "score_greedy": sorted(records, key=lambda item: (-item.score, item.index)),
        "depth_pruning": sorted(records, key=lambda item: (item.owner_depth > 1, item.index)),
    }
    result: dict[str, dict[str, Any]] = {}
    for name, ordered in policies.items():
        revealed = ordered[:budget]
        selected = max(revealed, key=lambda item: item.score, default=None)
        result[name] = {
            "revealed_candidate_ids": [item.id for item in revealed],
            "selected_candidate_id": selected.id if selected else None,
            "score": selected.score if selected else None,
            "evaluation_cost": sum(item.evaluation_cost for item in revealed),
        }
    return result


__all__ = ["CandidateLedger", "CandidateRecord", "replay_discovery_policies"]
