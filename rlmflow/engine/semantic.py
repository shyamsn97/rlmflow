"""Choose once among semantic transitions using a bounded graph read-set."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from rlmflow.graph.nodes import LLMUsage, Node, TransitionDecision

DEFAULT_READ_SET = 12

CONTINUE = "continue"


async def choose(
    flow: Any,
    node: Node,
    candidates: tuple[tuple[str, str, str], ...],
) -> TransitionDecision:
    """Ask ``flow``'s structured engine to select one eligible edge or none."""
    if not candidates:
        raise ValueError("semantic transition choice requires at least one candidate")
    candidate_map = {rule_id: condition for rule_id, _label, condition in candidates}
    labels = {label: rule_id for rule_id, label, _condition in candidates}
    if len(candidate_map) != len(candidates):
        raise ValueError("semantic transition candidates must have unique rule IDs")
    if len(labels) != len(candidates):
        raise ValueError("semantic transition candidates must have unique labels")

    limit = node.parent_agent.config.keep_n_messages or DEFAULT_READ_SET
    read_nodes = list(node.iter_backwards())[:limit]
    read_nodes.reverse()
    state = [{"id": item.id, "type": item.type, "content": item.content} for item in read_nodes]
    encoded = json.dumps(state, sort_keys=True, separators=(",", ":"))
    schema: dict[str, Any] = {
        "type": "string",
        "enum": [*labels, CONTINUE],
    }
    options = "\n".join(f"- `{label}`: {condition}" for _rule_id, label, condition in candidates)
    instruction = (
        "Choose exactly one transition whose condition is satisfied by the graph "
        f"nodes below, or `{CONTINUE}` to resume deterministic transition "
        "resolution. Return only that transition label. When uncertain, choose "
        f"`{CONTINUE}`.\n\nTransitions:\n"
        f"{options}"
    )
    started_at = time.time()
    result = await flow.structured_query(
        {
            "read_set": state,
            "candidates": [
                {"transition": label, "condition": condition}
                for _rule_id, label, condition in candidates
            ],
        },
        output_schema=schema,
        instruction=instruction,
    )
    value = result.value
    if not isinstance(value, str):
        raise TypeError("semantic transition selector must return a string")
    selected = None if value == CONTINUE else labels[value]
    usage = result.metadata.get("usage", {})
    usage = usage if isinstance(usage, dict) else {}
    return TransitionDecision(
        candidates=candidate_map,
        selected_rule_id=selected,
        source_node_id=node.id,
        read_set_node_ids=[item.id for item in read_nodes],
        read_set_hash=hashlib.sha256(encoded.encode()).hexdigest(),
        model=str(result.metadata.get("model", "default")),
        prompt_hash=hashlib.sha256(instruction.encode()).hexdigest(),
        engine_metadata=result.metadata,
        usage=LLMUsage(
            input_tokens=int(usage.get("input_tokens", 0) or 0),
            output_tokens=int(usage.get("output_tokens", 0) or 0),
            reasoning_tokens=int(usage.get("reasoning_tokens", 0) or 0),
        ),
        started_at=started_at,
        finished_at=time.time(),
    )


__all__ = ["choose"]
