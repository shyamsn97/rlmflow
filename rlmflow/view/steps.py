"""Read a finished graph back as the ordered run it was.

Every node is stamped when it is created, so the order a run happened in is just
those stamps sorted. Ties keep tree order, because the sort is stable and
``walk`` yields parents before children.
"""

from __future__ import annotations

from dataclasses import dataclass

from rlmflow.graph.nodes import (
    AgentStart,
    Node,
    Submission,
    TransitionDecision,
    VerificationFailure,
)

DETAIL_LIMIT = 1600


@dataclass
class Step:
    """One node, with what it was and where it sits in the run."""

    index: int
    node: Node
    agent: str
    kind: str
    summary: str
    detail: str
    elapsed: float

    @property
    def title(self) -> str:
        return f"{self.agent} · {self.kind}"


def timeline(root: Node) -> list[Node]:
    """Every node from ``root`` down, in the order it was created."""
    return sorted(root.walk(), key=lambda node: node.created_at)


def summarize(node: Node, limit: int = 96) -> str:
    """A single line describing what this node did."""
    if isinstance(node, AgentStart):
        text = f"start {node.config.path} · model {node.config.model}"
    elif isinstance(node, TransitionDecision):
        choice = (
            node.candidates[node.selected_rule_id]
            if node.selected_rule_id is not None
            else "continue"
        )
        text = f"selected {choice}"
    elif isinstance(node, VerificationFailure):
        text = f"{node.kind} rejected: {node.feedback}"
    elif isinstance(node, Submission):
        text = f"submission {_stringify(node.result)}"
    elif node.type == "done_output":
        text = f"done {_stringify(getattr(node, 'result', None) or node.content)}"
    elif node.type == "error_output":
        text = f"{getattr(node, 'error', 'error')}: {node.content}"
    else:
        text = getattr(node, "code", "") or node.content or ""
    text = " ".join(str(text).split())
    return text[: limit - 1] + "…" if len(text) > limit else text


def node_detail(node: Node, limit: int = DETAIL_LIMIT) -> str:
    """The content worth reading for this node, trimmed."""
    blocks: list[str] = []
    if isinstance(node, AgentStart):
        blocks.append(f"query:\n{node.content}")
        if node.args:
            blocks.append(f"args: {', '.join(node.args)}")
    code = getattr(node, "code", "")
    if code:
        blocks.append(f"code:\n{code}")
    if node.content and not isinstance(node, AgentStart):
        blocks.append(f"content:\n{node.content}")
    if isinstance(node, TransitionDecision):
        choices = "\n".join(
            f"- {rule_id}: {condition}" for rule_id, condition in node.candidates.items()
        )
        blocks.append(
            f"choices:\n{choices}\n"
            f"selected: {node.selected_rule_id or 'continue'}\n"
            f"read set: {', '.join(node.read_set_node_ids)}"
        )
    if isinstance(node, VerificationFailure):
        blocks.append(
            f"kind: {node.kind}\n"
            f"candidate: {_stringify(node.candidate)}\n"
            f"feedback: {node.feedback}"
        )
    result = getattr(node, "result", None)
    if result is not None:
        blocks.append(f"result:\n{_stringify(result)}")
    error = getattr(node, "error", "")
    if error:
        blocks.append(f"error: {error}")
    text = "\n\n".join(blocks).strip()
    return text[:limit] + "\n… (truncated)" if len(text) > limit else text


def _stringify(value: object) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else repr(value)


def steps(root: Node) -> list[Step]:
    """The run as a list of steps, ready to render."""
    ordered = timeline(root)
    start = ordered[0].created_at if ordered else 0.0
    return [
        Step(
            index=index,
            node=node,
            agent=node.parent_agent.config.path if node.parent_agent else "root",
            kind=node.type,
            summary=summarize(node),
            detail=node_detail(node),
            elapsed=node.created_at - start,
        )
        for index, node in enumerate(ordered)
    ]
