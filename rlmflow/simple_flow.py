from dataclasses import dataclass, field
import uuid
from typing import Any, Literal

def new_id() -> str:
    return f"n_{uuid.uuid4().hex[:8]}"


@dataclass
class Node:
    type: str
    id: str = field(default_factory=new_id)
    agent_id: str = ""
    seq: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    children: list[Node] = field(default_factory=list)

    @property
    def terminal(self) -> bool:
        return False


def mark_injected(node: Node) -> Node:
    """Stamp controller-edit provenance on ``node.metadata`` (in place)."""
    node.metadata["injected"] = True
    return node


@dataclass
class ObservationNode(Node):
    content: str = ""


@dataclass
class ActionNode(Node):
    pass


@dataclass
class UserQuery(ObservationNode):
    type: Literal["user_query"] = "user_query"


@dataclass
class LLMOutput(ObservationNode):
    type: Literal["llm_output"] = "llm_output"
    code: str = ""


@dataclass
class ExecAction(ActionNode):
    type: Literal["exec_action"] = "exec_action"
    code: str = ""


@dataclass
class ExecOutput(ObservationNode):
    type: Literal["exec_output"] = "exec_output"
    output: str = ""


@dataclass
class SupervisingOutput(ObservationNode):
    type: Literal["supervising_output"] = "supervising_output"
    output: str = ""
    waiting_on: list[str] = field(default_factory=list)


@dataclass
class ResumeAction(ActionNode):
    type: Literal["resume_action"] = "resume_action"
    resumed_from: list[str] = field(default_factory=list)


@dataclass
class ErrorOutput(ObservationNode):
    type: Literal["error_output"] = "error_output"
    error: str = ""
    output: str = ""


@dataclass
class DoneOutput(ObservationNode):
    type: Literal["done_output"] = "done_output"
    result: str = ""
    output: str = ""

    @property
    def terminal(self) -> bool:
        return True
