"""Regression coverage for short queries with supporting REPL inputs."""

from __future__ import annotations

from helpers import StubLLM

from rlmflow import Flow, LLMOutput, PlanQuery
from rlmflow.graph.nodes import PLANNING_ACTION


def repl(code: str) -> str:
    return f"```repl\n{code}\n```"


def test_input_backed_tasks_remain_free_form():
    calls = []

    def reply(messages):
        calls.append(messages)
        system = messages[0]["content"]
        assert "## Turn Guidance" not in system
        assert "Inspection turn only" not in system
        assert "Post-inspection orchestration turn" not in system
        assert "## REPL and Delegation" in system
        assert messages[-1]["content"] == PLANNING_ACTION
        return repl("""
text = INPUTS["context"]
answer = {"characters": len(text), "contains_requirement": "required" in text}
finish(answer)
""".strip())

    flow = Flow(StubLLM(reply))
    root = flow.start(
        "inspect the context",
        inputs={"context": "A required supporting value."},
        max_depth=1,
        max_iters=2,
        output_schema={
            "type": "object",
            "properties": {
                "characters": {"type": "integer"},
                "contains_requirement": {"type": "boolean"},
            },
            "required": ["characters", "contains_requirement"],
        },
    )

    try:
        result = flow.run(root)
    finally:
        flow.runtime.close_repls()

    outputs = [node for node in root.walk() if isinstance(node, LLMOutput)]
    assert len(calls) == 1
    assert len(outputs) == 1
    assert sum(isinstance(node, PlanQuery) for node in root.transcript()) == 1
    assert result == {"characters": 28, "contains_requirement": True}
