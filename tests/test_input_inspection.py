"""Regression coverage for short queries with supporting REPL inputs."""

from __future__ import annotations

from helpers import StubLLM

from rlmflow import Flow, LLMOutput, PlanQuery
from rlmflow.graph.nodes import EXECUTION_POLICY, FIRST_TURN_SAFEGUARD, PLANNING_ACTION


def repl(code: str) -> str:
    return f"```repl\n{code}\n```"


def test_input_backed_tasks_plan_inspect_and_act_iteratively():
    calls = []

    def reply(messages):
        calls.append(messages)
        system = messages[0]["content"]
        assert "## Turn Guidance" not in system
        assert "Inspection turn only" not in system
        assert "Post-inspection orchestration turn" not in system
        assert "Recursive Language Model" in system
        assert EXECUTION_POLICY in system
        assert PLANNING_ACTION in messages[-1]["content"]
        if len(calls) == 1:
            assert messages[-1]["content"].startswith(FIRST_TURN_SAFEGUARD)
            assert "Turn 1/" in messages[-1]["content"]
        else:
            assert FIRST_TURN_SAFEGUARD not in messages[-1]["content"]
            assert messages[-1]["content"].startswith("Turn ")
        if len(calls) == 1:
            return repl('print({"characters": len(INPUTS["context"]), "kind": "text"})')
        if len(calls) == 2:
            return repl(
                """
text = INPUTS["context"]
answer = {"characters": len(text), "contains_requirement": "required" in text}
print(answer)
""".strip()
            )
        return repl("finish(answer)")

    flow = Flow(StubLLM(reply))
    root = flow.start(
        "inspect the context",
        inputs={"context": "A required supporting value."},
        max_depth=1,
        max_iters=4,
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
    assert len(calls) == 3
    assert len(outputs) == 3
    assert sum(isinstance(node, PlanQuery) for node in root.transcript()) == 3
    assert result == {"characters": 28, "contains_requirement": True}
