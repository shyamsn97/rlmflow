from helpers import StubLLM
from pydantic import BaseModel

from rlmflow import AgentConfig, ErrorOutput, ExecOutput, Flow, LLMOutput, start


def test_messages_are_a_pure_projection_of_the_transcript():
    flow = Flow(StubLLM(lambda _messages: "unused"))
    root = start("query")
    root.append(LLMOutput(content="assistant", code="print('x')")).append(
        ExecOutput(content="observation")
    )
    before = [node.id for node in root.transcript()]

    messages = flow.messages(root.frontier)

    assert [node.id for node in root.transcript()] == before
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert messages[-1]["content"] == "observation"


def test_errors_are_projected_as_user_observations():
    flow = Flow(StubLLM(lambda _messages: "unused"))
    root = start("query")
    root.append(ErrorOutput(content="NameError: missing"))

    assert flow.messages(root.frontier)[-1]["role"] == "user"
    assert flow.messages(root.frontier)[-1]["content"].endswith("NameError: missing")


def test_explicit_output_schema_is_in_system_prompt():
    class Answer(BaseModel):
        value: str

    flow = Flow(StubLLM(lambda _messages: "unused"))
    root = start("query", output_schema=Answer.model_json_schema())

    assert '"value"' in flow.messages(root.frontier)[0]["content"]


def test_prompt_documents_the_complete_builtin_repl_api():
    flow = Flow(StubLLM(lambda _messages: "unused"))
    root = start("query", max_depth=1)

    prompt = flow.messages(root.frontier)[0]["content"]

    assert "# Built-in REPL API" in prompt
    for field in ("goal", "name", "inputs", "model", "output_schema", "prompt_profile"):
        assert f"`{field}`" in prompt
    assert "`await launch_subagent(...) -> AgentHandle`" in prompt
    assert "`wait_for_result()`" in prompt
    assert "Every launch starts the child in the background" in prompt
    assert "wait: bool" not in prompt
    assert "`finish(answer: object) -> NoReturn`" in prompt
    assert "done(" not in prompt
    assert "`INPUTS: dict[str, str]`" in prompt
    assert "`ENV: dict[str, object]`" in prompt
    assert "`print(...)` is the observation channel" in prompt


def test_leaf_prompt_omits_delegation_guidance():
    flow = Flow(
        StubLLM(lambda _messages: "unused"),
        config=AgentConfig(max_depth=1),
    )
    root = flow.start("root")
    leaf = start("leaf", config=root.config.child("leaf"))

    prompt = flow.messages(leaf.frontier)[0]["content"]

    assert "launch_subagent" not in prompt
    assert "AgentHandle" not in prompt
    assert "Act as an orchestrator" not in prompt
    assert "Fan out slices" not in prompt
    assert "Work directly on the assigned task" in prompt
    assert "`finish(answer: object) -> NoReturn`" in prompt
    assert "Do not catch tool exceptions" in prompt
